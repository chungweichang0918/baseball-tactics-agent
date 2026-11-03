"""One-upload, two-branch analysis workspace with a separate field simulation."""
from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np

from ..agent.orchestrator import run_agent
from ..analysis.statcast import candidates, load_statcast, pitch_markdown
from ..config import gemini_api_key, load_config, project_path
from ..llm.gemini_client import GeminiClient
from ..vision.pose_scan import merge_fragments, run_pose_scan
from ..vision.video import probe_video
from .app import BASES, OUTS, CURVES_HELP, _supported, make_game, preview_situation
from .studio import (branch_cards, field_board, identity_warning, overview_cards, prepare_clip,
                     simulation_summary, statcast_panel, timeline_rows)
from .view import RunView

HEADER = '''<div class="studio-header"><div class="brand"><div class="brand-mark"><span>◇</span></div>
<div><strong>DIAMOND STUDIO</strong><p>棒球戰術研究室</p></div></div>
<div class="header-label"><span class="live-dot"></span> VIDEO TO INSIGHT · 棒球影像分析</div></div>
<div class="studio-intro"><div><div class="eyebrow">SEE THE PLAY. FIND THE EDGE.</div>
<h1>每個 Play，都有下一步。</h1><p>從一段精華，拆解投球與揮棒，再推演場上的每個選擇。</p></div>
<div class="workflow"><b>01 上傳精華</b><i>／</i><span>02 分析動作</span><i>／</i><span>03 推演戰術</span></div></div>'''
DEFAULT_GAME = (7, "bottom", 1, ["1"], 0, 0, 0, "offense", "auto", "", "", "")


PALETTE = {"bg": "#0d1519", "block": "#142025", "raised": "#1a282d", "input": "#101b1f", "border": "#29373b",
           "text": "#e8eeea", "subdued": "#9baaa5", "label": "#afbeb5", "accent": "#c8f36b", "accent_soft": "#263a29"}


def studio_theme() -> gr.themes.Base:
    """深色主題。淺色與深色模式都設成同一組顏色，表格、檔案列表這類內建元件才不會在淺色模式下變白。"""
    c = PALETTE
    values = {
        "body_background_fill": c["bg"], "body_text_color": c["text"], "body_text_color_subdued": c["subdued"],
        "background_fill_primary": c["block"], "background_fill_secondary": c["raised"],
        "border_color_primary": c["border"], "border_color_accent": c["accent"], "border_color_accent_subdued": "#56713f",
        "color_accent": c["accent"], "color_accent_soft": c["accent_soft"], "code_background_fill": c["input"],
        "link_text_color": "#d1ef9b", "link_text_color_hover": "#e6ffb8", "link_text_color_visited": "#d1ef9b",
        "block_background_fill": c["block"], "block_border_color": c["border"], "block_label_background_fill": c["block"],
        "block_label_border_color": c["border"], "block_label_text_color": c["label"], "block_title_text_color": "#e8eee6",
        "block_info_text_color": c["subdued"], "panel_background_fill": c["block"], "panel_border_color": c["border"],
        "accordion_text_color": "#c9d6cb", "stat_background_fill": c["accent_soft"],
        "input_background_fill": c["input"], "input_background_fill_focus": c["input"], "input_background_fill_hover": c["input"],
        "input_border_color": "#344346", "input_border_color_focus": "#7e9b58", "input_border_color_hover": "#46594d",
        "input_placeholder_color": "#6f817a",
        "table_even_background_fill": c["block"], "table_odd_background_fill": "#17262b",
        "table_border_color": c["border"], "table_text_color": "#dbe5dc",
        "checkbox_background_color": c["input"], "checkbox_background_color_selected": c["accent"],
        "checkbox_border_color": "#56695e", "checkbox_border_color_selected": c["accent"],
        "checkbox_label_background_fill": "#1b2b2d", "checkbox_label_background_fill_hover": "#223538",
        "checkbox_label_background_fill_selected": "#30442c", "checkbox_label_border_color": "#3b4e43",
        "checkbox_label_border_color_selected": "#7e9b58", "checkbox_label_text_color": "#c7d6c7",
        "checkbox_label_text_color_selected": "#e0f5be",
        "button_primary_background_fill": c["accent"], "button_primary_background_fill_hover": "#d9ff8c",
        "button_primary_text_color": "#172119", "button_primary_border_color": c["accent"],
        "button_secondary_background_fill": "#213136", "button_secondary_background_fill_hover": "#2a4143",
        "button_secondary_text_color": "#e6eee6", "button_secondary_border_color": "#354846",
        "slider_color": c["accent"], "loader_color": c["accent"],
        "error_background_fill": "#3a1f1f", "error_border_color": "#7a3a3a", "error_text_color": "#ffd9d6",
    }
    theme = gr.themes.Base(primary_hue="lime", neutral_hue="slate", font=["Segoe UI", "Microsoft JhengHei", "sans-serif"])
    allowed = _supported(theme.set, **values, **{f"{k}_dark": v for k, v in values.items()})
    return theme.set(**allowed)


def runs_root() -> Path:
    return project_path(load_config()["paths"]["runs"])


def section(n: str, title: str, subtitle: str) -> str:
    return f'<div class="section-heading"><span><span class="section-number">{n}</span>{title}</span><small>{subtitle}</small></div>'


def frame_at(path, timestamp):
    if not path:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, int(float(timestamp) * fps)), max(0, count - 1)))
        ok, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()


def upload_video(path):
    if not path:
        return (gr.update(value=None, visible=False), "尚未載入影片", 0, 15,
                gr.update(choices=[], value=None), gr.update(maximum=15, value=0), None)
    try:
        meta = probe_video(path)
    except (ValueError, FileNotFoundError) as e:
        raise gr.Error(str(e)) from e
    edges = [0, *meta.scene_cuts, meta.frame_count]
    shots = []
    for i, (a, b) in enumerate(zip(edges, edges[1:]), 1):
        if (b-a)/meta.fps < .5:
            continue
        start, end = a/meta.fps, min(b/meta.fps, a/meta.fps+60)
        shots.append((f"鏡頭 {i:02d} · {start:.1f}–{end:.1f} 秒", f"{start:.6f},{end:.6f}"))
    info = f"**{meta.width} × {meta.height}** · {meta.fps:.1f} fps · {meta.duration_s:.1f} 秒"
    if meta.scene_cuts:
        info += f"\n\n偵測到 {len(meta.scene_cuts)} 處可能切鏡。請選連續鏡頭，避免把重播算成另一球。"
    else:
        info += "\n\n選取完整動作：準備姿勢 → 出手／揮棒 → 收尾。"
    if meta.duration_s > 60:
        info += "\n\n已選取前 60 秒；可在片段設定改選其他區間。"
    return (gr.update(value=path, visible=True), info, 0, round(min(meta.duration_s, 60), 3),
            gr.update(choices=shots, value=None), gr.update(maximum=round(meta.duration_s, 2), value=0), frame_at(path, 0))


def select_shot(value):
    return tuple(map(float, value.split(","))) if value else (gr.update(), gr.update())


def identify_people(video, start, end):
    if not video:
        raise gr.Error("請先上傳影片")
    try:
        clip, _ = prepare_clip(video, start, end, runs_root() / "clips")
        meta = probe_video(clip)
        scan, _ = run_pose_scan(clip, meta.fps, meta.frame_count)
        scan = merge_fragments(scan)
        images, choices = [], [("自動選擇", -1)]
        cap = cv2.VideoCapture(str(clip))
        try:
            for tid, track in scan.tracks.items():
                if track.coverage < .2:
                    continue
                idx = int(np.nanargmax(track.conf))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(scan.frame_indices[idx]))
                ok, frame = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = track.boxes[idx].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (107, 243, 200), 3)
                cv2.putText(frame, f"ID {tid}", (max(0, x1), max(25, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, .8, (107, 243, 200), 2)
                images.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"人物 #{tid} · 出現 {track.coverage:.0%}"))
                choices.append((f"人物 #{tid}", tid))
        finally:
            cap.release()
        return (images, gr.update(choices=choices, value=-1), gr.update(choices=choices, value=-1),
                f"找到 {len(images)} 位可追蹤人物。請對照框線指定投手／打者；同一人物不能充當兩個角色。")
    except Exception as e:
        raise gr.Error(f"人物辨識失敗：{e}") from e


def reset_people():
    return [], gr.update(choices=[("自動選擇", -1)], value=-1), gr.update(choices=[("自動選擇", -1)], value=-1), "片段已變更，請重新辨識人物。"


def pitch_choices(pitches, *game_values):
    """候選球清單：與目前賽況越吻合越前面，標出哪裡不同，讓使用者自己確認是哪一球。"""
    try:
        game = make_game(*game_values)
    except ValueError:
        return [(p.label(), p.row) for p in pitches[:30]]
    return [(("✓ 賽況相符 · " if not diff else f"與賽況不同：{'、'.join(diff)} · ") + p.label(), p.row)
            for p, diff in candidates(pitches, game)]


def import_statcast(path, *game_values):
    if not path:
        return [], gr.update(choices=[], value=None), "尚未匯入。", None, statcast_panel(None)
    try:
        pitches = load_statcast(path)
    except (ValueError, UnicodeDecodeError, OSError) as e:
        raise gr.Error(f"Statcast CSV 讀取失敗：{e}") from e
    choices = pitch_choices(pitches, *game_values)
    matched = sum(label.startswith("✓") for label, _ in choices)
    note = (f"讀到 {len(pitches)} 球，其中 {matched} 球與目前賽況相符。" if matched else
            f"讀到 {len(pitches)} 球，沒有與目前賽況完全相符的；清單依吻合程度排序。") + "請對照影片選出那一球。"
    return pitches, gr.update(choices=choices, value=None), note, None, statcast_panel(None)


def refresh_pitches(pitches, current, *game_values):
    if not pitches:
        return gr.update()
    choices = pitch_choices(pitches, *game_values)
    return gr.update(choices=choices, value=current if any(v == current for _, v in choices) else None)


def select_pitch(pitches, row):
    pitch = next((p for p in pitches or [] if p.row == row), None)
    info = pitch.to_dict() if pitch else None
    return info, statcast_panel(info)


def apply_pitch_context(pitches, row):
    """把選定那一球的局數、球數、壘況、投打慣用手帶進賽況；CSV 沒有的項目保持不動。"""
    pitch = next((p for p in pitches or [] if p.row == row), None)
    if not pitch:
        raise gr.Error("請先從候選清單選一球")
    def keep(value):
        return gr.update() if value is None else value
    bases = pitch.bases
    hand = {"R": "R", "L": "L"}
    return (keep(pitch.inning), keep(pitch.half), keep(pitch.outs),
            gr.update() if bases is None else [k for k, on in zip("123", bases) if on],
            keep(pitch.balls), keep(pitch.strikes),
            keep(hand.get(pitch.get("p_throws") or "")), keep(hand.get(pitch.get("stand") or "")))


def _snapshot(views, active, status, offset, finished=False):
    p, b = views.get("pitcher", RunView()), views.get("batter", RunView())
    labels = {"pitcher": "投球分析", "batter": "打擊分析", "auto": "自動分析"}
    current = views.get(active, RunView())
    report = "\n\n---\n\n".join(f"## {labels[k]}\n\n{v.report_md()}" for k, v in views.items())
    if views:
        report = f"> {identity_warning(views)}\n\n" + report
    reviews = "\n\n".join(f"### {labels[k]}\n{v.review_md()}" for k, v in views.items())
    files = [f for v in views.values() for f in v.files()]
    return (status, overview_cards(views), branch_cards(p, "pitcher"), branch_cards(b, "batter"),
            identity_warning(views), timeline_rows(views, offset), p.keyframes or [], b.keyframes or [],
            p.curves, b.curves, report or "分析完成後，這裡會整理可追溯的投打報告。", reviews, files or None,
            current.plan_md(), list(current.messages), gr.update(selected="report") if finished else gr.update())


def analyze_workspace(video, start, end, mode, pitcher_id, batter_id, *args):
    game_values, statcast, demo_fault, no_cache = args[:-3], args[-3], args[-2], args[-1]
    if not video:
        raise gr.Error("先上傳一段棒球影片，再開始分析")
    if mode == "full" and pitcher_id is not None and pitcher_id >= 0 and pitcher_id == batter_id:
        raise gr.Error("投手與打者不能指定相同人物；請更正人物編號或切換單一分支")
    try:
        game = make_game(*game_values)
        # Validate before clearing the previous result.
        start, end = float(start), float(end)
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or end-start > 60:
            raise ValueError("片段需介於 0–60 秒，結束時間必須大於開始時間")
    except (ValueError, TypeError) as e:
        raise gr.Error(str(e)) from e
    views = {}
    yield _snapshot(views, "pitcher", "準備分析片段…", start)
    try:
        clip, offset = prepare_clip(video, start, end, runs_root() / "clips")
    except Exception as e:
        yield _snapshot(views, "pitcher", f"片段準備失敗：{e}", start)
        return
    session = runs_root() / "studio" / uuid4().hex[:12]
    branches = ("pitcher", "batter") if mode == "full" else (mode,)
    for n, focus in enumerate(branches, 1):
        view = RunView()
        views[focus] = view
        selected = pitcher_id if focus == "pitcher" else batter_id
        branch_game = game.model_copy(update={"analysis_target": focus,
                                              "subject_track_id": int(selected) if selected is not None and selected >= 0 else None})
        name = "投球" if focus == "pitcher" else "打擊"
        yield _snapshot(views, focus, f"{n}/{len(branches)} · {name}分析準備中", offset)
        try:
            client = GeminiClient(cache_mode="off" if no_cache else None)
            last = 0.0
            for ev in run_agent(clip, branch_game, client=client, demo_fault=demo_fault, run_dir=session / focus):
                view.apply(ev)
                if ev.kind in ("report_delta", "progress") and time.monotonic()-last < .2:
                    continue
                last = time.monotonic()
                yield _snapshot(views, focus, f"{n}/{len(branches)} · {name}分析 · {view.status}", offset)
        except Exception as e:
            view.status = f"{name}分析失敗：{type(e).__name__}: {e}"
            view.report_header = view.status
            view.finished = True
            view.final = {"aborted": True}
            yield _snapshot(views, focus, view.status, offset)
    failures = sum(bool(v.final and v.final.get("aborted")) for v in views.values())
    limited = any(e.get("status") in ("degraded", "failed") for v in views.values() for e in v.evidence.values())
    status = "分析結束 · 部分分支無法完成，請查看報告" if failures else "分析完成 · 部分證據受限，請確認關鍵幀" if limited else "分析完成 · 投打報告已就緒"
    # Persist source timing and identity limitations alongside both branch reports.
    import json
    session.mkdir(parents=True, exist_ok=True)
    summary = session / "session.json"
    summary.write_text(json.dumps({"source": Path(video).name, "clip_start_s": offset, "clip_end_s": end,
                                   "mode": mode, "identity_note": identity_warning(views),
                                   "fielding": "manual_scenario_not_video_tracking",
                                   "timeline": timeline_rows(views, offset), "statcast": statcast},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    result = list(_snapshot(views, focus, status, offset, finished=True))
    if statcast:
        result[10] += "\n\n---\n\n" + pitch_markdown(statcast)
    combined = session / "combined_report.md"
    combined.write_text(f"# Diamond Studio 整合報告\n\n分析片段：{offset:.3f}–{end:.3f} 秒\n\n"
                        f"賽況：{game.describe_zh()}\n\n{result[10]}\n", encoding="utf-8")
    result[12] = [str(combined)] + (result[12] or []) + [str(summary)]
    yield tuple(result)


def build_workspace():
    from .app import CSS
    theme = studio_theme()
    with gr.Blocks(**_supported(gr.Blocks.__init__, title="Diamond Studio · 棒球戰術研究室", css=CSS, theme=theme,
                              analytics_enabled=False)) as demo:
        gr.HTML(HEADER)
        with gr.Row(elem_id="studio-layout"):
            with gr.Column(scale=1, min_width=290, elem_id="controls"):
                with gr.Group(elem_classes="panel"):
                    gr.HTML(section("01", "載入比賽精華", "YOUR FOOTAGE"))
                    video = gr.File(label="拖曳影片到這裡，或點擊上傳", file_types=["video"], type="filepath", elem_id="upload")
                    gr.Markdown("MP4 / MOV / AVI · 建議選取 10–15 秒完整動作", elem_classes="helper")
                    video_info = gr.Markdown("尚未載入影片", elem_classes="helper")
                    with gr.Accordion("片段選取與鏡頭切換", open=False):
                        shot = gr.Dropdown(label="依切鏡選取", choices=[], interactive=True)
                        with gr.Row():
                            start = gr.Number(label="開始（秒）", value=0, minimum=0)
                            end = gr.Number(label="結束（秒）", value=15, minimum=0)
                        gr.Markdown("單次最多 60 秒。時間軸顯示原影片秒數。", elem_classes="helper")
                    mode = gr.Radio([("完整投打", "full"), ("投球", "pitcher"), ("打擊", "batter")], value="full", label="分析模式")
                    gr.Markdown("完整模式依序分析投手與打者；守備頁提供獨立情境模擬。", elem_id="mode-note")
                with gr.Group(elem_classes="panel"):
                    gr.HTML(section("02", "比賽情境", "GAME CONTEXT"))
                    situation = gr.HTML(preview_situation(*DEFAULT_GAME), elem_id="situation")
                    with gr.Accordion("調整局數、球數與壘況", open=False):
                        inning = gr.Slider(1, 15, value=7, step=1, label="局數")
                        half = gr.Radio([("上半", "top"), ("下半", "bottom")], value="bottom", label="半局")
                        outs = gr.Radio(OUTS, value=1, label="出局數")
                        bases = gr.CheckboxGroup(BASES, value=["1"], label="壘上跑者")
                        balls = gr.Radio([0, 1, 2, 3], value=0, label="壞球")
                        strikes = gr.Radio([0, 1, 2], value=0, label="好球")
                        score = gr.Number(value=0, precision=0, label="比分差（我方 − 對方）")
                        role = gr.Radio([("我方進攻", "offense"), ("我方防守", "defense")], value="offense", label="我方角色")
                    target = gr.State("auto")
                    with gr.Accordion("球員設定與人物辨識", open=False):
                        hand = gr.Radio([("未知", ""), ("右投", "R"), ("左投", "L")], value="", label="投球手")
                        batter_hand = gr.Radio([("未知", ""), ("右打", "R"), ("左打", "L")], value="", label="打擊方向")
                        people_btn = gr.Button("辨識片段中的人物", size="sm")
                        people = gr.Gallery(label="人物編號預覽", columns=1, height=220, object_fit="contain")
                        people_note = gr.Markdown("多人轉播建議先辨識，對照框線指定角色。", elem_classes="helper")
                        pitcher_id = gr.Dropdown(choices=[("自動選擇", -1)], value=-1, label="投手人物", interactive=True)
                        batter_id = gr.Dropdown(choices=[("自動選擇", -1)], value=-1, label="打者人物", interactive=True)
                    with gr.Accordion("匯入 Statcast 逐球數據", open=False):
                        gr.Markdown("到 Baseball Savant 的 Statcast Search 查出那場比賽，下載逐球 CSV。"
                                    "球速、轉速、擊球初速以這份資料為準。", elem_classes="helper")
                        statcast_file = gr.File(label="Statcast CSV", file_types=[".csv"], type="filepath", elem_id="statcast-upload")
                        statcast_note = gr.Markdown("尚未匯入。", elem_classes="helper")
                        pitch_pick = gr.Dropdown(label="對應影片中的那一球", choices=[], interactive=True)
                        apply_pitch = gr.Button("把這一球的局數、球數與壘況帶入賽況", size="sm")
                    pitches = gr.State([])
                    statcast = gr.State(None)
                    notes = gr.Textbox(label="觀察重點（選填）", placeholder="例如：留意出手節奏與打者跨步", lines=2)
                run_btn = gr.Button("開始分析精華  ↗", variant="primary", size="lg", elem_id="run-button")
                stop_btn = gr.Button("停止分析", size="sm")
                with gr.Accordion("執行設定", open=False):
                    demo_fault = gr.Checkbox(label="展示 Critic 退回修正流程", value=False)
                    no_cache = gr.Checkbox(label="重新呼叫 Gemini，不使用快取", value=False)
                gr.Markdown("● Gemini 已設定" if gemini_api_key() else "○ 基礎分析模式 · 未設定 Gemini 金鑰，仍可執行骨架分析與證據彙整。", elem_classes="helper")
            with gr.Column(scale=3, min_width=340, elem_id="workspace"):
                gr.HTML('<div class="workspace-caption">比賽分析工作台 <small>ANALYSIS WORKSPACE</small></div>')
                status = gr.Markdown("準備就緒 · 上傳影片，開始拆解這一個 play", elem_id="status")
                player = gr.Video(label="原始精華 · 播放與重播", interactive=False, visible=False, height=340, elem_id="player")
                overview = gr.HTML(overview_cards({}))
                with gr.Tabs(elem_id="analysis-tabs") as tabs:
                    with gr.Tab("全場總覽", id="overview"):
                        overview_field = gr.HTML(field_board(bases=["1"]))
                        gr.Markdown("球場為手動情境預覽，可至 **守備模擬** 調整球路與回放進度。", elem_classes="helper")
                        quality = gr.Markdown(identity_warning({}), elem_id="quality-note")
                        with gr.Accordion("動作時間軸與逐幀預覽", open=False):
                            timeline = gr.Dataframe(headers=["原片秒數", "分支", "動作", "階段", "備註"], datatype=["number", "str", "number", "str", "str"], value=[], interactive=False, type="array", label="點選動作跳到對應畫面", elem_id="timeline")
                            scrub = gr.Slider(0, 15, step=.01, value=0, label="原影片時間（秒）")
                            frame = gr.Image(label="對照原始畫面", interactive=False, height=330)
                    with gr.Tab("投球分析", id="pitcher"):
                        pitch_cards = gr.HTML(branch_cards(RunView(), "pitcher"))
                        gr.Markdown("#### 投球動作關鍵幀\n準備 → 啟動 → 抬腿 → 著地 → 出手 → 收尾", elem_classes="helper")
                        pitch_gallery = gr.Gallery(label="投手關鍵幀與球軌跡", columns=3, height="auto", object_fit="contain")
                        with gr.Accordion("動作曲線與量測說明", open=False):
                            gr.Markdown(CURVES_HELP)
                            pitch_curves = gr.Image(label="投球曲線", type="filepath", interactive=False)
                    with gr.Tab("打擊分析", id="batter"):
                        bat_cards = gr.HTML(branch_cards(RunView(), "batter"))
                        gr.Markdown("#### 揮棒動作關鍵幀\n準備 → 引棒 → 跨步 → 擊球代理點 → 收棒", elem_classes="helper")
                        bat_gallery = gr.Gallery(label="打者關鍵幀", columns=3, height="auto", object_fit="contain")
                        with gr.Accordion("動作曲線與量測說明", open=False):
                            gr.Markdown(CURVES_HELP)
                            bat_curves = gr.Image(label="打擊曲線", type="filepath", interactive=False)
                    with gr.Tab("逐球數據", id="statcast"):
                        gr.Markdown("#### 動作看影片，數字看 Statcast。\n影片負責動作階段與時序；這一頁顯示你匯入並確認的那一球的官方逐球數據。", elem_classes="helper")
                        statcast_view = gr.HTML(statcast_panel(None))
                    with gr.Tab("守備模擬", id="field"):
                        gr.Markdown("#### 把下一個選擇，放上球場。\n選擇假設球路與站位，拖曳回放進度觀察球與守備員。")
                        with gr.Row():
                            scenario = gr.Dropdown([("游擊方向滾地球", "ground"), ("中外野飛球", "fly"), ("三壘側短打", "bunt")], value="ground", label="假設擊球方向", interactive=True)
                            alignment = gr.Dropdown([("標準站位", "standard"), ("內野趨前", "in"), ("雙殺準備", "double")], value="standard", label="守備配置", interactive=True)
                        field = gr.HTML(field_board(bases=["1"]))
                        progress = gr.Slider(0, 100, value=0, step=1, label="模擬回放 · 投球 0–20% / 擊球 20–60% / 回傳 60–100%")
                        with gr.Row():
                            rewind = gr.Button("回到投球", size="sm")
                            contact = gr.Button("查看擊球", size="sm")
                            catch = gr.Button("查看接球", size="sm")
                            finish = gr.Button("查看回傳", size="sm")
                        simulation = gr.Markdown(simulation_summary(make_game(*DEFAULT_GAME), "ground", "standard"))
                    with gr.Tab("分析報告", id="report"):
                        report = gr.Markdown("### 讓證據串起每一個判斷\n\n上傳影片並開始分析後，投球與打擊報告會顯示於此。\n\n每個分支保留獨立報告、關鍵幀與審查紀錄。", elem_id="report-body")
                        files = gr.File(label="下載報告與執行紀錄", file_count="multiple", interactive=False, elem_id="downloads")
                        with gr.Accordion("證據與審查歷程", open=False):
                            reviews = gr.Markdown("尚未開始審查")
                    with gr.Tab("執行紀錄", id="process"):
                        plan = gr.Markdown("開始分析後，這裡會顯示目前分支的執行步驟。", elem_id="plan")
                        chat = gr.Chatbot(label="分析步驟與工具回傳", height=450, **_supported(gr.Chatbot.__init__, type="messages"))
        gr.HTML('<div id="footnote">DIAMOND STUDIO / 棒球影像研究工作台　 ·　投打：影片證據　／　守備：假設情境　／　賽況：手動輸入</div>')
        game_inputs = [inning, half, outs, bases, balls, strikes, score, role, target, hand, batter_hand, notes]
        upload_event = video.change(upload_video, video, [player, video_info, start, end, shot, scrub, frame])
        shot.change(select_shot, shot, [start, end])
        people_outputs = [people, pitcher_id, batter_id, people_note]
        gr.on([video.change, start.change, end.change], reset_people, outputs=people_outputs)
        people_event = people_btn.click(identify_people, [video, start, end], people_outputs, concurrency_id="analysis", concurrency_limit=1)
        def seek_frame(path, timestamp):
            return frame_at(path, timestamp), gr.update(**_supported(gr.Video.__init__, playback_position=float(timestamp)))
        scrub.release(seek_frame, [video, scrub], [frame, player])
        def jump(rows, evt: gr.SelectData):
            if rows is None or len(rows) == 0:
                return gr.update()
            return float(rows[evt.index[0]][0])
        timeline.select(jump, timeline, scrub).then(seek_frame, [video, scrub], [frame, player])
        gr.on([c.change for c in game_inputs if not isinstance(c, gr.State)], preview_situation, game_inputs, situation, queue=False)
        statcast_file.change(import_statcast, [statcast_file, *game_inputs],
                             [pitches, pitch_pick, statcast_note, statcast, statcast_view], queue=False)
        gr.on([c.change for c in game_inputs if not isinstance(c, gr.State)], refresh_pitches,
              [pitches, pitch_pick, *game_inputs], pitch_pick, queue=False)
        pitch_pick.change(select_pitch, [pitches, pitch_pick], [statcast, statcast_view], queue=False)
        apply_pitch.click(apply_pitch_context, [pitches, pitch_pick],
                          [inning, half, outs, bases, balls, strikes, hand, batter_hand], queue=False)
        def update_field(s, a, t, *values):
            game = make_game(*values)
            occupied = [k for k, on in zip(["1", "2", "3"], [game.on_first, game.on_second, game.on_third]) if on]
            board = field_board(s, a, t, occupied)
            return board, board, simulation_summary(game, s, a)
        gr.on([scenario.change, alignment.change, progress.change, bases.change, outs.change], update_field,
              [scenario, alignment, progress, *game_inputs], [field, overview_field, simulation], queue=False)
        for button, pct in [(rewind, 0), (contact, 20), (catch, 60), (finish, 100)]:
            button.click(lambda value=pct: value, outputs=progress, queue=False)
        outputs = [status, overview, pitch_cards, bat_cards, quality, timeline, pitch_gallery, bat_gallery,
                   pitch_curves, bat_curves, report, reviews, files, plan, chat, tabs]
        analysis_event = run_btn.click(analyze_workspace, [video, start, end, mode, pitcher_id, batter_id,
                                                          *game_inputs, statcast, demo_fault, no_cache], outputs,
                                      concurrency_id="analysis", concurrency_limit=1, trigger_mode="once")
        stop_btn.click(lambda: "已停止更新 · 可調整片段後重新分析", outputs=status,
                       cancels=[analysis_event, people_event], queue=False)
        gr.on([video.change, start.change, end.change],
              lambda: _snapshot({}, "pitcher", "片段已更新 · 請開始新的分析", 0), outputs=outputs,
              cancels=[analysis_event, people_event], queue=False)
        # A result is bound to its uploaded source. Changing it clears all prior evidence.
        upload_event.then(lambda: _snapshot({}, "pitcher", "準備就緒 · 已更新影片，請選取片段", 0), outputs=outputs)
    # Gradio 6 moved theme/css from Blocks to launch. The launcher uses this theme.
    demo.studio_theme = theme
    return demo
