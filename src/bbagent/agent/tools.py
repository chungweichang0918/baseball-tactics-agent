"""ToolRegistry：Agent 能呼叫的工具。每個工具都是確定性的 Python 函式（Gemini 工具除外），
回傳統一的 ToolResult，中間物件放進 state.runtime 給後續工具使用。

工具不丟例外：失敗時回傳 status="failed" 與原因，讓 Planner 決定怎麼處理。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..analysis.batting_metrics import CONSISTENCY_THRESHOLDS, METRIC_INFO as BATTING_INFO, compute_batting_metrics
from ..analysis.keypoints import build_series
from ..analysis.pitching_metrics import compute_pitching_metrics
from ..analysis.plots import plot_motion_curves
from ..analysis.segmentation import PHASE_ZH, PITCH_PHASES, segment_pitcher
from ..analysis.statcast import evidence_lines
from ..analysis.subject import select_subject
from ..analysis.swing import SWING_PHASE_ZH, SWING_PHASES, segment_batter
from ..analysis.tipping import compare_pitches
from ..llm.gemini_client import GeminiClient, model_name
from ..schemas import ToolResult
from ..vision.camera_motion import boxes_from_scan, estimate_camera_motion
from ..vision.pose_scan import merge_fragments, run_pose_scan
from ..vision.video import probe_video
from .state import FOCUS_ZH, AgentState
from ..tools.ball_track import track_ball
from ..tools.gemini_inspect import gemini_inspect
from ..tools.keyframes import render_keyframes
from ..tools.re24 import re24_evaluate
from ..tools.scout_overview import scout_overview

MAX_INSPECT_IMAGES = 6


@dataclass
class ToolContext:
    state: AgentState
    client: GeminiClient
    progress: Callable[[str], None]


@dataclass
class ToolSpec:
    name: str
    description: str                 # 給 Planner 看
    args_doc: str
    requires: tuple[str, ...]        # 必須先成功的工具
    critical: bool                   # 失敗時要不要觸發重新規劃
    run: Callable[[ToolContext, dict], ToolResult]
    planner_visible: bool = True     # probe_video、scout_overview 由 Orchestrator 在規劃前固定執行
    focus: tuple[str, ...] = ("pitcher", "batter")   # 適用的分析分支


class MissingRuntime(RuntimeError):
    pass


def _need(ctx: ToolContext, key: str) -> Any:
    if key not in ctx.state.runtime:
        raise MissingRuntime(key)
    return ctx.state.runtime[key]


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _metric_lines(m: dict, per_key: str) -> list[str]:
    unit = "球" if per_key == "per_pitch" else "次"
    lines = []
    for key, s in m["summary"].items():
        per = "、".join(_fmt(p[key]) for p in m[per_key])
        spread = f"，標準差 {s['std']}" if s["std"] is not None else ""
        lines.append(f"{s['name']}：平均 {s['mean']} {s['unit']}{spread}（各{unit} {per}）")
    return lines + [f"※ {n}" for n in m["notes"]]


# ---------- 偵察 ----------

def _probe_video(ctx: ToolContext, args: dict) -> ToolResult:
    try:
        meta = probe_video(ctx.state.video)
    except (FileNotFoundError, ValueError) as e:
        return ToolResult(status="failed", summary="讀不到影片", error=str(e))
    ctx.state.meta = meta
    summary = f"{meta.width}x{meta.height}，{meta.fps:.2f} fps，{meta.frame_count} 幀（{meta.duration_s:.1f} 秒）"
    if meta.scene_cuts:
        summary += f"\n⚠️ 偵測到場景切換（第 {meta.scene_cuts} 幀），可能含特寫或重播"
    return ToolResult(status="degraded" if meta.scene_cuts else "ok", summary=summary, data=meta.to_dict())


def _scout_overview(ctx: ToolContext, args: dict) -> ToolResult:
    return scout_overview(ctx.state.video, ctx.state.meta, client=ctx.client, model=model_name("fast"))


def _statcast_pitch(ctx: ToolContext, args: dict) -> ToolResult:
    info = ctx.state.statcast
    if not info:
        return ToolResult(status="failed", summary="沒有匯入 Statcast 逐球數據", error="使用者沒有選定對應的那一球")
    have = [v for v in info["values"] if v["value"] is not None]
    return ToolResult(status="ok" if have else "degraded", summary="\n".join(evidence_lines(info)), data=info)


# ---------- 視覺 ----------

def _progress(ctx: ToolContext, label: str) -> Callable[[int, int], None]:
    """每 10% 回報一次進度。"""
    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = done * 100 // max(total, 1)
        if pct // 10 != last[0] // 10 or done == total:
            last[0] = pct
            ctx.progress(f"{label}：{done}/{total} 幀（{pct}%）")

    return progress


def _coarse_pose_scan(ctx: ToolContext, args: dict) -> ToolResult:
    meta = ctx.state.meta
    scan, hit = run_pose_scan(ctx.state.video, meta.fps, meta.frame_count, progress=_progress(ctx, "姿勢模型"))
    n_raw = len(scan.tracks)
    scan = merge_fragments(scan)
    who = FOCUS_ZH[ctx.state.focus]
    choice = select_subject(scan, ctx.state.focus)
    requested = ctx.state.game.subject_track_id
    if requested is not None:
        candidate = next((c for c in choice.candidates if c.track_id == requested), None)
        if candidate is None:
            return ToolResult(status="failed", summary=f"指定人物 #{requested} 在此片段沒有足夠的追蹤資料",
                              error="請重新辨識此片段的人物，或改回自動選擇")
        choice.track_id, choice.confident = requested, True
        choice.reason = "使用者依人物預覽指定"
        choice.candidates = [candidate] + [c for c in choice.candidates if c.track_id != requested]
    ctx.state.runtime.update(scan=scan, subject_choice=choice, track_id=choice.track_id)
    data = {"cache_hit": hit, "stride": scan.params["stride"], "samples": len(scan.frame_indices),
            "tracks_raw": n_raw, "tracks": len(scan.tracks), "subject": choice.to_dict()}
    head = f"追蹤到 {len(scan.tracks)} 個人（{'命中快取' if hit else '已重新計算'}，每 {scan.params['stride']} 幀取 1）"
    if choice.track_id is None:
        return ToolResult(status="failed", summary=head, data=data, error=f"找不到可分析的{who}：{choice.reason}")
    best = choice.candidates[0]
    summary = (f"{head}\n選定編號 {choice.track_id} 為{who}：出現 {best.coverage:.0%}、大小 {best.size:.2f}、"
               f"手腕活動量 {best.activity:.2f}；{choice.reason}")
    return ToolResult(status="ok" if choice.confident else "degraded", summary=summary, data=data)


def _camera_motion(ctx: ToolContext, args: dict) -> ToolResult:
    scan = _need(ctx, "scan")
    cm = estimate_camera_motion(ctx.state.video, ctx.state.meta.fps, boxes_from_scan(scan))
    ctx.state.runtime["camera_motion"] = cm
    if cm.moving is None:
        return ToolResult(status="degraded", summary=f"背景特徵不足，無法判斷鏡頭是否移動（有效 {cm.valid_pairs}/{cm.pairs} 組）",
                          data=cm.to_dict())
    detail = f"每秒平移約畫面對角線的 {cm.p80_shift:.1%}、每秒縮放約 {cm.p80_scale:.1%}（第 80 百分位）"
    summary = ("鏡頭有明顯移動：" if cm.moving else "鏡頭固定：") + detail
    return ToolResult(status="ok", summary=summary, data=cm.to_dict())


def _segment_motions(ctx: ToolContext, args: dict) -> ToolResult:
    scan = _need(ctx, "scan")
    track_id = _need(ctx, "track_id")
    state = ctx.state
    series = build_series(scan.tracks[track_id], scan.times, scan.frame_indices)
    if state.focus == "batter":
        seg = segment_batter(series, batting_side=state.game.batter_hand)
        side = {"R": "右打", "L": "左打"}.get(seg.batting_side or "", "打擊方向未指定")
        lines = [f"打者（{side}）｜偵測到 {len(seg.cycles)} 次揮棒｜修正左右標反 {series.lr_fixes} 幀"]
        phases, zh, unit, missing = SWING_PHASES, SWING_PHASE_ZH, "次揮棒", "影片中找不到揮棒動作"
    else:
        seg = segment_pitcher(series, throwing_side=state.game.pitcher_hand)
        hand = "右投" if seg.throwing_side == "R" else "左投"
        source = "使用者指定" if seg.side_source == "user" else "依骨架判斷"
        lines = [f"{hand}（{source}）｜偵測到 {len(seg.cycles)} 球｜修正左右標反 {series.lr_fixes} 幀"]
        phases, zh, unit, missing = PITCH_PHASES, PHASE_ZH, "球", "影片中找不到完整的投球動作"
    state.runtime.update(series=series, seg=seg)
    for c in seg.cycles:
        parts = [f"{zh[ph]} {c.times[ph]:.2f}s" for ph in phases if c.times[ph] is not None]
        lines.append(f"第 {c.index} {unit}：" + "、".join(parts))
        lines += [f"  ⚠️ {w}" for w in c.warnings]
    lines += [f"⚠️ {w}" for w in seg.warnings]
    data = seg.to_dict() | {"focus": state.focus}
    artifacts = []
    try:   # 曲線圖只給 UI 看，畫失敗不影響分段結果；找不到動作時也畫，方便判斷原因
        artifacts.append(str(plot_motion_curves(series, seg, state.run_dir / "motion_curves.png", figsize=(12, 8))))
    except Exception as e:  # noqa: BLE001
        lines.append(f"⚠️ 動作曲線圖繪製失敗：{type(e).__name__}: {e}")
    if not seg.cycles:
        return ToolResult(status="failed", summary="\n".join(lines), data=data, artifacts=artifacts, error=missing)
    degraded = seg.warnings or any(c.warnings for c in seg.cycles)
    return ToolResult(status="degraded" if degraded else "ok", summary="\n".join(lines), data=data,
                      artifacts=artifacts)


def _pitching_metrics(ctx: ToolContext, args: dict) -> ToolResult:
    series, seg = _need(ctx, "series"), _need(ctx, "seg")
    state = ctx.state
    m = compute_pitching_metrics(series, seg, state.camera_view)
    if state.camera_view != "side":
        m["notes"].append("非側拍視角：出手時間、手臂角度等姿勢類指標可靠度低")
    if state.camera_moving:
        m["notes"].append("鏡頭在移動：速度類訊號受到干擾，指標可靠度降低")
    state.runtime["metrics"] = m
    reliable = state.camera_view == "side" and not state.camera_moving
    return ToolResult(status="ok" if reliable else "degraded", summary="\n".join(_metric_lines(m, "per_pitch")),
                      data=m)


def _batting_metrics(ctx: ToolContext, args: dict) -> ToolResult:
    series, seg = _need(ctx, "series"), _need(ctx, "seg")
    state = ctx.state
    m = compute_batting_metrics(series, seg, state.camera_view)
    if state.camera_moving:
        m["notes"].append("鏡頭在移動：速度類訊號受到干擾，指標可靠度降低")
    state.runtime["metrics"] = m
    reliable = state.camera_view == "side" and not state.camera_moving
    return ToolResult(status="ok" if reliable else "degraded", summary="\n".join(_metric_lines(m, "per_swing")),
                      data=m)


def _swing_compare(ctx: ToolContext, args: dict) -> ToolResult:
    m = _need(ctx, "metrics")
    t = compare_pitches(m["per_swing"], CONSISTENCY_THRESHOLDS, BATTING_INFO, unit_word="次揮棒",
                        notes=["只比較揮棒動作的一致性，沒有球的位置與球種，不能判斷打者對哪種球較弱"])
    lines = [f"比較 {t['n_pitches']} 次揮棒（信心上限：{t['confidence']}）"]
    lines += [f"- 差異：{f['text']}" for f in t["flags"]] or ["- 各次揮棒的時間與姿勢沒有超過門檻的差異"]
    lines += [f"※ {n}" for n in t["notes"]]
    return ToolResult(status="degraded" if t["n_pitches"] < 2 else "ok", summary="\n".join(lines), data=t)


def _track_ball(ctx: ToolContext, args: dict) -> ToolResult:
    state = ctx.state
    speed_ok = state.camera_view == "side" and not state.camera_moving
    result = track_ball(state.video, state.meta, _need(ctx, "series"), _need(ctx, "seg"), _need(ctx, "metrics"),
                        state.run_dir, speed_ok, progress=_progress(ctx, "細掃找球"))
    state.runtime["ball"] = result.data
    return result


def _tipping_compare(ctx: ToolContext, args: dict) -> ToolResult:
    m = _need(ctx, "metrics")
    t = compare_pitches(m["per_pitch"])
    lines = [f"比較 {t['n_pitches']} 球（信心上限：{t['confidence']}）"]
    lines += [f"- 疑似差異：{f['text']}" for f in t["flags"]] or ["- 各球動作沒有超過門檻的差異"]
    lines += [f"※ {n}" for n in t["notes"]]
    return ToolResult(status="degraded" if t["n_pitches"] < 2 else "ok", summary="\n".join(lines), data=t)


def _render_keyframes(ctx: ToolContext, args: dict) -> ToolResult:
    result = render_keyframes(ctx.state.video, _need(ctx, "scan"), _need(ctx, "track_id"), _need(ctx, "seg"),
                              ctx.state.run_dir)
    ctx.state.runtime["keyframes"] = result.data.get("keyframes", [])
    return result


def _gemini_inspect(ctx: ToolContext, args: dict) -> ToolResult:
    keyframes = _need(ctx, "keyframes")
    if ctx.state.focus == "batter":
        order = {"contact": 0, "foot_plant": 1, "load": 2, "stance": 3, "follow_through": 4}
    else:
        order = {"release": 0, "set": 1, "peak_leg_lift": 2, "foot_strike": 3, "follow_through": 4}
    chosen = sorted(keyframes, key=lambda k: (order[k["phase"]], k["pitch"]))[:MAX_INSPECT_IMAGES]
    chosen.sort(key=lambda k: (k["pitch"], order[k["phase"]]))
    state = ctx.state
    context = [state.game.describe_zh()]
    if state.game.notes:
        context.append(f"補充：{state.game.notes}")
    m = state.runtime.get("metrics")
    if m:
        context += [f"{s['name']}：平均 {s['mean']} {s['unit']}" for s in m["summary"].values()]
    return gemini_inspect([k["path"] for k in chosen], args.get("focus", f"{FOCUS_ZH[state.focus]}的動作有什麼值得注意的地方"),
                          [k["caption"] for k in chosen], "\n".join(context),
                          client=ctx.client, model=model_name("pro"))


def _re24(ctx: ToolContext, args: dict) -> ToolResult:
    return re24_evaluate(ctx.state.game, args.get("options", []))


PITCHER, BATTER = ("pitcher",), ("batter",)
REGISTRY: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("probe_video", "讀取影片基本資訊並偵測場景切換", "（無）", (), True, _probe_video, False),
    ToolSpec("scout_overview", "讓 Gemini 判斷視角、主體與拍攝條件", "（無）", ("probe_video",), False,
             _scout_overview, False),
    ToolSpec("statcast_pitch", "讀取使用者匯入並確認的 Statcast 逐球數據（球速、轉速、擊球初速等）", "（無）", (), False,
             _statcast_pitch, False),
    ToolSpec("coarse_pose_scan", "用 YOLO 骨架模型掃描影片、追蹤所有人並選出分析對象（必要）", "（無）",
             ("probe_video",), True, _coarse_pose_scan),
    ToolSpec("camera_motion", "用人物以外的背景特徵點量測鏡頭是否移動（OpenCV，比 Gemini 判斷可靠）", "（無）",
             ("coarse_pose_scan",), False, _camera_motion),
    ToolSpec("segment_motions", "找出每一球的 6 個投球階段，或每次揮棒的 5 個階段（必要）", "（無）",
             ("coarse_pose_scan",), True, _segment_motions),
    ToolSpec("pitching_metrics", "計算投手指標：固定姿勢時間、到本壘時間（估）、抬腿高度、手臂角度等（投手必要）", "（無）",
             ("segment_motions",), False, _pitching_metrics, focus=PITCHER),
    ToolSpec("track_ball", "在每一球出手後逐幀找球、擬合軌跡，粗估球速與到本壘時間（側拍且鏡頭固定時才換算球速）",
             "（無）", ("pitching_metrics",), False, _track_ball, focus=PITCHER),
    ToolSpec("tipping_compare", "比較各球的動作差異，找出疑似洩漏球種的小動作（至少要 2 球）", "（無）",
             ("pitching_metrics",), False, _tipping_compare, focus=PITCHER),
    ToolSpec("batting_metrics", "計算打者指標：引棒到擊球時間、著地到擊球時間、跨步長度、頭部移動量、手速（打者必要）",
             "（無）", ("segment_motions",), False, _batting_metrics, focus=BATTER),
    ToolSpec("swing_compare", "比較各次揮棒的時間與姿勢是否一致（至少要 2 次揮棒）", "（無）",
             ("batting_metrics",), False, _swing_compare, focus=BATTER),
    ToolSpec("render_keyframes", "把每一球或每次揮棒的關鍵階段畫成疊圖（UI 顯示、給 Gemini 看）", "（無）",
             ("segment_motions",), False, _render_keyframes),
    ToolSpec("gemini_inspect", "讓 Gemini 看關鍵幀，回答一個具體的觀察重點（握球、手套、視線、手的位置等骨架看不到的細節）",
             "inspect_focus：觀察重點（一句話）", ("render_keyframes",), False, _gemini_inspect),
    ToolSpec("re24_evaluate", "用 RE24 得分期望值評估候選戰術，算出 ΔRE 與損益平衡成功率（必要）",
             "使用計畫中的 candidate_options", (), False, _re24),
]}


def planner_catalog() -> str:
    lines = []
    for s in REGISTRY.values():
        if s.planner_visible:
            req = "、".join(s.requires) or "無"
            branch = "、".join(FOCUS_ZH[f] for f in s.focus)
            lines.append(f"- {s.name}：{s.description}｜適用：{branch}｜前置：{req}｜參數：{s.args_doc}")
    return "\n".join(lines)


def dependents_of(tool: str) -> set[str]:
    """所有直接或間接依賴 tool 的工具。"""
    out: set[str] = set()
    frontier = {tool}
    while frontier:
        nxt = {s.name for s in REGISTRY.values() if set(s.requires) & frontier} - out
        out |= nxt
        frontier = nxt
    return out
