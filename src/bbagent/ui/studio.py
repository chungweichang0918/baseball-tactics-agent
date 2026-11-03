"""Presentation helpers for measured evidence and explicitly hypothetical field scenarios."""
from __future__ import annotations

import hashlib
import math
from html import escape
from pathlib import Path

import cv2

from ..analysis.segmentation import PHASE_ZH
from ..analysis.swing import SWING_PHASE_ZH
from ..schemas import GameState
from ..tools.re24 import BaseOutState, get_re24
from ..vision.video import iter_frames, probe_video
from .view import RunView


def card(label, value="—", unit="", detail="等待影片分析", tone="") -> str:
    return (f'<div class="metric {tone}"><span>{escape(label)}</span>'
            f'<div class="metric-value">{escape(str(value))}<small>{escape(unit)}</small></div>'
            f'<p>{escape(detail)}</p></div>')


def evidence_data(view: RunView, tool: str) -> dict:
    evidence = view.evidence.get(tool, {})
    return evidence.get("data", {}) if evidence.get("status") != "failed" else {}


def branch_cards(view: RunView, focus: str) -> str:
    tool = "pitching_metrics" if focus == "pitcher" else "batting_metrics"
    keys = ([('delivery_time_s', '啟動 → 出手', '秒'), ('leg_lift_height', '抬腿高度', '身高'),
             ('arm_slot_deg', '出手手臂角度', '°')]
            if focus == "pitcher" else [('load_to_contact_s', '引棒 → 擊球代理點', '秒'),
                                       ('peak_hand_speed', '最大手腕速度', '身高/秒'),
                                       ('head_movement', '頭部移動量', '身高')])
    evidence = view.evidence.get(tool, {})
    data = evidence_data(view, tool)
    summary = data.get("summary", {})
    cards = []
    for key, title, unit in keys:
        item = summary.get(key, {})
        value = item.get("mean")
        detail = (f"{item.get('n', 0)} 次有效動作的平均 · "
                  + ("條件受限" if evidence.get("status") == "degraded" else "骨架估計"))
        if value is None:
            detail = "無法判定／證據不足" if view.finished else "等待影片分析"
        cards.append(card(title, f"{value:.3g}" if value is not None else "—", unit, detail))
    notes = " · ".join(data.get("notes", []))
    subject = evidence_data(view, "coarse_pose_scan").get("subject", {})
    if subject.get("track_id") is not None:
        notes = f"人物 #{subject['track_id']} · {subject.get('reason', '')}。 " + notes
    if focus == "batter":
        notes += " 擊球代理點取手腕速度峰值，並非已偵測到球棒碰球；手腕速度不是棒速。"
    return '<div class="metrics">' + ''.join(cards) + '</div><p class="evidence-note">' + escape(notes) + '</p>'


def overview_cards(views: dict[str, RunView]) -> str:
    p, b = views.get("pitcher", RunView()), views.get("batter", RunView())
    def count(view, key, list_key):
        data = evidence_data(view, key)
        return str(len(data[list_key])) if list_key in data else "—"
    balls = evidence_data(p, "track_ball").get("pitches", [])
    speeds = [x["speed_kmh_est"] for x in balls if x.get("found") and x.get("speed_kmh_est") is not None]
    return '<div class="metrics overview-metrics">' + ''.join([
        card("投球動作", count(p, "pitching_metrics", "per_pitch"), "球", "骨架分段 · 待確認人物", "lime"),
        card("揮棒動作", count(b, "batting_metrics", "per_swing"), "次", "手腕峰值 · 待確認人物"),
        card("球速粗估", f"{sum(speeds)/len(speeds):.0f}" if speeds else "—", "km/h",
             "可信軌跡的平均估值" if speeds else "需可信軌跡與固定側拍"),
        card("守備戰術板", "情境", "", "手動設定 · 非影片追蹤"),
    ]) + '</div>'


def identity_warning(views: dict[str, RunView]) -> str:
    ids = [evidence_data(views.get(k, RunView()), "coarse_pose_scan").get("subject", {}).get("track_id")
           for k in ("pitcher", "batter")]
    if ids[0] is not None and ids[0] == ids[1]:
        return (f"⚠️ 投手與打者都選到人物 #{ids[0]}。這些是同一人物的兩種動作分析，不能視為完整投打結果。"
                "請在「人物辨識」預覽編號，分別指定投手／打者後重新分析。")
    return "投打數據來自影片候選動作；請在關鍵幀確認人物。守備配置與球路為手動情境示意。"


def timeline_rows(views: dict[str, RunView], offset: float = 0) -> list:
    """每個動作階段一列；同一球（或同一次揮棒）的警告只放在它最早的那一列，避免每列重複同一段長文字。"""
    rows = []
    for focus, view in views.items():
        seg = evidence_data(view, "segment_motions")
        labels = PHASE_ZH if focus == "pitcher" else SWING_PHASE_ZH
        branch = "投球" if focus == "pitcher" else "打擊"
        for cycle in seg.get("cycles", []):
            note = "；".join(cycle.get("warnings", [])) or "候選動作"
            stamps = sorted((t, phase) for phase, t in cycle.get("times", {}).items() if t is not None)
            for k, (timestamp, phase) in enumerate(stamps):
                rows.append([round(offset + timestamp, 3), branch, cycle["index"], labels.get(phase, phase),
                             note if k == 0 else ""])
    return sorted(rows, key=lambda row: row[0])


def prepare_clip(path: str, start: float, end: float, root: Path) -> tuple[Path, float]:
    """Cache a frame-exact analysis clip. Never modifies the uploaded source."""
    start, end = float(start), float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("請設定有效片段：開始 ≥ 0，結束必須大於開始")
    if end - start > 60:
        raise ValueError("單次片段最多 60 秒，請選取一個完整 play 再分析")
    meta = probe_video(path)
    if start >= meta.duration_s or end > meta.duration_s + 0.1:
        raise ValueError(f"片段超出影片長度（{meta.duration_s:.2f} 秒）")
    first, stop = int(start * meta.fps), min(meta.frame_count, math.ceil(end * meta.fps))
    if stop - first < 2:
        raise ValueError("片段太短，請至少選取兩幀")
    if first == 0 and stop == meta.frame_count:
        return Path(path), 0.0
    source = Path(path)
    key = hashlib.sha256(f"{source.resolve()}:{source.stat().st_mtime_ns}:{source.stat().st_size}:{first}:{stop}".encode()).hexdigest()[:20]
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"clip_{key}.avi"
    if not output.exists():
        temp = root / f"clip_{key}.partial.avi"
        writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"MJPG"), meta.fps, (meta.width, meta.height))
        try:
            if not writer.isOpened():
                raise ValueError("無法建立分析片段，請確認影片編碼")
            count = 0
            for _, frame in iter_frames(path, start=first, end=stop):
                writer.write(frame)
                count += 1
            if count != stop - first:
                raise ValueError("片段解碼不完整，請重新選取或轉換影片")
        except BaseException:
            writer.release()
            temp.unlink(missing_ok=True)
            raise
        else:
            writer.release()
            temp.replace(output)
    return output, first / meta.fps


FIELDERS = {'P': (300, 315), 'C': (300, 441), '1B': (402, 310), '2B': (359, 240),
            'SS': (241, 240), '3B': (198, 310), 'LF': (150, 160), 'CF': (300, 100), 'RF': (450, 160)}
SCENARIOS = {"ground": ("游擊方向滾地球", (224, 273), "SS", (405, 324), "游擊手接球 → 傳一壘"),
             "fly": ("中外野飛球", (325, 134), "CF", (300, 218), "中外野手接球 → 回傳內野"),
             "bunt": ("三壘側短打", (258, 366), "3B", (405, 324), "三壘手趨前 → 傳一壘")}


def field_board(scenario="ground", alignment="standard", progress=0, bases=None) -> str:
    title, landing, receiver, destination, _ = SCENARIOS.get(scenario, SCENARIOS["ground"])
    t = max(0, min(100, float(progress))) / 100
    positions = dict(FIELDERS)
    for key in ("1B", "2B", "SS", "3B"):
        x, y = positions[key]
        if alignment == "in":
            positions[key] = (300 + (x-300)*0.8, y+40)
        elif alignment == "double":
            positions[key] = (300 + (x-300)*0.83, y+12)
    def mix(a, b, f):
        return a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f
    home, mound = (300, 428), (300, 315)
    if t < .2:
        ball = mix(mound, home, t/.2)
    elif t < .6:
        ball = mix(home, landing, (t-.2)/.4)
    else:
        ball = mix(landing, destination, (t-.6)/.4)
    positions[receiver] = mix(positions[receiver], landing, min(1, max(0, (t-.2)/.4)))
    phase = "投球" if t < .2 else "擊球／移動" if t < .6 else "接球／回傳"
    marks = []
    for name, (x, y) in positions.items():
        active = name == receiver
        marks.append(f'<g><circle cx="{x}" cy="{y}" r="17" fill="{"#c8f36b" if active else "#183e36"}" '
                     f'stroke="{"#e1ffa4" if active else "#689080"}"/><text x="{x}" y="{y+4}" '
                     f'text-anchor="middle" font-size="11" font-weight="700" fill="{"#172522" if active else "#e8f3ee"}">{name}</text></g>')
    for key, (x, y) in {"1": (405, 324), "2": (300, 219), "3": (195, 324)}.items():
        marks.append(f'<rect x="{x-6}" y="{y-6}" width="12" height="12" transform="rotate(45 {x} {y})" '
                     f'fill="{"#ffc978" if key in (bases or []) else "#d8e6dc"}"/>')
    return f'''<div class="field-wrap"><div class="field-heading"><span>FIELD VIEW <b>守備戰術板</b></span><em>情境模擬 · 非實測</em></div>
      <svg viewBox="0 0 600 490" role="img" aria-label="{escape(title)}，{phase}，手動情境模擬">
      <defs><pattern id="grass" width="48" height="48" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="24" height="48" fill="#25493c"/><rect x="24" width="24" height="48" fill="#214436"/></pattern></defs>
      <path d="M300 428 L47 175 Q300 -101 553 175 Z" fill="url(#grass)" stroke="#608374" stroke-width="2"/>
      <path d="M300 428 L170 298 Q300 148 430 298 Z" fill="#9d8060" opacity=".7"/>
      <path d="M300 416 L208 324 L300 232 L392 324 Z" fill="#315645"/>
      <path d="M47 175 L300 428 L553 175 M195 324 L300 219 L405 324" fill="none" stroke="#c5d6c6" stroke-width="1.5" opacity=".6"/>
      <circle cx="300" cy="315" r="17" fill="#af9874"/><path d="M293 423 h14 v8 l-7 6 -7 -6Z" fill="#fff"/>
      <path d="M300 315 L300 428 L{landing[0]} {landing[1]} L{destination[0]} {destination[1]}" fill="none"
        stroke="#d5ee97" stroke-dasharray="5 7" opacity=".7" stroke-width="2"/>
      {''.join(marks)}<circle cx="{ball[0]}" cy="{ball[1]}" r="6" fill="#fff" stroke="#ffc978" stroke-width="3"/>
      <text x="300" y="65" text-anchor="middle" fill="#99b4a5" font-size="11" letter-spacing="3">CENTER FIELD</text>
      <text x="30" y="465" fill="#c5d6c6" font-size="12">{escape(title)}</text>
      <text x="570" y="465" text-anchor="end" fill="#d4f398" font-size="12">{phase} · {int(t*100):02d}%</text>
      </svg><div class="field-legend"><span>● 守備員</span><span>● 重點守備員</span><span>○ 球</span><span>◆ 橘色壘包＝有跑者</span></div></div>'''


def simulation_summary(game: GameState, scenario: str, alignment: str) -> str:
    re = get_re24().value(BaseOutState.from_game(game))
    text = SCENARIOS.get(scenario, SCENARIOS["ground"])[4]
    alignment_note = {"standard": "標準站位", "in": "內野趨前，示意縮短處理短球的距離", "double": "雙殺準備站位，示意靠近內野的配置"}[alignment]
    return (f"**{text}**\n\n{alignment_note}。拖動進度查看投球、擊球、接球與回傳的位置變化。\n\n"
            f"目前賽況的進攻方 RE24：**{re:.3f} 分**（專案內建參考表）。\n\n"
            "守備位置、球路與移動比例由情境預設；未從影片還原，也不表示接殺、封殺或雙殺必然成功。"
            "RE24 只隨壘況與出局數改變，並非此守備配置的預測得分。")
