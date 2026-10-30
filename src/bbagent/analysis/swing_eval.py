"""用 MLB-YouTube 資料集（Piergiovanni & Ryoo, 2018）的標註驗證揮棒偵測。

資料集只有標註：每段約 7 秒的轉播片段，記錄 YouTube 網址、起訖秒數與動作標籤（swing、hit、foul…）。
影片要自己下載整場比賽，再依標註切成片段（cut_clip）。

轉播多是中外野視角：畫面中最大的人通常是投手，而每段片段都有投球。所以同時比較三種方式：
  agent      與 Agent 相同：select_subject(scan, "batter")，選分數最高的人，再用 segment_batter 找揮棒
  broadcast  排除身形最大的人（通常是投手），再從其餘人中選分數最高的，同樣用 segment_batter
  timed      轉播改良：先從投手骨架找出手時間，只看出手後 0.15–0.75 秒；
             其餘每個人中，雙手要同時快速移動、彼此靠近（握著球棒）、移動距離夠長，才算揮棒

timed 的門檻是事先依揮棒的動作特性設定的，不是用評估片段調出來的；
在同一批片段上比較只能說明改進的方向，要確認效果應換一批片段（--seed 或另一場比賽）再評估。
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..vision.pose_scan import PoseScan
from .keypoints import KP, SubjectSeries, build_series
from .segmentation import segment_pitcher
from .subject import MIN_COVERAGE, select_subject
from .swing import segment_batter

LABELS_IN_ZIP = "mlb-youtube-master/data/mlb-youtube-segmented.json"
STRATEGIES = ("agent", "broadcast", "timed")
STRATEGY_ZH = {"agent": "Agent 原本的規則（分數最高）", "broadcast": "轉播規則（排除最大的人）",
               "timed": "轉播改良（出手時間＋雙手同步）"}

# timed 策略的門檻（事先設定，見模組說明）
SWING_WINDOW_S = (0.15, 0.75)   # 出手後的這段時間內找揮棒：球約 0.4–0.5 秒到本壘，揮棒約在擊球前 0.15 秒開始
MIN_BOTH_HANDS_SPEED = 2.5      # 兩手中「較慢的那隻」也要達到這個速度（身高/秒）：只有一隻手動（捕手接球）不算
MIN_HAND_TRAVEL = 0.35          # 雙手中點在時間窗內至少移動這麼遠（身高）：骨架抖動通常遠小於此值
MAX_HAND_GAP = 0.3              # 速度最快時兩手腕距離不超過這個值（身高）：雙手握著同一支球棒


def load_labels(path: str | Path) -> dict[str, dict]:
    """讀 mlb-youtube-segmented.json；也可以直接給下載的 mlb-youtube-master.zip。"""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read(LABELS_IN_ZIP))
    return json.loads(path.read_text(encoding="utf-8"))


def video_id(entry: dict) -> str:
    return entry["url"].split("=")[-1]


def has_swing(entry: dict) -> bool:
    return "swing" in entry["labels"]


def contact(entry: dict) -> bool:
    """揮棒且碰到球（擊出或界外）。"""
    return has_swing(entry) and bool({"hit", "foul"} & set(entry["labels"]))


def games_by_size(labels: dict, subset: str = "testing") -> list[tuple[str, int]]:
    """各場比賽在指定子集中的片段數，多的在前。只下載前幾場就有足夠的片段。"""
    counts: dict[str, int] = {}
    for e in labels.values():
        if subset == "all" or e["subset"] == subset:
            counts[video_id(e)] = counts.get(video_id(e), 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def select_clips(labels: dict, subset: str = "testing", video_ids: set[str] | None = None,
                 limit: int | None = None, seed: int = 0) -> list[str]:
    """抽樣片段：有揮棒與沒揮棒各半（某一類不夠時用另一類補滿），結果依片段 id 排序。"""
    pool = [k for k, e in labels.items()
            if (subset == "all" or e["subset"] == subset) and (video_ids is None or video_id(e) in video_ids)]
    if limit is None or limit >= len(pool):
        return sorted(pool)
    rng = random.Random(seed)
    swing = sorted(k for k in pool if has_swing(labels[k]))
    other = sorted(k for k in pool if not has_swing(labels[k]))
    rng.shuffle(swing)
    rng.shuffle(other)
    half = limit // 2
    take_swing = min(len(swing), max(half, limit - len(other)))
    chosen = swing[:take_swing] + other[:limit - take_swing]
    return sorted(chosen)


def find_game_video(folder: Path, vid: str) -> Path | None:
    for ext in (".mkv", ".mp4", ".webm"):
        p = folder / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def cut_clip(src: str | Path, start: float, end: float, out: str | Path) -> Path:
    """用 OpenCV 從整場比賽切出片段（不需要 ffmpeg）。已存在就直接用。"""
    out = Path(out)
    if out.exists() and out.stat().st_size > 0:
        return out
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"讀不到影片：{src}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        written = 0
        for _ in range(round((end - start) * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        writer.release()
    finally:
        cap.release()
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{src} 在 {start:.1f}–{end:.1f} 秒讀不到畫面（影片可能比標註短）")
    tmp.replace(out)
    return out


def pick_batters(scan: PoseScan) -> dict[str, int | None]:
    """兩種策略各選出的打者編號。"""
    choice = select_subject(scan, "batter")
    picks: dict[str, int | None] = {"agent": choice.track_id, "broadcast": None}
    if len(choice.candidates) >= 2:
        largest = max(choice.candidates, key=lambda c: c.size)
        rest = [c for c in choice.candidates if c.track_id != largest.track_id]   # 已依分數排序
        picks["broadcast"] = rest[0].track_id
    else:
        picks["broadcast"] = choice.track_id
    return picks


def detect_swings(scan: PoseScan, track_id: int | None) -> int:
    """回傳偵測到的揮棒次數；沒有可分析的人時回傳 0。"""
    if track_id is None:
        return 0
    series = build_series(scan.tracks[track_id], scan.times, scan.frame_indices)
    return len(segment_batter(series).cycles)


def release_times(scan: PoseScan, pitcher_id: int) -> list[float]:
    """投手的出手時間。分段找不到時，用手腕速度最快的時刻代替。"""
    series = build_series(scan.tracks[pitcher_id], scan.times, scan.frame_indices)
    cycles = segment_pitcher(series).cycles
    if cycles:
        return [c.times["release"] for c in cycles]
    speed = series.max_speed("wrist")
    return [float(series.times[int(np.nanargmax(speed))])] if np.isfinite(speed).any() else []


REASON_ZH = {
    "few_people": "可分析的人不到 2 個",
    "no_release": "找不到投手出手時間",
    "not_visible": "出手後的時間窗內沒有其他人（可能切換了鏡頭）",
    "no_wrists": "時間窗內手腕骨架點缺漏（遠處的人信心太低）",
    "slow_hands": "兩隻手沒有同時快速移動",
    "short_travel": "雙手移動距離不夠",
    "hands_apart": "兩手沒有靠在一起",
}
CHECKS = ("slow_hands", "short_travel", "hands_apart")


def swing_signature(series: SubjectSeries, t0: float, t1: float) -> dict | None:
    """時間窗內的揮棒特徵；這個人不在時間窗內時回傳 None。

    wrist_ok 是兩個手腕都有骨架點的取樣比例；兩手都沒有資料時 both_speed 為 NaN。
    """
    m = (series.times >= t0) & (series.times <= t1)
    present = m & np.isfinite(series.energy)
    if present.sum() < 3:
        return None
    lw, rw = series.pos[m, KP["l_wrist"]], series.pos[m, KP["r_wrist"]]
    ok = np.isfinite(lw).all(axis=1) & np.isfinite(rw).all(axis=1)
    sig = {"track": series.track_id, "both_speed": float("nan"), "travel": 0.0, "gap": float("nan"),
           "wrist_ok": round(float(ok.sum() / m.sum()), 2)}
    both = np.fmin(series.speed[m, KP["l_wrist"]], series.speed[m, KP["r_wrist"]])
    if not np.isfinite(both).any():
        return sig
    k = int(np.nanargmax(both))
    hands = ((lw + rw) / 2)[ok]
    sig["both_speed"] = round(float(both[k]), 2)
    if len(hands) >= 2:
        sig["travel"] = round(float(np.max(np.linalg.norm(hands - hands[0], axis=1))) / series.body_h, 3)
    sig["gap"] = round(float(np.linalg.norm(lw[k] - rw[k])) / series.body_h, 3)
    return sig


def failed_check(sig: dict | None) -> str | None:
    """依序檢查三個條件，回傳第一個沒通過的；全部通過回傳 None。"""
    if sig is None:
        return "not_visible"
    if not np.isfinite(sig["both_speed"]):
        return "no_wrists"
    if sig["both_speed"] < MIN_BOTH_HANDS_SPEED:
        return "slow_hands"
    if sig["travel"] < MIN_HAND_TRAVEL:
        return "short_travel"
    if not (np.isfinite(sig["gap"]) and sig["gap"] <= MAX_HAND_GAP):
        return "hands_apart"
    return None


def is_swing(sig: dict | None) -> bool:
    return failed_check(sig) is None


def _closeness(sig: dict | None) -> tuple:
    """離「判斷為揮棒」有多近：通過的檢查越多越近，同樣多時兩手速度越快越近。"""
    fail = failed_check(sig)
    passed = len(CHECKS) if fail is None else (CHECKS.index(fail) if fail in CHECKS else -1)
    speed = sig["both_speed"] if sig is not None and np.isfinite(sig["both_speed"]) else -1.0
    return passed, speed


@dataclass
class TimedResult:
    swings: int                       # 判斷為有揮棒的投球數
    batter: int | None                # 揮棒者的編號
    releases: list[float] = field(default_factory=list)
    reason: str | None = None         # 沒判斷成揮棒時，最接近通過的人卡在哪一關（REASON_ZH 的鍵）
    best: dict | None = None          # 最接近通過的人的特徵值


def timed_detect(scan: PoseScan) -> TimedResult:
    """身形最大的人當作投手；在每次出手後的時間窗內，檢查其餘每個人有沒有揮棒。"""
    people = [t for t in scan.tracks.values() if t.coverage >= MIN_COVERAGE]
    if len(people) < 2:
        return TimedResult(0, None, reason="few_people")
    pitcher = max(people, key=lambda t: float(np.nanmedian(t.heights)))
    others = [build_series(t, scan.times, scan.frame_indices) for t in people if t.track_id != pitcher.track_id]
    releases = release_times(scan, pitcher.track_id)
    if not releases:
        return TimedResult(0, None, reason="no_release")
    result = TimedResult(0, None, releases=[round(r, 2) for r in releases])
    best_hit = 0.0
    for r in releases:
        sigs = [swing_signature(s, r + SWING_WINDOW_S[0], r + SWING_WINDOW_S[1]) for s in others]
        hits = [g for g in sigs if is_swing(g)]
        if hits:
            result.swings += 1
            top = max(hits, key=lambda g: g["travel"])
            if top["travel"] > best_hit:
                best_hit, result.batter, result.best = top["travel"], top["track"], top
        elif result.swings == 0:
            closest = max(sigs, key=_closeness)
            if result.best is None or _closeness(closest) > _closeness(result.best):
                result.best = closest
    if result.swings == 0:
        result.reason = failed_check(result.best)
    return result


@dataclass
class ClipResult:
    clip_id: str
    labels: list[str]
    truth: bool
    contact: bool
    tracks: int = 0
    picks: dict[str, int | None] = field(default_factory=dict)
    swings: dict[str, int] = field(default_factory=dict)
    timed: dict = field(default_factory=dict)     # 轉播改良的診斷：出手時間、卡在哪一關、特徵值
    error: str | None = None

    def predicted(self, strategy: str) -> bool:
        return self.swings.get(strategy, 0) > 0

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_clip(clip_id: str, entry: dict, scan: PoseScan) -> ClipResult:
    r = ClipResult(clip_id, entry["labels"], has_swing(entry), contact(entry), tracks=len(scan.tracks))
    r.picks = pick_batters(scan)
    r.swings = {s: detect_swings(scan, r.picks[s]) for s in ("agent", "broadcast")}
    t = timed_detect(scan)
    r.swings["timed"], r.picks["timed"] = t.swings, t.batter
    r.timed = {"releases": t.releases, "reason": t.reason, "best": t.best}
    return r


def confusion(results: list[ClipResult], strategy: str, only=None) -> dict:
    """only：只計算符合條件的片段，例如 lambda r: r.contact。"""
    rows = [r for r in results if r.error is None and (only is None or only(r))]
    tp = sum(r.truth and r.predicted(strategy) for r in rows)
    fp = sum(not r.truth and r.predicted(strategy) for r in rows)
    fn = sum(r.truth and not r.predicted(strategy) for r in rows)
    tn = sum(not r.truth and not r.predicted(strategy) for r in rows)
    n = len(rows)

    def ratio(a, b):
        return round(a / b, 3) if b else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision and recall else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": ratio(tp + tn, n),
            "precision": precision, "recall": recall, "f1": f1}


def summarize(results: list[ClipResult]) -> dict:
    ok = [r for r in results if r.error is None]
    return {
        "clips": len(results), "evaluated": len(ok), "errors": len(results) - len(ok),
        "swing_share": round(sum(r.truth for r in ok) / len(ok), 3) if ok else None,
        "strategies": {s: {
            "all": confusion(results, s),
            "recall_contact": confusion(results, s, lambda r: r.contact)["recall"],
            "recall_miss": confusion(results, s, lambda r: r.truth and not r.contact)["recall"],
        } for s in STRATEGIES},
        "same_pick": sum(r.picks.get("agent") == r.picks.get("broadcast") for r in ok),
        "timed_diagnosis": timed_diagnosis(ok),
    }


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return round(float(np.median(vals)), 3) if vals else None


def timed_diagnosis(results: list[ClipResult]) -> dict:
    """轉播改良漏掉的揮棒卡在哪一關，以及有揮棒與沒揮棒片段的特徵值中位數。"""
    missed = [r for r in results if r.truth and not r.predicted("timed")]
    reasons: dict[str, int] = {}
    for r in missed:
        key = r.timed.get("reason") or "unknown"
        reasons[key] = reasons.get(key, 0) + 1

    def medians(rows):
        best = [r.timed.get("best") or {} for r in rows]
        return {k: _median([b.get(k) for b in best]) for k in ("both_speed", "travel", "gap", "wrist_ok")}

    return {"missed": len(missed), "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "swing": medians([r for r in results if r.truth]), "no_swing": medians([r for r in results if not r.truth])}


def summary_markdown(summary: dict, results: list[ClipResult]) -> str:
    def pct(x):
        return "—" if x is None else f"{x:.1%}"

    lines = [
        "# MLB-YouTube 揮棒偵測評估",
        "",
        f"評估 {summary['evaluated']} 段片段（失敗 {summary['errors']} 段），其中有揮棒的占 {pct(summary['swing_share'])}。"
        f"Agent 原本的規則與轉播規則選到同一個人的片段有 {summary['same_pick']} 段。",
        "",
        "| 選打者的方式 | 準確率 | 精確率 | 召回率 | F1 | 擊中球的召回率 | 揮空的召回率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, v in summary["strategies"].items():
        a = v["all"]
        lines.append(f"| {STRATEGY_ZH[s]} | {pct(a['accuracy'])} | {pct(a['precision'])} | {pct(a['recall'])} | "
                     f"{pct(a['f1'])} | {pct(v['recall_contact'])} | {pct(v['recall_miss'])} |")
    lines += [
        "",
        "- 準確率：判斷「有沒有揮棒」對的比例。只看準確率會被樣本比例影響，所以也列精確率與召回率。",
        "- 精確率：說有揮棒的片段中，真的有揮棒的比例。投手的投球動作被誤當成揮棒時會降低。",
        "- 召回率：真的有揮棒的片段中，被偵測到的比例。",
        "- 擊中球／揮空：把有揮棒的片段再分成擊出或界外（碰到球）與揮空，看哪一種比較容易漏掉。",
        "- 轉播改良的門檻是事先設定的；在同一批片段上比較只能看改進方向，"
        "要確認效果請換一批片段（--seed 或另一場比賽）再評估。",
        "",
        "限制：轉播多是中外野視角、常有切換鏡頭與重播，與本專案設計的側拍差很多；"
        "標註的「swing」也包含半揮棒等判定，骨架不一定看得出來。這份數字只能說明在轉播畫面上的表現。",
    ]
    diag = summary.get("timed_diagnosis")
    if diag:
        lines += ["", "## 轉播改良的診斷", "",
                  f"漏掉的揮棒共 {diag['missed']} 段，最接近通過的人卡在：", ""]
        lines += [f"- {REASON_ZH.get(k, k)}：{n} 段" for k, n in diag["reasons"].items()] or ["- （沒有漏掉）"]
        lines += ["", "最接近通過的人的特徵值（中位數）：", "",
                  "| | 兩手中較慢那隻的速度（身高/秒） | 雙手移動距離（身高） | 兩手距離（身高） | 手腕有骨架點的比例 |",
                  "|---|---|---|---|---|"]
        for key, name in (("swing", "有揮棒的片段"), ("no_swing", "沒揮棒的片段")):
            m = diag[key]
            lines.append(f"| {name} | {m['both_speed']} | {m['travel']} | {m['gap']} | {m['wrist_ok']} |")
        lines += ["", f"目前的門檻：速度 ≥ {MIN_BOTH_HANDS_SPEED}、移動距離 ≥ {MIN_HAND_TRAVEL}、兩手距離 ≤ {MAX_HAND_GAP}。"]
    errors = [r for r in results if r.error]
    if errors:
        lines += ["", "## 失敗的片段", ""] + [f"- {r.clip_id}：{r.error}" for r in errors]
    return "\n".join(lines) + "\n"
