"""用 MLB-YouTube 資料集（Piergiovanni & Ryoo, 2018）的標註驗證揮棒偵測。

資料集只有標註：每段約 7 秒的轉播片段，記錄 YouTube 網址、起訖秒數與動作標籤（swing、hit、foul…）。
影片要自己下載整場比賽，再依標註切成片段（cut_clip）。

轉播多是中外野視角：畫面中最大的人通常是投手，而每段片段都有投球。所以同時比較兩種選打者的方式：
  agent      與 Agent 相同：select_subject(scan, "batter")，選分數最高的人
  broadcast  排除身形最大的人（通常是投手），再從其餘人中選分數最高的
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

from ..vision.pose_scan import PoseScan
from .keypoints import build_series
from .subject import select_subject
from .swing import segment_batter

LABELS_IN_ZIP = "mlb-youtube-master/data/mlb-youtube-segmented.json"
STRATEGIES = ("agent", "broadcast")


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


@dataclass
class ClipResult:
    clip_id: str
    labels: list[str]
    truth: bool
    contact: bool
    tracks: int = 0
    picks: dict[str, int | None] = field(default_factory=dict)
    swings: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def predicted(self, strategy: str) -> bool:
        return self.swings.get(strategy, 0) > 0

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_clip(clip_id: str, entry: dict, scan: PoseScan) -> ClipResult:
    r = ClipResult(clip_id, entry["labels"], has_swing(entry), contact(entry), tracks=len(scan.tracks))
    r.picks = pick_batters(scan)
    r.swings = {s: detect_swings(scan, r.picks[s]) for s in STRATEGIES}
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
    }


def summary_markdown(summary: dict, results: list[ClipResult]) -> str:
    def pct(x):
        return "—" if x is None else f"{x:.1%}"

    lines = [
        "# MLB-YouTube 揮棒偵測評估",
        "",
        f"評估 {summary['evaluated']} 段片段（失敗 {summary['errors']} 段），其中有揮棒的占 {pct(summary['swing_share'])}。"
        f"兩種策略選到同一個人的片段有 {summary['same_pick']} 段。",
        "",
        "| 選打者的方式 | 準確率 | 精確率 | 召回率 | F1 | 擊中球的召回率 | 揮空的召回率 |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {"agent": "Agent 原本的規則（分數最高）", "broadcast": "轉播規則（排除最大的人）"}
    for s, v in summary["strategies"].items():
        a = v["all"]
        lines.append(f"| {names[s]} | {pct(a['accuracy'])} | {pct(a['precision'])} | {pct(a['recall'])} | "
                     f"{pct(a['f1'])} | {pct(v['recall_contact'])} | {pct(v['recall_miss'])} |")
    lines += [
        "",
        "- 準確率：判斷「有沒有揮棒」對的比例。只看準確率會被樣本比例影響，所以也列精確率與召回率。",
        "- 精確率：說有揮棒的片段中，真的有揮棒的比例。投手的投球動作被誤當成揮棒時會降低。",
        "- 召回率：真的有揮棒的片段中，被偵測到的比例。",
        "- 擊中球／揮空：把有揮棒的片段再分成擊出或界外（碰到球）與揮空，看哪一種比較容易漏掉。",
        "",
        "限制：轉播多是中外野視角、常有切換鏡頭與重播，與本專案設計的側拍差很多；"
        "標註的「swing」也包含半揮棒等判定，骨架不一定看得出來。這份數字只能說明在轉播畫面上的表現。",
    ]
    errors = [r for r in results if r.error]
    if errors:
        lines += ["", "## 失敗的片段", ""] + [f"- {r.clip_id}：{r.error}" for r in errors]
    return "\n".join(lines) + "\n"
