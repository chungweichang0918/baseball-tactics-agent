"""用 MLB-YouTube 資料集的標註，評估打者揮棒偵測的準確度。

資料集：https://github.com/piergiaj/mlb-youtube （只有標註，影片要自己從 YouTube 取得）
這些是 MLB 轉播畫面，有版權；只下載少量片段做自己的測試，不要放進公開的 repo。

步驟：
    1. 看要下載哪幾場（測試集片段最多的前幾場）：
       python scripts/eval_mlb_youtube.py plan --labels ~/Downloads/mlb-youtube-master.zip --games 2
    2. 把那幾場整場影片放進同一個資料夾，檔名用 YouTube 影片 id，例如 91YyEVUeO8I.mp4（.mkv、.webm 也可以）
    3. 執行評估（片段會自動切好並快取在 cache/mlb_clips/）：
       python scripts/eval_mlb_youtube.py run --labels ~/Downloads/mlb-youtube-master.zip --videos ~/mlb_games --limit 40

結果輸出到 runs/eval/mlb_youtube/<時間>/：summary.md（中文摘要）、summary.json、results.csv（每段片段）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.vision import device  # noqa: E402,F401  必須在 torch 之前 import

from bbagent.analysis.swing_eval import (STRATEGIES, ClipResult, cut_clip, evaluate_clip, find_game_video,  # noqa: E402
                                         games_by_size, has_swing, load_labels, select_clips, summarize,
                                         summary_markdown, video_id)
from bbagent.config import load_config, project_path  # noqa: E402
from bbagent.vision.pose_scan import merge_fragments, run_pose_scan  # noqa: E402
from bbagent.vision.video import probe_video  # noqa: E402


def cmd_plan(args) -> int:
    labels = load_labels(args.labels)
    games = games_by_size(labels, args.subset)
    print(f"{args.subset} 子集共 {sum(n for _, n in games)} 段片段、{len(games)} 場比賽。建議先下載下面 {args.games} 場：\n")
    for vid, n in games[:args.games]:
        clips = [e for e in labels.values() if video_id(e) == vid and (args.subset == "all" or e["subset"] == args.subset)]
        swings = sum(has_swing(e) for e in clips)
        print(f"  {vid}  {n} 段（有揮棒 {swings} 段）  https://www.youtube.com/watch?v={vid}")
    print("\n存檔時用影片 id 當檔名，例如 <影片id>.mp4，放在同一個資料夾，再執行 run。")
    return 0


def cmd_run(args) -> int:
    labels = load_labels(args.labels)
    videos = Path(args.videos).expanduser()
    available = {vid for vid, _ in games_by_size(labels, "all") if find_game_video(videos, vid)}
    if not available:
        print(f"在 {videos} 找不到任何比賽影片；檔名要是 YouTube 影片 id，例如 91YyEVUeO8I.mp4。先執行 plan 看要下載哪幾場。")
        return 1
    clip_ids = select_clips(labels, args.subset, available, args.limit, args.seed)
    if not clip_ids:
        print(f"已下載的 {len(available)} 場比賽在 {args.subset} 子集中沒有片段；試試 --subset all。")
        return 1
    cfg = load_config()
    clip_dir = project_path(cfg["paths"]["cache"]) / "mlb_clips"
    out_dir = project_path(cfg["paths"]["runs"]) / "eval" / "mlb_youtube" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"評估 {len(clip_ids)} 段片段（來自 {len(available)} 場比賽），結果寫到 {out_dir}\n")

    results: list[ClipResult] = []
    for i, cid in enumerate(clip_ids, 1):
        e = labels[cid]
        try:
            game = find_game_video(videos, video_id(e))
            clip = cut_clip(game, e["start"], e["end"], clip_dir / f"{cid}.mp4")
            meta = probe_video(clip)
            scan, _ = run_pose_scan(clip, meta.fps, meta.frame_count, use_cache=not args.no_cache)
            r = evaluate_clip(cid, e, merge_fragments(scan))
        except Exception as ex:  # noqa: BLE001  一段失敗不影響其他片段
            r = ClipResult(cid, e["labels"], has_swing(e), False, error=f"{type(ex).__name__}: {ex}")
        results.append(r)
        mark = "有揮棒" if r.truth else "沒揮棒"
        got = r.error or "、".join(f"{s} {'✔' if r.predicted(s) == r.truth else '✘'}" for s in STRATEGIES)
        print(f"[{i}/{len(clip_ids)}] {cid}（標註：{mark}，{'/'.join(e['labels']) or '無'}）→ {got}")

    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(summary, results), encoding="utf-8")
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "labels", "truth_swing", "contact", "tracks"]
                   + [f"{s}_{k}" for s in STRATEGIES for k in ("track", "swings", "correct")]
                   + ["timed_releases", "timed_reason", "timed_both_speed", "timed_travel", "timed_gap",
                      "timed_wrist_ok", "error"])
        for r in results:
            w.writerow([r.clip_id, "/".join(r.labels), r.truth, r.contact, r.tracks]
                       + [x for s in STRATEGIES for x in (r.picks.get(s), r.swings.get(s), r.predicted(s) == r.truth)]
                       + [" ".join(map(str, r.timed.get("releases", []))), r.timed.get("reason") or ""]
                       + [(r.timed.get("best") or {}).get(k, "") for k in ("both_speed", "travel", "gap", "wrist_ok")]
                       + [r.error or ""])
    print("\n" + summary_markdown(summary, results))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        s = sub.add_parser(name)
        s.add_argument("--labels", required=True, help="mlb-youtube-master.zip 或 mlb-youtube-segmented.json")
        s.add_argument("--subset", choices=["testing", "training", "all"], default="testing")
    sub.choices["plan"].add_argument("--games", type=int, default=2, help="列出片段最多的前幾場")
    run = sub.choices["run"]
    run.add_argument("--videos", required=True, help="放整場比賽影片的資料夾（檔名為 YouTube 影片 id）")
    run.add_argument("--limit", type=int, default=40, help="最多評估幾段（有揮棒與沒揮棒各半）")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-cache", action="store_true", help="重新掃描骨架，不用快取")
    args = p.parse_args()
    return cmd_plan(args) if args.cmd == "plan" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
