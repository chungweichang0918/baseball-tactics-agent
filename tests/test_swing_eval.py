import importlib.util
import json
import zipfile
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic import make_batter, make_pitcher, make_scan

from bbagent.analysis import swing_eval as se
from bbagent.analysis.keypoints import KP
from bbagent.vision.pose_scan import PoseTrack

ROOT = Path(__file__).resolve().parents[1]


def entry(vid, start, labels, subset="testing"):
    return {"url": f"https://www.youtube.com/watch?v={vid}", "start": start, "end": start + 2.0,
            "labels": labels, "subset": subset, "type": "fastball", "speed": 92.0}


LABELS = {
    "A1": entry("gameA", 1.0, ["swing", "hit", "strike"]),
    "A2": entry("gameA", 4.0, ["ball"]),
    "A3": entry("gameA", 7.0, ["swing", "strike"]),            # 揮空
    "A4": entry("gameA", 10.0, ["strike"]),
    "B1": entry("gameB", 1.0, ["swing", "foul"]),
    "B2": entry("gameB", 4.0, ["ball"], subset="training"),
}


def test_load_labels_from_json_and_zip(tmp_path):
    js = tmp_path / "seg.json"
    js.write_text(json.dumps(LABELS), encoding="utf-8")
    assert se.load_labels(js) == LABELS
    zp = tmp_path / "mlb-youtube-master.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(se.LABELS_IN_ZIP, json.dumps(LABELS))
    assert se.load_labels(zp) == LABELS


def test_games_and_clip_selection():
    assert se.games_by_size(LABELS) == [("gameA", 4), ("gameB", 1)]
    assert se.contact(LABELS["A1"]) and se.contact(LABELS["B1"]) and not se.contact(LABELS["A3"])
    picked = se.select_clips(LABELS, "testing", {"gameA"}, limit=2, seed=1)
    assert len(picked) == 2 and sum(se.has_swing(LABELS[k]) for k in picked) == 1   # 有揮棒與沒揮棒各一
    assert picked == se.select_clips(LABELS, "testing", {"gameA"}, limit=2, seed=1)  # 同一個 seed 結果相同
    assert se.select_clips(LABELS, "all") == sorted(LABELS)
    assert se.select_clips(LABELS, "training") == ["B2"]


def write_counter_video(path, seconds=6, fps=30):
    """每一幀的亮度等於幀號（除以 2），用來檢查切出的片段從哪一幀開始。"""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(seconds * fps):
        writer.write(np.full((48, 64, 3), i // 2, np.uint8))
    writer.release()


def test_cut_clip(tmp_path):
    src = tmp_path / "gameA.avi"
    write_counter_video(src)
    out = se.cut_clip(src, 2.0, 3.0, tmp_path / "clips" / "A1.mp4")
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    assert ok and abs(n - 30) <= 2
    assert abs(float(first.mean()) - 30) <= 3              # 第 60 幀的亮度約 30
    assert se.cut_clip(src, 2.0, 3.0, out) == out          # 已存在就直接用
    with pytest.raises(ValueError):
        se.cut_clip(src, 100.0, 101.0, tmp_path / "late.mp4")


def broadcast_scan():
    """中外野轉播的簡化版：前景的投手身形大、會投球；遠處的打者身形小、會揮棒。"""
    pitcher, _, frames, _ = make_pitcher(n_pitches=1, stride=2, track_id=1)
    batter, _, bframes, _ = make_batter(n_swings=1, stride=2, track_id=2)
    n = min(len(frames), len(bframes))
    small = PoseTrack(2, batter.boxes[:n] * 0.4 + 300, batter.conf[:n], batter.kpts[:n] * 0.4 + 300, batter.kconf[:n])
    big = PoseTrack(1, pitcher.boxes[:n], pitcher.conf[:n], pitcher.kpts[:n], pitcher.kconf[:n])
    return make_scan([big, small], frames[:n])


def test_pick_batters_and_detect_swings():
    scan = broadcast_scan()
    picks = se.pick_batters(scan)
    assert picks == {"agent": 1, "broadcast": 2}           # Agent 的規則選到投手，轉播規則選到打者
    assert se.detect_swings(scan, picks["broadcast"]) == 1
    assert se.detect_swings(scan, picks["agent"]) >= 1     # 投手的投球也會被當成揮棒：這正是要量的誤判
    assert se.detect_swings(scan, None) == 0


def test_summary_metrics():
    results = []
    for cid, truth, agent, broadcast in [("A1", True, 1, 1), ("A2", False, 1, 0), ("A3", True, 1, 0), ("A4", False, 1, 0)]:
        r = se.ClipResult(cid, LABELS[cid]["labels"], truth, se.contact(LABELS[cid]))
        r.picks = {"agent": 1, "broadcast": 2}
        r.swings = {"agent": agent, "broadcast": broadcast}
        results.append(r)
    results.append(se.ClipResult("B1", ["swing"], True, True, error="ValueError: 讀不到影片"))
    s = se.summarize(results)
    assert s["evaluated"] == 4 and s["errors"] == 1 and s["swing_share"] == 0.5
    agent, bc = s["strategies"]["agent"], s["strategies"]["broadcast"]
    assert agent["all"]["recall"] == 1.0 and agent["all"]["precision"] == 0.5     # 每段都說有揮棒
    assert bc["all"]["precision"] == 1.0 and bc["all"]["recall"] == 0.5
    assert bc["recall_contact"] == 1.0 and bc["recall_miss"] == 0.0              # 漏掉的是揮空
    md = se.summary_markdown(s, results)
    assert "評估 4 段片段" in md and "B1：ValueError" in md


def test_script_runs_end_to_end(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("eval_mlb_youtube", ROOT / "scripts" / "eval_mlb_youtube.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    videos = tmp_path / "games"
    videos.mkdir()
    write_counter_video(videos / "gameA.avi", seconds=14)
    (videos / "gameA.mp4").write_bytes((videos / "gameA.avi").read_bytes())    # 檔名用影片 id
    labels = tmp_path / "seg.json"
    labels.write_text(json.dumps(LABELS), encoding="utf-8")
    scan = broadcast_scan()
    monkeypatch.setattr(mod, "run_pose_scan", lambda *a, **k: (scan, True))
    monkeypatch.setattr(mod, "project_path", lambda p: tmp_path / p)

    args = Namespace(labels=str(labels), videos=str(videos), subset="testing", limit=None, seed=0, no_cache=False,
                     only=None, thresholds=None)
    assert mod.cmd_run(args) == 0
    out = next((tmp_path / "runs" / "eval" / "mlb_youtube").iterdir())
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluated"] == 4                       # gameA 在測試集的 4 段
    assert (out / "results.csv").read_text(encoding="utf-8-sig").count("\n") == 5
    assert "MLB-YouTube 揮棒偵測評估" in (out / "summary.md").read_text(encoding="utf-8")


def pad_to(track, k, n, track_id=None):
    """把整條軌跡往後延 k 個取樣（前面補第一幀），再補到 n 個取樣。"""
    def pad(a):
        a = np.concatenate([np.repeat(a[:1], k, axis=0), a])[:n]
        return np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)])
    return PoseTrack(track_id or track.track_id, pad(track.boxes), pad(track.conf), pad(track.kpts), pad(track.kconf))


def scaled(track, factor, offset, track_id):
    return PoseTrack(track_id, track.boxes * factor + offset, track.conf, track.kpts * factor + offset, track.kconf)


def timed_scene(batter_swings=True):
    """投手 3.6 秒出手；打者在出手後約 0.23 秒揮棒（或站著不動）；捕手在出手後 0.4 秒單手接球。"""
    pitcher, _, frames, _ = make_pitcher(n_pitches=1, stride=2, track_id=1)
    n = len(frames)
    batter, _, bframes, _ = make_batter(n_swings=1, stride=2, track_id=2)
    batter = pad_to(batter, 28, n)                                     # 擊球 2.9 秒 → 約 3.83 秒
    if not batter_swings:
        batter = pad_to(PoseTrack(2, batter.boxes[:1], batter.conf[:1], batter.kpts[:1], batter.kconf[:1]), 0, n)
    catcher = pad_to(PoseTrack(3, batter.boxes[:1], batter.conf[:1], batter.kpts[:1].copy(), batter.kconf[:1]), 0, n)
    t = frames / 60
    u = np.clip((t - 4.0) / 0.15, 0, 1)[:, None]                         # 4.0–4.15 秒左手伸出接球
    catcher.kpts[:, KP["l_wrist"]] += u * np.array([250.0, 0.0])
    tracks = [pitcher, scaled(batter, 0.4, 300, 2), scaled(catcher, 0.4, 520, 3)]
    return make_scan(tracks, frames)



def test_timed_detects_batter_swing_after_release():
    scan = timed_scene()
    assert se.release_times(scan, 1) == [pytest.approx(3.6, abs=0.05)]
    found = se.timed_detect(scan)
    assert (found.swings, found.batter, found.reason) == (1, 2, None)   # 一次揮棒，揮棒者是打者
    assert found.best["travel"] >= se.MIN_HAND_TRAVEL and found.best["wrist_ok"] == 1.0
    sig = se.swing_signature(se.build_series(scan.tracks[3], scan.times, scan.frame_indices), 3.75, 4.35)
    assert sig["both_speed"] < se.MIN_BOTH_HANDS_SPEED and not se.is_swing(sig)  # 捕手只有一隻手動


def test_timed_ignores_still_batter_and_catcher():
    still = se.timed_detect(timed_scene(batter_swings=False))
    assert still.swings == 0 and still.batter is None and still.reason == "slow_hands"
    r = se.evaluate_clip("A1", LABELS["A1"], timed_scene(batter_swings=False))
    diag = se.timed_diagnosis([r])
    assert diag["missed"] == 1 and diag["reasons"] == {"slow_hands": 1}
    assert "兩隻手沒有同時快速移動：1 段" in se.summary_markdown(se.summarize([r]), [r])
    r = se.evaluate_clip("A2", LABELS["A2"], timed_scene(batter_swings=False))
    assert r.swings["timed"] == 0 and r.predicted("agent")              # 舊規則把投手的投球當成揮棒


def test_failed_check_order():
    base = {"track": 2, "both_speed": 3.0, "travel": 0.5, "gap": 0.1, "wrist_ok": 1.0}
    assert se.failed_check(None) == "not_visible"
    assert se.failed_check(base | {"both_speed": float("nan")}) == "no_wrists"
    assert se.failed_check(base | {"both_speed": 1.0, "travel": 0.0}) == "slow_hands"   # 依序回報第一個沒過的
    assert se.failed_check(base | {"travel": 0.1}) == "short_travel"
    assert se.failed_check(base | {"gap": 0.6}) == "hands_apart"
    assert se.failed_check(base) is None


def test_thresholds_roundtrip(tmp_path):
    th = se.Thresholds(1.8, 0.1, float("inf"), "測試")
    th.save(tmp_path / "t.json")
    back = se.Thresholds.load(tmp_path / "t.json")
    assert back == th and "兩手距離 不限" in back.describe()
    assert se.failed_check({"track": 1, "both_speed": 2.0, "travel": 0.2, "gap": 0.9, "wrist_ok": 1.0}, th) is None


def scene_for(path):
    """依片段 id 給不同的合成畫面：A1、A3 有揮棒，A2、A4 沒揮棒。"""
    return timed_scene(batter_swings=Path(path).stem in ("A1", "A3"))


def test_calibrate_separates_swings():
    results = [se.evaluate_clip(cid, LABELS[cid], scene_for(f"{cid}.mp4")) for cid in ("A1", "A2", "A3", "A4")]
    th, m = se.calibrate(results, "測試")
    assert m["balanced_accuracy"] == 1.0 and th.source == "測試"
    assert [r.predicted("timed") for r in results] == [True, False, True, False]   # 校正後套用到每段


def test_calibrate_then_validate_script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("eval_mlb_youtube", ROOT / "scripts" / "eval_mlb_youtube.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    videos = tmp_path / "games"
    videos.mkdir()
    write_counter_video(videos / "gameA.mp4", seconds=14)
    labels = tmp_path / "seg.json"
    labels.write_text(json.dumps(LABELS), encoding="utf-8")
    monkeypatch.setattr(mod, "run_pose_scan", lambda path, *a, **k: (scene_for(path), True))
    monkeypatch.setattr(mod, "project_path", lambda p: tmp_path / p)
    base = dict(labels=str(labels), videos=str(videos), subset="testing", limit=0, seed=0, no_cache=False)

    assert mod.cmd_calibrate(Namespace(**base, only=["gameA"])) == 0
    saved = next((tmp_path / "runs" / "eval" / "mlb_youtube").glob("*/thresholds.json"))
    assert "gameA" in se.Thresholds.load(saved).source
    assert mod.cmd_run(Namespace(**base, only=["gameB"], thresholds=None)) == 1      # gameB 沒有下載
    assert mod.cmd_run(Namespace(**base, only=["gameA"], thresholds=str(saved))) == 0
