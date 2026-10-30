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

    args = Namespace(labels=str(labels), videos=str(videos), subset="testing", limit=None, seed=0, no_cache=False)
    assert mod.cmd_run(args) == 0
    out = next((tmp_path / "runs" / "eval" / "mlb_youtube").iterdir())
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluated"] == 4                       # gameA 在測試集的 4 段
    assert (out / "results.csv").read_text(encoding="utf-8-sig").count("\n") == 5
    assert "MLB-YouTube 揮棒偵測評估" in (out / "summary.md").read_text(encoding="utf-8")
