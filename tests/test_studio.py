from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import pytest

from bbagent.schemas import AgentEvent
from bbagent.ui.studio import (branch_cards, field_board, identity_warning, overview_cards,
                              prepare_clip, timeline_rows)
from bbagent.ui.view import RunView


def result(view, tool, data, status="ok"):
    view.apply(AgentEvent(stage="executor", kind="tool_result", title=tool,
                         payload={"tool": tool, "data": data, "status": status}))


def test_measured_cards_do_not_invent_missing_values():
    v = RunView()
    result(v, "pitching_metrics", {"summary": {"delivery_time_s": {"mean": 1.234, "n": 2}}})
    html = branch_cards(v, "pitcher")
    assert "1.23" in html and "2 次有效動作" in html
    assert html.count('metric-value">—') == 2
    result(v, "pitching_metrics", {"summary": {"delivery_time_s": {"mean": 999}}}, "failed")
    assert "999" not in branch_cards(v, "pitcher")
    assert "情境" in overview_cards({}) and "非影片追蹤" in overview_cards({})


def test_same_subject_warning_and_html_escaping():
    p, b = RunView(), RunView()
    for v in (p, b):
        result(v, "coarse_pose_scan", {"subject": {"track_id": 7, "reason": "<script>alert(1)</script>"}})
    assert "同一人物" in identity_warning({"pitcher": p, "batter": b})
    assert "<script>" not in branch_cards(p, "pitcher")
    result(b, "coarse_pose_scan", {"subject": {"track_id": 8}})
    assert "同一人物" not in identity_warning({"pitcher": p, "batter": b})


def test_timeline_uses_source_time_and_preserves_branches():
    p, b = RunView(), RunView()
    result(p, "segment_motions", {"cycles": [{"index": 1, "times": {"release": 2.4, "set": None}}]})
    result(b, "segment_motions", {"cycles": [{"index": 1, "times": {"contact": 2.1}}]})
    rows = timeline_rows({"pitcher": p, "batter": b}, 10.5)
    assert [(r[0], r[1]) for r in rows] == [(12.6, "打擊"), (12.9, "投球")]


def test_timeline_note_only_on_first_row_of_each_cycle():
    p = RunView()
    result(p, "segment_motions", {"cycles": [
        {"index": 1, "times": {"release": 2.4, "first_move": 1.2, "set": None}, "warnings": ["無法判定前腳著地"]},
        {"index": 2, "times": {"release": 6.0, "first_move": 5.1}, "warnings": []}]})
    rows = timeline_rows({"pitcher": p})
    assert [(r[0], r[4]) for r in rows] == [(1.2, "無法判定前腳著地"), (2.4, ""), (5.1, "候選動作"), (6.0, "")]


def test_theme_keeps_tables_dark_in_light_mode():
    from bbagent.ui.workspace import PALETTE, studio_theme

    theme = studio_theme()
    for name in ("table_even_background_fill", "table_odd_background_fill", "background_fill_primary",
                 "input_background_fill"):
        light, dark = getattr(theme, name), getattr(theme, f"{name}_dark")
        assert light == dark and light.startswith("#") and light.lower() not in ("#ffffff", "#fff")
    assert theme.body_background_fill == PALETTE["bg"]


def test_controls_do_not_scroll_separately_from_page():
    from bbagent.ui.app import CSS

    controls = [rule for rule in CSS.split("}") if "#controls{" in rule.replace(" ", "")]
    assert controls and not any("sticky" in r or "overflow-y:auto" in r for r in controls)


@pytest.mark.parametrize("scenario", ["ground", "fly", "bunt"])
def test_simulation_valid_svg_and_progress(scenario):
    first = field_board(scenario, "standard", 0, ["1"])
    last = field_board(scenario, "in", 100, ["3"])
    assert first != last and "情境模擬 · 非實測" in first
    svg = last[last.index('<svg'):last.index('</svg>')+6]
    ElementTree.fromstring(svg)
    assert "接球／回傳" in last and "100%" in last


@pytest.fixture
def clip_video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
    for i in range(40):
        writer.write(np.full((64, 96, 3), i*5, dtype=np.uint8))
    writer.release()
    return path


def test_clip_frame_bounds_cache_and_original_preserved(clip_video, tmp_path):
    original = clip_video.read_bytes()
    clip, offset = prepare_clip(str(clip_video), 1.23, 2.3, tmp_path / "clips")
    assert offset == 1.2 and clip_video.read_bytes() == original
    cap = cv2.VideoCapture(str(clip))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 11
        ok, frame = cap.read()
        assert ok and frame.mean() == pytest.approx(60, abs=2)
    finally:
        cap.release()
    assert prepare_clip(str(clip_video), 1.23, 2.3, tmp_path / "clips")[0] == clip
    assert prepare_clip(str(clip_video), 0, 4, tmp_path / "clips")[0] == clip_video


@pytest.mark.parametrize("start,end", [(2, 1), (-1, 2), (0, 61), (5, 6), (0, float('nan'))])
def test_invalid_clip_rejected(clip_video, tmp_path, start, end):
    with pytest.raises(ValueError):
        prepare_clip(str(clip_video), start, end, tmp_path / "clips")


def test_full_workspace_keeps_results_separate(clip_video, tmp_path, monkeypatch):
    from bbagent.ui import workspace as w
    calls = []
    def fake_agent(video, game, **kw):
        calls.append((game.analysis_target, game.subject_track_id, kw["run_dir"]))
        branch = game.analysis_target
        tool = "pitching_metrics" if branch == "pitcher" else "batting_metrics"
        key = "per_pitch" if branch == "pitcher" else "per_swing"
        yield AgentEvent(stage="executor", kind="tool_result", title=tool,
                         payload={"tool": tool, "status": "ok", "data": {key: [{"index": 1}], "summary": {}}})
        yield AgentEvent(stage="system", kind="final", title="完成", content=branch, payload={"passed": True})
    monkeypatch.setattr(w, "run_agent", fake_agent)
    monkeypatch.setattr(w, "runs_root", lambda: tmp_path / "runs")
    statcast = {"label": "2017-11-01 · 7 局下", "values": [
        {"key": "release_spin_rate", "label": "轉速", "unit": "rpm", "source": "Statcast 追蹤", "value": 2400.0},
        {"key": "launch_speed", "label": "擊球初速", "unit": "mph", "source": "Statcast 追蹤", "value": None}]}
    outputs = list(w.analyze_workspace(str(clip_video), 0, 4, "full", 3, 5, *w.DEFAULT_GAME, statcast, False, False))
    assert [x[:2] for x in calls] == [("pitcher", 3), ("batter", 5)]
    assert calls[0][2] != calls[1][2]
    final = outputs[-1]
    assert len(final) == 16 and "投球分析" in final[10] and "打擊分析" in final[10]
    assert Path(final[12][-1]).name == "session.json"
    assert "2400 rpm" in final[10] and "擊球初速" not in final[10]
    assert "2400 rpm" in Path(final[12][0]).read_text(encoding="utf-8")


def test_workspace_continues_after_branch_error(clip_video, tmp_path, monkeypatch):
    from bbagent.ui import workspace as w
    def fake_agent(video, game, **kw):
        if game.analysis_target == "pitcher":
            raise RuntimeError("missing pitcher")
        yield AgentEvent(stage="system", kind="final", title="完成", content="batter result", payload={"passed": True})
    monkeypatch.setattr(w, "run_agent", fake_agent)
    monkeypatch.setattr(w, "runs_root", lambda: tmp_path / "runs")
    final = list(w.analyze_workspace(str(clip_video), 0, 4, "full", -1, -1, *w.DEFAULT_GAME, None, False, False))[-1]
    assert "部分分支無法完成" in final[0]
    assert "missing pitcher" in final[10] and "batter result" in final[10]


def test_manual_subject_selection_overrides_ranking(monkeypatch):
    from types import SimpleNamespace
    from bbagent.agent import tools
    from bbagent.analysis.subject import SubjectCandidate, SubjectChoice
    from bbagent.schemas import GameState
    scan = SimpleNamespace(tracks={1: object(), 8: object()}, params={"stride": 1}, frame_indices=[0, 1])
    candidates = [SubjectCandidate(1, 1, .8, 3, 10), SubjectCandidate(8, .8, .2, 2, 2)]
    monkeypatch.setattr(tools, "run_pose_scan", lambda *a, **kw: (scan, True))
    monkeypatch.setattr(tools, "merge_fragments", lambda s: s)
    monkeypatch.setattr(tools, "select_subject", lambda *a: SubjectChoice(1, True, "largest", list(candidates)))
    game = GameState(inning=1, half="top", outs=0, our_role="offense", subject_track_id=8)
    state = SimpleNamespace(game=game, focus="batter", video="clip.avi", meta=SimpleNamespace(fps=30, frame_count=2), runtime={})
    result = tools._coarse_pose_scan(tools.ToolContext(state, None, lambda message: None), {})
    assert result.status == "ok" and state.runtime["track_id"] == 8
    assert result.data["subject"]["candidates"][0]["track_id"] == 8
    game.subject_track_id = 99
    result = tools._coarse_pose_scan(tools.ToolContext(state, None, lambda message: None), {})
    assert result.status == "failed" and "99" in result.summary
