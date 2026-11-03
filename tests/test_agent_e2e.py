"""用合成投手與模擬的 Gemini 跑完整個 Agent：偵察 → 規劃 → 執行 → 撰寫 → 審查。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fake_gemini import FakeResponse
from synthetic import make_pitcher, make_scan

from bbagent.agent import tools as agent_tools
from bbagent.agent.orchestrator import run_agent
from bbagent.llm.gemini_client import GeminiClient
from bbagent.schemas import GameState
from bbagent.vision.pose_scan import PoseTrack

FPS = 60
OVERVIEW = {"camera_view": "side", "view_reason": "正側面", "subject": "pitcher", "camera_moving": False,
            "full_body_visible": True, "is_live_action": True, "scoreboard": {"visible": False}, "concerns": []}
PLAN = {
    "analysis_focus": "pitcher", "rationale": "一壘有人，評估盜二壘", "tactical_questions": ["要不要盜二壘"],
    "candidate_options": ["steal_2b", "sac_bunt", "ibb"],
    "steps": [
        {"id": "s1", "tool": "coarse_pose_scan", "purpose": "找投手", "depends_on": []},
        {"id": "s2", "tool": "camera_motion", "purpose": "鏡頭", "depends_on": ["s1"]},
        {"id": "s3", "tool": "segment_motions", "purpose": "分段", "depends_on": ["s1"]},
        {"id": "s4", "tool": "pitching_metrics", "purpose": "指標", "depends_on": ["s3"]},
        {"id": "s5", "tool": "tipping_compare", "purpose": "差異", "depends_on": ["s4"]},
        {"id": "s6", "tool": "render_keyframes", "purpose": "關鍵幀", "depends_on": ["s3"]},
        {"id": "s7", "tool": "gemini_inspect", "purpose": "看視線", "depends_on": ["s6"], "inspect_focus": "視線"},
        {"id": "s8", "tool": "re24_evaluate", "purpose": "RE24", "depends_on": []},
    ],
}
INSPECT = {"answer": "看不出明顯前兆。", "observations": [
    {"aspect": "視線", "finding": "固定姿勢時看向本壘", "confidence": "medium", "images": [1]}], "limitations": []}
# 證據編號：E01 probe、E02 scout、E03 粗掃、E04 鏡頭、E05 分段、E06 指標、E07 差異、E08 關鍵幀、E09 看圖、E10 RE24
REPORT = """# ⚾ 戰術建議報告
**情境**：7 局下｜一出局｜一壘有人｜球數 0-0｜平手｜我方進攻
**分析對象**：右投投手，側拍，分析 2 球

## 1. 結論
- 主建議：下一球盜二壘 [E06][E10]。
- 備案：正常強攻。

## 2. 影片觀察
| # | 觀察 | 數據 | 證據 |
|---|---|---|---|
| 1 | 到本壘時間偏慢 | 1.47 秒 | [E06] |

## 3. 數據依據（RE24）
| 戰術 | 結果 | ΔRE | 損益平衡成功率 |
|---|---|---|---|
| 盜二壘 | 成功 | +0.155 | 72.6% |
| 盜二壘 | 失敗 | -0.411 | 72.6% |

## 4. 戰術建議與推論
到本壘時間偏慢 [E06]，盜二壘成功率可望超過 72.6% [E10]。

## 5. 信心與限制
- 只分析 2 球，樣本少。
"""


class ScriptedModels:
    """依結構化輸出的型別決定回覆，不依賴呼叫順序。"""

    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config):
        name = getattr(config.response_schema, "__name__", None)
        self.calls.append(name)
        reply = {"VideoOverview": OVERVIEW, "PlanDraft": PLAN, "InspectResult": INSPECT,
                 "CriticReview": {"passed": True, "issues": []}}[name]
        return FakeResponse(json.dumps(reply, ensure_ascii=False))

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append("stream")
        for i in range(0, len(REPORT), 40):
            yield FakeResponse(REPORT[i:i + 40])


class ScriptedSDK:
    def __init__(self):
        self.models = ScriptedModels()


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (80, 60))
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8), (0, 0), 1)
    for _ in range(round(10.5 * FPS)):
        writer.write(bg)
    writer.release()
    return path


def patch_scan(monkeypatch, tracks, frames):
    scan = make_scan(tracks, frames, FPS)
    scan.params = {"stride": 2}
    monkeypatch.setattr(agent_tools, "run_pose_scan", lambda *a, **k: (scan, True))


def run(video, tmp_path, demo_fault=False):
    client = GeminiClient(cache_mode="off", cache_dir=tmp_path / "llm", sdk_client=ScriptedSDK(), sleep=lambda s: None)
    game = GameState(inning=7, half="bottom", outs=1, on_first=True, our_role="offense", pitcher_hand="R")
    events = list(run_agent(video, game, client=client, run_dir=tmp_path / "run", demo_fault=demo_fault))
    return events, client


def test_full_run(video, tmp_path, monkeypatch):
    track, t, frames, _ = make_pitcher(n_pitches=2, stride=2)
    patch_scan(monkeypatch, [track], frames)
    events, client = run(video, tmp_path)

    final = events[-1]
    assert final.kind == "final" and final.payload["passed"], final.content
    assert {e.stage for e in events} == {"system", "planner", "executor", "synthesizer", "critic"}
    results = [e for e in events if e.kind == "tool_result"]
    assert [e.payload["evidence_id"] for e in results] == [f"E{i:02d}" for i in range(1, 11)]
    assert all(e.payload["status"] != "failed" for e in results), [(e.title, e.content) for e in results]
    assert any("ibb" in e.content for e in events if e.stage == "planner")       # 不適用的戰術被移除

    report = Path(final.payload["report_path"]).read_text(encoding="utf-8")
    assert "## 6. Agent 執行摘要" in report and "Gemini 規劃" in report and "退回 0 次" in report
    trace = json.loads(Path(final.payload["trace_path"]).read_text(encoding="utf-8"))
    assert len(trace["evidence"]) == 10 and trace["plan_source"] == "llm"
    assert all(Path(p).exists() for p in final.payload["keyframes"]) and final.payload["keyframes"]
    assert client._sdk.models.calls.count("CriticReview") == 1


def test_demo_fault_triggers_reflection(video, tmp_path, monkeypatch):
    track, t, frames, _ = make_pitcher(n_pitches=2, stride=2)
    patch_scan(monkeypatch, [track], frames)
    events, _ = run(video, tmp_path, demo_fault=True)
    verdicts = [e.payload["passed"] for e in events if e.kind == "critique" and "passed" in e.payload]
    assert verdicts == [False, True]
    assert events[-1].payload["passed"]
    assert "退回 1 次" in events[-1].content and "+0.999" in events[-1].content     # 退回原因寫進摘要


def test_no_pitch_found_triggers_replan(video, tmp_path, monkeypatch):
    n = 315
    kpts = np.tile(np.linspace([400, 300], [420, 700], 17), (n, 1, 1))
    still = PoseTrack(1, np.tile([380.0, 280, 440, 720], (n, 1)), np.full(n, 0.9), kpts, np.full((n, 17), 0.9))
    patch_scan(monkeypatch, [still], np.arange(0, 2 * n, 2))
    events, _ = run(video, tmp_path)

    titles = [e.title for e in events]
    assert "重新規劃" in titles
    failed = [e for e in events if e.kind == "tool_result" and e.payload["status"] == "failed"]
    assert failed[0].payload["tool"] == "segment_motions"
    final = events[-1]
    assert not final.payload["passed"]                    # 報告引用了不存在的證據，修正上限後仍未通過
    assert "審查未通過的項目" in final.content and "重新規劃 1 次" in final.content


BATTER_PLAN = {
    "analysis_focus": "batter", "rationale": "打擊練習影片，分析打者揮棒", "tactical_questions": ["強攻還是觸擊"],
    "candidate_options": ["steal_2b", "sac_bunt"],
    "steps": [
        {"id": "s1", "tool": "coarse_pose_scan", "purpose": "找打者", "depends_on": []},
        {"id": "s2", "tool": "segment_motions", "purpose": "分段", "depends_on": ["s1"]},
        {"id": "s3", "tool": "batting_metrics", "purpose": "指標", "depends_on": ["s2"]},
        {"id": "s4", "tool": "swing_compare", "purpose": "一致性", "depends_on": ["s3"]},
        {"id": "s5", "tool": "render_keyframes", "purpose": "關鍵幀", "depends_on": ["s2"]},
        {"id": "s6", "tool": "gemini_inspect", "purpose": "看手", "depends_on": ["s5"], "inspect_focus": "手的位置"},
        {"id": "s7", "tool": "re24_evaluate", "purpose": "RE24", "depends_on": []},
    ],
}
# 證據編號：E01 probe、E02 scout、E03 粗掃、E04 分段、E05 指標、E06 一致性、E07 關鍵幀、E08 看圖、E09 RE24
BATTER_REPORT = REPORT.replace("[E06]", "[E05]").replace("[E10]", "[E09]")


def test_batter_branch_end_to_end(video, tmp_path, monkeypatch):
    import test_agent_e2e as me
    from synthetic import make_batter

    track, t, frames, _ = make_batter(n_swings=2, stride=2)
    patch_scan(monkeypatch, [track], frames)
    monkeypatch.setattr(me, "PLAN", BATTER_PLAN)
    monkeypatch.setattr(me, "OVERVIEW", OVERVIEW | {"subject": "batter"})
    monkeypatch.setattr(me, "REPORT", BATTER_REPORT)
    events, _ = run(video, tmp_path)

    results = {e.payload["tool"]: e for e in events if e.kind == "tool_result"}
    assert "pitching_metrics" not in results
    assert results["segment_motions"].payload["data"]["focus"] == "batter"
    assert "偵測到 2 次揮棒" in results["segment_motions"].content
    assert "著地到擊球時間" in results["batting_metrics"].content
    assert all(k["caption"].startswith("第 ") and "次揮棒" in k["caption"]
               for k in results["render_keyframes"].payload["data"]["keyframes"])
    final = events[-1]
    assert final.payload["passed"], final.content


def test_statcast_becomes_evidence_and_is_reviewed(video, tmp_path, monkeypatch):
    """匯入的那一球在偵察階段變成證據；報告沒有引用它時，Critic 會退回。"""
    from test_critic_rules import STATCAST
    track, t, frames, _ = make_pitcher(n_pitches=2, stride=2)
    patch_scan(monkeypatch, [track], frames)
    client = GeminiClient(cache_mode="off", cache_dir=tmp_path / "llm", sdk_client=ScriptedSDK(), sleep=lambda s: None)
    game = GameState(inning=7, half="bottom", outs=1, on_first=True, our_role="offense", pitcher_hand="R")
    events = list(run_agent(video, game, client=client, run_dir=tmp_path / "run", statcast=STATCAST))
    results = [e for e in events if e.kind == "tool_result"]
    assert results[2].payload["tool"] == "statcast_pitch" and results[2].payload["evidence_id"] == "E03"
    assert "投球速度 87.3 mph" in results[2].content
    critiques = [e for e in events if e.kind == "critique" and "passed" in e.payload]
    assert critiques and not critiques[0].payload["passed"]
    assert any(i["rule"] == "statcast" for i in critiques[0].payload["issues"])
    trace = json.loads(Path(events[-1].payload["trace_path"]).read_text(encoding="utf-8"))
    assert trace["statcast"]["label"] == STATCAST["label"]
