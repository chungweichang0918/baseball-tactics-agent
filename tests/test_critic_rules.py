from pathlib import Path
from types import SimpleNamespace

import pytest

from bbagent.agent.critic_rules import check_rules, cited_ids, split_sections
from bbagent.agent.orchestrator import inject_fault
from bbagent.agent.state import AgentState
from bbagent.schemas import GameState, ToolResult
from bbagent.tools.re24 import re24_evaluate

GOOD = """# ⚾ 戰術建議報告
**情境**：7 局下｜一出局｜一壘有人｜球數 1-1｜平手｜我方進攻
**分析對象**：右投投手，側拍，分析 2 球

## 1. 結論
- 主建議：若投手固定姿勢超過 1.5 秒，下一球盜二壘 [E02][E03]。
- 備案：正常強攻，不建議犧牲觸擊。

## 2. 影片觀察
| # | 觀察 | 數據 | 證據 |
|---|---|---|---|
| 1 | 到本壘時間偏慢 | 1.52 秒 | [E02] |
| 2 | 固定姿勢時間穩定 | 1.0 秒 | [E02、E01] |

## 3. 數據依據（RE24）
| 戰術 | 結果 | ΔRE | 損益平衡成功率 |
|---|---|---|---|
| 盜二壘 | 成功 | +0.155 | 72.6% |
| 盜二壘 | 失敗 | -0.411 | 72.6% |
| 犧牲觸擊 | 成功 | -0.190 | — |

## 4. 戰術建議與推論
到本壘時間 1.52 秒 [E02]，盜二壘成功率可望高於損益平衡成功率 72.6% [E03]，目前得分期望值 0.509。

## 5. 信心與限制
- 只分析 2 球，樣本少。
"""


def make_state(tmp_path, view="side", moving=False, **game) -> AgentState:
    g = GameState(**({"inning": 7, "half": "bottom", "outs": 1, "on_first": True, "our_role": "offense"} | game))
    state = AgentState(video=Path("x.mp4"), game=g, run_dir=tmp_path)
    state.add_evidence("r2", "scout_overview", {},
                       ToolResult(status="ok", summary="概況", data={"plan_view": view, "camera_moving": moving}))
    state.add_evidence("s4", "pitching_metrics", {}, ToolResult(status="ok", summary="指標"))
    state.add_evidence("s8", "re24_evaluate", {}, re24_evaluate(g, ["steal_2b", "sac_bunt", "swing_away"]))
    return state


def rules(issues) -> list[str]:
    return [i["rule"] for i in issues]


def test_good_report_passes(tmp_path):
    assert check_rules(make_state(tmp_path), GOOD) == []


def test_helpers():
    assert sorted(split_sections(GOOD)) == [1, 2, 3, 4, 5]
    assert cited_ids("見 [E02][E10、E11] 與 E05") == ["E02", "E10", "E11"]


def test_missing_section(tmp_path):
    report = GOOD.split("## 5.")[0]
    assert "sections" in rules(check_rules(make_state(tmp_path), report))


def test_unknown_and_missing_citations(tmp_path):
    state = make_state(tmp_path)
    issues = check_rules(state, GOOD.replace("[E03]。", "[E09]。"))
    assert any("E09" in i["message"] for i in issues)
    issues = check_rules(state, GOOD.replace("| 1.52 秒 | [E02] |", "| 1.52 秒 | — |"))
    assert any(i["rule"] == "citations" and "第 2 節" == i["location"] for i in issues)


@pytest.mark.parametrize("old,new", [
    ("| +0.155 |", "| +0.160 |"),            # 表格中的 ΔRE
    ("72.6% [E03]", "75% [E03]"),           # 內文的損益平衡成功率
    ("得分期望值 0.509", "得分期望值 0.600"),
])
def test_wrong_re24_numbers(tmp_path, old, new):
    assert "re24_numbers" in rules(check_rules(make_state(tmp_path), GOOD.replace(old, new)))


@pytest.mark.parametrize("phrase", ["得分期望值提升（+0.155）", "得分期望值變化 -0.190", "得分期望值增加 0.155"])
def test_re_change_is_checked_as_delta(tmp_path, phrase):
    # Mac 實測：「得分期望值提升（+0.155）」被當成 RE 本身，連退 2 次仍無法通過
    report = GOOD.replace("目前得分期望值 0.509。", f"盜二壘成功時{phrase}。")
    assert check_rules(make_state(tmp_path), report) == []
    wrong = GOOD.replace("目前得分期望值 0.509。", "盜二壘成功時得分期望值提升（+0.200）。")
    issues = check_rules(make_state(tmp_path), wrong)
    assert [i["rule"] for i in issues] == ["re24_numbers"] and "+0.200" in issues[0]["message"]


def test_issue_quotes_the_offending_text(tmp_path):
    issues = check_rules(make_state(tmp_path), GOOD.replace("得分期望值 0.509", "得分期望值 0.600"))
    assert "目前得分期望值 0.600" in issues[0]["location"] and "0.509" in issues[0]["message"]
    issues = check_rules(make_state(tmp_path), GOOD.replace("| +0.155 |", "| +0.160 |"))
    assert issues[0]["location"].startswith("第 3 節表格") and "盜二壘" in issues[0]["location"]


def test_rounded_percentage_is_accepted(tmp_path):
    assert check_rules(make_state(tmp_path), GOOD.replace("72.6% [E03]", "73% [E03]")) == []


def test_demo_fault_is_caught(tmp_path):
    issues = check_rules(make_state(tmp_path), inject_fault(GOOD))
    assert any("+0.999" in i["message"] for i in issues)


def test_illegal_tactics(tmp_path):
    two_outs = make_state(tmp_path, outs=2)
    bad = GOOD.replace("- 備案：正常強攻，不建議犧牲觸擊。", "- 備案：犧牲觸擊推進跑者。")
    assert any("兩出局時不能犧牲觸擊" in i["message"] for i in check_rules(two_outs, bad))
    assert "tactics" not in rules(check_rules(two_outs, GOOD))      # 「不建議犧牲觸擊」不算建議

    walk = GOOD.replace("- 備案：正常強攻，不建議犧牲觸擊。", "- 備案：故意四壞下一棒。")
    assert any("防守方的戰術" in i["message"] for i in check_rules(make_state(tmp_path), walk))


def test_tipping_needs_hedging(tmp_path):
    state = make_state(tmp_path)
    claim = GOOD.replace("目前得分期望值 0.509。", "目前得分期望值 0.509。投手有洩漏球種的小動作 [E02]。")
    assert "tipping" in rules(check_rules(state, claim))
    hedged = claim.replace("投手有洩漏", "投手疑似有洩漏")
    assert check_rules(state, hedged) == []
    # 沒有 Gemini 時的證據彙整報告會列出工具名稱，不算提到洩漏球種
    listed = GOOD.replace("目前得分期望值 0.509。", "目前得分期望值 0.509。tipping_compare：比較 1 球 [E02]。")
    assert check_rules(state, listed) == []


def test_limitations_required_for_bad_footage(tmp_path):
    assert "limitations" in rules(check_rules(make_state(tmp_path, view="other"), GOOD))
    fixed = GOOD + "- 影片不是側拍，出手時間可靠度低。\n"
    assert check_rules(make_state(tmp_path, view="other"), fixed) == []

    state = make_state(tmp_path)
    state.runtime["camera_motion"] = SimpleNamespace(moving=True)   # OpenCV 量到鏡頭移動
    assert any("鏡頭在移動" in i["message"] for i in check_rules(state, GOOD))


STATCAST = {"label": "2017-11-01 · 7 局下 · Slider 87.3 mph", "context": {"description": "swinging_strike"}, "values": [
    {"key": "release_speed", "label": "投球速度", "unit": "mph", "source": "Statcast 追蹤", "value": 87.3},
    {"key": "release_spin_rate", "label": "轉速", "unit": "rpm", "source": "Statcast 追蹤", "value": 2510.0},
    {"key": "launch_speed", "label": "擊球初速", "unit": "mph", "source": "Statcast 追蹤", "value": None}]}


def statcast_state(tmp_path):
    from bbagent.agent.tools import REGISTRY, ToolContext
    state = make_state(tmp_path)
    state.statcast = STATCAST
    state.add_evidence("r3", "statcast_pitch", {}, REGISTRY["statcast_pitch"].run(ToolContext(state, None, print), {}))
    return state


def test_statcast_numbers_must_match_and_be_cited(tmp_path):
    state = statcast_state(tmp_path)
    assert "這一球沒有的數據：擊球初速" in state.evidence["E04"].result.summary
    cited = GOOD.replace("| 2 | 固定姿勢時間穩定 | 1.0 秒 | [E02、E01] |",
                         "| 2 | 固定姿勢時間穩定 | 1.0 秒 | [E02、E01] |\n| 3 | 揮空的滑球，投球速度 87.3 mph、轉速 2,510 rpm | — | [E04] |")
    assert check_rules(state, cited) == []
    assert "statcast" in rules(check_rules(state, GOOD))                      # 有數據卻沒引用
    wrong = [i["message"] for i in check_rules(state, cited.replace("87.3 mph", "88.0 mph"))]
    assert any("投球速度應為 87.3 mph" in m for m in wrong)
    missing = cited.replace("轉速 2,510 rpm", "轉速 2,510 rpm、擊球初速 95 mph")
    assert any("沒有擊球初速數據" in i["message"] for i in check_rules(state, missing))


def test_fallback_report_cites_statcast(tmp_path):
    """沒有 Gemini 時的證據彙整報告也要通過 Statcast 規則：有引用、數字照抄。"""
    from bbagent.agent.synthesizer import fallback_report
    state = statcast_state(tmp_path)
    issues = check_rules(state, fallback_report(state, "沒有金鑰"))
    assert not [i for i in issues if i["rule"] == "statcast"], issues


def test_mph_without_statcast_is_rejected(tmp_path):
    claim = GOOD.replace("目前得分期望值 0.509。", "目前得分期望值 0.509。這球轉速約 2400 rpm [E02]。")
    assert any("沒有 Statcast" in i["message"] for i in check_rules(make_state(tmp_path), claim))
