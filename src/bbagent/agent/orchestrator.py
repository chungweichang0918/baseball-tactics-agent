"""Orchestrator：Agent 主迴圈（純 Python generator）。

  偵察   probe_video → scout_overview（規劃前固定執行，讓 Planner 有依據）
  規劃   Planner 產生計畫（Gemini 決策 + 程式檢查，失敗時退回預設計畫）
  執行   Executor 依相依順序呼叫工具；關鍵工具失敗時重新規劃一次
  綜合   Synthesizer 串流撰寫報告
  審查   Critic 規則檢查 + LLM 檢查；不通過就退回修正，最多 2 次

每一步都 yield AgentEvent，UI 與 CLI 只需要消費事件。
"""

from __future__ import annotations

import json
import queue
import re
from collections.abc import Generator, Iterator
from datetime import datetime
from pathlib import Path

from ..config import load_config, project_path
from ..llm.gemini_client import GeminiClient
from ..schemas import AgentEvent, GameState, PlanStep
from .critic import Critic, Verdict
from .executor import Executor
from .planner import Planner
from .state import AgentState
from .synthesizer import Synthesizer

MAX_REPLANS = 1
MAX_REVISIONS = 2
FAULT_TEXT = "\n補充：依 RE24 計算，盜二壘的期望值變化為 ΔRE +0.999。\n"


def inject_fault(report: str) -> str:
    """示範模式：在第 4 節插入一個錯誤的 RE24 數字，保證 Critic 會退回，用來展示 Reflection。"""
    m = re.search(r"^## 5\.", report, flags=re.MULTILINE)
    return report[:m.start()] + FAULT_TEXT + "\n" + report[m.start():] if m else report + FAULT_TEXT


def execution_summary(state: AgentState, client: GeminiClient, verdict: Verdict) -> str:
    statuses = [ev.result.status for ev in state.evidence.values()]
    rounds = len(state.critiques)
    returned = [c for c in state.critiques if not c["passed"]]
    reasons = "；".join(i["message"] for c in returned for i in c["issues"] if i["severity"] == "high")
    lines = [
        "## 6. Agent 執行摘要",
        f"- 計畫：{len(state.plan.steps) if state.plan else 0} 個步驟（{'Gemini 規劃' if state.plan_source == 'llm' else '預設計畫'}，"
        f"規劃嘗試 {state.counters['plan_attempts']} 次）｜重新規劃 {state.counters['replans']} 次",
        f"- 工具呼叫：{len(statuses)} 次（成功 {statuses.count('ok')}、部分成功 {statuses.count('degraded')}、"
        f"失敗 {statuses.count('failed')}）",
        f"- Critic 審查：{rounds} 輪，退回 {len(returned)} 次，最終{'通過' if verdict.passed else '未通過'}"
        + (f"（退回原因：{reasons}）" if reasons else ""),
        f"- Gemini：實際呼叫 {client.stats['api_calls']} 次、快取重播 {client.stats['cache_hits']} 次",
    ]
    return "\n".join(lines)


def compose_final(state: AgentState, draft: str, verdict: Verdict, client: GeminiClient) -> str:
    draft = re.split(r"^## 6\.", draft, maxsplit=1, flags=re.MULTILINE)[0].rstrip()   # 第 6 節一律由程式產生
    head = ""
    if not verdict.passed:
        head = "> ⚠️ **審查未通過的項目**（已達修正上限，請自行判讀）\n" + "\n".join(
            f"> - {i['location']}：{i['message']}" for i in verdict.blocking) + "\n\n"
    return head + draft + "\n\n" + execution_summary(state, client, verdict) + "\n"


def default_run_dir(video: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_path(load_config()["paths"]["runs"]) / "agent" / f"{stamp}_{video.stem}"


def run_agent(video: str | Path, game: GameState, *, client: GeminiClient | None = None,
              run_dir: str | Path | None = None, demo_fault: bool = False,
              statcast: dict | None = None) -> Iterator[AgentEvent]:
    """statcast：使用者從 Savant CSV 確認的那一球（Pitch.to_dict()），有的話在偵察階段變成一筆證據。"""
    video = Path(video)
    run_dir = Path(run_dir) if run_dir else default_run_dir(video)
    run_dir.mkdir(parents=True, exist_ok=True)
    notices: queue.Queue = queue.Queue()
    client = client or GeminiClient()
    client.on_retry = lambda n, wait, reason: notices.put(f"Gemini 第 {n} 次重試：{reason}，{wait:g} 秒後再試")
    state = AgentState(video=video, game=game, run_dir=run_dir, statcast=statcast)
    for event in _run(state, client, notices, demo_fault):
        state.events.append(event)
        yield event


def _run(state: AgentState, client: GeminiClient, notices: queue.Queue, demo_fault: bool) -> Generator[AgentEvent]:
    executor = Executor(client, notices)
    planner, synthesizer, critic = Planner(client), Synthesizer(client), Critic(client)

    yield AgentEvent(stage="system", kind="thought", title="開始分析",
                     content=f"影片：{state.video.name}\n賽況：{state.game.describe_zh()}",
                     payload={"game": state.game.model_dump()})

    # 偵察：有匯入 Statcast 時一併讀入，讓 Planner 規劃時就知道有哪些逐球數字
    recon = [("probe_video", "讀取影片資訊"), ("scout_overview", "判斷視角與拍攝條件")]
    if state.statcast:
        recon.append(("statcast_pitch", "讀取使用者確認的 Statcast 逐球數據"))
    for i, (tool, purpose) in enumerate(recon):
        ev = yield from executor.run_step(state, PlanStep(id=f"r{i + 1}", tool=tool, purpose=purpose))
        if tool == "probe_video" and ev.result.status == "failed":
            yield AgentEvent(stage="system", kind="final", title="分析中止", content=ev.result.error or "讀不到影片",
                             payload={"aborted": True})
            return

    # 規劃與執行
    plan = yield from planner.plan(state)
    yield from executor.drain("planner")
    state.plan = plan
    failure = yield from executor.run_plan(state, plan)
    if failure and state.counters["replans"] < MAX_REPLANS:
        plan2 = yield from planner.replan(state, failure)
        yield from executor.drain("planner")
        if plan2.steps:
            failure = yield from executor.run_plan(state, plan2)

    # 綜合與審查
    feedback, draft, verdict = None, None, Verdict(passed=False)
    for round_no in range(MAX_REVISIONS + 1):
        draft = yield from synthesizer.write(state, feedback, draft, round_no)
        yield from executor.drain("synthesizer")
        if demo_fault and round_no == 0:
            draft = inject_fault(draft)
            yield AgentEvent(stage="system", kind="thought", title="示範模式：在草稿中注入錯誤",
                             content=f"刻意加入一個錯誤的 RE24 數字：{FAULT_TEXT.strip()}")
        verdict = yield from critic.review(state, draft, round_no)
        yield from executor.drain("critic")
        if verdict.passed or round_no == MAX_REVISIONS:
            break
        state.counters["revisions"] += 1
        feedback = [f"{i['location']}：{i['message']}" for i in verdict.blocking]
        yield AgentEvent(stage="critic", kind="thought", title=f"退回 Synthesizer 修正（第 {round_no + 1}/{MAX_REVISIONS} 次）",
                         content="\n".join(feedback))

    state.report = compose_final(state, draft, verdict, client)
    report_path = state.run_dir / "report.md"
    trace_path = state.run_dir / "trace.json"
    report_path.write_text(state.report, encoding="utf-8")
    trace = state.to_trace() | {"report": state.report, "gemini": client.stats}
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    yield AgentEvent(stage="system", kind="final", title="分析完成" if verdict.passed else "分析完成（審查未完全通過）",
                     content=state.report,
                     payload={"passed": verdict.passed, "run_dir": str(state.run_dir), "report_path": str(report_path),
                              "trace_path": str(trace_path),
                              "keyframes": [k["path"] for k in state.runtime.get("keyframes", [])]})
