"""Planner：根據賽況與影片概況產生分析計畫，交給 Executor 執行。

Gemini 負責「決策」（分支、工具、觀察重點、候選戰術），程式負責「檢查」：
工具是否存在、前置條件是否滿足、必要步驟是否齊全、戰術是否符合賽況。
檢查不通過就把錯誤回饋給 Gemini 重新規劃；Gemini 無法使用時改用預設計畫，Agent 不會因此停擺。
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Literal

from pydantic import BaseModel, ValidationError

from ..llm.gemini_client import GeminiClient, GeminiError, model_name
from ..llm.prompts import load_prompt
from ..schemas import AgentEvent, Plan, PlanStep
from ..tools.re24 import BaseOutState, get_re24
from .state import FOCUS_ZH, AgentState
from .tools import REGISTRY, dependents_of, planner_catalog

REQUIRED = {
    "pitcher": ["coarse_pose_scan", "segment_motions", "pitching_metrics", "re24_evaluate"],
    "batter": ["coarse_pose_scan", "segment_motions", "batting_metrics", "re24_evaluate"],
}
DEFAULT_TOOLS = {
    "pitcher": ["coarse_pose_scan", "camera_motion", "segment_motions", "pitching_metrics", "track_ball",
                "tipping_compare", "render_keyframes", "gemini_inspect", "re24_evaluate"],
    "batter": ["coarse_pose_scan", "camera_motion", "segment_motions", "batting_metrics", "swing_compare",
               "render_keyframes", "gemini_inspect", "re24_evaluate"],
}
MAX_PLAN_RETRIES = 2
MAX_INSPECTS = 2
BASELINE = {"offense": "swing_away", "defense": "pitch_to_batter"}

Events = Generator[AgentEvent, None, Plan]


class PlanStepDraft(BaseModel):
    id: str
    tool: str
    purpose: str
    depends_on: list[str]
    inspect_focus: str | None = None


class PlanDraft(BaseModel):
    """Gemini 的結構化輸出。工具參數用具名欄位而不是任意 dict，結構化輸出才穩定。"""

    analysis_focus: Literal["pitcher", "batter"]
    rationale: str
    tactical_questions: list[str]
    candidate_options: list[str]
    steps: list[PlanStepDraft]


def hands_zh(g) -> str:
    pitcher = {"R": "右投", "L": "左投"}.get(g.pitcher_hand or "", "未指定")
    batter = {"R": "右打", "L": "左打"}.get(g.batter_hand or "", "未指定")
    return f"投手：{pitcher}｜打者：{batter}"


def valid_options(state: AgentState) -> tuple[list[str], dict[str, str]]:
    """回傳 (可評估的戰術, 不可評估的戰術與原因)。"""
    re24 = get_re24()
    s = BaseOutState.from_game(state.game)
    ok, bad = [], {}
    for opt in re24.options:
        reason = re24.check_option(s, opt, state.game.our_role)
        if reason:
            bad[opt] = reason
        else:
            ok.append(opt)
    return ok, bad


def default_options(state: AgentState) -> list[str]:
    ok, _ = valid_options(state)
    baseline = BASELINE[state.game.our_role]
    return [o for o in ok if o != baseline][:3] + [baseline]


def default_focus(state: AgentState) -> str:
    g = state.game
    if state.focus == "batter":
        if g.our_role == "defense":
            return "準備姿勢與引棒時雙手的位置與高度，以及擊球瞬間頭部是否穩定（判斷內外角、高低球的應對）"
        return "跨步著地與擊球瞬間的手部位置、頭部是否穩定，揮棒路徑是否過大"
    if g.our_role == "offense" and (g.on_first or g.on_second):
        return "固定姿勢時投手的頭部方向與視線，以及開始投球前是否有固定的小動作（牽制或啟動前兆）"
    return "固定姿勢時手套的位置與手腕角度，以及出手前是否有容易被看出的小動作"


def default_plan(state: AgentState, excluded: set[str] = frozenset(), done: set[str] = frozenset()) -> Plan:
    """不依賴 Gemini 的預設計畫：依序執行該分支所有可用的工具。"""
    focus = state.focus
    wanted = DEFAULT_TOOLS[focus]
    steps: list[PlanStep] = []
    ids: dict[str, str] = {}
    satisfied = {"probe_video", "scout_overview"} | set(done)   # 偵察步驟在規劃前已執行
    for tool in wanted:
        if tool in excluded or tool in done:
            continue
        spec = REGISTRY[tool]
        if any(r not in ids and r not in satisfied for r in spec.requires):
            continue
        sid = f"s{len(steps) + 1}"
        args = {}
        if tool == "gemini_inspect":
            args = {"focus": default_focus(state)}
        elif tool == "re24_evaluate":
            args = {"options": default_options(state)}
        steps.append(PlanStep(id=sid, tool=tool, args=args, purpose=spec.description,
                              depends_on=[ids[r] for r in spec.requires if r in ids]))
        ids[tool] = sid
    rationale = ("使用預設計畫：依序執行所有可用的分析工具。" if steps
                 else "沒有其他可以執行的步驟，直接以現有證據撰寫報告。")
    return Plan(analysis_focus=focus, camera_view=state.camera_view, rationale=rationale,
                tactical_questions=[], candidate_options=default_options(state), steps=steps)


def validate_draft(draft: PlanDraft, state: AgentState, required: list[str] | None = None,
                   excluded: set[str] = frozenset(), done: set[str] = frozenset()) -> tuple[Plan | None, list[str], list[str]]:
    """回傳 (計畫, 錯誤, 警告)。有錯誤時計畫為 None。"""
    focus = draft.analysis_focus
    required = REQUIRED[focus] if required is None else required
    errors: list[str] = []
    warnings: list[str] = []
    target = state.game.analysis_target
    if target != "auto" and focus != target:
        errors.append(f"使用者指定分析{FOCUS_ZH[target]}，analysis_focus 必須是 {target}")
    if done and state.plan is not None and focus != state.plan.analysis_focus:
        errors.append(f"重新規劃不能更換分析分支，analysis_focus 必須是 {state.plan.analysis_focus}")

    steps: list[PlanStep] = []
    tool_of: dict[str, str] = {}
    inspects = 0
    for d in draft.steps:
        spec = REGISTRY.get(d.tool)
        if spec is None or not spec.planner_visible:
            if d.tool in ("probe_video", "scout_overview", "statcast_pitch"):
                warnings.append(f"{d.tool} 已在規劃前執行，略過")
            else:
                errors.append(f"步驟 {d.id} 使用了不存在的工具 {d.tool}")
            continue
        if d.tool in excluded:
            errors.append(f"步驟 {d.id} 使用了已失敗或無法執行的工具 {d.tool}")
            continue
        if focus not in spec.focus:
            errors.append(f"步驟 {d.id} 的 {d.tool} 不適用於{FOCUS_ZH[focus]}分析")
            continue
        if d.tool == "gemini_inspect":
            inspects += 1
            if inspects > MAX_INSPECTS:
                warnings.append(f"gemini_inspect 超過 {MAX_INSPECTS} 次，略過步驟 {d.id}")
                continue
        # 容許模型把 depends_on 寫成工具名稱
        deps = []
        for dep in d.depends_on:
            if dep in tool_of:
                deps.append(dep)
            else:
                match = [sid for sid, t in tool_of.items() if t == dep]
                if match:
                    deps.append(match[-1])
                elif dep not in done:
                    errors.append(f"步驟 {d.id} 依賴不存在或更晚的步驟 {dep}")
        earlier = set(tool_of.values()) | set(done)
        for req in spec.requires:
            if req not in earlier and req not in ("probe_video", "scout_overview"):
                errors.append(f"步驟 {d.id}（{d.tool}）需要先執行 {req}")
        args = {}
        if d.tool == "gemini_inspect":
            args = {"focus": d.inspect_focus or default_focus(state)}
        steps.append(PlanStep(id=d.id, tool=d.tool, args=args, purpose=d.purpose, depends_on=deps))
        tool_of[d.id] = d.tool

    present = set(tool_of.values())
    for req in required:
        if req not in present and req not in done:
            errors.append(f"缺少必要步驟 {req}")

    ok, bad = valid_options(state)
    options = []
    for opt in draft.candidate_options:
        if opt in ok:
            options.append(opt)
        else:
            warnings.append(f"候選戰術 {opt} 不適用（{bad.get(opt, '未知的戰術')}），已移除")
    baseline = BASELINE[state.game.our_role]
    if baseline not in options:
        options.append(baseline)
    for s in steps:
        if s.tool == "re24_evaluate":
            s.args = {"options": options}

    if errors:
        return None, errors, warnings
    try:
        plan = Plan(analysis_focus=focus, camera_view=state.camera_view, rationale=draft.rationale,
                    tactical_questions=draft.tactical_questions, candidate_options=options, steps=steps)
    except ValidationError as e:
        return None, [err["msg"] for err in e.errors()], warnings
    return plan, [], warnings


def planner_input(state: AgentState) -> str:
    g = state.game
    ok, bad = valid_options(state)
    re24 = get_re24()
    target = {"auto": "自動判斷（依影片概況的主體）", "pitcher": "投手", "batter": "打者"}[g.analysis_target]
    lines = [
        f"【賽況】{g.describe_zh()}｜{hands_zh(g)}",
        f"【使用者指定的分析對象】{target}",
        f"【使用者補充】{g.notes or '（無）'}",
        "【已完成的偵察】",
        *[ev.brief() for ev in state.evidence.values()],
        "【可用工具】",
        planner_catalog(),
        "【可評估的戰術】",
        *[f"- {o}：{re24.label(o)}" for o in ok],
        "【不可評估的戰術（不要列入）】",
        *[f"- {o}：{reason}" for o, reason in bad.items()],
    ]
    return "\n".join(lines)


def plan_text(plan: Plan) -> str:
    lines = [plan.rationale]
    if plan.tactical_questions:
        lines.append("戰術問題：" + "；".join(plan.tactical_questions))
    lines.append("候選戰術：" + "、".join(get_re24().label(o) for o in plan.candidate_options))
    for s in plan.execution_order():
        dep = f"（依賴 {', '.join(s.depends_on)}）" if s.depends_on else ""
        focus = f"：{s.args['focus']}" if s.tool == "gemini_inspect" else ""
        lines.append(f"{s.id}. {s.tool}{dep} — {s.purpose}{focus}")
    return "\n".join(lines)


class Planner:
    def __init__(self, client: GeminiClient, model: str | None = None):
        self.client = client
        self.model = model or model_name("fast")

    def plan(self, state: AgentState) -> Events:
        yield AgentEvent(stage="planner", kind="thought", title="規劃分析步驟",
                         content="根據賽況與影片概況，決定要呼叫哪些工具、評估哪些戰術。")
        plan = yield from self._ask(state, planner_input(state))
        return plan

    def replan(self, state: AgentState, failure: dict) -> Events:
        state.counters["replans"] += 1
        failed = failure["tool"]
        excluded = {failed} | dependents_of(failed)
        done = {ev.tool for ev in state.evidence.values() if ev.result.status != "failed"}
        yield AgentEvent(stage="planner", kind="thought", title="重新規劃",
                         content=f"{failed} 失敗（{failure['error']}），以下工具無法再使用：{'、'.join(sorted(excluded))}。"
                                 "規劃剩餘還能執行的步驟。")
        feedback = (f"\n【執行狀況】上一個計畫在 {failed} 失敗：{failure['error']}\n"
                    f"已完成的工具：{'、'.join(sorted(done)) or '無'}（不要再列入，可以當作已滿足的前置條件）\n"
                    f"不可使用的工具：{'、'.join(sorted(excluded))}\n請只規劃剩餘還能執行的步驟。")
        required = [t for t in REQUIRED[state.focus] if t not in excluded and t not in done]
        plan = yield from self._ask(state, planner_input(state) + feedback, required, excluded, done)
        return plan

    def _ask(self, state: AgentState, prompt: str, required: list[str] | None = None,
             excluded: set[str] = frozenset(), done: set[str] = frozenset()) -> Events:
        feedback: list[str] = []
        for attempt in range(MAX_PLAN_RETRIES + 1):
            state.counters["plan_attempts"] += 1
            text = prompt
            if feedback:
                text += "\n【上一份計畫沒有通過檢查，請修正】\n" + "\n".join(f"- {e}" for e in feedback)
            try:
                draft, _ = self.client.generate_structured(self.model, [text], PlanDraft, system=load_prompt("planner"))
            except GeminiError as e:
                yield AgentEvent(stage="planner", kind="error", title="Gemini 無法規劃", content=str(e))
                break
            plan, errors, warnings = validate_draft(draft, state, required, excluded, done)
            for w in warnings:
                yield AgentEvent(stage="planner", kind="thought", title="計畫調整", content=w)
            if plan is not None:
                state.plan_source = "llm"
                yield AgentEvent(stage="planner", kind="thought", title=f"計畫完成（{len(plan.steps)} 個步驟）",
                                 content=plan_text(plan), payload={"plan": plan.model_dump()})
                return plan
            feedback = errors
            yield AgentEvent(stage="planner", kind="critique", title=f"計畫未通過檢查（第 {attempt + 1} 次）",
                             content="\n".join(errors), payload={"errors": errors})

        plan = default_plan(state, excluded, done)
        state.plan_source = "default"
        yield AgentEvent(stage="planner", kind="thought", title=f"改用預設計畫（{len(plan.steps)} 個步驟）",
                         content=plan_text(plan), payload={"plan": plan.model_dump()})
        return plan
