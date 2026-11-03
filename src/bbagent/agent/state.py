"""AgentState：Planner、Executor、Synthesizer、Critic 共用的狀態。

可序列化的部分（賽況、計畫、證據摘要、事件）會存成執行紀錄 trace.json；
粗掃結果、骨架序列這類大型中間物件放在 runtime，不寫進紀錄。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import AgentEvent, CameraView, GameState, Plan, Subject, ToolResult
from ..vision.video import VideoMeta

STATUS_ZH = {"ok": "成功", "degraded": "部分成功", "failed": "失敗"}
FOCUS_ZH = {"pitcher": "投手", "batter": "打者"}


@dataclass
class Evidence:
    id: str
    step_id: str
    tool: str
    args: dict
    result: ToolResult

    def brief(self) -> str:
        return f"[{self.id}] {self.tool}（{STATUS_ZH[self.result.status]}）：{self.result.summary}"

    def to_dict(self) -> dict:
        r = self.result
        return {"id": self.id, "step_id": self.step_id, "tool": self.tool, "args": self.args, "status": r.status,
                "summary": r.summary, "error": r.error, "artifacts": r.artifacts, "data": r.data}


@dataclass
class AgentState:
    video: Path
    game: GameState
    run_dir: Path
    meta: VideoMeta | None = None
    plan: Plan | None = None
    plan_source: str = ""                                  # "llm" 或 "default"
    statcast: dict | None = None                           # 使用者確認的那一球（analysis.statcast.Pitch.to_dict）
    evidence: dict[str, Evidence] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)
    report: str | None = None
    critiques: list[dict] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=lambda: {"replans": 0, "plan_attempts": 0, "revisions": 0})

    def add_evidence(self, step_id: str, tool: str, args: dict, result: ToolResult) -> Evidence:
        eid = f"E{len(self.evidence) + 1:02d}"
        result.evidence_id = eid
        ev = Evidence(eid, step_id, tool, args, result)
        self.evidence[eid] = ev
        return ev

    def latest(self, tool: str, include_failed: bool = False) -> Evidence | None:
        for ev in reversed(self.evidence.values()):
            if ev.tool == tool and (include_failed or ev.result.status != "failed"):
                return ev
        return None

    def succeeded(self, tool: str) -> bool:
        return self.latest(tool) is not None

    @property
    def overview(self) -> dict | None:
        ev = self.latest("scout_overview")
        return ev.result.data if ev else None

    @property
    def focus(self) -> Subject:
        """分析分支：以計畫為準；還沒有計畫時依使用者指定，再依影片概況判斷，預設投手。"""
        if self.plan is not None:
            return self.plan.analysis_focus
        if self.game.analysis_target != "auto":
            return self.game.analysis_target
        return "batter" if (self.overview or {}).get("subject") == "batter" else "pitcher"

    @property
    def camera_view(self) -> CameraView:
        return (self.overview or {}).get("plan_view", "other")

    @property
    def camera_moving(self) -> bool | None:
        """以 OpenCV 量測為準；沒有量測結果時才參考 Gemini 的判斷。"""
        cm = self.runtime.get("camera_motion")
        if cm is not None and cm.moving is not None:
            return cm.moving
        return (self.overview or {}).get("camera_moving")

    def to_trace(self) -> dict:
        return {
            "video": str(self.video),
            "game": self.game.model_dump(),
            "meta": self.meta.to_dict() if self.meta else None,
            "plan": self.plan.model_dump() if self.plan else None,
            "plan_source": self.plan_source,
            "statcast": self.statcast,
            "counters": self.counters,
            "evidence": [ev.to_dict() for ev in self.evidence.values()],
            "critiques": self.critiques,
            "events": [e.model_dump() for e in self.events if e.kind != "report_delta"],
        }
