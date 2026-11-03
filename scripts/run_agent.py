"""在終端機執行完整的 Agent：偵察 → 規劃 → 執行 → 撰寫報告 → 審查。

用法：
    python scripts/run_agent.py data/samples/p_side_02.mp4 --inning 7 --half bottom --outs 1 --bases 1 \\
        --balls 1 --strikes 1 --score-diff 0 --role offense --hand R
    python scripts/run_agent.py clip.mp4 --outs 2 --bases 13 --role offense --demo-fault   # 示範 Critic 退回
    python scripts/run_agent.py data/samples/b_side_01.mp4 --target batter --role defense --batter-hand L
    python scripts/run_agent.py clip.mp4 --statcast savant_data.csv --pitch-row 12   # 併入 Statcast 逐球數據

--bases 寫有人的壘號，例如 1、13、123；沒人就省略。
報告與執行紀錄存在 runs/agent/<時間>_<影片檔名>/。
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bbagent.vision import device  # noqa: E402,F401  必須在 torch 之前 import

from bbagent.agent.orchestrator import run_agent  # noqa: E402
from bbagent.llm.gemini_client import GeminiClient  # noqa: E402
from bbagent.schemas import AgentEvent, GameState  # noqa: E402

ICONS = {"planner": "🧭", "executor": "🔧", "synthesizer": "✍️ ", "critic": "🔍", "system": "⚙️ "}
STATUS_ICON = {"ok": "✅", "degraded": "⚠️ ", "failed": "❌"}


def indent(text: str, n: int = 5) -> str:
    return textwrap.indent(text, " " * n)


def show(ev: AgentEvent, state: dict) -> None:
    icon = ICONS[ev.stage]
    if ev.kind == "report_delta":
        if not state.get("streaming"):
            print("\n" + "─" * 60)
            state["streaming"] = True
        print(ev.content, end="", flush=True)
        return
    if state.get("streaming"):
        print("\n" + "─" * 60)
        state["streaming"] = False
    if ev.kind == "tool_call":
        print(f"\n{icon} {ev.title}：{ev.content}")
    elif ev.kind == "progress":
        print(f"     ⏳ {ev.content}")
    elif ev.kind == "tool_result":
        print(f"   {STATUS_ICON[ev.payload['status']]} {ev.title}")
        print(indent(ev.content))
    elif ev.kind == "final":
        print(f"\n{icon} {ev.title}")
        if ev.payload.get("report_path"):
            print(f"   報告：{ev.payload['report_path']}\n   執行紀錄：{ev.payload['trace_path']}")
        else:
            print(indent(ev.content))
    else:
        print(f"\n{icon} {ev.title}")
        if ev.content:
            print(indent(ev.content))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--inning", type=int, default=7)
    p.add_argument("--half", choices=["top", "bottom"], default="bottom")
    p.add_argument("--outs", type=int, default=0)
    p.add_argument("--bases", default="", help="有人的壘號，例如 1、13、123")
    p.add_argument("--balls", type=int, default=0)
    p.add_argument("--strikes", type=int, default=0)
    p.add_argument("--score-diff", type=int, default=0, help="我方減對方")
    p.add_argument("--role", choices=["offense", "defense"], default="offense")
    p.add_argument("--target", choices=["auto", "pitcher", "batter"], default="auto", help="分析對象（預設依影片判斷）")
    p.add_argument("--hand", choices=["R", "L"], help="投手的投球手")
    p.add_argument("--batter-hand", choices=["R", "L"], help="打者的打擊方向（右打 R、左打 L）")
    p.add_argument("--notes", default="", help="補充說明，例如跑者腳程、打者特性")
    p.add_argument("--demo-fault", action="store_true", help="示範模式：在第一版報告注入錯誤，展示 Critic 退回修正")
    p.add_argument("--no-cache", action="store_true", help="略過 Gemini 快取")
    p.add_argument("--statcast", type=Path, help="Baseball Savant 逐球 CSV；搭配 --pitch-row 指定影片中的那一球")
    p.add_argument("--pitch-row", type=int, help="那一球在 CSV 的資料列（第 1 列是第一筆資料，不含表頭）")
    args = p.parse_args()

    bad = set(args.bases) - set("123")
    if bad:
        p.error("--bases 只能包含 1、2、3")
    statcast = None
    if args.statcast:
        from bbagent.analysis.statcast import load_statcast
        pitches = load_statcast(args.statcast)
        if args.pitch_row is None or not 1 <= args.pitch_row <= len(pitches):
            p.error(f"--statcast 需要搭配 --pitch-row（1–{len(pitches)}）")
        statcast = pitches[args.pitch_row - 1].to_dict()
    game = GameState(inning=args.inning, half=args.half, outs=args.outs, on_first="1" in args.bases,
                     on_second="2" in args.bases, on_third="3" in args.bases, balls=args.balls,
                     strikes=args.strikes, score_diff=args.score_diff, our_role=args.role,
                     analysis_target=args.target, pitcher_hand=args.hand, batter_hand=args.batter_hand,
                     notes=args.notes)
    client = GeminiClient(cache_mode="off" if args.no_cache else None)
    state: dict = {}
    final = None
    for ev in run_agent(args.video, game, client=client, demo_fault=args.demo_fault, statcast=statcast):
        show(ev, state)
        if ev.kind == "final":
            final = ev
    return 0 if final and final.payload.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
