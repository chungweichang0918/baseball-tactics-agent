"""Critic 的規則檢查（純 Python，不呼叫 LLM）：便宜、精確，也最適合在報告中展示 Reflection。

檢查項目：
  sections     第 1–5 節是否齊全
  citations    引用的證據編號是否存在；第 2 節每一列是否都有引用
  re24_numbers 報告中的 ΔRE、損益平衡成功率、得分期望值是否與 RE24 工具算出的一致
  tactics      第 1、4 節建議的戰術是否違反賽況（例如兩出局觸擊、進攻方敬遠）
  tipping      提到洩漏球種時，是否用「疑似」並說明樣本數
  limitations  非側拍或鏡頭移動時，第 5 節是否寫出這項限制
  statcast     mph／rpm 數字是否照抄使用者匯入的 Statcast 逐球數據；有這份數據時是否引用
"""

from __future__ import annotations

import re

from ..tools.re24 import BaseOutState, get_re24
from .state import AgentState

CITE_BLOCK = re.compile(r"\[(E\d{2,}(?:\s*[、,，]\s*E\d{2,})*)\]")
NUM = r"[+\-−]?\d+\.\d+"
TACTIC_WORDS = {
    "steal_2b": ["盜二壘"],
    "steal_3b": ["盜三壘"],
    "sac_bunt": ["犧牲觸擊", "犧牲短打", "觸擊推進"],
    "ibb": ["故意四壞", "敬遠"],
}
NEGATION = re.compile(r"(不建議|不宜|避免|不要|不應|勿|不適合|不採取|放棄|而非|不必)[^。；\n]{0,8}$")
TIPPING = re.compile(r"tipping(?!_compare)|洩漏|小動作", re.IGNORECASE)   # 工具名稱 tipping_compare 不算提到
LIMIT_WORDS = ("視角", "鏡頭", "拍攝", "側拍")
DELTA_RE = re.compile(rf"ΔRE[^\d\n+\-−]{{0,8}}({NUM})")
PCT_RE = re.compile(r"損益平衡[^\d\n]{0,10}?(\d+(?:\.\d+)?)\s*%")
RE_VALUE_RE = re.compile(r"得分期望值([^\d\n+\-−]{0,8})([+\-−]?\d+\.\d{3})")
CHANGE_WORDS = re.compile(r"提升|提高|增加|上升|減少|降低|下降|變化|改變|損失")
TOL_DELTA = 0.0015
TOL_PCT = 0.15
STATCAST_NUM = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(mph|rpm)\b", re.IGNORECASE)
TOL_STATCAST = {"mph": 0.051, "rpm": 0.51}


def split_sections(report: str) -> dict[int, str]:
    sections: dict[int, str] = {}
    current, buf = None, []
    for line in report.splitlines():
        m = re.match(r"^##\s*(\d+)\.", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf)
            current, buf = int(m.group(1)), []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf)
    return sections


def cited_ids(text: str) -> list[str]:
    return [eid for block in CITE_BLOCK.findall(text) for eid in re.findall(r"E\d{2,}", block)]


def _num(s: str) -> float:
    return float(s.replace("−", "-"))


def _table(text: str) -> tuple[list[str], list[list[str]]]:
    """回傳第一個 Markdown 表格的 (表頭, 資料列)。"""
    rows = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    if not rows:
        return [], []
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    header = cells[0]
    body = [c for c in cells[1:] if not all(set(x) <= set("-: ") for x in c)]
    return header, body


def _context(report: str, m: re.Match, width: int = 15) -> str:
    """數字所在的原文片段，例如「…盜壘成功的得分期望值提升（+0.155）優於…」。"""
    start, end = max(0, m.start() - width), min(len(report), m.end() + width)
    snippet = report[start:end].replace("\n", " ").strip()
    return f"「…{snippet}…」"


def _issue(rule: str, location: str, message: str) -> dict:
    return {"source": "rule", "rule": rule, "severity": "high", "location": location, "message": message}


def _allowed_numbers(state: AgentState) -> tuple[set[float], set[float], set[float]]:
    deltas, break_evens, values = set(), set(), set()
    for ev in state.evidence.values():
        if ev.tool != "re24_evaluate" or ev.result.status == "failed":
            continue
        d = ev.result.data
        values.add(d["re_before"])
        for opt in d["options"]:
            if not opt.get("valid"):
                continue
            for o in opt["outcomes"]:
                deltas.add(abs(o["delta"]))
                values.add(o["re_after"])
            if opt.get("break_even") is not None:
                break_evens.add(round(opt["break_even"] * 100, 1))
            if opt.get("expected_delta") is not None:
                deltas.add(abs(opt["expected_delta"]))
    return deltas, break_evens, values


def _pct_ok(v: float, allowed: set[float]) -> bool:
    return any(abs(v - a) <= TOL_PCT or (v == int(v) and round(a) == v) for a in allowed)


def _statcast_issues(state: AgentState, report: str) -> list[dict]:
    """mph／rpm 只可能來自 Statcast：影片沒有這份數據時不能出現，有的話必須和證據中的值一致。"""
    ev = state.latest("statcast_pitch")
    found = list(STATCAST_NUM.finditer(report))
    issues = []
    if ev is None:
        for m in found:
            issues.append(_issue("statcast", _context(report, m),
                                 f"寫了 {m.group(1)} {m.group(2)}，但沒有 Statcast 逐球數據；影片無法量測轉速與擊球初速，"
                                 "球速只有 km/h 粗估"))
        return issues
    if ev.id not in cited_ids(report):
        issues.append(_issue("statcast", "第 2 節", f"有使用者確認的 Statcast 逐球數據 [{ev.id}]，"
                                                  "請在第 2 節引用，並在第 4 節和影片的動作證據對照"))
    fields = ev.result.data["values"]
    for m in found:
        value, unit = float(m.group(1).replace(",", "")), m.group(2).lower()
        before = report[max(0, m.start() - 20):m.start()]
        named = [f for f in fields if f["unit"] == unit and f["label"] in before]
        if named:   # 數字前面寫了欄位名稱，就要和那一個欄位相符
            f = max(named, key=lambda f: before.rfind(f["label"]))
            if f["value"] is None:
                message = f"這一球沒有{f['label']}數據，不能寫出 {value:g} {unit}"
            elif abs(value - f["value"]) > TOL_STATCAST[unit]:
                message = f"{f['label']}應為 {f['value']:g} {unit}（Statcast），報告寫成 {value:g} {unit}"
            else:
                continue
        else:
            allowed = [f["value"] for f in fields if f["unit"] == unit and f["value"] is not None]
            if any(abs(value - a) <= TOL_STATCAST[unit] for a in allowed):
                continue
            message = (f"{value:g} {unit} 不是 Statcast 證據中的數值；請照抄證據 [{ev.id}]"
                       + (f"（可用：{'、'.join(f'{a:g} {unit}' for a in allowed)}）" if allowed else f"（這一球沒有 {unit} 數據）"))
        issues.append(_issue("statcast", _context(report, m), message))
    return issues


def check_rules(state: AgentState, report: str) -> list[dict]:
    issues: list[dict] = []
    sections = split_sections(report)

    # sections
    for n in range(1, 6):
        if n not in sections:
            issues.append(_issue("sections", f"第 {n} 節", f"缺少第 {n} 節，請依範本補上"))

    # citations
    ids = cited_ids(report)
    if not ids:
        issues.append(_issue("citations", "全文", "報告沒有引用任何證據，請用 [E05] 格式標示依據"))
    unknown = sorted(set(ids) - set(state.evidence))
    if unknown:
        issues.append(_issue("citations", "全文", f"引用了不存在的證據 {'、'.join(unknown)}；"
                                                 f"可用的證據為 {'、'.join(state.evidence)}"))
    if 2 in sections:
        _, body = _table(sections[2])
        bullets = [l for l in sections[2].splitlines() if re.match(r"^\s*[-*]\s+", l)]
        for row in ["|".join(r) for r in body] + bullets:
            if not CITE_BLOCK.search(row):
                issues.append(_issue("citations", "第 2 節", f"這一項觀察沒有引用證據：「{row.strip()[:40]}」"))

    # re24_numbers：每個數字都記下原文片段，退回時 Synthesizer 才找得到要改哪裡
    deltas, break_evens, values = _allowed_numbers(state)
    found_delta = [(_num(m.group(1)), _context(report, m)) for m in DELTA_RE.finditer(report)]
    found_pct = [(float(m.group(1)), _context(report, m)) for m in PCT_RE.finditer(report)]
    found_re = []
    for m in RE_VALUE_RE.finditer(report):
        # 「得分期望值提升（+0.155）」「得分期望值變化 -0.190」說的是 ΔRE，不是 RE 本身
        gap, value = m.group(1), m.group(2)
        is_change = value[0] in "+-−" or CHANGE_WORDS.search(gap)
        (found_delta if is_change else found_re).append((_num(value), _context(report, m)))
    if 3 in sections:
        header, body = _table(sections[3])
        d_cols = [i for i, h in enumerate(header) if "ΔRE" in h]
        p_cols = [i for i, h in enumerate(header) if "損益平衡" in h]
        for row in body:
            where = "第 3 節表格：「" + " | ".join(row)[:40] + "」"
            for i in d_cols:
                if i < len(row):
                    found_delta += [(_num(x), where) for x in re.findall(NUM, row[i])]
            for i in p_cols:
                if i < len(row):
                    found_pct += [(float(x), where) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", row[i])]
    if (found_delta or found_pct or found_re) and not (deltas or values):
        issues.append(_issue("re24_numbers", "第 3 節", "報告引用了 RE24 數字，但沒有成功的 re24_evaluate 證據"))
    else:
        bad = [(f"ΔRE {v:+.3f} 不是 RE24 工具算出的值，請照抄證據中的數字", where)
               for v, where in found_delta if not any(abs(abs(v) - a) <= TOL_DELTA for a in deltas)]
        bad += [(f"損益平衡成功率 {v:g}% 與 RE24 工具的結果不符", where)
                for v, where in found_pct if not _pct_ok(v, break_evens)]
        bad += [(f"得分期望值 {v:.3f} 與 RE24 表不符（得分期望值指局面本身的值，證據中有 "
                 f"{'、'.join(f'{x:.3f}' for x in sorted(values))}；變化量請寫成 ΔRE）", where)
                for v, where in found_re if not any(abs(v - a) <= TOL_DELTA for a in values)]
        seen = set()
        for message, where in bad:
            if message not in seen:
                seen.add(message)
                issues.append(_issue("re24_numbers", where, message))

    # tactics
    re24 = get_re24()
    s = BaseOutState.from_game(state.game)
    for n in (1, 4):
        text = sections.get(n, "")
        for option, words in TACTIC_WORDS.items():
            reason = re24.check_option(s, option, state.game.our_role)
            if not reason:
                continue
            for w in words:
                for m in re.finditer(w, text):
                    if NEGATION.search(text[max(0, m.start() - 12):m.start()]):
                        continue
                    issues.append(_issue("tactics", f"第 {n} 節", f"建議了「{w}」，但{reason}"))
                    break

    # tipping
    if TIPPING.search(report):
        if "疑似" not in report:
            issues.append(_issue("tipping", "全文", "提到洩漏球種或小動作時，必須用「疑似」表述"))
        if not re.search(r"\d+\s*球|樣本", sections.get(5, "")):
            issues.append(_issue("tipping", "第 5 節", "提到洩漏球種時，第 5 節必須說明分析了幾球與樣本不足的限制"))

    # limitations
    if state.camera_view != "side" or state.camera_moving:
        if not any(w in sections.get(5, "") for w in LIMIT_WORDS):
            what = "鏡頭在移動" if state.camera_moving else "影片不是側拍"
            issues.append(_issue("limitations", "第 5 節", f"證據顯示{what}，第 5 節必須寫出這項限制與影響"))

    # statcast：同一句話只回報一次
    seen = {(i["location"], i["message"]) for i in issues}
    for issue in _statcast_issues(state, report):
        if (issue["location"], issue["message"]) not in seen:
            seen.add((issue["location"], issue["message"]))
            issues.append(issue)
    return issues
