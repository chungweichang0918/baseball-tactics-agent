"""Baseball Savant（Statcast）逐球 CSV：讀取、依賽況列出候選球、整理成有來源標示的數據。

影片只能粗估球速，轉速與擊球初速更無法從一般轉播畫面取得；這些數字改由使用者匯入的
Savant 逐球資料提供。單憑一段剪輯無法保證自動對到正確的那一球，所以這裡只列候選、由使用者確認。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# (欄位別名, 中文名稱, 單位, 來源)。Savant 歷年欄名略有不同，依序取第一個存在的欄位。
MEASURED = "Statcast 追蹤"
BATTED = "Statcast 追蹤（少數未追蹤到的擊球為官方估計值）"
MODELED = "Statcast 模型估計"
FIELDS = [
    (("release_speed",), "投球速度", "mph", MEASURED),
    (("release_spin_rate", "release_spin"), "轉速", "rpm", MEASURED),
    (("spin_axis",), "旋轉軸", "°", MEASURED),
    (("release_extension",), "出手延伸", "ft", MEASURED),
    (("pfx_x",), "水平位移", "ft", MEASURED),
    (("pfx_z",), "垂直位移", "ft", MEASURED),
    (("plate_x",), "進壘水平位置", "ft", MEASURED),
    (("plate_z",), "進壘高度", "ft", MEASURED),
    (("bat_speed",), "棒速", "mph", MEASURED),
    (("swing_length",), "揮棒長度", "ft", MEASURED),
    (("launch_speed",), "擊球初速", "mph", BATTED),
    (("launch_angle",), "擊球仰角", "°", BATTED),
    (("hit_distance_sc", "hit_distance"), "擊球距離", "ft", MEASURED),
    (("estimated_ba_using_speedangle",), "預期打擊率 xBA", "", MODELED),
    (("estimated_woba_using_speedangle",), "預期 wOBA", "", MODELED),
]
REQUIRED = ("inning", "inning_topbot", "balls", "strikes", "pitch_number")
EMPTY = {"", "null", "na", "nan", "none"}


def _clean(value) -> str | None:
    value = (value or "").strip()
    return None if value.lower() in EMPTY else value


def _int(value) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


def _float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class Pitch:
    row: int
    raw: dict = field(repr=False, compare=False)

    def get(self, *names: str) -> str | None:
        return next((v for n in names if (v := _clean(self.raw.get(n))) is not None), None)

    @property
    def inning(self) -> int | None: return _int(self.get("inning"))
    @property
    def half(self) -> str | None:
        side = (self.get("inning_topbot") or "").lower()
        return "top" if side.startswith("top") else "bottom" if side.startswith("bot") else None
    @property
    def balls(self) -> int | None: return _int(self.get("balls"))
    @property
    def strikes(self) -> int | None: return _int(self.get("strikes"))
    @property
    def outs(self) -> int | None: return _int(self.get("outs_when_up"))
    @property
    def bases(self) -> tuple[bool, bool, bool] | None:
        if not any(n in self.raw for n in ("on_1b", "on_2b", "on_3b")):
            return None
        return tuple(self.get(n) is not None for n in ("on_1b", "on_2b", "on_3b"))

    def values(self) -> list[dict]:
        """所有進階數據欄位；CSV 沒有或該球沒有的數值保留為 None，不補值。"""
        out = []
        for names, label, unit, source in FIELDS:
            out.append({"key": names[0], "label": label, "unit": unit, "source": source,
                        "value": _float(self.get(*names))})
        return out

    def label(self) -> str:
        half = {"top": "上", "bottom": "下"}.get(self.half, "?")
        count = f"{self.balls}-{self.strikes}" if self.balls is not None and self.strikes is not None else "?-?"
        speed = _float(self.get("release_speed"))
        parts = [self.get("game_date") or "日期未知",
                 f"{self.inning or '?'} 局{half}",
                 f"第 {self.get('at_bat_number') or '?'} 打席第 {self.get('pitch_number') or '?'} 球",
                 f"{count}・{self.outs if self.outs is not None else '?'} 出局",
                 " ".join(x for x in (self.get("pitch_name", "pitch_type"),
                                     f"{speed:.1f} mph" if speed is not None else None) if x),
                 self.get("description") or ""]
        return " · ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        keys = ("game_pk", "game_date", "home_team", "away_team", "player_name", "pitcher", "batter",
                "at_bat_number", "pitch_number", "inning", "inning_topbot", "balls", "strikes", "outs_when_up",
                "p_throws", "stand", "pitch_type", "pitch_name", "description", "events")
        return {"label": self.label(), "row": self.row,
                "context": {k: self.get(k) for k in keys if self.get(k) is not None},
                "values": self.values()}


def load_statcast(path: str | Path) -> list[Pitch]:
    """讀取 Baseball Savant 匯出的逐球 CSV。缺少賽況欄位就不是逐球資料，直接拒絕。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"不是 Baseball Savant 逐球 CSV：缺少欄位 {', '.join(missing)}")
        pitches = [Pitch(i, row) for i, row in enumerate(reader, 1)]
    if not pitches:
        raise ValueError("CSV 沒有任何逐球資料")
    return pitches


def mismatches(pitch: Pitch, game) -> list[str]:
    """這一球與目前手動賽況不同的項目；CSV 沒提供的項目不算不符。"""
    checks = [("局數", pitch.inning, game.inning), ("半局", pitch.half, game.half),
              ("出局數", pitch.outs, game.outs), ("壞球", pitch.balls, game.balls),
              ("好球", pitch.strikes, game.strikes),
              ("壘況", pitch.bases, (game.on_first, game.on_second, game.on_third))]
    return [name for name, have, want in checks if have is not None and have != want]


def candidates(pitches: list[Pitch], game, limit: int = 30) -> list[tuple[Pitch, list[str]]]:
    """依與賽況的吻合程度排序，賽況完全相同的排最前面；同分時維持 CSV 內比賽與打席順序。"""
    def order(p):
        return (_int(p.get("game_pk")) or 0, _int(p.get("at_bat_number")) or 0, _int(p.get("pitch_number")) or 0)
    scored = [(p, mismatches(p, game)) for p in pitches]
    scored.sort(key=lambda x: (len(x[1]), order(x[0])))
    return scored[:limit]


def pitch_markdown(info: dict | None) -> str:
    """報告用：一球的 Statcast 數據表，每列標示來源。"""
    if not info:
        return ""
    rows = [f"| {v['label']} | {v['value']:g} {v['unit']} | {v['source']} |"
            for v in info["values"] if v["value"] is not None]
    table = "\n".join(["| 項目 | 數值 | 來源 |", "|---|---|---|", *rows]) if rows else "這一球的 CSV 沒有進階數據欄位。"
    return (f"## 逐球數據（使用者匯入的 Statcast CSV）\n\n對應的那一球：{info['label']}\n\n"
            "這一球由使用者從候選清單確認，影片無法自動證明兩者是同一球。\n\n" + table)
