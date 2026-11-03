import csv

import pytest

from bbagent.analysis.statcast import candidates, load_statcast, pitch_markdown
from bbagent.schemas import GameState
from bbagent.ui.studio import statcast_panel

COLUMNS = ["pitch_type", "game_date", "release_speed", "player_name", "description", "stand", "p_throws",
           "balls", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "inning_topbot",
           "launch_speed", "launch_angle", "release_spin_rate", "game_pk", "at_bat_number", "pitch_number",
           "pitch_name", "estimated_ba_using_speedangle"]
ROWS = [
    # 7 局下、一出局、一壘有人、1-2：揮空，沒有擊球數據
    {"pitch_type": "SL", "game_date": "2017-11-01", "release_speed": "87.3", "description": "swinging_strike",
     "stand": "L", "p_throws": "R", "balls": "1", "strikes": "2", "on_1b": "605141", "outs_when_up": "1",
     "inning": "7", "inning_topbot": "Bot", "release_spin_rate": "2510", "game_pk": "526517",
     "at_bat_number": "50", "pitch_number": "4", "pitch_name": "Slider"},
    # 同一打席的上一球，球數不同
    {"pitch_type": "FF", "game_date": "2017-11-01", "release_speed": "95.1", "description": "foul",
     "stand": "L", "p_throws": "R", "balls": "1", "strikes": "1", "on_1b": "605141", "outs_when_up": "1",
     "inning": "7", "inning_topbot": "Bot", "release_spin_rate": "2290", "launch_speed": "",
     "game_pk": "526517", "at_bat_number": "50", "pitch_number": "3", "pitch_name": "4-Seam Fastball"},
    # 別的局：擊出球
    {"pitch_type": "CU", "game_date": "2017-11-01", "release_speed": "80.2", "description": "hit_into_play",
     "stand": "R", "p_throws": "L", "balls": "0", "strikes": "0", "outs_when_up": "0", "inning": "3",
     "inning_topbot": "Top", "launch_speed": "101.4", "launch_angle": "24", "game_pk": "526517",
     "at_bat_number": "20", "pitch_number": "1", "pitch_name": "Curveball", "estimated_ba_using_speedangle": ".612"},
]


@pytest.fixture
def savant_csv(tmp_path):
    path = tmp_path / "savant_data.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(ROWS)
    return path


def game(**kw):
    base = dict(inning=7, half="bottom", outs=1, on_first=True, balls=1, strikes=2, our_role="offense")
    return GameState(**{**base, **kw})


def test_candidates_rank_matching_situation_first(savant_csv):
    pitches = load_statcast(savant_csv)
    ranked = candidates(pitches, game())
    assert [(p.get("pitch_number"), diff) for p, diff in ranked[:2]] == [("4", []), ("3", ["好球"])]
    assert "局數" in ranked[2][1] and "半局" in ranked[2][1]
    assert "7 局下" in ranked[0][0].label() and "Slider 87.3 mph" in ranked[0][0].label()


def test_values_keep_gaps_and_sources(savant_csv):
    slider, _, curve = load_statcast(savant_csv)
    values = {v["key"]: v for v in slider.values()}
    assert values["release_spin_rate"]["value"] == 2510 and values["launch_speed"]["value"] is None
    assert values["bat_speed"]["value"] is None          # 2017 年的 CSV 沒有棒速欄位
    batted = {v["key"]: v for v in curve.values()}
    assert batted["launch_speed"]["value"] == 101.4 and "估計" in batted["launch_speed"]["source"]
    assert batted["estimated_ba_using_speedangle"]["source"] == "Statcast 模型估計"
    report = pitch_markdown(slider.to_dict())
    assert "2510 rpm" in report and "擊球初速" not in report and "使用者從候選清單確認" in report


def test_panel_shows_blanks_instead_of_video_estimates(savant_csv):
    slider = load_statcast(savant_csv)[0]
    html = statcast_panel(slider.to_dict())
    assert "2510" in html and "這一球沒有此數據" in html and "<script" not in html
    assert "影片無法量測" in statcast_panel(None)


def test_rejects_non_savant_csv(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("name,score\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少欄位"):
        load_statcast(path)


def test_apply_pitch_context_fills_game_state(savant_csv):
    from bbagent.ui.workspace import apply_pitch_context, import_statcast
    pitches, choices, note, *_ = import_statcast(str(savant_csv), 7, "bottom", 1, ["1"], 1, 2, 0, "offense",
                                                 "auto", "", "", "")
    assert choices["choices"][0][0].startswith("✓ 賽況相符") and "1 球與目前賽況相符" in note
    inning, half, outs, bases, balls, strikes, hand, batter_hand = apply_pitch_context(pitches, 3)
    assert (inning, half, outs, bases, balls, strikes, hand, batter_hand) == (3, "top", 0, [], 0, 0, "L", "R")
