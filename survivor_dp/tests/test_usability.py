import time
from pathlib import Path

import pytest

from survivor import data_io, dp
from survivor.config import ConfigError, load_config
from survivor.team_registry import TeamRegistry
from survivor.cli import main as cli_main

EXAMPLE = Path(__file__).resolve().parent.parent / "example_data"


def game(week, away, home, pa, completed="", gid=""):
    return {"Week": str(week), "Away Team": away, "Home Team": home,
            "Away Win %": f"{pa}%", "Home Win %": f"{100 - pa}%",
            "Completed": completed, "Game ID": gid}


# ---- team name resolution --------------------------------------------------

def test_resolver_accepts_full_name_case_nickname_and_abbreviation():
    rows = [game(1, "Denver Broncos", "Kansas City Chiefs", 43, gid="2026-W01-DEN@KC"),
            game(1, "New York Giants", "New York Jets", 50, gid="2026-W01-NYG@NYJ")]
    r = data_io.TeamNameResolver(rows)
    assert r.resolve("Kansas City Chiefs") == "Kansas City Chiefs"
    assert r.resolve("kansas city chiefs") == "Kansas City Chiefs"
    assert r.resolve("Chiefs") == "Kansas City Chiefs"
    assert r.resolve("KC") == "Kansas City Chiefs"
    assert r.resolve("nyj") == "New York Jets"


def test_resolver_refuses_to_guess_ambiguous_or_unknown_names():
    rows = [game(1, "New York Giants", "New York Jets", 50)]
    r = data_io.TeamNameResolver(rows)
    assert r.resolve("New York") is None       # ambiguous
    assert r.resolve("Gaints") is None          # typo
    assert r.resolve("") is None


# ---- current week detection ------------------------------------------------

def test_detect_current_week_is_first_week_with_an_incomplete_game():
    rows = [game(1, "A", "B", 40, "Yes"), game(2, "A", "C", 40, "Yes"),
            game(3, "A", "D", 40, "Yes"), game(3, "B", "C", 40, ""),  # Thursday done, Sunday not
            game(4, "A", "B", 40, "")]
    assert data_io.detect_current_week(rows) == 3


def test_detect_current_week_none_without_completed_column():
    rows = [{"Week": "1", "Away Team": "A", "Home Team": "B",
             "Away Win %": "40%", "Home Win %": "60%"}]
    assert data_io.detect_current_week(rows) is None


# ---- load-report guardrails --------------------------------------------------

def base_games():
    return [game(1, "A", "B", 30, "Yes"), game(1, "C", "D", 40, "Yes"),
            game(2, "A", "C", 35, "Yes"), game(2, "B", "D", 45, "Yes"),
            game(3, "A", "D", 20, ""), game(3, "B", "C", 50, "")]


def test_missing_own_pick_is_blocking():
    picks = [{"Player": "Me", "Week 1": "B", "Week 2": ""}]
    pool = data_io.build_pool_from_rows(base_games(), picks)
    assert pool.load_report.current_week == 3
    assert any("week(s) 2" in m for m in pool.load_report.blocking)


def test_misspelled_own_pick_is_blocking():
    picks = [{"Player": "Me", "Week 1": "B", "Week 2": "Zzz"}]
    pool = data_io.build_pool_from_rows(base_games(), picks)
    assert any("'Zzz'" in m for m in pool.load_report.blocking)


def test_opponent_gap_is_a_warning_not_blocking():
    picks = [{"Player": "Me", "Week 1": "B", "Week 2": "D"},
             {"Player": "Opp", "Week 1": "D", "Week 2": ""}]
    pool = data_io.build_pool_from_rows(base_games(), picks)
    assert pool.load_report.blocking == []
    assert any("Opp has no pick" in m for m in pool.load_report.warnings)


def test_clean_sheet_has_no_problems_and_blocks_reuse():
    picks = [{"Player": "Me", "Week 1": "B", "Week 2": "D"},
             {"Player": "Opp", "Week 1": "D", "Week 2": "C"}]
    pool = data_io.build_pool_from_rows(base_games(), picks)
    assert pool.load_report.blocking == []
    rec = pool.get_recommendations()
    recommended = {c.team for c in rec.pure_survival_top}
    assert "B" not in recommended and "D" not in recommended  # already used


def test_listed_player_with_no_picks_yet_still_counts_as_opponent():
    picks = [{"Player": "Me", "Week 1": "B", "Week 2": "D"},
             {"Player": "Newbie", "Week 1": "", "Week 2": ""}]
    pool = data_io.build_pool_from_rows(base_games(), picks)
    assert "Newbie" in pool.load_report.opponents_alive


# ---- robust sheet reading ------------------------------------------------------

def test_value_grid_reader_tolerates_blank_and_trailing_header_columns():
    values = [["Player", "Week 1", "", ""],
              ["Me", "Chiefs", "", "stray"],
              ["", "", "", ""]]
    rows = data_io._rows_from_value_grid(values)
    assert rows == [{"Player": "Me", "Week 1": "Chiefs"}]


def test_spreadsheet_id_extracted_from_url_or_passed_through():
    url = "https://docs.google.com/spreadsheets/d/1AbC_def-123/edit#gid=0"
    assert data_io.spreadsheet_id_from(url) == "1AbC_def-123"
    assert data_io.spreadsheet_id_from("1AbC_def-123") == "1AbC_def-123"


# ---- config ------------------------------------------------------------------

def test_config_placeholder_gives_a_helpful_error(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[sheet]\nspreadsheet = "PASTE_YOUR_SHEET_URL_HERE"\n')
    with pytest.raises(ConfigError, match="paste your Google Sheet"):
        load_config(str(cfg))


def test_config_credentials_resolved_relative_to_config_file(tmp_path):
    (tmp_path / "proj").mkdir()
    cfg = tmp_path / "proj" / "config.toml"
    cfg.write_text('[sheet]\nspreadsheet = "https://docs.google.com/spreadsheets/d/X/edit"\n'
                   'credentials = "../nfl_prob_grid/service_account.json"\n')
    c = load_config(str(cfg))
    assert c.credentials == (tmp_path / "nfl_prob_grid" / "service_account.json").resolve()


# ---- CLI (offline mode) ------------------------------------------------------

def test_cli_check_and_recommend_on_example_data(capsys):
    assert cli_main(["check", "--csv-dir", str(EXAMPLE)]) == 0
    assert "No problems found" in capsys.readouterr().out
    assert cli_main(["recommend", "--csv-dir", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "Ranking A" in out and "Ranking B" in out and "Next steps" in out


def test_cli_recommend_refuses_when_blocking_problems(tmp_path, capsys):
    (tmp_path / "master_game_table.csv").write_text(
        (EXAMPLE / "master_game_table.csv").read_text())
    (tmp_path / "pool_picks.csv").write_text("Player,Week 1,Week 2\nMe,Buffalo Bills,\n")
    assert cli_main(["recommend", "--csv-dir", str(tmp_path)]) == 1
    assert "Not producing a recommendation" in capsys.readouterr().out


# ---- scale -------------------------------------------------------------------

def test_pure_survival_solves_full_season_horizon_quickly():
    import random
    from survivor.vegas import Game, WeekSlate
    random.seed(0)
    teams = [f"T{i}" for i in range(32)]
    slates = {}
    for wk in range(1, 19):
        playing = teams[:]
        random.shuffle(playing)
        if 5 <= wk <= 14:
            playing = playing[:28]  # 4 teams on bye
        slates[wk] = WeekSlate(wk, [Game.from_probability(wk, playing[i], playing[i + 1],
                                                          random.uniform(0.1, 0.9))
                                    for i in range(0, len(playing), 2)])
    t = time.time()
    value, policy = dp.pure_survival_value(TeamRegistry(), 0, 1, slates)
    assert time.time() - t < 2.0
    assert len(policy) == 18 and len(set(policy.values())) == 18
    assert 0.0 < value < 1.0
