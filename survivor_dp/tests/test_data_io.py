import math
from survivor import data_io


def test_parse_master_game_table_uses_probabilities_directly():
    rows = [
        {"Week": "2", "Date": "2026-09-17", "Away Team": "Detroit Lions",
         "Home Team": "Buffalo Bills", "Away Win %": "31.5%", "Home Win %": "68.5%",
         "Source": "Fresh line"},
    ]
    slates, skipped = data_io.parse_master_game_table_rows(rows)
    assert skipped == []
    assert 2 in slates
    probs = slates[2].win_probs()
    assert probs["Buffalo Bills"] > probs["Detroit Lions"]
    assert math.isclose(probs["Buffalo Bills"], 0.685, abs_tol=1e-9)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


def test_parse_master_game_table_accepts_numeric_fraction_not_just_percent_string():
    # gspread can hand back the underlying numeric value (0.685) rather than
    # the display string ("68.5%") depending on how the sheet is read.
    rows = [
        {"Week": "2", "Away Team": "Detroit Lions", "Home Team": "Buffalo Bills",
         "Away Win %": 0.315, "Home Win %": 0.685},
    ]
    slates, skipped = data_io.parse_master_game_table_rows(rows)
    assert skipped == []
    probs = slates[2].win_probs()
    assert math.isclose(probs["Buffalo Bills"], 0.685, abs_tol=1e-9)


def test_parse_master_game_table_infers_missing_side_from_complement():
    rows = [
        {"Week": "2", "Away Team": "Detroit Lions", "Home Team": "Buffalo Bills",
         "Away Win %": "31.5%", "Home Win %": ""},
    ]
    slates, skipped = data_io.parse_master_game_table_rows(rows)
    assert skipped == []
    probs = slates[2].win_probs()
    assert math.isclose(probs["Buffalo Bills"], 0.685, abs_tol=1e-9)


def test_parse_master_game_table_skips_genuinely_malformed_rows():
    rows = [
        {"Week": "5", "Away Team": "Team A", "Home Team": "Team B",
         "Away Win %": "45.0%", "Home Win %": "55.0%"},
        {"Week": "5", "Away Team": "Team C", "Home Team": "Team D",
         "Away Win %": "", "Home Win %": ""},  # no probability at all -- malformed
        {"Week": "", "Away Team": "Team E", "Home Team": "Team F",
         "Away Win %": "50.0%", "Home Win %": "50.0%"},  # missing week number
    ]
    slates, skipped = data_io.parse_master_game_table_rows(rows)
    assert 5 in slates
    assert len(slates[5].games) == 1
    assert slates[5].games[0].team_a == "Team A"
    assert len(skipped) == 2


def test_master_game_table_pairing_preserved_for_correlated_elimination():
    """The whole reason this schema is used over the plain Team x Week grid:
    pool_math.py needs to know WHICH TWO teams share a game."""
    rows = [
        {"Week": "3", "Away Team": "Atlanta Falcons", "Home Team": "Green Bay Packers",
         "Away Win %": "32.0%", "Home Win %": "68.0%"},
    ]
    slates, _ = data_io.parse_master_game_table_rows(rows)
    g = slates[3].game_for_team("Atlanta Falcons")
    assert g.opponent_of("Atlanta Falcons") == "Green Bay Packers"


def test_parse_pool_picks_wide_format_and_no_lookahead():
    rows = [
        {"Player": "Player 1", "Week 1": "Chiefs", "Week 2": "Ravens", "Week 3": "Eagles"},
        {"Player": "Player 2", "Week 1": "Bills", "Week 2": "", "Week 3": "Cowboys"},
        {"Player": "Me", "Week 1": "Ravens", "Week 2": "Chiefs", "Week 3": "Bills"},
    ]
    # current_week = 3 -> week 3 data must NOT be loaded (no look-ahead)
    opp_hist, my_hist, eliminated = data_io.parse_pool_picks_rows(rows, current_week=3)

    assert opp_hist["Player 1"] == [(1, "Chiefs"), (2, "Ravens")]
    assert opp_hist["Player 2"] == [(1, "Bills")]
    assert my_hist == [(1, "Ravens"), (2, "Chiefs")]
    assert "Me" not in opp_hist
    assert eliminated == {}
    # Week 3 picks should be nowhere in the output despite being in the rows.
    for _, team in opp_hist["Player 1"]:
        assert team != "Eagles"
    assert ("Bills", 3) not in my_hist


def test_parse_pool_picks_elimination_markers():
    rows = [
        {"Player": "Player 1", "Week 1": "Chiefs", "Week 2": "OUT", "Week 3": ""},
        {"Player": "Me", "Week 1": "Ravens", "Week 2": "Chiefs", "Week 3": ""},
    ]
    opp_hist, my_hist, eliminated = data_io.parse_pool_picks_rows(rows, current_week=3)
    assert eliminated == {"Player 1": 2}
    # The OUT marker itself shouldn't show up as a "team pick"
    assert all(team != "OUT" for _, team in opp_hist.get("Player 1", []))


def test_parse_pool_picks_blank_cell_is_not_treated_as_elimination():
    rows = [
        {"Player": "Player 1", "Week 1": "Chiefs", "Week 2": "", "Week 3": ""},
        {"Player": "Me", "Week 1": "Ravens", "Week 2": "Chiefs", "Week 3": ""},
    ]
    _, _, eliminated = data_io.parse_pool_picks_rows(rows, current_week=3)
    assert eliminated == {}


def test_build_pool_from_rows_end_to_end():
    game_rows = [
        {"Week": "1", "Away Team": "Detroit Lions", "Home Team": "Buffalo Bills",
         "Away Win %": "31.5%", "Home Win %": "68.5%"},
        {"Week": "1", "Away Team": "Houston Texans", "Home Team": "Baltimore Ravens",
         "Away Win %": "20.0%", "Home Win %": "80.0%"},
        {"Week": "2", "Away Team": "Cincinnati Bengals", "Home Team": "Houston Texans",
         "Away Win %": "43.6%", "Home Win %": "56.4%"},
    ]
    picks_rows = [
        {"Player": "Dave", "Week 1": "Buffalo Bills", "Week 2": ""},
        {"Player": "Me", "Week 1": "Baltimore Ravens", "Week 2": ""},
    ]
    pool = data_io.build_pool_from_rows(game_rows, picks_rows, current_week=2,
                                         verbose=False)
    assert pool.current_week == 2
    assert pool.registry.is_used(pool.my_used_mask, "Baltimore Ravens")
    assert "Dave" in pool.opponents
    assert pool.opponents["Dave"].used_mask & pool.registry.bit("Buffalo Bills")
    # Week 2 has a line loaded (for the NEXT decision), current week works.
    rec_ready = 2 in pool.future_slates
    assert rec_ready
