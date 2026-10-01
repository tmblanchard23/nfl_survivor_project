import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from make_demo_data import FULL, INIT_AS_OF, build_demo, run_as_of  # noqa: E402
from nfl_prob_grid import apply_init, load_config  # noqa: E402
from nfl_prob_grid.probability import logit, sigmoid  # noqa: E402
from nfl_prob_grid.tabular import read_table  # noqa: E402


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    return build_demo(tmp_path_factory.mktemp("demo"), seed=11)


@pytest.fixture()
def cfg(tmp_path):
    return load_config(ROOT / "config.toml", paths__state_dir=str(tmp_path / "state"),
                       paths__output_dir=str(tmp_path / "out"), time__input_timezone="UTC")  # engine tests are tz-agnostic


@pytest.fixture()
def init_state(demo, cfg):
    """(master, rating history) for a freshly initialised season, in memory."""
    d = demo["dir"]
    res = apply_init(read_table(str(d / "initial_lines.csv")), read_table(str(d / "win_totals.csv")),
                     cfg, as_of=pd.Timestamp(INIT_AS_OF), current_week=1)
    return res.master, res.hist


def _odds(p: float, vig: float = 1.045):
    return round(1.0 / (p * vig), 4), round(1.0 / ((1.0 - p) * vig), 4)


def line_rows(rows: list[tuple], ts: str) -> pd.DataFrame:
    """rows: (week, away, away_odds, home, home_odds) using canonical ids."""
    return pd.DataFrame([{"Week": w, "Away Team": a, "Away Odds": ao, "Home Team": h,
                          "Home Odds": ho, "Timestamp": ts} for w, a, ao, h, ho in rows])


def implied_lines(master, hist, weeks, ts, shifts: dict | None = None, use_anchor=False) -> pd.DataFrame:
    """Fresh lines for every game in ``weeks``.  Each is the BASELINE-RATING-implied probability
    (zero innovation) plus ``shifts[team]`` logit in that team's favour, so tests control exactly
    how much new information the filter sees.  use_anchor=True starts from the stored line instead."""
    r = hist.ratings(0)
    rows = []
    for g in master[master.week.isin(weeks)].itertuples():
        base = (logit(g.latest_vegas_home_prob) if use_anchor
                else r[g.home_team] - r[g.away_team] + hist.hfa)
        base += (shifts or {}).get(g.home_team, 0.0) - (shifts or {}).get(g.away_team, 0.0)
        oh, oa = _odds(float(sigmoid(base)))
        rows.append((g.week, g.away_team, oa, g.home_team, oh))
    return line_rows(rows, ts)


def pick_game(master: pd.DataFrame, *, min_week=10, team=None, home=None):
    m = master[master.week >= min_week]
    if team:
        m = m[(m.home_team == team) | (m.away_team == team)]
        if home is not None:
            m = m[(m.home_team == team) == home]
    return m.iloc[0]


def team_playing_in(master, weeks) -> str:
    """A team with a game in every one of ``weeks``."""
    sets = [set(master[master.week == w].home_team) | set(master[master.week == w].away_team) for w in weeks]
    return sorted(set.intersection(*sets))[0]
