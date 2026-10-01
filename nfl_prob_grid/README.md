# NFL Season-Long Probability Grid — Refresh Engine

Maintains a continuously updated **Team × Week (1–18) win-probability grid** from an initial
full-season set of Vegas lines, irregular Vegas refreshes, and preseason win totals (used to anchor a
market-perceived team-strength rating implied by the game lines). It produces the grid only — no
Survivor selection or optimisation. Methodology and weaknesses: **[DESIGN.md](DESIGN.md)**.
All assumptions: **[config.toml](config.toml)**.

## Quick start
```bash
pip install -r requirements.txt            # Python >= 3.11
python demo/run_demo.py                    # 8-week end-to-end run on SYNTHETIC data
python -m pytest tests -q                  # 95 tests
```

## Google Sheets (recommended)
The sheet can be the whole interface: your odds script fills **Game Lines**, and `python -m nfl_prob_grid sheets build`
writes the grid, a dated archive tab, the Master Game Table, Team Ratings and a Run Log. Setup: **[SHEETS_SETUP.md](SHEETS_SETUP.md)**.

## Real use (files)
```bash
# 0. Edit config.toml: column names, paths (or Google Sheets URLs), current_week, parameters.
# 1. Preseason: build master v1 from the full-season lines + preseason win totals.
#    Prints the win-total-vs-preseason-lines gap and the weight win totals get in the baseline.
python -m nfl_prob_grid init    --as-of 2026-08-25T00:00:00Z
# 2. Every refresh: (--current-week = first week NOT yet completed; earlier weeks freeze)
python -m nfl_prob_grid refresh --current-week 3 --lines <path-or-sheet-url>
python -m nfl_prob_grid refresh --current-week 3 --dry-run        # compute + report, write nothing
python -m nfl_prob_grid status | validate | grid
```
Use `--as-of <ISO time>` for reproducible reruns; default is "now". `--no-lines` runs a
refresh with no new lines (time passes; ratings' uncertainty grows; ages advance). Each run prints a summary (lines refreshed, ignored, rejected
rows with reasons, stale-adjusted count); full detail is in `state/runs/run_v000N.json`.

## Inputs
| | Shape | Notes |
|---|---|---|
| A: initial lines | one row per game: week, home, away, home odds, away odds, timestamp (+ optional game id, date) | must be complete; any bad row aborts `init` |
| B: refreshed lines | same shape, any number of rows/weeks, both teams' odds on one row | headers configurable (defaults `Week, Away Team, Away Odds, Home Team, Home Odds, Timestamp`) |
| C: preseason win totals | one row per team: team, win total (+ optional over/under decimal odds) | read **once, at `init`**; all 32 teams required; cannot be refreshed mid-season and does not need to be |

Decimal odds. Team names may be full names, nicknames, unambiguous cities or abbreviations; add
extras under `[team_aliases]`. Google Sheets: paste the sheet URL as the path (must be link-shared or
published); **this path is untested in my environment — local files are tested.**

## Outputs
* `output/prob_grid_latest.csv` (+ `prob_grid_v000N.csv`): `Team, Week 1 … Week 18`; probability team wins;
  bye weeks blank. Opposing cells are complementary. Derived from the master; never edited by hand.
* `output/team_ratings_latest.csv` / `team_ratings_history.csv`: the market-perceived power ratings (logit units,
  plus spread-point equivalents): current, baseline, win-total-only, change since baseline, uncertainty.
* `output/master_game_table_latest.csv` = `state/master_current.csv`: the game-level source of truth with
  full lineage (which probability came from a fresh line, an older line, or a stale line adjusted for
  strength movement, and every input to that adjustment). Field dictionary: DESIGN.md §5.
* `state/versions/`, `state/changes/`, `state/runs/`, `state/manifest.json`: full history and audit trail.

## Safety behaviour
Weeks before `--current-week` are frozen and can never change (verified by exact equality every run;
lowering `current_week` aborts). Only strictly newer timestamps replace a line. Any validation error
aborts with the previous master untouched; state is hash-verified on load; commits are atomic with the
manifest written last. Re-initialising requires `--force` and moves old state aside rather than deleting it.

## Python API
```python
from nfl_prob_grid import load_config, run_init, run_refresh, regenerate_grid
cfg = load_config("config.toml")
run_refresh(cfg, current_week=4, as_of="2026-09-29T15:00:00Z", lines_source="refresh.csv")
```
`apply_init` / `apply_refresh` are pure (DataFrames in, DataFrames out) for notebooks and tests.

## Testing on real data
**Backtested on 2012–2025 real data — see [BACKTEST_RESULTS.md](BACKTEST_RESULTS.md).** Reproduce with
`python backtest/run_backtest.py` and `python backtest/replay_engine.py` (data included in `backtest/data/`).
See **[BACKTEST_PLAN.md](BACKTEST_PLAN.md)**: data sources (nflverse closing lines verified; win totals from Pro-Football-Reference),
a test ladder, and a real-data pilot (`backtest/pilot_real_data.py`, output in `backtest/pilot_output.txt`).

## Layout
`nfl_prob_grid/` package · `config.toml` · `backtest/` (real-data pilot) · `demo/` (synthetic data generator, demo run, sanity evaluation) ·
`tests/`. **Demo data are synthetic: schedule, strengths and odds are simulated, not real NFL lines.**
