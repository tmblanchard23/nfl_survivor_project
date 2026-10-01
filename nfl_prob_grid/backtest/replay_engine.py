"""L4 full-engine replay: push real seasons through the PRODUCTION package (run_init + weekly run_refresh,
real files, hash-verified state) and score it against real closing lines.

Per season:
  * init: win totals (real) + a full-season "preseason line" table.  Historical preseason game lines do not
    exist, so these are the WIN-TOTAL-IMPLIED lines (real win totals, real schedule, walk-forward home field).
  * run for current_week = k (k = 1..17): the refresh table holds the REAL closing moneylines of weeks k and k+1,
    written with raw nflverse team codes (exercises alias handling), stamped the Tuesday of week k.
  * after each run: every game >= 2 weeks beyond the refreshed slate is scored - stale anchor
    (latest_vegas_home_prob) vs the engine's current probability (home_win_prob) vs that game's real closing line.
Caveat: refreshed week-(k+1) lines are closing lines, i.e. slightly later information than a Tuesday API pull.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))
import run_backtest as B  # noqa: E402
from nfl_prob_grid import load_config, run_init, run_refresh  # noqa: E402
from nfl_prob_grid.master import StateStore  # noqa: E402
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))  # vectorised

SEASONS = [2021, 2022, 2023, 2024, 2025]
CONFIGS = {   # "previous" = pre-backtest defaults, pinned explicitly; "backtested" = the shipped defaults
    "previous_defaults": dict(aging__half_weight_weeks=3.0, aging__max_weight=0.8, strength__threshold_up=0.12,
                              adjust__cap=0.4),
    "backtested_defaults": {},
}


def american_to_decimal(ml):
    return np.where(ml > 0, 1 + ml / 100.0, 1 + 100.0 / np.abs(ml))


def replay(season: int, cfg_name: str, g_raw: pd.DataFrame, g: pd.DataFrame, hfa: dict, work: Path) -> pd.DataFrame:
    d = work / f"{season}_{cfg_name}"
    shutil.rmtree(d, ignore_errors=True)
    (d / "in").mkdir(parents=True)
    gs = g[g.season == season].copy()
    raw = g_raw[(g_raw.season == season) & (g_raw.game_type == "REG")].copy()
    tue1 = pd.Timestamp(gs[gs.week == 1].gameday.min(), tz="UTC") - pd.Timedelta(days=2) + pd.Timedelta(hours=15)
    t_run = lambda k: tue1 + pd.Timedelta(days=7 * (k - 1))  # noqa: E731
    pre_ts = tue1 - pd.Timedelta(days=12)

    # --- inputs: win totals (real) + win-total-implied preseason lines
    shutil.copy(B.DATA / f"win_totals_{season}.csv", d / "in" / "win_totals.csv")
    r = B.win_total_ratings(g, season, hfa[season])
    p = sigmoid(r[gs.hi] - r[gs.ai] + np.where(gs.neutral, 0.0, hfa[season]))
    pre = pd.DataFrame({"week": gs.week, "date": gs.gameday, "away_team": gs.away_team, "home_team": gs.home_team,
                        "home_odds": 1 / (p * 1.045), "away_odds": 1 / ((1 - p) * 1.045),
                        "timestamp": pre_ts.strftime("%Y-%m-%d %H:%M:%S")})
    pre.to_csv(d / "in" / "initial.csv", index=False)
    neutral = [f"{season}-W{w:02d}-{a}@{h}" for w, a, h in
               zip(gs[gs.neutral].week, gs[gs.neutral].away_team, gs[gs.neutral].home_team)]

    cfg = load_config(ROOT / "config.toml", paths__state_dir=str(d / "state"), paths__output_dir=str(d / "out"),
                      season__season=season, season__games_per_team=17 if season >= 2021 else 16,
                      rating__hfa_mode="fixed", rating__hfa_logit=hfa[season],
                      rating__neutral_site_games=neutral, **CONFIGS[cfg_name])
    cfg.team_aliases = {"LA": "LAR"}                           # nflverse uses LA for the Rams
    cfg.columns_initial.game_id, cfg.columns_initial.date = "", "date"
    cfg.columns_win_totals.over_odds = cfg.columns_win_totals.under_odds = ""
    cfg.columns_win_totals.total = "Win Total"
    run_init(cfg, as_of=pre_ts + pd.Timedelta(days=1), current_week=1, lines_source=str(d / "in" / "initial.csv"),
             win_totals_source=str(d / "in" / "win_totals.csv"))

    truth = gs.set_index(gs.apply(lambda x: f"{season}-W{x.week:02d}-{x.away_team}@{x.home_team}", axis=1))["y"]
    scored = []
    for k in range(1, 18):
        wk = raw[raw.week.isin([k, k + 1])]
        tbl = pd.DataFrame({"Week": wk.week, "Away Team": wk.away_team, "Home Team": wk.home_team,
                            "Away Odds": american_to_decimal(wk.away_moneyline.to_numpy()).round(4),
                            "Home Odds": american_to_decimal(wk.home_moneyline.to_numpy()).round(4),
                            "Timestamp": (t_run(k) - pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")})
        f = d / "in" / f"refresh_{k:02d}.csv"
        tbl.to_csv(f, index=False)
        rep = run_refresh(cfg, as_of=t_run(k), current_week=k, lines_source=str(f))
        if rep.rejected_rows:
            raise RuntimeError(f"{season} week {k}: rejected rows {rep.rejected_rows[:3]}")
        m, _, _ = StateStore(cfg.paths.state_dir).load()
        fut = m[m.week >= k + 2]
        lg = lambda s_: np.log(s_ / (1 - s_))  # noqa: E731
        scored.append(pd.DataFrame({"season": season, "origin": k, "game_id": fut.game_id,
                                    "y": truth.reindex(fut.game_id).to_numpy(),
                                    "stale": lg(fut.latest_vegas_home_prob).to_numpy(),
                                    "engine": lg(fut.home_win_prob).to_numpy(),
                                    "adj_active": fut.adj_active.to_numpy(),
                                    "skill": [rep.details.get("adjustment_skill")] * len(fut)}))
    man = StateStore(cfg.paths.state_dir).read_manifest()
    assert man["current_version"] == 18, man["current_version"]
    return pd.concat(scored, ignore_index=True)


def main() -> None:
    g_raw = pd.read_csv(B.DATA / "nflverse_games.csv")
    g = B.load_games()
    hfa, _ = B.walk_forward_hfa(g)
    work = Path(tempfile.mkdtemp(prefix="replay_"))
    res = {}
    for name in CONFIGS:
        res[name] = pd.concat([replay(s, name, g_raw, g, hfa, work) for s in SEASONS], ignore_index=True)
    print(f"L4 FULL-ENGINE REPLAY 2021-2025: {len(SEASONS)} seasons x 18 engine versions x {len(CONFIGS)} configs, "
          "all runs validated, 0 rejected rows")
    rows = []
    for name, x in res.items():
        x = x.dropna(subset=["y"])
        e_st, e_en = B.rmse(x.y - x.stale), B.rmse(x.y - x.engine)
        by = {s: 1 - B.rmse(z.y - z.engine) / B.rmse(z.y - z.stale) for s, z in x.groupby("season")}
        rows.append({"config": name, "predictions": len(x), "stale_rmse": round(e_st, 4), "engine_rmse": round(e_en, 4),
                     "error_cut": f"{1 - e_en / e_st:.1%}", "share_adjusted": f"{x.adj_active.mean():.0%}",
                     "per_season_cut": {s: f"{v:.1%}" for s, v in by.items()}})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    skill = res["backtested_defaults"].dropna(subset=["skill"]).drop_duplicates(["season", "origin"])
    sk = pd.DataFrame(list(skill.skill))
    if len(sk):
        print(f"built-in live skill check (backtested defaults): {int(sk.n.sum())} adjusted games later got fresh lines; "
              f"mean error {sk.mean_abs_err_stale_logit.mean():.3f} (unadjusted) -> {sk.mean_abs_err_adjusted_logit.mean():.3f} (adjusted)")
    B.OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(B.OUT / "t4_engine_replay.csv", index=False)


if __name__ == "__main__":
    main()
