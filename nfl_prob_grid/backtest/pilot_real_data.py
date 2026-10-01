"""PILOT on REAL NFL closing lines (nflverse): calibrate the rating filter and test the adjustment policy.

What this can and cannot do
  * Uses REAL weekly closing moneylines (2010-2025).  No look-ahead: predictions for a game use only
    lines from weeks strictly before it; parameters are reported for the full sample AND walk-forward.
  * The preseason anchor is a STAND-IN (prior-season market ratings, regressed) because historical
    preseason win totals could not be fetched from this environment.  Swap in win totals with
    `--anchor win_totals --win-totals-dir DIR` (files `win_totals_<season>.csv`, columns Team, Win Total).
  * Game-specific stale lines do not exist historically, so the stale anchor for every game is the
    rating-implied line at the preseason.  The COMPARISON adjusted-vs-stale is still valid: a game's
    idiosyncratic offset is common to both predictors and cancels.

Usage: python backtest/pilot_real_data.py [--games PATH]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nfl_prob_grid import load_config  # noqa: E402
from nfl_prob_grid.adjustment import age_weight, cap_adjustment  # noqa: E402
from nfl_prob_grid.ratings import (N, TEAMS, estimate_hfa, kalman_update, normalize_win_totals,  # noqa: E402
                                   solve_win_total_ratings, team_idx)
from nfl_prob_grid.strength import strength_shift  # noqa: E402
from nfl_prob_grid.teams import TeamNormalizer  # noqa: E402

ANCHOR = {"mode": "carryover", "dir": None}     # set from the command line
REMAP = {"OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR"}
URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def load_games(path: str | None) -> pd.DataFrame:
    g = pd.read_csv(path or URL)
    g = g[(g.game_type == "REG") & g.home_moneyline.notna() & g.away_moneyline.notna() & g.home_score.notna()].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(REMAP)
    dec = lambda ml: np.where(ml > 0, 1 + ml / 100.0, 1 + 100.0 / np.abs(ml))  # noqa: E731
    qh, qa = 1 / dec(g.home_moneyline.to_numpy()), 1 / dec(g.away_moneyline.to_numpy())
    g["p_home"] = qh / (qh + qa)                                  # proportional de-vig (same as the engine)
    g["y"] = np.log(g.p_home / (1 - g.p_home))
    g["hi"], g["ai"] = team_idx(g.home_team), team_idx(g.away_team)
    g["neutral"] = g.location.eq("Neutral")
    return g[g.season >= 2010].reset_index(drop=True)


def season_hfa(g: pd.DataFrame) -> dict:
    """Walk-forward home field: mean regression intercept of the previous 3 seasons (2020 excluded)."""
    raw = {s: estimate_hfa(d.y.to_numpy(), d.hi.to_numpy(), d.ai.to_numpy(), d.neutral.to_numpy())
           for s, d in g.groupby("season")}
    out = {}
    for s in raw:
        prev = [raw[p] for p in range(s - 3, s) if p in raw and p != 2020]
        out[s] = float(np.mean(prev)) if prev else float("nan")
    return out, raw


def prior_ratings(gs_prev: pd.DataFrame, hfa_prev: float, rho: float) -> np.ndarray:
    """STAND-IN preseason anchor: last season's closing-line ratings, regressed toward the mean."""
    y = gs_prev.y.to_numpy() - np.where(gs_prev.neutral, 0.0, hfa_prev)
    r, _, _ = kalman_update(np.zeros(N), 100 * np.eye(N), gs_prev.hi.to_numpy(), gs_prev.ai.to_numpy(), y,
                            np.full(len(y), 0.15 ** 2))
    return rho * (r - r.mean())


def anchor_ratings(g: pd.DataFrame, s: int, hfa_by: dict, rho: float) -> np.ndarray:
    """Preseason ratings for season s: prior-season carryover (stand-in) or PRESEASON WIN TOTALS.

    Win-total mode: totals -> vig removed (additive) -> Newton solve over the season's ACTUAL schedule
    with the walk-forward home field.  Uses no game results from season s."""
    if ANCHOR["mode"] == "win_totals":
        wt = pd.read_csv(Path(ANCHOR["dir"]) / f"win_totals_{s}.csv")
        nz = TeamNormalizer({"LA": "LAR"})
        tot = pd.Series({nz(t): float(v) for t, v in zip(wt["Team"], wt["Win Total"])}).reindex(TEAMS)
        if tot.isna().any():
            raise SystemExit(f"win_totals_{s}.csv is missing teams: {list(tot[tot.isna()].index)}")
        gs = g[g.season == s]
        target, _ = normalize_win_totals(tot.to_numpy(), len(gs), "additive")
        return solve_win_total_ratings(target, gs.hi.to_numpy(), gs.ai.to_numpy(),
                                       np.where(gs.neutral, 0.0, hfa_by[s]))
    prev = hfa_by.get(s - 1, np.nan)
    return prior_ratings(g[g.season == s - 1], 0.25 if np.isnan(prev) else prev, rho)


def run_season(gs: pd.DataFrame, r0: np.ndarray, hfa: float, q: float, sd_r: float, prior_sd: float):
    """Walk-forward Kalman over one season.  Returns (one-step errors, ratings after each week)."""
    r, cov = r0.copy(), prior_sd ** 2 * np.eye(N)
    after, errs = {0: r0.copy()}, []
    for k in sorted(gs.week.unique()):
        d = gs[gs.week == k]
        if k > 1:
            cov = cov + q ** 2 * np.eye(N)
        h = np.where(d.neutral, 0.0, hfa)
        mu = r[d.hi] - r[d.ai] + h
        errs.append(d.y.to_numpy() - mu)
        r, cov, _ = kalman_update(r, cov, d.hi.to_numpy(), d.ai.to_numpy(), d.y.to_numpy() - h,
                                  np.full(len(d), sd_r ** 2))
        after[int(k)] = r.copy()
    return np.concatenate(errs), after


def one_step_rmse(g, hfa_by, params, seasons):
    q, sd_r, prior_sd, rho = params
    se = []
    for s in seasons:
        r0 = anchor_ratings(g, s, hfa_by, rho)
        e, _ = run_season(g[g.season == s], r0, hfa_by[s], q, sd_r, prior_sd)
        se.append(e ** 2)
    return float(np.sqrt(np.mean(np.concatenate(se))))


def multi_horizon(g, hfa_by, params, seasons, cfg):
    q, sd_r, prior_sd, rho = params
    rows = []
    for s in seasons:
        gs = g[g.season == s]
        r0 = anchor_ratings(g, s, hfa_by, rho)
        _, after = run_season(gs, r0, hfa_by[s], q, sd_r, prior_sd)
        for k in range(2, 15):                                    # origin: info through week k
            tgt = gs[gs.week >= k + 2]                            # weeks beyond the refreshed slate
            if tgt.empty:
                continue
            h = np.where(tgt.neutral, 0.0, hfa_by[s])
            rk = after[k]
            mv = (rk - r0) - np.median(rk - r0)                    # centred team movement since the anchor
            mu_a = r0[tgt.hi] - r0[tgt.ai] + h
            mu_f = rk[tgt.hi] - rk[tgt.ai] + h
            age = k + 1.7                                          # weeks since preseason lines at run k
            w = age_weight(age, cfg.aging)
            adj = [cap_adjustment(w * (strength_shift(mv[a], cfg.strength) - strength_shift(mv[b], cfg.strength)), cfg.adjust)
                   for a, b in zip(tgt.hi, tgt.ai)]
            rows.append(pd.DataFrame({"season": s, "origin": k, "target_week": tgt.week.to_numpy(), "y": tgt.y.to_numpy(),
                                      "mu_stale": mu_a, "mu_full": mu_f, "mu_sys": mu_a + np.array(adj), "age": age}))
    return pd.concat(rows, ignore_index=True)


def boot_ci(df, fn, n=400, seed=0):
    rng = np.random.default_rng(seed)
    by = {s: d for s, d in df.groupby("season")}
    keys = list(by)
    vals = [fn(pd.concat([by[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    return np.percentile(vals, [5, 95])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=None)
    ap.add_argument("--anchor", choices=("carryover", "win_totals"), default="carryover")
    ap.add_argument("--win-totals-dir", default=None, help="dir with win_totals_<season>.csv (Team, Win Total)")
    ap.add_argument("--first", type=int, default=2012)
    ap.add_argument("--last", type=int, default=2025)
    a = ap.parse_args()
    ANCHOR.update(mode=a.anchor, dir=a.win_totals_dir)
    cfg = load_config(ROOT / "config.toml")
    g = load_games(a.games)
    hfa_by, raw_hfa = season_hfa(g)
    print(f"real games: {len(g)}  seasons {g.season.min()}-{g.season.max()}  | regression HFA by season (logit): "
          + ", ".join(f"{s}:{v:.2f}" for s, v in raw_hfa.items() if s in (2010, 2015, 2019, 2020, 2021, 2024, 2025)))
    seasons = list(range(a.first, a.last + 1))
    if a.anchor == "win_totals":
        seasons = [x for x in seasons if (Path(a.win_totals_dir) / f"win_totals_{x}.csv").exists()]
        if len(seasons) < 2:
            raise SystemExit("need win_totals_<season>.csv for at least 2 seasons")
    print(f"anchor: {a.anchor} | seasons evaluated: {seasons[0]}-{seasons[-1]} ({len(seasons)})")

    print("\n[0] ANCHOR ACCURACY: RMSE (logit) of the preseason anchor against the REAL Week-1 / Weeks-1-2 closing lines")
    for mode in ("carryover",) + (("win_totals",) if a.anchor == "win_totals" else ()):
        ANCHOR["mode"] = mode
        r1, r12 = [], []
        for x in seasons:
            r0 = anchor_ratings(g, x, hfa_by, 0.75)
            gx = g[(g.season == x) & (g.week <= 2)]
            e = gx.y.to_numpy() - (r0[gx.hi] - r0[gx.ai] + np.where(gx.neutral, 0.0, hfa_by[x]))
            r1.append(e[gx.week.to_numpy() == 1]); r12.append(e)
        print(f"    {mode:<11} week 1: {np.sqrt(np.mean(np.concatenate(r1) ** 2)):.4f}   weeks 1-2: {np.sqrt(np.mean(np.concatenate(r12) ** 2)):.4f}"
              f"   ({len(np.concatenate(r12))} games)")
    ANCHOR["mode"] = a.anchor

    print("\n[1] Calibrate filter dynamics on REAL one-step-ahead closing-line prediction (RMSE, logit units)")
    grid = list(itertools.product((0.06, 0.09, 0.13, 0.18), (0.14, 0.18, 0.23), (0.20, 0.30, 0.45), (0.6, 0.75)))
    res = sorted(((one_step_rmse(g, hfa_by, p, seasons), p) for p in grid))
    for rm, p in res[:5]:
        print(f"    q={p[0]:.2f}/wk  line_sd={p[1]:.2f}  prior_sd={p[2]:.2f}  carry_rho={p[3]:.2f}  -> RMSE {rm:.4f}")
    worst = res[-1]
    print(f"    (worst grid point RMSE {worst[0]:.4f}; package defaults are q=0.08/wk, line_sd=0.15)")
    best = res[0][1]
    # walk-forward check of the calibration (choose params on seasons < s, score season s)
    wf = []
    for s in seasons[len(seasons) // 2:]:
        tr = [x for x in seasons if x < s]
        bp = min(grid, key=lambda p: one_step_rmse(g, hfa_by, p, tr))
        wf.append(one_step_rmse(g, hfa_by, bp, [s]))
    print(f"    walk-forward (params chosen on prior seasons only, later half of the seasons) mean RMSE {np.mean(wf):.4f}")

    print("\n[2] Multi-horizon: does updating beat the stale preseason anchor? (games >=2 weeks past the refreshed slate)")
    d = multi_horizon(g, hfa_by, best, seasons, cfg)
    d["k"] = pd.cut(d.origin, [1, 3, 6, 10, 14], labels=["k2-3", "k4-6", "k7-10", "k11-14"])
    def rmse(x, c): return float(np.sqrt(np.mean((x.y - x[c]) ** 2)))
    out = []
    for k, x in d.groupby("k", observed=True):
        dm, dy = (x.mu_full - x.mu_stale).to_numpy(), (x.y - x.mu_stale).to_numpy()
        beta = float((dm * dy).sum() / (dm * dm).sum())
        out.append({"origin": k, "games": len(x), "age_wk": round(x.age.mean(), 1), "RMSE_stale": rmse(x, "mu_stale"),
                    "RMSE_full_passthrough": rmse(x, "mu_full"), "RMSE_system_defaults": rmse(x, "mu_sys"),
                    "best_passthrough_beta": beta})
    print(pd.DataFrame(out).round(4).to_string(index=False))
    tot = {c: rmse(d, c) for c in ("mu_stale", "mu_full", "mu_sys")}
    ci = boot_ci(d, lambda z: 1 - rmse(z, "mu_sys") / rmse(z, "mu_stale"))
    ci2 = boot_ci(d, lambda z: 1 - rmse(z, "mu_full") / rmse(z, "mu_stale"))
    print(f"    overall RMSE stale {tot['mu_stale']:.4f} | system defaults {tot['mu_sys']:.4f} "
          f"({1 - tot['mu_sys'] / tot['mu_stale']:.1%} better; 90% CI over seasons {ci[0]:.1%}..{ci[1]:.1%}) "
          f"| full pass-through {tot['mu_full']:.4f} ({1 - tot['mu_full'] / tot['mu_stale']:.1%}; {ci2[0]:.1%}..{ci2[1]:.1%})")

    print("\n[2b] Policy variants (same filter, same data): how much of the available gain does each keep?")
    variants = {
        "package defaults (start 3wk, max .8, thr .12, cap .4)": {},
        "no dead-zone in age (start 0, half .5wk, max .9)": dict(aging__start_age_weeks=0.0, aging__half_weight_weeks=0.5, aging__max_weight=0.9),
        "  + small gate (thr .03) and cap .8": dict(aging__start_age_weeks=0.0, aging__half_weight_weeks=0.5, aging__max_weight=0.9,
                                                    strength__threshold_up=0.03, adjust__cap=0.8),
        "constant .9 pass-through, no gate, cap 1.5": dict(aging__start_age_weeks=0.0, aging__half_weight_weeks=0.01, aging__max_weight=0.9,
                                                            strength__threshold_up=0.0, adjust__cap=1.5),
    }
    for name, ov in variants.items():
        cv = load_config(ROOT / "config.toml", **ov)
        dv = multi_horizon(g, hfa_by, best, seasons, cv)
        gain = 1 - rmse(dv, "mu_sys") / rmse(dv, "mu_stale")
        lo, hi_ = boot_ci(dv, lambda z: 1 - rmse(z, "mu_sys") / rmse(z, "mu_stale"))
        early = dv[dv.origin <= 3]
        print(f"    {name:<55} error cut {gain:5.1%}  (90% CI {lo:.1%}..{hi_:.1%});  at age~4wk: {1 - rmse(early, 'mu_sys') / rmse(early, 'mu_stale'):5.1%}")

    print("\n[3] Gate check: is the signal in SMALL movements too?  (slope of realised change on estimated change)")
    d["mv"] = (d.mu_full - d.mu_stale).abs()
    d["bin"] = pd.cut(d.mv, [0, 0.10, 0.20, 0.35, 0.6, 5], labels=["<0.10", "0.10-0.20", "0.20-0.35", "0.35-0.60", ">0.60"])
    rows = []
    for b, x in d.groupby("bin", observed=True):
        dm, dy = (x.mu_full - x.mu_stale).to_numpy(), (x.y - x.mu_stale).to_numpy()
        rows.append({"|game movement| (logit)": b, "games": len(x), "slope": round(float((dm * dy).sum() / (dm * dm).sum()), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
