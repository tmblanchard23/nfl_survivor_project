"""Backtest on REAL data: preseason win totals (SportsOddsHistory, 2012-2025) + nflverse closing moneylines.

Tests (see BACKTEST_PLAN.md):
  T1  anchor accuracy   - how well do win-total ratings predict the real Week 1-2 closing lines?
  T2  filter dynamics   - one-step-ahead prediction of each week's closing lines (calibrates q, line_sd, prior_sd)
  T3  adjustment policy - predicting closing lines >= 2 weeks past the information set (calibrates age curve,
                          gate, cap).  Stale anchor = the game's win-total-implied line.
Protocol: everything is chosen on TRAIN seasons 2012-2022 and scored on HOLDOUT 2023-2025.  A game is only ever
predicted from closing lines of EARLIER weeks; win totals use no results; home field is walk-forward.

Usage: python backtest/run_backtest.py            (writes backtest/results/)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nfl_prob_grid.adjustment import age_weight  # noqa: E402
from nfl_prob_grid.config import AgingCfg  # noqa: E402
from nfl_prob_grid.ratings import (N, TEAMS, estimate_hfa, kalman_update,  # noqa: E402
                                   normalize_win_totals, solve_win_total_ratings, team_idx)

DATA, OUT = ROOT / "backtest" / "data", ROOT / "backtest" / "results"
REMAP = {"OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR"}
TRAIN, HOLD = list(range(2012, 2023)), [2023, 2024, 2025]
ALL = TRAIN + HOLD
PRESEASON_LEAD_WEEKS = 1.7     # preseason lines are ~12 days before the Tuesday of week 1


# ------------------------------------------------------------------------------------------ data
def load_games() -> pd.DataFrame:
    g = pd.read_csv(DATA / "nflverse_games.csv")
    g = g[(g.game_type == "REG") & g.home_moneyline.notna() & g.away_moneyline.notna()].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(REMAP)
    dec = lambda ml: np.where(ml > 0, 1 + ml / 100.0, 1 + 100.0 / np.abs(ml))  # noqa: E731
    qh, qa = 1 / dec(g.home_moneyline.to_numpy()), 1 / dec(g.away_moneyline.to_numpy())
    g["p_home"] = qh / (qh + qa)                          # proportional de-vig, identical to the engine
    g["y"] = np.log(g.p_home / (1 - g.p_home))
    g["hi"], g["ai"] = team_idx(g.home_team), team_idx(g.away_team)
    g["neutral"] = g.location.eq("Neutral")
    return g[g.season >= 2008].reset_index(drop=True)


def walk_forward_hfa(g) -> tuple[dict, dict]:
    raw = {s: estimate_hfa(d.y.to_numpy(), d.hi.to_numpy(), d.ai.to_numpy(), d.neutral.to_numpy())
           for s, d in g.groupby("season")}
    wf = {s: float(np.mean([raw[p] for p in range(s - 3, s) if p in raw and p != 2020])) for s in raw if s > 2008}
    return wf, raw


def win_total_ratings(g, s, hfa, mode="additive") -> np.ndarray:
    wt = pd.read_csv(DATA / f"win_totals_{s}.csv").set_index("Team")["Win Total"].reindex(TEAMS)
    gs = g[g.season == s]
    target, _ = normalize_win_totals(wt.to_numpy(float), len(gs), mode)
    return solve_win_total_ratings(target, gs.hi.to_numpy(), gs.ai.to_numpy(), np.where(gs.neutral, 0.0, hfa))


def carryover_ratings(g, s, hfa_prev, rho=0.75) -> np.ndarray:
    d = g[g.season == s - 1]
    y = d.y.to_numpy() - np.where(d.neutral, 0.0, hfa_prev)
    r, _, _ = kalman_update(np.zeros(N), 100 * np.eye(N), d.hi.to_numpy(), d.ai.to_numpy(), y, np.full(len(y), 0.15 ** 2))
    return rho * (r - r.mean())


def rmse(e) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def season_boot(per_season: dict, fn, n=2000, seed=1):
    rng, keys = np.random.default_rng(seed), list(per_season)
    vals = [fn(pd.concat([per_season[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    return np.percentile(vals, [5, 95])


# ------------------------------------------------------------------------------------------ T1
def t1_anchor(g, hfa, anchors) -> dict:
    rows, per = [], {}
    for s in ALL:
        gs = g[(g.season == s) & (g.week <= 2)]
        h = np.where(gs.neutral, 0.0, hfa[s])
        wt, co = anchors[s]["wt"], anchors[s]["carry"]
        preds = {"win_totals": wt, "win_totals_proportional": anchors[s]["wt_prop"], "carryover": co}
        for w in (0.25, 0.5, 0.75):
            preds[f"blend_{int(w * 100)}wt"] = w * wt + (1 - w) * co
        for c in (0.8, 0.9, 1.1, 1.2):
            preds[f"win_totals_x{c}"] = c * wt
        d = {"season": s, "week": gs.week.to_numpy()}
        for k, r in preds.items():
            d[k] = gs.y.to_numpy() - (r[gs.hi] - r[gs.ai] + h)
        per[s] = pd.DataFrame(d)
    all_ = pd.concat(per.values())
    cols = [c for c in all_.columns if c not in ("season", "week")]
    for c in cols:
        rows.append({"anchor": c, "week1_rmse": rmse(all_[all_.week == 1][c]), "weeks1_2_rmse": rmse(all_[c]),
                     "train_w1_2": rmse(all_[all_.season.isin(TRAIN)][c]), "holdout_w1_2": rmse(all_[all_.season.isin(HOLD)][c])})
    tab = pd.DataFrame(rows).sort_values("weeks1_2_rmse")
    ci = season_boot(per, lambda z: rmse(z.carryover) - rmse(z.win_totals))
    by_season = pd.DataFrame({s: {"win_totals": rmse(d.win_totals), "carryover": rmse(d.carryover)} for s, d in per.items()}).T
    return {"table": tab, "wt_minus_carry_ci": ci, "by_season": by_season,
            "wt_better_seasons": int((by_season.win_totals < by_season.carryover).sum())}


# ------------------------------------------------------------------------------------------ filter
def run_season(gs, r0, hfa, q, sd_r, prior_sd):
    r, cov = r0.copy(), prior_sd ** 2 * np.eye(N)
    after, errs = {0: r0.copy()}, {}
    for k in sorted(gs.week.unique()):
        d = gs[gs.week == k]
        if k > 1:
            cov = cov + q ** 2 * np.eye(N)
        h = np.where(d.neutral, 0.0, hfa)
        errs[int(k)] = d.y.to_numpy() - (r[d.hi] - r[d.ai] + h)
        r, cov, _ = kalman_update(r, cov, d.hi.to_numpy(), d.ai.to_numpy(), d.y.to_numpy() - h, np.full(len(d), sd_r ** 2))
        after[int(k)] = r.copy()
    return errs, after


def t2_filter(g, hfa, anchors) -> dict:
    grid = list(itertools.product((0.06, 0.09, 0.13, 0.18), (0.14, 0.18, 0.23), (0.10, 0.15, 0.20, 0.30, 0.45)))
    cache = {}                                     # (anchor, params, season) -> squared errors by week
    for an in ("wt", "carry"):
        for p in grid:
            for s in ALL:
                errs, _ = run_season(g[g.season == s], anchors[s][an], hfa[s], *p)
                cache[(an, p, s)] = errs
    def score(an, p, seasons, weeks=None):
        e = [v for s in seasons for k, v in cache[(an, p, s)].items() if weeks is None or k in weeks]
        return rmse(np.concatenate(e))
    res = {}
    for an in ("wt", "carry"):
        best = min(grid, key=lambda p: score(an, p, TRAIN))
        ranked = sorted(grid, key=lambda p: score(an, p, TRAIN))
        res[an] = {"best_train": best, "train": score(an, best, TRAIN), "holdout": score(an, best, HOLD),
                   "holdout_w1_4": score(an, best, HOLD, range(1, 5)), "holdout_w5_18": score(an, best, HOLD, range(5, 19)),
                   "top5": [(p, round(score(an, p, TRAIN), 4)) for p in ranked[:5]]}
    # expanding-window walk-forward for the WT anchor (params chosen only on seasons before s)
    wf = []
    for s in ALL[4:]:
        tr = [x for x in ALL if x < s]
        p = min(grid, key=lambda p: score("wt", p, tr))
        wf.append({"season": s, "params": p, "rmse": score("wt", p, [s])})
    res["walk_forward"] = pd.DataFrame(wf)
    # sensitivity of holdout RMSE to prior_sd with the other two at their best
    q, sd, _ = res["wt"]["best_train"]
    res["prior_sd_curve"] = {ps: (round(score("wt", (q, sd, ps), TRAIN), 4), round(score("wt", (q, sd, ps), HOLD, range(1, 5)), 4))
                             for ps in (0.10, 0.15, 0.20, 0.30, 0.45)}
    res["defaults_like"] = {"note": "package defaults q=0.08 line_sd=0.15 prior_sd=0.15; nearest grid cell (0.09,0.14,0.15)",
                            "train": score("wt", (0.09, 0.14, 0.15), TRAIN), "holdout": score("wt", (0.09, 0.14, 0.15), HOLD)}
    return res


# ------------------------------------------------------------------------------------------ T3
def build_multi_horizon(g, hfa, anchors, params) -> pd.DataFrame:
    rows = []
    for s in ALL:
        gs = g[g.season == s]
        r0 = anchors[s]["wt"]
        _, after = run_season(gs, r0, hfa[s], *params)
        for k in range(1, 16):                                 # information: closing lines through week k
            tgt = gs[gs.week >= k + 2]
            if tgt.empty:
                continue
            mv = after[k] - r0
            mv = mv - np.median(mv)
            h = np.where(tgt.neutral, 0.0, hfa[s])
            rows.append(pd.DataFrame({
                "season": s, "origin": k, "target_week": tgt.week.to_numpy(), "y": tgt.y.to_numpy(),
                "mu_stale": r0[tgt.hi] - r0[tgt.ai] + h, "mv_h": mv[tgt.hi], "mv_a": mv[tgt.ai],
                "age": k + PRESEASON_LEAD_WEEKS}))
    return pd.concat(rows, ignore_index=True)


def policy_predict(d, start, half, wmax, thr, cap, curve="hyperbolic"):
    a = AgingCfg(curve=curve, start_age_weeks=start, half_weight_weeks=half, max_weight=wmax)
    wmap = {x: age_weight(x, a) for x in d.age.unique()}
    w = d.age.map(wmap).to_numpy()
    sh = lambda m: np.sign(m) * np.maximum(np.abs(m) - thr, 0.0)  # noqa: E731
    x = w * (sh(d.mv_h.to_numpy()) - sh(d.mv_a.to_numpy()))
    adj = cap * np.tanh(x / cap) if cap > 0 else 0.0
    return d.mu_stale.to_numpy() + adj


def t3_policy(d) -> dict:
    tr, ho = d[d.season.isin(TRAIN)], d[d.season.isin(HOLD)]
    grid = list(itertools.product((0, 1, 2, 3), (0.25, 0.5, 1.0, 2.0, 3.0), (0.6, 0.7, 0.8, 0.9, 1.0),
                                  (0.0, 0.02, 0.05, 0.08, 0.12), (0.4, 0.6, 0.8, 1.0, 1.5)))
    base_tr, base_ho = rmse(tr.y - tr.mu_stale), rmse(ho.y - ho.mu_stale)
    scores = sorted(((rmse(tr.y - policy_predict(tr, *p)), p) for p in grid))
    best = scores[0][1]
    def cut(x, p):
        return 1 - rmse(x.y - policy_predict(x, *p)) / rmse(x.y - x.mu_stale)
    named = {
        "previous defaults (start 3, half 3, max .8, thr .12, cap .4)": (3, 3.0, 0.8, 0.12, 0.4),
        "SHIPPED defaults (start 3, half .5, max 1, thr .02, cap 1)": (3, 0.5, 1.0, 0.02, 1.0),
        "best on train": best,
        "full pass-through (w=1 always, no gate, no cap)": (0, 1e-9, 1.0, 0.0, 50.0),
    }
    # simplest policy within 0.1% of the best train RMSE -> preferred (robust to the flat surface)
    tol = scores[0][0] * 1.001
    simple = [p for s_, p in scores if s_ <= tol]
    simple.sort(key=lambda p: (-p[0], p[3] != 0, -p[1], p[4]))   # prefer keeping a start age, no gate, gentle ramp
    named["recommended (simplest within 0.1% of best)"] = simple[0]
    table = []
    per_season = {s: x for s, x in d.groupby("season")}
    for name, p in named.items():
        ci = season_boot({s: x for s, x in per_season.items() if s in HOLD or True}, lambda z: cut(z, p), n=1000)
        table.append({"policy": name, "params (start,half,max,thr,cap)": p, "train_cut": cut(tr, p),
                      "holdout_cut": cut(ho, p), "all_seasons_90ci": f"{ci[0]:.1%}..{ci[1]:.1%}"})
    # pass-through slope by age on train (unconstrained: realised change on estimated change)
    d = d.assign(dm=d.mv_h - d.mv_a, dy=d.y - d.mu_stale)
    slope = d[d.season.isin(TRAIN)].groupby(pd.cut(d.age, [2, 4, 6, 8, 11, 17])).apply(
        lambda z: float((z.dm * z.dy).sum() / (z.dm ** 2).sum()), include_groups=False)
    small = d[d.season.isin(TRAIN)].groupby(pd.cut(d.dm.abs(), [0, 0.05, 0.10, 0.20, 0.4, 5])).apply(
        lambda z: float((z.dm * z.dy).sum() / (z.dm ** 2).sum()), include_groups=False)
    return {"table": pd.DataFrame(table), "stale_rmse": (base_tr, base_ho), "slope_by_age": slope,
            "slope_by_size": small, "best": best, "recommended": named["recommended (simplest within 0.1% of best)"],
            "n_train": len(tr), "n_hold": len(ho)}


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_games()
    hfa, raw_hfa = walk_forward_hfa(g)
    anchors = {s: {"wt": win_total_ratings(g, s, hfa[s]), "wt_prop": win_total_ratings(g, s, hfa[s], "proportional"),
                   "carry": carryover_ratings(g, s, hfa[s - 1])} for s in ALL}
    n = len(g[g.season.isin(ALL)])
    print(f"REAL DATA: {n} games with closing moneylines, seasons {ALL[0]}-{ALL[-1]} | train {TRAIN[0]}-{TRAIN[-1]}, holdout {HOLD[0]}-{HOLD[-1]}")
    print("walk-forward home field (logit): " + ", ".join(f"{s}:{hfa[s]:.2f}" for s in (2012, 2016, 2020, 2021, 2023, 2025)))

    t1 = t1_anchor(g, hfa, anchors)
    print("\n[T1] ANCHOR ACCURACY vs real closing lines (RMSE, logit)")
    print(t1["table"].round(4).to_string(index=False))
    lo, hi = t1["wt_minus_carry_ci"]
    print(f"  win totals beat carryover in {t1['wt_better_seasons']}/{len(ALL)} seasons; RMSE advantage 90% CI {lo:.3f}..{hi:.3f}")
    t1["table"].to_csv(OUT / "t1_anchor.csv", index=False)
    t1["by_season"].to_csv(OUT / "t1_by_season.csv")

    t2 = t2_filter(g, hfa, anchors)
    print("\n[T2] FILTER DYNAMICS: one-step-ahead RMSE (logit), params = (q per week, line_sd, prior_sd)")
    for an, lab in (("wt", "win-total anchor"), ("carry", "carryover anchor")):
        r = t2[an]
        print(f"  {lab:<17} best on train {r['best_train']}: train {r['train']:.4f} | holdout {r['holdout']:.4f} "
              f"(weeks 1-4 {r['holdout_w1_4']:.4f}, weeks 5-18 {r['holdout_w5_18']:.4f})")
    print("  top-5 (win-total anchor, train):", t2["wt"]["top5"])
    print("  prior_sd curve (train all weeks, holdout weeks 1-4):", t2["prior_sd_curve"])
    print(f"  near-default cell: train {t2['defaults_like']['train']:.4f} | holdout {t2['defaults_like']['holdout']:.4f}")
    print("  expanding walk-forward:", t2["walk_forward"].assign(rmse=lambda x: x.rmse.round(4)).to_dict("records"))

    params = t2["wt"]["best_train"]
    d = build_multi_horizon(g, hfa, anchors, params)
    t3 = t3_policy(d)
    print(f"\n[T3] ADJUSTMENT POLICY: predicting closing lines >= 2 weeks past the information "
          f"(train {t3['n_train']} / holdout {t3['n_hold']} game-predictions); stale RMSE train {t3['stale_rmse'][0]:.4f}, holdout {t3['stale_rmse'][1]:.4f}")
    print(t3["table"].to_string(index=False, formatters={"train_cut": "{:.1%}".format, "holdout_cut": "{:.1%}".format}))
    print("  unconstrained pass-through slope by line age (weeks), train:", t3["slope_by_age"].round(3).to_dict())
    print("  slope by |movement| size (logit), train:", t3["slope_by_size"].round(3).to_dict())

    summary = {"t1": t1["table"].to_dict("records"), "t1_wt_better_seasons": t1["wt_better_seasons"],
               "t1_ci": list(map(float, t1["wt_minus_carry_ci"])),
               "t2": {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv)) for kk, vv in v.items()}
                      for k, v in t2.items() if k in ("wt", "carry")},
               "t2_prior_sd_curve": {str(k): v for k, v in t2["prior_sd_curve"].items()},
               "t3": t3["table"].to_dict("records"), "t3_best": t3["best"], "t3_recommended": t3["recommended"],
               "filter_params": params, "hfa_walk_forward": hfa}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    t3["table"].to_csv(OUT / "t3_policy.csv", index=False)


if __name__ == "__main__":
    main()
