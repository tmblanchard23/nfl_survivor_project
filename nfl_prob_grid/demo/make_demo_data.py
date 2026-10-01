"""Generate a SYNTHETIC season for demonstration and testing.

Nothing here is real NFL data: the schedule is random, team strengths are random, and all odds
are simulated.  It exists to exercise every code path (varied team-name formats, variable-size
refresh tables, preseason win totals with over/under prices, evolving strengths with a few
injury/breakout shocks).

Latent truth: each team has a market strength r0 at preseason that random-walks and receives
shocks from week 3.  Preseason game lines are thin (noise sd 0.12 logit); fresh lines are
liquid (noise sd 0.08).  Win totals are the books' expected wins over the schedule + noise.

Timeline: preseason lines / win totals on 2026-08-20; the run for ``current_week = w`` happens
on the Tuesday of week w (2026-09-08 + 7*(w-1) days) and ships lines for weeks w and w+1.
"""
from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from nfl_prob_grid.ratings import TEAMS, expected_wins, team_idx
from nfl_prob_grid.teams import CANONICAL_TEAMS, TEAM_TABLE

FULL = {a: f"{c} {n}" for a, c, n in TEAM_TABLE}
NICK = {a: n for a, c, n in TEAM_TABLE}
CITY_ONLY = {a: c for a, c, n in TEAM_TABLE}   # ambiguous for NY/LA - tests only use unambiguous ones
PRESEASON_TS = pd.Timestamp("2026-08-20T12:00:00Z")
INIT_AS_OF = "2026-08-25T00:00:00Z"
HFA = 0.25
PRESEASON_NOISE, FRESH_NOISE = 0.12, 0.08
_ND = NormalDist()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_as_of(week: int) -> pd.Timestamp:
    return pd.Timestamp("2026-09-08T15:00:00Z") + pd.Timedelta(days=7 * (week - 1))


def make_schedule(rng: np.random.Generator) -> pd.DataFrame:
    teams = list(CANONICAL_TEAMS)
    bye_counts = [2, 4, 4, 4, 4, 4, 4, 2, 2, 2]          # weeks 5..14, sums to 32
    order, byes, i = list(rng.permutation(teams)), {}, 0
    for wk, c in zip(range(5, 15), bye_counts):
        for t in order[i:i + c]:
            byes[t] = wk
        i += c
    rows = []
    for wk in range(1, 19):
        active = [t for t in teams if byes.get(t) != wk]
        rng.shuffle(active)
        for a, b in zip(active[::2], active[1::2]):
            home, away = (a, b) if rng.random() < 0.5 else (b, a)
            date = (pd.Timestamp("2026-09-13") + pd.Timedelta(days=7 * (wk - 1))).strftime("%Y-%m-%d")
            rows.append({"week": wk, "date": date, "away": away, "home": home})
    return pd.DataFrame(rows)


def game_odds(r_home: float, r_away: float, rng, noise: float, vig: float = 1.045):
    p = float(sigmoid(r_home - r_away + HFA + rng.normal(0, noise)))
    return round(1.0 / (p * vig), 2), round(1.0 / ((1.0 - p) * vig), 2)


def build_demo(out_dir: str | Path, seed: int = 7, run_weeks=range(2, 10), with_prices: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sched = make_schedule(rng)
    r0 = {t: float(rng.normal(0, 0.6)) for t in CANONICAL_TEAMS}
    r0 = {t: v - float(np.mean(list(r0.values()))) for t, v in r0.items()}

    shocked = [str(t) for t in rng.choice(CANONICAL_TEAMS, size=6, replace=False)]
    shock = {t: (-0.55 if i < 3 else 0.45) for i, t in enumerate(shocked)}   # from week 3 on
    drift = {t: rng.normal(0, 0.03, 20) for t in CANONICAL_TEAMS}

    def ratings_at(week: int) -> dict:
        return {t: r0[t] + float(np.sum(drift[t][:week])) + (shock.get(t, 0.0) if week >= 3 else 0.0)
                for t in CANONICAL_TEAMS}

    # ---- Input A: full-season preseason lines (thin: noisy) ----------------------------------
    rows = []
    for k, g in sched.iterrows():
        oh, oa = game_odds(r0[g.home], r0[g.away], rng, PRESEASON_NOISE)
        rows.append({"game_id": f"SRC{k:04d}", "week": g.week, "date": g.date,
                     "away_team": FULL[g.away], "home_team": FULL[g.home],
                     "away_odds": oa, "home_odds": oh,
                     "timestamp": PRESEASON_TS.strftime("%Y-%m-%d %H:%M:%S")})
    pd.DataFrame(rows).to_csv(out / "initial_lines.csv", index=False)

    # ---- Input C: preseason win totals (expected wins over the real schedule + book noise) ------
    hi, ai = team_idx(sched["home"]), team_idx(sched["away"])
    e_true, _ = expected_wins(np.array([r0[t] for t in TEAMS]), hi, ai, np.full(len(sched), HFA))
    belief = e_true + rng.normal(0, 0.35, len(TEAMS))
    totals = np.round((belief + 0.15) * 2) / 2                      # books shade up ~0.15 and post halves
    wt = {"Team": [FULL[t] for t in TEAMS], "Win Total": totals}
    if with_prices:
        p_over = np.array([_ND.cdf((b - x) / 2.1) for b, x in zip(belief, totals)])
        p_over = np.clip(p_over, 0.2, 0.8)
        wt["Over Odds"] = np.round(1.0 / (p_over * 1.045), 2)
        wt["Under Odds"] = np.round(1.0 / ((1 - p_over) * 1.045), 2)
    pd.DataFrame(wt).to_csv(out / "win_totals.csv", index=False)

    # ---- Input B per weekly run (nicknames only; variable number of games) ---------------------
    for w in run_weeks:
        as_of, r = run_as_of(w), ratings_at(w)
        stamp = (as_of - pd.Timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        sub = sched[sched.week.isin([w, w + 1])]
        rows = []
        for _, g in sub.iterrows():
            oh, oa = game_odds(r[g.home], r[g.away], rng, FRESH_NOISE)
            rows.append({"Week": g.week, "Away Team": NICK[g.away], "Away Odds": oa,
                         "Home Team": NICK[g.home], "Home Odds": oh, "Timestamp": stamp})
        pd.DataFrame(rows).to_csv(out / f"refresh_w{w:02d}.csv", index=False)

    return {"schedule": sched, "shocks": shock, "dir": out, "seed": seed,
            "run_weeks": list(run_weeks), "true_ratings": {w: ratings_at(w) for w in run_weeks},
            "r0": r0, "belief_wins": dict(zip(TEAMS, belief)), "e_true": dict(zip(TEAMS, e_true))}


if __name__ == "__main__":
    import sys
    info = build_demo(sys.argv[1] if len(sys.argv) > 1 else "demo/data")
    print("wrote synthetic demo data to", info["dir"], "| shocked teams:", info["shocks"])
