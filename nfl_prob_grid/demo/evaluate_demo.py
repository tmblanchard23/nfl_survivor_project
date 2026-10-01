"""Sanity check on SYNTHETIC data: (1) do stale-line adjustments move probabilities toward the truth,
and (2) how do baseline choices compare (pure win totals / ~50-50 blend / lines-heavy)?

CAVEAT: the generator builds preseason lines, fresh lines AND win totals from the same latent
strengths, so this validates the MECHANISM only; it says nothing about real-world accuracy.  Real
calibration needs historical lines + win totals (DESIGN.md section 7)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "demo"))
import numpy as np, pandas as pd  # noqa: E402
from make_demo_data import HFA, INIT_AS_OF, build_demo, run_as_of, sigmoid  # noqa: E402
from nfl_prob_grid import load_config, run_init, run_refresh  # noqa: E402
from nfl_prob_grid.master import StateStore  # noqa: E402
from nfl_prob_grid.ratings import TEAMS  # noqa: E402

VARIANTS = {
    "pure win totals (reconcile off)": dict(rating__reconcile_preseason_lines=False),
    "default: ~50% win totals": {},
    "lines-heavy (~15% win totals)": dict(rating__preseason_line_sd=0.25),
}


def evaluate(seed: int, **overrides) -> dict:
    tmp = Path(tempfile.mkdtemp())
    info = build_demo(tmp / "d", seed=seed); d = info["dir"]
    cfg = load_config(ROOT / "config.toml", paths__state_dir=str(tmp / "s"),
                      paths__output_dir=str(tmp / "o"), **overrides)
    run_init(cfg, as_of=INIT_AS_OF, current_week=1, lines_source=str(d / "initial_lines.csv"),
             win_totals_source=str(d / "win_totals.csv"))
    for w in info["run_weeks"]:
        run_refresh(cfg, as_of=run_as_of(w), current_week=w, lines_source=str(d / f"refresh_w{w:02d}.csv"))
    m, hist, _ = StateStore(cfg.paths.state_dir).load()
    last = info["run_weeks"][-1]
    r_true = info["true_ratings"][last]
    fut = m[(~m.is_frozen) & (m.week >= 11) & (m.refresh_count == 0)]           # never-refreshed future games
    truth = np.array([sigmoid(r_true[h] - r_true[a] + HFA) for h, a in zip(fut.home_team, fut.away_team)])
    stale, adj = fut.original_vegas_home_prob.to_numpy(), fut.home_win_prob.to_numpy()
    est = hist.ratings(hist.latest_id()).reindex(TEAMS).to_numpy()
    tru = np.array([r_true[t] for t in TEAMS]); tru -= tru.mean()
    return {"mae_stale": np.abs(stale - truth).mean(), "mae_adjusted": np.abs(adj - truth).mean(),
            "rating_rmse_vs_truth": float(np.sqrt(np.mean((est - tru) ** 2))),
            "adjusted_share": float(fut.adj_active.mean())}


if __name__ == "__main__":
    seeds = range(1, 9)
    rows = []
    for name, ov in VARIANTS.items():
        res = pd.DataFrame([evaluate(s, **ov) for s in seeds]).mean()
        rows.append({"baseline": name, **res.to_dict()})
    df = pd.DataFrame(rows).set_index("baseline")
    df["error_reduction_vs_stale"] = 1 - df["mae_adjusted"] / df["mae_stale"]
    pd.set_option("display.width", 200)
    print(f"Mean over {len(list(seeds))} synthetic seasons, never-refreshed games in weeks 11-18 (MAE in probability):\n")
    print(df.round(4).to_string())
