"""End-to-end demo on SYNTHETIC data:  python demo/run_demo.py

Builds a fake season, runs `init` (win totals + preseason lines) and then eight weekly refreshes (weeks 2-9),
printing what changed each run and showing the audit trail of a few stale-adjusted games."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

import pandas as pd  # noqa: E402

from make_demo_data import INIT_AS_OF, build_demo, run_as_of  # noqa: E402
from nfl_prob_grid import load_config, run_init, run_refresh  # noqa: E402
from nfl_prob_grid.master import StateStore  # noqa: E402


def main() -> None:
    data = ROOT / "demo" / "data"
    info = build_demo(data)
    cfg = load_config(ROOT / "config.toml",
                      paths__state_dir=str(ROOT / "demo" / "state"),
                      paths__output_dir=str(ROOT / "demo" / "output"))
    print("Synthetic shocks (strength change from week 3):", info["shocks"], "\n")

    rep = run_init(cfg, as_of=INIT_AS_OF, current_week=1, force=True,
                   lines_source=str(data / "initial_lines.csv"),
                   win_totals_source=str(data / "win_totals.csv"))
    print(rep.summary(), "\n")

    for w in info["run_weeks"]:
        rep = run_refresh(cfg, as_of=run_as_of(w), current_week=w,
                          lines_source=str(data / f"refresh_w{w:02d}.csv"))
        print(rep.summary(), "\n")

    master, _, man = StateStore(cfg.paths.state_dir).load()
    adj = master[master["adj_active"]].copy()
    adj["abs_pp"] = adj["adj_home_prob_delta"].abs()
    cols = ["game_id", "line_age_weeks", "age_weight", "home_rating_move", "away_rating_move",
            "matchup_shift", "adj_logit", "latest_vegas_home_prob", "home_win_prob"]
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"Final master v{man['current_version']}: {len(adj)} stale-adjusted games. Largest:")
    print(adj.sort_values("abs_pp", ascending=False)[cols].head(8).round(3).to_string(index=False))
    print("\nSource of current probability, active games:")
    print(master[~master["is_frozen"]]["prob_source"].value_counts().to_string())
    print("\nGrid preview:")
    print(pd.read_csv(ROOT / "demo" / "output" / "prob_grid_latest.csv").iloc[:6, :10].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
