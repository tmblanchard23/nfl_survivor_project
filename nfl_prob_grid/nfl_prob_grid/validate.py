"""Master-table validation.  Any ``error`` issue aborts a run before anything is written."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .errors import Issue, OddsError
from .master import MASTER_COLUMNS, PROB_SOURCES
from .probability import devig_two_way, shift_probability
from .teams import CANONICAL_TEAMS

# Fields that must be byte-for-byte unchanged once a game is frozen.
FROZEN_FIELDS = ["home_win_prob", "away_win_prob", "prob_source", "latest_vegas_home_prob",
                 "latest_home_odds", "latest_away_odds", "last_vegas_refresh_ts", "adj_logit"]
# Fields that must never change for ANY game after initialisation.
IMMUTABLE_FIELDS = ["season", "week", "game_date", "away_team", "home_team",
                    "original_vegas_home_prob", "original_vegas_ts"]


def validate_master(m: pd.DataFrame, cfg: Config, prev: pd.DataFrame | None = None,
                    current_week: int | None = None) -> list[Issue]:
    issues: list[Issue] = []

    def err(code: str, msg: str, gid: str = "") -> None:
        issues.append(Issue("error", code, msg, gid))

    missing = [c for c in MASTER_COLUMNS if c not in m.columns]
    if missing:
        err("missing_columns", f"master table lacks columns {missing}")
        return issues
    tol = cfg.validation.prob_tol
    ids = m["game_id"]

    if ids.duplicated().any():
        err("duplicate_game_id", f"duplicate game ids: {sorted(ids[ids.duplicated()].unique())[:10]}")
    dup_key = m.duplicated(["week", "away_team", "home_team"], keep=False)
    if dup_key.any():
        err("duplicate_game", f"{int(dup_key.sum())} rows repeat a (week, away, home) game")
    bad_team = ~m["home_team"].isin(CANONICAL_TEAMS) | ~m["away_team"].isin(CANONICAL_TEAMS)
    for gid in ids[bad_team]:
        err("unknown_team", "team not in canonical list", gid)
    for gid in ids[m["home_team"] == m["away_team"]]:
        err("same_team", "home == away", gid)
    for gid in ids[~m["week"].between(1, cfg.season.n_weeks)]:
        err("unexpected_week", f"week outside 1..{cfg.season.n_weeks}", gid)

    long = pd.concat([m[["week", "home_team", "game_id"]].rename(columns={"home_team": "team"}),
                      m[["week", "away_team", "game_id"]].rename(columns={"away_team": "team"})])
    twice = long[long.duplicated(["team", "week"], keep=False)]
    for (team, week), g in twice.groupby(["team", "week"]):
        err("team_plays_twice", f"{team} has {len(g)} games in week {week}: {sorted(g['game_id'])}")
    if cfg.season.games_per_team:
        counts = long["team"].value_counts()
        for team in CANONICAL_TEAMS:
            if counts.get(team, 0) != cfg.season.games_per_team:
                issues.append(Issue("warning", "games_per_team",
                                    f"{team} has {counts.get(team, 0)} games, expected "
                                    f"{cfg.season.games_per_team}"))

    for col in ("home_win_prob", "away_win_prob", "latest_vegas_home_prob", "original_vegas_home_prob"):
        v = m[col].to_numpy(dtype=float)
        bad = ~np.isfinite(v) | (v <= 0.0) | (v >= 1.0)
        for gid in ids[bad]:
            err("impossible_probability", f"{col} outside (0, 1) or not finite", gid)
    off = (m["home_win_prob"] + m["away_win_prob"] - 1.0).abs() > tol
    for gid in ids[off]:
        err("probabilities_not_complementary", "home + away != 1", gid)
    for col in ("original_vegas_ts", "last_vegas_refresh_ts"):
        for gid in ids[m[col].isna()]:
            err("missing_timestamp", f"{col} is null", gid)
    for gid in ids[~m["prob_source"].isin(PROB_SOURCES)]:
        err("bad_prob_source", "unrecognised prob_source", gid)

    # -- lineage consistency: current prob must be reproducible from stored fields --------
    for r in m.itertuples():
        try:
            p_dv, _ = devig_two_way(r.latest_home_odds, r.latest_away_odds, cfg.devig)
        except OddsError as exc:
            err("stored_odds_invalid", str(exc), r.game_id)
            continue
        if abs(p_dv - r.latest_vegas_home_prob) > 1e-9:
            err("devig_mismatch", f"latest_vegas_home_prob {r.latest_vegas_home_prob:.6f} != "
                f"de-vig of stored odds {p_dv:.6f}", r.game_id)
        expect = shift_probability(r.latest_vegas_home_prob, r.adj_logit, cfg.adjust.link) \
            if r.adj_active else r.latest_vegas_home_prob
        if abs(expect - r.home_win_prob) > 1e-9:
            err("lineage_mismatch", f"home_win_prob {r.home_win_prob:.6f} not reproducible from "
                f"anchor + adjustment ({expect:.6f})", r.game_id)
        if r.adj_active != (r.prob_source == "VEGAS_STALE_ADJUSTED"):
            err("source_flag_mismatch", "adj_active disagrees with prob_source", r.game_id)
        if not r.adj_active and r.adj_logit != 0.0:
            err("adjustment_flag_mismatch", "adj_logit != 0 but adj_active is False", r.game_id)

    # -- comparison with previous version -----------------------------------------------
    if prev is not None:
        pi, mi = prev.set_index("game_id"), m.set_index("game_id")
        if set(pi.index) != set(mi.index):
            err("game_set_changed", f"added={sorted(set(mi.index) - set(pi.index))[:5]} "
                f"removed={sorted(set(pi.index) - set(mi.index))[:5]}")
        else:
            mi = mi.loc[pi.index]
            for col in IMMUTABLE_FIELDS:
                diff = ~((pi[col] == mi[col]) | (pi[col].isna() & mi[col].isna()))
                for gid in pi.index[diff.to_numpy()]:
                    err("immutable_field_changed", f"{col} changed", gid)
            cw = current_week if current_week is not None else 0
            frozen = (pi["is_frozen"] | (pi["week"] < cw)).to_numpy()
            for col in FROZEN_FIELDS:
                a, b = pi.loc[frozen, col], mi.loc[frozen, col]
                # EXACT equality: a frozen week is permanently static, not "almost" static
                same = ((a == b) | (a.isna() & b.isna())).to_numpy()
                for gid in a.index[~same]:
                    err("frozen_week_modified", f"{col} changed in a frozen week", gid)
            still = mi.loc[~mi["is_frozen"] & (mi["week"] < cw)]
            for gid in still.index:
                err("freeze_not_applied", "completed week not marked frozen", gid)
    return issues
