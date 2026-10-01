"""Market-perceived team strength implied by game lines, anchored on preseason win totals.

Model (all in logit units):   logit P(home wins) = r_home - r_away + hfa        (hfa = 0 if neutral)

1. ANCHOR (t = 0).  Preseason win totals -> expected wins (vig removed) -> ratings that reproduce
   each team's expected wins over its ACTUAL schedule (opponent- and home/away-adjusted).
   These are the prior:  r ~ N(r_win, prior_sd^2 I).
2. RECONCILE (optional, recommended).  The stale lines being adjusted were set by the GAME market,
   not the win-total market.  Any gap between the two would later masquerade as "movement", so the
   preseason game lines are folded in as noisy observations (noise = preseason_line_sd).  Larger
   noise => win totals dominate; 1e6 => pure win totals.  A diagnostic quantifies the gap.
3. UPDATE (each refresh).  A Kalman filter: random-walk process noise for elapsed time, then a
   measurement update from every newly refreshed game line (y = logit(p) - hfa = r_h - r_a + e).
   One line only identifies a DIFFERENCE, so the full covariance is carried between runs and
   evidence accumulates across weeks; the win-total prior is used once and its influence fades.

Movement for a stale game = rating(now) - rating(snapshot that already contained the game's line),
centred on the league median.  See DESIGN.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, RatingCfg, StrengthCfg
from .errors import InputError
from .probability import logit
from .tabular import atomic_write_csv, atomic_write_text, iso
from .teams import CANONICAL_TEAMS

TEAMS = list(CANONICAL_TEAMS)
IDX = {t: i for i, t in enumerate(TEAMS)}
N = len(TEAMS)


def team_idx(names) -> np.ndarray:
    return np.array([IDX[t] for t in names], dtype=int)


# ---------------------------------------------------------------------------------------
# Home-field advantage
# ---------------------------------------------------------------------------------------
def estimate_hfa(y_logit: np.ndarray, hi: np.ndarray, ai: np.ndarray, neutral: np.ndarray) -> float:
    """Intercept of the regression  logit(p_home) = r_home - r_away + hfa * [not neutral].

    NOT the plain mean of home logits: that mean also contains the average strength gap between
    the teams that happen to be at home and away, which is schedule imbalance, not home field."""
    m = len(y_logit)
    d = np.zeros((m, N))
    d[np.arange(m), hi] = 1.0
    d[np.arange(m), ai] = -1.0
    x = np.column_stack([d, (~neutral).astype(float)])
    return float(np.linalg.lstsq(x, y_logit, rcond=None)[0][-1])


# ---------------------------------------------------------------------------------------
# Win totals -> ratings
# ---------------------------------------------------------------------------------------
def expected_wins(r: np.ndarray, hi: np.ndarray, ai: np.ndarray, hfa: np.ndarray):
    """Expected wins per team and per-game home win probability."""
    p = 1.0 / (1.0 + np.exp(-(r[hi] - r[ai] + hfa)))
    e = np.bincount(hi, weights=p, minlength=N) + np.bincount(ai, weights=1.0 - p, minlength=N)
    return e, p


def normalize_win_totals(totals: np.ndarray, n_games: int, mode: str) -> tuple[np.ndarray, float]:
    """Books shade totals so they sum to more than the number of games.  Returns
    (target expected wins, average wins removed per team)."""
    total = float(totals.sum())
    if mode == "additive":
        shift = (total - n_games) / len(totals)
        return totals - shift, shift
    if mode == "proportional":
        return totals * (n_games / total), total / len(totals) - n_games / len(totals)
    return totals.copy(), 0.0


def solve_win_total_ratings(target: np.ndarray, hi, ai, hfa: np.ndarray,
                            tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """Newton solve for ratings (mean zero) whose schedule-expected wins equal ``target``."""
    r = np.zeros(N)
    for _ in range(max_iter):
        e, p = expected_wins(r, hi, ai, hfa)
        resid = target - e
        if np.max(np.abs(resid - resid.mean())) < tol:
            return r - r.mean()
        w = p * (1.0 - p)
        jac = np.zeros((N, N))
        np.add.at(jac, (hi, hi), w)
        np.add.at(jac, (ai, ai), w)
        np.add.at(jac, (hi, ai), -w)
        np.add.at(jac, (ai, hi), -w)
        step = np.linalg.pinv(jac) @ resid
        big = float(np.max(np.abs(step)))
        if big > 1.0:                       # damp wild steps
            step = step / big
        r = r + step
        r = r - r.mean()
    raise InputError("win-total -> rating solve did not converge; check the win totals "
                     "(are they plausible for this schedule?)")


# ---------------------------------------------------------------------------------------
# Kalman machinery
# ---------------------------------------------------------------------------------------
def kalman_update(r: np.ndarray, cov: np.ndarray, hi, ai, y: np.ndarray, var: np.ndarray):
    """Measurement update for observations  y_k = r[hi_k] - r[ai_k] + e_k,  e_k ~ N(0, var_k)."""
    m = len(y)
    h = np.zeros((m, N))
    h[np.arange(m), hi] = 1.0
    h[np.arange(m), ai] = -1.0
    s = h @ cov @ h.T + np.diag(var)
    k = np.linalg.solve(s, h @ cov).T             # K = P H' S^-1  (P, S symmetric)
    innov = y - h @ r
    r_new = r + k @ innov
    ikh = np.eye(N) - k @ h
    cov_new = ikh @ cov @ ikh.T + k @ np.diag(var) @ k.T     # Joseph form (stays PSD)
    return r_new - r_new.mean(), (cov_new + cov_new.T) / 2.0, innov


@dataclass
class Baseline:
    ratings: np.ndarray
    cov: np.ndarray
    hfa: float
    anchor: pd.DataFrame                  # per-team audit table of the baseline construction
    diagnostics: dict = field(default_factory=dict)


def build_baseline(wins: pd.DataFrame, hi, ai, pre_logit_home: np.ndarray, neutral: np.ndarray,
                   cfg: RatingCfg) -> Baseline:
    """Anchor on win totals, optionally reconcile with the preseason game lines."""
    n_games = len(hi)
    hfa = (estimate_hfa(pre_logit_home, hi, ai, neutral) if cfg.hfa_mode == "estimate"
           else cfg.hfa_logit)
    hfa_vec = np.where(neutral, 0.0, hfa)
    exp_w = wins.set_index("team").reindex(TEAMS)["expected_wins"].to_numpy(float)
    target, shift = normalize_win_totals(exp_w, n_games, cfg.normalize)
    r_win = solve_win_total_ratings(target, hi, ai, hfa_vec)
    prior_cov = cfg.prior_sd ** 2 * np.eye(N)
    y = pre_logit_home - hfa_vec
    # game-lines-only fit (diffuse prior).  Fixed noise so this yardstick does NOT depend on
    # preseason_line_sd, the very setting it is used to evaluate.
    r_lines, _, _ = kalman_update(np.zeros(N), 100.0 * np.eye(N), hi, ai, y,
                                  np.full(n_games, cfg.fresh_line_sd ** 2))
    if cfg.reconcile_preseason_lines:
        r0, cov0, _ = kalman_update(r_win, prior_cov, hi, ai, y,
                                    np.full(n_games, cfg.preseason_line_sd ** 2))
    else:
        r0, cov0 = r_win.copy(), prior_cov
    gap = r_win - r_lines
    denom = float(np.sum(gap ** 2))
    weight = 1.0 if not cfg.reconcile_preseason_lines else (
        float(np.sum((r0 - r_lines) * gap) / denom) if denom > 1e-12 else 1.0)
    e_lines, _ = expected_wins(r_lines, hi, ai, hfa_vec)
    e0, _ = expected_wins(r0, hi, ai, hfa_vec)
    anchor = pd.DataFrame({
        "team": TEAMS, "win_total": wins.set_index("team").reindex(TEAMS)["win_total"].to_numpy(),
        "expected_wins_used": target, "win_total_rating": r_win, "lines_only_rating": r_lines,
        "baseline_rating": r0, "baseline_sd": np.sqrt(np.diag(cov0)),
        "baseline_implied_wins": e0, "lines_implied_wins": e_lines})
    worst = anchor.assign(gap_wins=lambda d: d["lines_implied_wins"] - d["expected_wins_used"]) \
        .reindex(columns=["team", "gap_wins"]).sort_values("gap_wins", key=np.abs, ascending=False).head(5)
    diag = {
        "hfa_logit": round(hfa, 4), "hfa_mode": cfg.hfa_mode,
        "neutral_games": int(neutral.sum()),
        "win_total_sum": round(float(exp_w.sum()), 2), "games": n_games,
        "vig_removed_wins_per_team": round(shift, 3),
        "rms_gap_win_totals_vs_preseason_lines_logit": round(float(np.sqrt(np.mean(gap ** 2))), 4),
        "rms_gap_win_totals_vs_preseason_lines_wins": round(float(np.sqrt(np.mean((e_lines - target) ** 2))), 3),
        "reconciled": bool(cfg.reconcile_preseason_lines),
        "win_total_weight_in_baseline": round(weight, 3),
        "rms_baseline_shift_from_win_totals_logit": round(float(np.sqrt(np.mean((r0 - r_win) ** 2))), 4),
        "largest_win_total_vs_lines_gaps_wins": [
            {"team": t, "gap_wins": round(float(g), 2)} for t, g in zip(worst["team"], worst["gap_wins"])],
    }
    return Baseline(r0, cov0, hfa, anchor, diag)


# ---------------------------------------------------------------------------------------
# History (snapshots + filter state)
# ---------------------------------------------------------------------------------------
HIST_COLS = ["snapshot_id", "snapshot_ts", "kind", "team", "rating", "sd"]


class RatingHistory:
    """Append-only rating snapshots plus the filter's current covariance and hfa."""

    def __init__(self, df: pd.DataFrame | None = None, cov: np.ndarray | None = None,
                 hfa: float = 0.0, anchor: pd.DataFrame | None = None):
        if df is None:
            df = pd.DataFrame({"snapshot_id": pd.Series(dtype="int64"),
                               "snapshot_ts": pd.Series(dtype="datetime64[ns, UTC]"),
                               "kind": pd.Series(dtype="object"), "team": pd.Series(dtype="object"),
                               "rating": pd.Series(dtype="float64"), "sd": pd.Series(dtype="float64")})
        self.df = df.reset_index(drop=True)
        self.cov = cov if cov is not None else np.eye(N)
        self.hfa = hfa
        self.anchor = anchor

    # -- persistence -------------------------------------------------------------------
    @classmethod
    def load(cls, hist_path, state_path) -> "RatingHistory":
        df = pd.read_csv(hist_path, dtype={"team": str, "kind": str}, float_precision="round_trip")
        df["snapshot_id"] = df["snapshot_id"].astype("int64")
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
        st = json.loads(open(state_path).read())
        if st["teams"] != TEAMS:
            raise InputError("rating_state.json team list does not match this version of the code")
        anchor = pd.DataFrame(st["anchor"]) if st.get("anchor") else None
        return cls(df[HIST_COLS], np.array(st["cov"], dtype=float), float(st["hfa"]), anchor)

    def save(self, hist_path, state_path) -> None:
        out = self.df.copy()
        out["snapshot_ts"] = out["snapshot_ts"].map(iso)
        atomic_write_csv(out[HIST_COLS], hist_path)
        state = {"hfa": self.hfa, "latest_snapshot_id": self.latest_id(), "teams": TEAMS,
                 "cov": self.cov.tolist(),
                 "anchor": None if self.anchor is None else self.anchor.to_dict("records")}
        atomic_write_text(state_path, json.dumps(state))

    def copy(self) -> "RatingHistory":
        return RatingHistory(self.df.copy(), self.cov.copy(), self.hfa,
                             None if self.anchor is None else self.anchor.copy())

    # -- snapshots ---------------------------------------------------------------------------
    @property
    def empty(self) -> bool:
        return self.df.empty

    def snapshots(self) -> pd.DataFrame:
        return (self.df.drop_duplicates("snapshot_id")[["snapshot_id", "snapshot_ts", "kind"]]
                .sort_values("snapshot_id").reset_index(drop=True))

    def baseline_id(self) -> int:
        return 0

    def latest_id(self) -> int:
        return int(self.df["snapshot_id"].max())

    def latest_ts(self) -> pd.Timestamp:
        return self.snapshots().iloc[-1]["snapshot_ts"]

    def add_snapshot(self, ratings: np.ndarray, cov: np.ndarray, ts: pd.Timestamp, kind: str) -> int:
        sid = 0 if self.empty else self.latest_id() + 1
        add = pd.DataFrame({"snapshot_id": sid, "snapshot_ts": pd.Timestamp(ts), "kind": kind,
                            "team": TEAMS, "rating": ratings.astype(float),
                            "sd": np.sqrt(np.diag(cov))})
        self.df = pd.concat([self.df, add], ignore_index=True)
        self.cov = cov
        return sid

    def ratings(self, snapshot_id: int) -> pd.Series:
        d = self.df[self.df["snapshot_id"] == snapshot_id]
        return pd.Series(d["rating"].to_numpy(float), index=d["team"].to_numpy())

    def movement(self, ref_id: int, now_id: int, cfg: StrengthCfg) -> pd.Series:
        """Per-team rating change between two snapshots, centred on the league (logit units)."""
        ref, now = self.ratings(ref_id), self.ratings(now_id).reindex(self.ratings(ref_id).index)
        raw = now - ref
        center = {"median": raw.median(), "mean": raw.mean(), "none": 0.0}[cfg.center]
        return raw - float(center)


def advance_ratings(hist: RatingHistory, obs: list[dict], as_of: pd.Timestamp, cfg: RatingCfg,
                    current_week: int) -> tuple[int, dict]:
    """Time-update to ``as_of`` then assimilate newly refreshed lines; appends a snapshot.

    obs: dicts with home, away (canonical ids), p_home (fair prob), week, neutral (bool)."""
    prev_r = hist.ratings(hist.latest_id()).reindex(TEAMS).to_numpy()
    dt_weeks = max(0.0, (as_of - hist.latest_ts()).total_seconds() / (7 * 86400.0))
    cov = hist.cov + cfg.process_sd_per_week ** 2 * dt_weeks * np.eye(N)
    r, stats = prev_r.copy(), {"n_obs": len(obs), "dt_weeks": round(dt_weeks, 3)}
    if obs:
        hi = team_idx([o["home"] for o in obs])
        ai = team_idx([o["away"] for o in obs])
        y = np.array([logit(o["p_home"]) - (0.0 if o["neutral"] else hist.hfa) for o in obs])
        look = np.array([max(0, o["week"] - (current_week + 1)) for o in obs], dtype=float)
        var = cfg.fresh_line_sd ** 2 + (cfg.lookahead_sd_per_week * look) ** 2
        r, cov, innov = kalman_update(prev_r, cov, hi, ai, y, var)
        stats["rms_innovation_logit"] = round(float(np.sqrt(np.mean(innov ** 2))), 4)
    chg = r - prev_r
    stats["max_abs_rating_change"] = round(float(np.max(np.abs(chg))), 4)
    if len(obs) >= 8 and np.std(prev_r) > 1e-9:       # dispersion-drift diagnostic (report only)
        stats["scale_drift_slope"] = round(float(np.polyfit(prev_r, chg, 1)[0]), 4)
    sid = hist.add_snapshot(r, cov, as_of, "update")
    stats["snapshot_id"] = sid
    return sid, stats


def ratings_tables(hist: RatingHistory, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(latest power-rating table, wide history) for output files."""
    now = hist.ratings(hist.latest_id())
    last = hist.df[hist.df["snapshot_id"] == hist.latest_id()].set_index("team")["sd"]
    a = hist.anchor.set_index("team") if hist.anchor is not None else None
    lp = cfg.output.logit_per_point
    t = pd.DataFrame({"Team": TEAMS, "CurrentRating": now.reindex(TEAMS).to_numpy(),
                      "RatingSD": last.reindex(TEAMS).to_numpy()})
    if a is not None:
        t["WinTotal"] = a.reindex(TEAMS)["win_total"].to_numpy()
        t["WinTotalRating"] = a.reindex(TEAMS)["win_total_rating"].to_numpy()
        t["BaselineRating"] = a.reindex(TEAMS)["baseline_rating"].to_numpy()
        t["ChangeSinceBaseline"] = t["CurrentRating"] - t["BaselineRating"]
    t["CurrentSpreadEquivPoints"] = t["CurrentRating"] / lp
    t = t.sort_values("CurrentRating", ascending=False).round(4)
    wide = hist.df.pivot(index="team", columns="snapshot_id", values="rating").reindex(TEAMS)
    stamps = hist.snapshots().set_index("snapshot_id")["snapshot_ts"]
    wide.columns = [f"S{c} {stamps[c]:%Y-%m-%d}" for c in wide.columns]
    wide.index.name = "Team"
    return t, wide.round(4)
