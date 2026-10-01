"""The stateful refresh engine.

``apply_init`` / ``apply_refresh`` are PURE (dataframes in -> dataframes out, nothing is
written) so they can be unit-tested.  ``run_init`` / ``run_refresh`` wrap them with file I/O,
integrity checks and the atomic commit of the new master version.

Refresh sequence (maps to spec section 19; freezing is done FIRST so that nothing below can
touch a completed week - the outcome is identical, the safety is stronger):

  1  load previous master (verified)          2  freeze weeks < current_week
  3  read incoming lines                      4  match + keep only genuinely newer lines
  5  de-vig -> replace probability            6  update last-refresh timestamp
  7  (freeze, done at step 2)                 8  Kalman update of team ratings from the new lines
  9  age of every active game's Vegas line    10 rating movement since that line + meaningful gate
  11 age-weighted adjustment (anchored)       12 validate everything
  13 commit new master + grid + rating tables
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .adjustment import age_weight, compute_adjustment
from .config import Config
from .errors import InputError, Issue, StateError, ValidationError
from .grid import build_grid
from .ingest import (ingest_win_totals, match_to_master, resolve_duplicates,
                     standardize_lines)
from .master import StateStore, build_initial_master
from .probability import logit, shift_probability
from .ratings import (RatingHistory, advance_ratings, build_baseline, ratings_tables,
                      team_idx)
from .tabular import atomic_write_csv, iso, parse_timestamp, read_table
from .teams import TeamNormalizer
from .validate import validate_master

NAN = float("nan")


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------
@dataclass
class RunReport:
    run_type: str
    version: int
    as_of: str
    current_week: int
    counts: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    rejected_rows: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> dict:
        d = {"run_type": self.run_type, "version": self.version, "as_of": self.as_of,
             "current_week": self.current_week, "counts": self.counts,
             "details": self.details, "issues": [vars(i) for i in self.issues],
             "rejected_rows": self.rejected_rows}
        return d

    def summary(self) -> str:
        lines = [f"{self.run_type} -> master v{self.version}  (as_of {self.as_of}, "
                 f"current_week {self.current_week})"]
        lines += [f"  {k:<32}{v}" for k, v in self.counts.items()]
        b = self.details.get("baseline")
        if b:
            lines.append(f"  baseline: hfa={b['hfa_logit']} logit | win totals vs preseason lines RMS gap "
                         f"{b['rms_gap_win_totals_vs_preseason_lines_wins']} wins | weight on win totals "
                         f"{b['win_total_weight_in_baseline']}")
        rs = self.details.get("ratings")
        if rs:
            lines.append(f"  ratings: snapshot {rs['snapshot_id']}, {rs['n_obs']} new lines assimilated, "
                         f"max rating change {rs['max_abs_rating_change']} logit")
        sk = self.details.get("adjustment_skill")
        if sk:
            lines.append(f"  adjustment skill on {sk['n']} adjusted games that just got fresh lines: error "
                         f"{sk['mean_abs_err_stale_logit']} (unadjusted) -> {sk['mean_abs_err_adjusted_logit']} "
                         f"(adjusted); improved in {sk['share_improved']:.0%}")
        by_code: dict[str, int] = {}
        for r in self.rejected_rows:
            by_code[r["code"]] = by_code.get(r["code"], 0) + 1
        if by_code:
            lines.append(f"  rejected rows by reason:   {by_code}")
        warn = [i for i in self.issues if i.level in ("warning", "error")]
        for i in warn[:15]:
            lines.append(f"  ! {i}")
        if len(warn) > 15:
            lines.append(f"  ! ... {len(warn) - 15} more (see run report JSON)")
        return "\n".join(lines)


@dataclass
class RefreshResult:
    master: pd.DataFrame
    hist: RatingHistory
    report: RunReport
    changes: pd.DataFrame


def resolve_as_of(cfg: Config, as_of) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp.now(tz="UTC").floor("s")
    return parse_timestamp(as_of, cfg.time.input_timezone, cfg.time.timestamp_format)


def _raise_on_errors(issues: list[Issue]) -> None:
    if any(i.level == "error" for i in issues):
        raise ValidationError(issues)


# ---------------------------------------------------------------------------------------
# Init (master v1)
# ---------------------------------------------------------------------------------------
def apply_init(lines_raw: pd.DataFrame, wt_raw: pd.DataFrame, cfg: Config, *,
               as_of: pd.Timestamp, current_week: int) -> RefreshResult:
    rep = RunReport("init", 1, iso(as_of), current_week)
    nz = TeamNormalizer(cfg.team_aliases)
    std, rejected = standardize_lines(lines_raw, cfg.columns_initial, nz, cfg, as_of=as_of,
                                      label="initial")
    rep.bump("rows_in", len(lines_raw))
    if rejected:
        detail = "\n  ".join(f"row {r['row']}: {r['code']} - {r['detail']}" for r in rejected[:20])
        raise InputError(f"Initial lines contain {len(rejected)} unusable row(s); the master "
                         f"table must be complete:\n  {detail}")
    std["game_id"] = [f"{cfg.season.season}-W{w:02d}-{a}@{h}" for w, a, h in
                      zip(std["week"], std["away_team"], std["home_team"])]
    std["swapped"] = False
    kept, _, dup_issues = resolve_duplicates(std)
    rep.issues += dup_issues
    _raise_on_errors(dup_issues)
    if len(kept) < len(std):
        rep.bump("duplicate_rows_collapsed", len(std) - len(kept))

    master = build_initial_master(kept, cfg, version=1, current_week=current_week)
    rep.bump("games", len(master))
    rep.bump("frozen_games", int(master["is_frozen"].sum()))

    # ---- rating baseline: win totals (anchor) + preseason game lines (reconciliation) ------------
    wins, wt_issues, wt_ts = ingest_win_totals(wt_raw, cfg, nz, n_games=len(master))
    rep.issues += wt_issues
    unknown = sorted(set(cfg.rating.neutral_site_games) - set(master["game_id"]))
    if unknown:
        rep.add(Issue("warning", "unknown_neutral_game", f"neutral_site_games not in schedule: {unknown}"))
    neutral = master["game_id"].isin(cfg.rating.neutral_site_games).to_numpy()
    hi, ai = team_idx(master["home_team"]), team_idx(master["away_team"])
    pre_logit = np.array([logit(p) for p in master["home_win_prob"]])
    base = build_baseline(wins, hi, ai, pre_logit, neutral, cfg.rating)
    hist = RatingHistory(hfa=base.hfa, anchor=base.anchor)
    b_ts = master["original_vegas_ts"].min()
    if wt_ts is not None:
        b_ts = min(b_ts, wt_ts)
    hist.add_snapshot(base.ratings, base.cov, b_ts, "baseline")
    rep.details["baseline"] = base.diagnostics

    issues = validate_master(master, cfg, prev=None, current_week=current_week)
    rep.issues += issues
    _raise_on_errors(issues)
    return RefreshResult(master, hist, rep, master.iloc[0:0])


# ---------------------------------------------------------------------------------------
# Refresh (master v_n -> v_n+1)
# ---------------------------------------------------------------------------------------
def apply_refresh(prev: pd.DataFrame, hist_in: RatingHistory, lines_raw: pd.DataFrame | None,
                  cfg: Config, *, version: int, current_week: int, as_of: pd.Timestamp,
                  prev_current_week: int | None = None) -> RefreshResult:
    rep = RunReport("refresh", version, iso(as_of), current_week)
    nz = TeamNormalizer(cfg.team_aliases)
    hist = hist_in.copy()
    m = prev.copy()
    m.index = pd.Index(m["game_id"].to_numpy())  # index by game id (no column-name clash)

    if not (1 <= current_week <= cfg.season.n_weeks + 1):
        raise ValidationError([Issue("error", "bad_current_week",
                                     f"current_week {current_week} outside 1..{cfg.season.n_weeks + 1}")])
    if prev_current_week is not None and current_week < prev_current_week:
        raise ValidationError([Issue("error", "frozen_week_modified",
                                     f"current_week {current_week} < previous {prev_current_week}: "
                                     "that would un-freeze completed weeks")])

    # ---- Step 2/7: freeze completed weeks (before anything can touch them) -------------------
    newly = (m["week"] < current_week) & ~m["is_frozen"]
    m.loc[newly, "is_frozen"] = True
    m.loc[newly, "frozen_at_version"] = version
    rep.bump("newly_frozen", int(newly.sum()))

    # ---- Steps 3-6: incoming Vegas lines ----------------------------------------------------
    refreshed: set[str] = set()
    n_in = 0 if lines_raw is None else len(lines_raw)
    rep.bump("rows_in", n_in)
    if n_in:
        std, rejected = standardize_lines(lines_raw, cfg.columns_refresh, nz, cfg, as_of=as_of,
                                          label="refresh")
        rep.rejected_rows += rejected
        matched, rej_m, iss_m = match_to_master(std, m, cfg)
        rep.rejected_rows += rej_m
        rep.issues += iss_m
        kept, rej_d, iss_d = resolve_duplicates(matched)
        rep.rejected_rows += rej_d
        rep.issues += iss_d
        rep.bump("rows_rejected", len(rep.rejected_rows))
        rep.bump("rows_matched", len(matched))
        for r in rep.rejected_rows:
            rep.add(Issue("warning", r["code"], f"row {r['row']}: {r['detail']}"))
        if len(rep.rejected_rows) / n_in > cfg.validation.max_reject_fraction:
            raise ValidationError([Issue("error", "too_many_rejected_rows",
                                         f"{len(rep.rejected_rows)}/{n_in} incoming rows rejected "
                                         "(> validation.max_reject_fraction); likely a schema or "
                                         "team-mapping problem. Nothing was written.")] + rep.issues)
        if cfg.validation.strict and rep.rejected_rows:
            raise ValidationError([Issue("error", "strict_mode_rejections",
                                         f"{len(rep.rejected_rows)} rejected rows in strict mode")]
                                  + rep.issues)

        for r in kept.itertuples(index=False):
            gid = r.game_id
            if m.at[gid, "is_frozen"]:
                rep.bump("ignored_frozen_week")
                rep.add(Issue("warning", "frozen_week_line_ignored",
                              f"incoming line (week {r.week}) targets a frozen week", gid))
                continue
            last = m.at[gid, "last_vegas_refresh_ts"]
            if r.ts < last:
                rep.bump("ignored_older")
                rep.add(Issue("warning", "incoming_older_than_existing",
                              f"incoming {iso(r.ts)} < stored {iso(last)}", gid))
                continue
            if r.ts == last:
                rep.bump("ignored_same_timestamp")
                if abs(r.home_prob - m.at[gid, "latest_vegas_home_prob"]) > 1e-9:
                    rep.add(Issue("warning", "same_timestamp_different_odds",
                                  "same timestamp as stored line but different odds; ignored", gid))
                continue
            m.at[gid, "latest_vegas_home_prob"] = r.home_prob
            m.at[gid, "latest_home_odds"] = r.home_odds
            m.at[gid, "latest_away_odds"] = r.away_odds
            m.at[gid, "last_vegas_refresh_ts"] = r.ts
            m.at[gid, "refresh_count"] = int(m.at[gid, "refresh_count"]) + 1
            m.at[gid, "last_refresh_version"] = version
            if r.source_game_id and not m.at[gid, "source_game_id"]:
                m.at[gid, "source_game_id"] = r.source_game_id
            refreshed.add(gid)
        rep.bump("lines_refreshed", len(refreshed))

    # ---- live validation: did last run's ADJUSTED probabilities anticipate the lines that just arrived?
    pi = prev.set_index("game_id")
    skill = []
    for gid in refreshed:
        if bool(pi.at[gid, "adj_active"]):
            new_l = logit(float(m.at[gid, "latest_vegas_home_prob"]))
            skill.append((abs(new_l - logit(float(pi.at[gid, "home_win_prob"]))),
                          abs(new_l - logit(float(pi.at[gid, "latest_vegas_home_prob"])))))
    if skill:
        a = np.array(skill)
        rep.details["adjustment_skill"] = {
            "n": len(a), "mean_abs_err_adjusted_logit": round(float(a[:, 0].mean()), 4),
            "mean_abs_err_stale_logit": round(float(a[:, 1].mean()), 4),
            "share_improved": float((a[:, 0] < a[:, 1]).mean())}

    # ---- Step 8: update team ratings from the lines refreshed in THIS run ------------------------
    neutral_ids = set(cfg.rating.neutral_site_games)
    obs = [{"home": m.at[g, "home_team"], "away": m.at[g, "away_team"],
            "p_home": float(m.at[g, "latest_vegas_home_prob"]), "week": int(m.at[g, "week"]),
            "neutral": g in neutral_ids} for g in sorted(refreshed)]
    snap_id, rstats = advance_ratings(hist, obs, as_of, cfg.rating, current_week)
    rep.details["ratings"] = rstats
    for g in refreshed:
        m.at[g, "anchor_rating_snapshot_id"] = snap_id     # this snapshot already contains the line
    now_id = hist.latest_id()

    # ---- Steps 9-11: age, rating movement since the line, gate, anchored adjustment ----------------
    move_cache: dict[tuple[int, int], pd.Series] = {}
    unit = 86400.0 * cfg.time.age_unit_days
    n_active = n_adj = 0
    for gid in m.index[~m["is_frozen"].to_numpy()]:
        n_active += 1
        anchor = float(m.at[gid, "latest_vegas_home_prob"])
        last_ts = m.at[gid, "last_vegas_refresh_ts"]
        age = max(0.0, (as_of - last_ts).total_seconds() / unit)
        w = age_weight(age, cfg.aging)
        adj = 0.0
        f = dict(home_rating_move=NAN, away_rating_move=NAN, home_strength_shift=NAN,
                 away_strength_shift=NAN, matchup_shift=NAN)
        ref_id = None
        if w > 0:
            ref_id = (hist.baseline_id() if cfg.strength.reference == "preseason"
                      else int(m.at[gid, "anchor_rating_snapshot_id"]))
            key = (ref_id, now_id)
            if key not in move_cache:
                move_cache[key] = hist.movement(ref_id, now_id, cfg.strength)
            mv = move_cache[key]
            mh, ma = float(mv[m.at[gid, "home_team"]]), float(mv[m.at[gid, "away_team"]])
            res = compute_adjustment(mh, ma, w, cfg.strength, cfg.adjust)
            adj = res.adj_logit
            f = dict(home_rating_move=mh, away_rating_move=ma, home_strength_shift=res.home_shift,
                     away_strength_shift=res.away_shift, matchup_shift=res.delta)
        active = abs(adj) > 1e-12
        p_home = shift_probability(anchor, adj, cfg.adjust.link) if active else anchor
        n_adj += int(active)
        is_fresh = gid in refreshed
        source = ("VEGAS_STALE_ADJUSTED" if active else "VEGAS_FRESH" if is_fresh else
                  "VEGAS_OLDER" if int(m.at[gid, "refresh_count"]) > 0 else "VEGAS_PRESEASON")
        m.at[gid, "home_win_prob"] = p_home
        m.at[gid, "away_win_prob"] = 1.0 - p_home
        m.at[gid, "prob_source"] = source
        m.at[gid, "adj_active"] = active
        m.at[gid, "adj_logit"] = adj if active else 0.0
        m.at[gid, "adj_home_prob_delta"] = p_home - anchor
        m.at[gid, "line_age_weeks"] = age
        m.at[gid, "age_weight"] = w
        for k, v in f.items():
            m.at[gid, k] = v
        m.at[gid, "rating_ref_snapshot_id"] = pd.NA if ref_id is None else ref_id
        m.at[gid, "rating_now_snapshot_id"] = pd.NA if ref_id is None else now_id
    rep.bump("active_games", n_active)
    rep.bump("stale_adjusted", n_adj)

    # ---- last_updated_version: bump where the probability actually moved --------------------
    pprev = pd.Series(prev["home_win_prob"].to_numpy(), index=prev["game_id"].to_numpy())
    prob_moved = ((m["home_win_prob"] - pprev.reindex(m.index)).abs() > 1e-12).to_numpy()
    moved = prob_moved | m.index.isin(list(refreshed))
    m.loc[moved, "last_updated_version"] = version
    rep.bump("probabilities_changed", int(prob_moved.sum()))

    # ---- Step 12: validate ---------------------------------------------------------------------
    m = m.reset_index(drop=True)
    issues = validate_master(m, cfg, prev=prev, current_week=current_week)
    rep.issues += issues
    _raise_on_errors(issues)
    changes = _changes(prev, m)
    return RefreshResult(m, hist, rep, changes)


def _changes(prev: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    p, n = prev.set_index("game_id"), new.set_index("game_id").loc[prev["game_id"].to_numpy()]
    delta = n["home_win_prob"] - p["home_win_prob"]
    mask = ((delta.abs() > 1e-12) | (n["prob_source"] != p["prob_source"])).to_numpy()
    out = pd.DataFrame({
        "game_id": p.index, "week": p["week"].to_numpy(), "away_team": p["away_team"].to_numpy(),
        "home_team": p["home_team"].to_numpy(), "prev_home_prob": p["home_win_prob"].to_numpy(),
        "new_home_prob": n["home_win_prob"].to_numpy(), "delta_pp": (delta * 100).to_numpy(),
        "prev_source": p["prob_source"].to_numpy(), "new_source": n["prob_source"].to_numpy(),
    })
    return out[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# I/O wrappers
# ---------------------------------------------------------------------------------------
def _write_outputs(cfg: Config, master: pd.DataFrame, version: int, hist=None) -> pd.DataFrame:
    grid = build_grid(master, cfg)
    out = cfg.paths.output_dir
    from pathlib import Path
    from .master import master_to_csv_frame
    atomic_write_csv(grid, Path(out) / f"prob_grid_v{version:04d}.csv", index=True)
    atomic_write_csv(grid, Path(out) / "prob_grid_latest.csv", index=True)
    atomic_write_csv(master_to_csv_frame(master), Path(out) / "master_game_table_latest.csv")
    if hist is not None:
        latest, wide = ratings_tables(hist, cfg)
        atomic_write_csv(latest, Path(out) / "team_ratings_latest.csv")
        atomic_write_csv(wide, Path(out) / "team_ratings_history.csv", index=True)
    return grid


def run_init(cfg: Config, *, as_of=None, current_week: int | None = None, force: bool = False,
             lines_source: str | None = None, win_totals_source: str | None = None,
             dry_run: bool = False, lines_df: pd.DataFrame | None = None,
             win_totals_df: pd.DataFrame | None = None, details: dict | None = None) -> RunReport:
    as_of = resolve_as_of(cfg, as_of)
    cw = current_week or cfg.season.current_week
    store = StateStore(cfg.paths.state_dir)
    if store.exists() and not force and not dry_run:
        raise StateError(f"State already exists in {store.dir}. Use --force to archive it and "
                         "start over (the old state is moved aside, not deleted).")
    lines = lines_df if lines_df is not None else read_table(lines_source or cfg.paths.initial_lines, cfg.paths.sheet_name)
    wt = win_totals_df if win_totals_df is not None else read_table(win_totals_source or cfg.paths.win_totals, cfg.paths.sheet_name)
    res = apply_init(lines, wt, cfg, as_of=as_of, current_week=cw)
    res.report.details.update(details or {})
    if dry_run:
        return res.report
    if store.exists():
        store.archive_existing()
    store.commit(res.master, res.hist, version=1, run_type="init", as_of=as_of, current_week=cw,
                 report=res.report.to_dict(), changes=res.changes, cfg=cfg)
    _write_outputs(cfg, res.master, 1, res.hist)
    return res.report


def run_refresh(cfg: Config, *, as_of=None, current_week: int | None = None,
                lines_source: str | None = None, use_lines: bool = True,
                dry_run: bool = False, lines_df: pd.DataFrame | None = None,
                details: dict | None = None) -> RunReport:
    as_of = resolve_as_of(cfg, as_of)
    store = StateStore(cfg.paths.state_dir)
    prev, hist, man = store.load(verify=cfg.validation.verify_integrity)
    cw = current_week or cfg.season.current_week
    lines_raw = None
    if lines_df is not None:
        lines_raw = lines_df
    elif use_lines:
        lines_raw = read_table(lines_source or cfg.paths.refresh_lines, cfg.paths.sheet_name)
    version = man["current_version"] + 1
    res = apply_refresh(prev, hist, lines_raw, cfg, version=version, current_week=cw,
                        as_of=as_of, prev_current_week=man.get("current_week"))
    res.report.details.update(details or {})
    if dry_run:
        return res.report
    store.commit(res.master, res.hist, version=version, run_type="refresh", as_of=as_of,
                 current_week=cw, report=res.report.to_dict(), changes=res.changes, cfg=cfg)
    _write_outputs(cfg, res.master, version, res.hist)
    return res.report


def regenerate_grid(cfg: Config) -> pd.DataFrame:
    store = StateStore(cfg.paths.state_dir)
    master, hist, man = store.load(verify=cfg.validation.verify_integrity)
    return _write_outputs(cfg, master, man["current_version"], hist)


def status(cfg: Config) -> dict:
    store = StateStore(cfg.paths.state_dir)
    master, hist, man = store.load(verify=cfg.validation.verify_integrity)
    act = master[~master["is_frozen"]]
    return {
        "version": man["current_version"], "current_week": man["current_week"],
        "games": len(master), "frozen_games": int(master["is_frozen"].sum()),
        "active_games": len(act),
        "by_source": act["prob_source"].value_counts().to_dict(),
        "max_line_age_weeks": float(act["line_age_weeks"].max()) if len(act) else None,
        "rating_snapshots": int(hist.snapshots().shape[0]), "hfa_logit": round(hist.hfa, 4),
        "last_as_of": man["versions"][-1]["as_of"],
    }
