"""Turn raw input tables into validated, canonical rows and match them to master games.

Row-level problems never raise: the row is *rejected* (with a machine-readable code) and
reported, so one bad row cannot corrupt or block the rest.  Table-level problems (missing
columns, an unusable Super Bowl table) raise ``InputError``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, LinesColumns
from .errors import InputError, Issue, OddsError, RowError
from .probability import devig_two_way, parse_decimal_odds, probit
from .tabular import parse_game_date, parse_timestamp, parse_week
from .teams import CANONICAL_TEAMS, TeamNormalizer


def _lookup(df: pd.DataFrame) -> dict:
    return {str(c).strip().lower(): c for c in df.columns}


def _resolve(df: pd.DataFrame, wanted: dict[str, str], required: set[str], label: str) -> dict:
    """Map logical names -> actual dataframe columns (case/whitespace-insensitive)."""
    look, out, missing = _lookup(df), {}, []
    for logical, name in wanted.items():
        if name and name.strip().lower() in look:
            out[logical] = look[name.strip().lower()]
        elif logical in required and name:
            missing.append(name)
    if missing:
        raise InputError(f"{label}: missing column(s) {missing}. Available: {list(df.columns)}")
    return out


# ---------------------------------------------------------------------------------------
# Game-line tables (Input A and Input B share this shape: one row per game)
# ---------------------------------------------------------------------------------------
def standardize_lines(raw: pd.DataFrame, cols: LinesColumns, nz: TeamNormalizer, cfg: Config, *,
                      as_of: pd.Timestamp, label: str) -> tuple[pd.DataFrame, list[dict]]:
    wanted = {k: getattr(cols, k) for k in
              ("game_id", "week", "date", "home_team", "away_team", "home_odds", "away_odds",
               "timestamp")}
    required = {"week", "home_team", "away_team", "home_odds", "away_odds"}
    if cols.timestamp:
        required.add("timestamp")
    c = _resolve(raw, wanted, required, label)

    rows, rejected = [], []
    tz, fmt = cfg.time.input_timezone, cfg.time.timestamp_format
    latest_ok = as_of + pd.Timedelta(hours=cfg.time.max_future_ts_hours)
    for i, rec in enumerate(raw.to_dict("records")):
        rownum = i + 2  # spreadsheet-style row number (header = row 1)
        try:
            home, away = nz(rec[c["home_team"]]), nz(rec[c["away_team"]])
            if home is None or away is None:     # checked first: a bad name is the most useful message
                bad = [rec[c[k]] for k, t in (("home_team", home), ("away_team", away)) if t is None]
                raise RowError("unknown_team", f"unrecognised team name(s): {bad} (add it under [team_aliases])")
            week = parse_week(rec[c["week"]])
            if not (1 <= week <= cfg.season.n_weeks):
                raise RowError("unexpected_week", f"week {week} outside 1..{cfg.season.n_weeks}")
            if home == away:
                raise RowError("same_team", f"home and away are both {home}")
            oh, oa = rec[c["home_odds"]], rec[c["away_odds"]]
            p_home, overround = devig_two_way(oh, oa, cfg.devig)
            ts = parse_timestamp(rec[c["timestamp"]], tz, fmt) if "timestamp" in c else as_of
            if ts > latest_ok:
                raise RowError("timestamp_in_future", f"timestamp {ts} is after as_of {as_of}")
            rows.append({
                "row": rownum,
                "source_game_id": str(rec[c["game_id"]]).strip() if "game_id" in c else "",
                "week": week,
                "game_date": parse_game_date(rec[c["date"]]) if "date" in c else "",
                "home_team": home, "away_team": away,
                "home_odds": parse_decimal_odds(oh, cfg.devig),
                "away_odds": parse_decimal_odds(oa, cfg.devig),
                "home_prob": p_home, "overround": overround, "ts": ts,
            })
        except RowError as exc:  # includes OddsError
            rejected.append({"source": label, "row": rownum, "code": exc.code,
                             "detail": str(exc), "raw": {k: str(v) for k, v in rec.items()}})
    cols_out = ["row", "source_game_id", "week", "game_date", "home_team", "away_team",
                "home_odds", "away_odds", "home_prob", "overround", "ts"]
    return pd.DataFrame(rows, columns=cols_out), rejected


def match_to_master(std: pd.DataFrame, master: pd.DataFrame, cfg: Config
                    ) -> tuple[pd.DataFrame, list[dict], list[Issue]]:
    """Attach each incoming row to a master ``game_id`` (never by row order).

    Priority: (1) explicit game id, (2) (week, away, home[, date]) key, (3) same key with
    home/away reversed (optional, warns and flips the odds), else rejected with a reason.
    Returned rows are expressed in the MASTER's home/away orientation.
    """
    use_date = cfg.matching.use_date_in_key
    by_key, by_src, by_pair = {}, {}, {}
    for r in master.itertuples():
        k = (r.week, r.away_team, r.home_team) + ((r.game_date,) if use_date else ())
        by_key[k] = r.Index
        if r.source_game_id:
            by_src[r.source_game_id] = r.Index
        by_pair.setdefault(frozenset((r.away_team, r.home_team)), []).append(int(r.week))

    out, rejected, issues = [], [], []

    def emit(r, gid: str, swapped: bool) -> None:
        d = r._asdict()
        d.pop("Index", None)
        d["game_id"], d["swapped"] = gid, swapped
        if swapped:
            d["home_team"], d["away_team"] = r.away_team, r.home_team
            d["home_odds"], d["away_odds"] = r.away_odds, r.home_odds
            d["home_prob"] = 1.0 - r.home_prob
            issues.append(Issue("warning", "swapped_home_away",
                                f"row {r.row}: home/away reversed vs schedule; odds flipped", gid))
        out.append(d)

    def reject(r, code: str, detail: str) -> None:
        rejected.append({"source": "refresh", "row": r.row, "code": code, "detail": detail,
                         "raw": {"week": r.week, "away": r.away_team, "home": r.home_team}})

    for r in std.itertuples(index=False):
        date_part = (r.game_date,) if use_date else ()
        key = (r.week, r.away_team, r.home_team) + date_part
        rkey = (r.week, r.home_team, r.away_team) + date_part
        if cfg.matching.prefer_explicit_game_id and r.source_game_id and r.source_game_id in by_src:
            gid = by_src[r.source_game_id]
            m = master.loc[gid]
            if r.week != m["week"] or {r.home_team, r.away_team} != {m["home_team"], m["away_team"]}:
                reject(r, "id_conflict", f"explicit id {r.source_game_id!r} maps to {gid} but "
                       "week/teams disagree")
            elif r.home_team == m["home_team"]:
                emit(r, gid, False)
            elif cfg.matching.resolve_swapped_home_away:
                emit(r, gid, True)
            else:
                reject(r, "swapped_home_away", "home/away reversed and resolution is disabled")
        elif key in by_key:
            emit(r, by_key[key], False)
        elif cfg.matching.resolve_swapped_home_away and rkey in by_key:
            emit(r, by_key[rkey], True)
        else:
            weeks = by_pair.get(frozenset((r.away_team, r.home_team)))
            if weeks:
                reject(r, "unexpected_week", f"{r.away_team}@{r.home_team} is scheduled in "
                       f"week(s) {sorted(weeks)}, not week {r.week}")
            else:
                reject(r, "unmatched_game", f"no scheduled game {r.away_team}@{r.home_team}")
    cols = ["row", "source_game_id", "week", "game_date", "home_team", "away_team", "home_odds",
            "away_odds", "home_prob", "overround", "ts", "game_id", "swapped"]
    return pd.DataFrame(out, columns=cols), rejected, issues


def resolve_duplicates(matched: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[Issue]]:
    """Collapse multiple rows for the same game.  Latest timestamp wins; identical duplicates
    are collapsed (warning); same-timestamp rows with different odds are a conflict and the
    game is dropped from this update (reported), never guessed."""
    keep, rejected, issues = [], [], []
    for gid, g in matched.groupby("game_id", sort=False):
        if len(g) == 1:
            keep.append(g.iloc[0])
            continue
        latest = g[g["ts"] == g["ts"].max()]
        if len(latest) > 1:
            if latest["home_prob"].max() - latest["home_prob"].min() > 1e-9:
                issues.append(Issue("error", "conflicting_duplicates",
                                    f"{len(latest)} rows share timestamp {latest['ts'].iloc[0]} "
                                    "with different odds; game skipped", gid))
                for r in latest.itertuples():
                    rejected.append({"source": "refresh", "row": r.row, "code": "conflicting_duplicates",
                                     "detail": "same timestamp, different odds", "raw": {"game_id": gid}})
                continue
            issues.append(Issue("warning", "duplicate_timestamp",
                                f"{len(latest)} identical rows at {latest['ts'].iloc[0]}", gid))
        keep.append(latest.iloc[0])
    df = pd.DataFrame(keep, columns=matched.columns) if keep else matched.iloc[0:0]
    return df.reset_index(drop=True), rejected, issues


# ---------------------------------------------------------------------------------------
# Preseason win totals (Input C)
# ---------------------------------------------------------------------------------------
def ingest_win_totals(raw: pd.DataFrame, cfg: Config, nz: TeamNormalizer, *, n_games: int
                      ) -> tuple[pd.DataFrame, list[Issue], pd.Timestamp | None]:
    """Validate the preseason win-total table (all teams, once each) and derive expected wins.

    If over/under prices are configured and present, the total is shaded to a MEAN:
        expected_wins = total + wins_sd * probit(P_over_fair)
    (an over priced above 50% means mean wins sit above the posted number).  Any problem refuses
    the whole table (InputError): a partial anchor would distort the league-wide solve.
    Returns (table[team, win_total, price_shift, expected_wins], issues, capture_ts or None).
    """
    wc, rc = cfg.columns_win_totals, cfg.rating
    c = _resolve(raw, {"team": wc.team, "total": wc.total, "over": wc.over_odds,
                       "under": wc.under_odds, "timestamp": wc.timestamp}, {"team", "total"}, "win_totals")
    use_price = rc.use_over_under_price and "over" in c and "under" in c
    problems, rows, stamps, n_priced = [], [], [], 0
    for i, rec in enumerate(raw.to_dict("records")):
        t = nz(rec[c["team"]])
        if t is None:
            problems.append(f"row {i + 2}: unknown team {rec[c['team']]!r}")
            continue
        try:
            total = float(str(rec[c["total"]]).strip())
        except ValueError:
            problems.append(f"row {i + 2} ({t}): win total {rec[c['total']]!r} is not numeric")
            continue
        if not (rc.win_total_min <= total <= rc.win_total_max):
            problems.append(f"row {i + 2} ({t}): win total {total} outside "
                            f"[{rc.win_total_min}, {rc.win_total_max}]")
            continue
        shift = 0.0
        if use_price and str(rec[c["over"]]).strip() and str(rec[c["under"]]).strip():
            try:
                p_over, _ = devig_two_way(rec[c["over"]], rec[c["under"]], cfg.devig)
                shift = rc.wins_sd * probit(p_over)
                n_priced += 1
            except RowError as exc:
                problems.append(f"row {i + 2} ({t}): over/under odds: {exc}")
                continue
        if "timestamp" in c and str(rec[c["timestamp"]]).strip():
            try:
                stamps.append(parse_timestamp(rec[c["timestamp"]], cfg.time.input_timezone,
                                              cfg.time.timestamp_format))
            except RowError as exc:
                problems.append(f"row {i + 2} ({t}): {exc}")
                continue
        rows.append({"team": t, "win_total": total, "price_shift": shift,
                     "expected_wins": total + shift})
    teams = [r["team"] for r in rows]
    dup = sorted({t for t in teams if teams.count(t) > 1})
    if dup:
        problems.append(f"duplicate teams: {dup}")
    missing = sorted(set(CANONICAL_TEAMS) - set(teams))
    if missing:
        problems.append(f"missing teams: {missing}")
    if problems:
        raise InputError("Win-total table refused:\n  " + "\n  ".join(problems[:20]))
    df = pd.DataFrame(rows)
    issues = []
    if abs(df["expected_wins"].sum() - n_games) > 12:
        issues.append(Issue("warning", "win_totals_sum_far_from_games",
                            f"win totals sum to {df['expected_wins'].sum():.1f} vs {n_games} games "
                            f"(normalize = {rc.normalize!r} will remove the difference)"))
    if use_price:
        issues.append(Issue("info", "win_total_price_adjustment",
                            f"over/under prices applied to {n_priced}/{len(df)} teams"))
    return df, issues, (min(stamps) if stamps else None)
