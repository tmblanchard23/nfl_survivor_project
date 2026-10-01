"""Sheets bridge: read input tabs -> run the engine -> publish output tabs.

Commands
  bootstrap  one-time: init from the full-season opening lines + preseason win totals, replay any catch-up
             line exports in time order, then run a normal build from the lines tab. Every version gets an
             archive tab, so the season's history is in the sheet from day one.
  build      weekly: read the lines tab (only lines - weeks are inferred, games matched by ID), refresh the
             engine, publish.
  publish    re-render the sheet from saved state (no engine run). Use it if a build's sheet write failed.
  status     check the input tabs and the state without changing anything.

Safety: input tabs are read-only; output tabs are rendered from committed state (idempotent); a failed engine
run writes nothing anywhere; a failed sheet write leaves the engine state valid and `publish` repairs the sheet.
"""
from __future__ import annotations

import copy
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..engine import RunReport, resolve_as_of, run_init, run_refresh
from ..errors import ConfigError, InputError, StateError
from ..master import StateStore, coerce_master_dtypes
from ..ratings import RatingHistory
from ..tabular import iso
from ..teams import TeamNormalizer
from ..weeks import assign_weeks, infer_current_week
from .backends import open_backend
from .render import archive_name, et_str, grid_tab, log_tab, master_tab, ratings_tab

ENGINE_LINES = ["Week", "Away Team", "Home Team", "Away Odds", "Home Odds", "Timestamp"]


# ------------------------------------------------------------------------------------------ reading
def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() in ("", "nan", "None")


def _norm(s) -> str:
    return " ".join(str(s).strip().lower().split())


def _cell_time(v, tz: str) -> str:
    """Datetime / Sheets serial / text -> 'YYYY-MM-DD HH:MM:SS' (naive, in the sheet's timezone) or ''."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
        return ""
    if isinstance(v, (datetime, pd.Timestamp)):
        return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, date):
        return pd.Timestamp(v).strftime("%Y-%m-%d 00:00:00")
    if isinstance(v, (int, float, np.integer, np.floating)) and 20000 < float(v) < 100000:
        return (pd.Timestamp("1899-12-30") + pd.to_timedelta(float(v), unit="D")).round("s").strftime("%Y-%m-%d %H:%M:%S")
    return str(v).strip()


def read_table(backend, tab: str, wanted: dict[str, str], required: set[str]) -> pd.DataFrame:
    """Find the header row (first row containing every required header), return {logical: column} rows below it."""
    if tab not in backend.tabs():
        raise InputError(f"Tab {tab!r} not found in the spreadsheet (tabs: {backend.tabs()})")
    grid = backend.read(tab)
    need = {_norm(wanted[k]) for k in required}
    for h, row in enumerate(grid[:25]):
        cells = [_norm(c) if c is not None else "" for c in row]
        if need <= set(cells):
            cols = {k: cells.index(_norm(v)) for k, v in wanted.items() if v and _norm(v) in cells}
            body = []
            for r in grid[h + 1:]:
                rec = {k: (r[c] if c < len(r) else None) for k, c in cols.items()}
                if all(_is_blank(rec.get(k)) for k in required):
                    continue
                body.append(rec)
            return pd.DataFrame(body, columns=list(cols))
    raise InputError(f"Tab {tab!r}: no header row with columns {sorted(wanted[k] for k in required)} in the first 25 rows")


def _lines_wanted(cfg: Config) -> dict:
    s = cfg.sheets
    return {"game_time": s.col_game_time, "away": s.col_away, "home": s.col_home, "away_odds": s.col_away_odds,
            "home_odds": s.col_home_odds, "updated": s.col_updated, "week": s.col_week}


def read_lines(backend, tab: str, cfg: Config) -> pd.DataFrame:
    df = read_table(backend, tab, _lines_wanted(cfg), {"away", "home", "away_odds", "home_odds", "updated"})
    tz = cfg.time.input_timezone
    for c in ("game_time", "updated"):
        if c in df:
            df[c] = [_cell_time(v, tz) for v in df[c]]
    if "game_time" not in df:
        df["game_time"] = ""
    return df


def _engine_cfg(cfg: Config) -> Config:
    """Copy of cfg whose column mappings point at the bridge's internal table layout."""
    c = copy.deepcopy(cfg)
    for cols in (c.columns_refresh, c.columns_initial):
        cols.game_id, cols.week, cols.home_team, cols.away_team = "", "Week", "Home Team", "Away Team"
        cols.home_odds, cols.away_odds, cols.timestamp = "Home Odds", "Away Odds", "Timestamp"
    c.columns_refresh.date = ""
    c.columns_initial.date = "Date"
    w = c.columns_win_totals
    w.team, w.total, w.timestamp = "team", "win_total", ""
    w.over_odds = "over" if cfg.sheets.col_wt_over else ""
    w.under_odds = "under" if cfg.sheets.col_wt_under else ""
    return c


def _to_engine(df: pd.DataFrame, with_date: bool = False) -> pd.DataFrame:
    out = pd.DataFrame({"Week": df["week"], "Away Team": df["away"], "Home Team": df["home"],
                        "Away Odds": df["away_odds"], "Home Odds": df["home_odds"], "Timestamp": df["updated"]})
    if with_date:
        out["Date"] = df["game_time"]
    return out.astype(str).replace({"None": "", "nan": ""})


def _latest_ts(df: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    ts = pd.to_datetime(df["updated"].replace("", np.nan).dropna(), errors="coerce").dropna()
    if ts.empty:
        raise InputError("No parseable 'Line Last Updated' timestamps in the lines table")
    return ts.max().tz_localize(cfg.time.input_timezone).tz_convert("UTC") + pd.Timedelta(minutes=1)


def prepare_refresh(backend, tab: str, cfg: Config, master: pd.DataFrame,
                    current_week: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Lines tab -> engine table. Set aside (reported, never guessed): rows with no odds, rows that are not a
    scheduled regular-season game, and lines for weeks already completed (last week's games still in the pull)."""
    raw = read_lines(backend, tab, cfg)
    blank = raw["away_odds"].map(_is_blank) & raw["home_odds"].map(_is_blank)
    aside = [{"row": int(i), "reason": "no_line_yet", "detail": f"{r.away} @ {r.home}: no odds posted"}
             for i, r in raw[blank].iterrows()]
    lines, aside2 = assign_weeks(raw[~blank].reset_index(drop=True), master, TeamNormalizer(cfg.team_aliases),
                                 cfg.time.input_timezone, cfg.sheets.week_match_max_days)
    done = 0
    if current_week is not None and len(lines):
        wk = pd.to_numeric(lines["week"], errors="coerce")
        past = wk < current_week
        done = int(past.sum())
        lines = lines[~past.fillna(False).to_numpy(bool)].reset_index(drop=True)
    info = {"rows_in_tab": int(len(raw)), "set_aside": len(aside) + len(aside2),
            "set_aside_rows": (aside + aside2)[:50], "completed_week_rows": done, "lines_tab": tab}
    return _to_engine(lines), info


# ------------------------------------------------------------------------------------------ commands
def _check_tab_roles(cfg: Config) -> None:
    s = cfg.sheets
    inputs = {s.lines_tab, s.initial_lines_tab, s.win_totals_tab}
    outputs = {s.grid_tab, s.master_tab, s.ratings_tab, s.log_tab}
    clash = inputs & outputs
    if clash:
        raise ConfigError(f"[sheets] output tab name(s) {sorted(clash)} are also input tabs - refusing to overwrite inputs")
    if any(t.startswith(s.archive_prefix + " ") for t in inputs):
        raise ConfigError("[sheets] archive_prefix would collide with an input tab name")


def _backend(cfg: Config, backend=None):
    _check_tab_roles(cfg)
    return backend if backend is not None else open_backend(cfg.sheets.spreadsheet, cfg.sheets.credentials)


def bootstrap(cfg: Config, *, catch_up: list[str] | None = None, as_of=None, force: bool = False,
              backend=None, run_build: bool = True) -> list[RunReport]:
    be = _backend(cfg, backend)
    store = StateStore(cfg.paths.state_dir)
    if store.exists() and not force:
        raise StateError(f"State already exists in {store.dir}. `sheets bootstrap --force` archives it and starts over.")
    ecfg, s = _engine_cfg(cfg), cfg.sheets
    # 1. init: full-season opening lines + preseason win totals
    init = read_lines(be, s.initial_lines_tab, cfg)
    if "week" not in init or init["week"].isin(["", None]).any():
        raise InputError(f"{s.initial_lines_tab!r} needs a {s.col_week!r} column for every game (it defines the schedule)")
    wt = read_table(be, s.win_totals_tab, {"team": s.col_wt_team, "win_total": s.col_wt_total,
                                           "over": s.col_wt_over, "under": s.col_wt_under}, {"team", "win_total"})
    wt = wt.astype(str).replace({"None": "", "nan": ""})
    init_as_of = _latest_ts(init, cfg)
    sched = pd.DataFrame({"week": init["week"].astype(float).astype(int),
                          "game_date": pd.to_datetime(init["game_time"]).dt.strftime("%Y-%m-%d")})
    cw0 = infer_current_week(sched, init_as_of, cfg.time.input_timezone, cfg.season.n_weeks)
    reps = [run_init(ecfg, as_of=init_as_of, current_week=cw0, force=force, lines_df=_to_engine(init, with_date=True),
                     win_totals_df=wt, details={"bridge": {"archive_tab": archive_name(cfg, 1, cw0),
                                                           "rows_in_tab": int(len(init)), "set_aside": 0,
                                                           "lines_tab": s.initial_lines_tab}})]
    # 2. catch-up exports in time order, each at its own timestamp
    tabs = sorted(catch_up or [], key=lambda t: _latest_ts(read_lines(be, t, cfg), cfg))
    for tab in tabs:
        master, _, man = store.load(cfg.validation.verify_integrity)
        t_as_of = _latest_ts(read_lines(be, tab, cfg), cfg)
        reps.append(_refresh(cfg, ecfg, be, tab, master, man, t_as_of, None, False))
    # 3. a normal build from the live lines tab
    if run_build:
        master, _, man = store.load(cfg.validation.verify_integrity)
        reps.append(_refresh(cfg, ecfg, be, s.lines_tab, master, man, resolve_as_of(cfg, as_of), None, False))
    publish(cfg, backend=be, versions="all")
    return reps


def _refresh(cfg, ecfg, be, tab, master, man, as_of, current_week, dry_run) -> RunReport:
    cw = current_week or max(infer_current_week(master, as_of, cfg.time.input_timezone, cfg.season.n_weeks),
                             int(man.get("current_week", 1)))
    lines, info = prepare_refresh(be, tab, cfg, master, cw)
    version = man["current_version"] + 1
    info["archive_tab"] = archive_name(cfg, version, cw) if cfg.sheets.archive_grids else ""
    return run_refresh(ecfg, as_of=as_of, current_week=cw, lines_df=lines, dry_run=dry_run, details={"bridge": info})


def build(cfg: Config, *, as_of=None, current_week: int | None = None, dry_run: bool = False,
          backend=None) -> RunReport:
    be = _backend(cfg, backend)
    store = StateStore(cfg.paths.state_dir)
    master, _, man = store.load(cfg.validation.verify_integrity)
    rep = _refresh(cfg, _engine_cfg(cfg), be, cfg.sheets.lines_tab, master, man, resolve_as_of(cfg, as_of),
                   current_week, dry_run)
    if not dry_run:
        publish(cfg, backend=be, versions="latest")
    return rep


def _version_state(store: StateStore, man: dict, version: int):
    entry = next(v for v in man["versions"] if v["version"] == version)
    m = coerce_master_dtypes(pd.read_csv(store.versions_dir / entry["file"], dtype={"source_game_id": str, "game_date": str},
                                         float_precision="round_trip"))
    run = json.loads((store.runs_dir / f"run_v{version:04d}.json").read_text())
    return m, run


def _meta(run: dict, master: pd.DataFrame, cfg: Config) -> dict:
    cw = int(run["current_week"])
    c = run.get("counts", {})
    bridge = run.get("details", {}).get("bridge", {})
    return {"built": et_str(pd.Timestamp(run["as_of"])), "version": int(run["version"]),
            "current_week_label": "season complete" if cw > cfg.season.n_weeks else f"Week {cw}",
            "frozen_label": "none" if cw <= 1 else ("1" if cw == 2 else f"1-{min(cw - 1, cfg.season.n_weeks)}"),
            "applied": c.get("lines_refreshed", c.get("games", 0) if run["run_type"] == "init" else 0),
            "adjusted": int(master["adj_active"].sum()),
            "problems": len(run.get("rejected_rows", [])) + int(bridge.get("set_aside", 0))}


def publish(cfg: Config, *, backend=None, versions: str = "latest") -> list[str]:
    """Render the sheet from committed state. versions: 'latest' (archive tab for the newest version) or
    'all' (archive tab for every version - used by bootstrap)."""
    be = _backend(cfg, backend)
    store = StateStore(cfg.paths.state_dir)
    master, hist, man = store.load(cfg.validation.verify_integrity)
    written = []
    latest_v = man["current_version"]
    _, run = _version_state(store, man, latest_v)
    spec = grid_tab(cfg.sheets.grid_tab, master, cfg, _meta(run, master, cfg))
    be.write(spec); written.append(spec.name)
    if cfg.sheets.archive_grids:
        todo = [v["version"] for v in man["versions"]] if versions == "all" else [latest_v]
        for v in todo:
            m_v, run_v = _version_state(store, man, v)
            name = run_v.get("details", {}).get("bridge", {}).get("archive_tab") or archive_name(cfg, v, int(run_v["current_week"]))
            be.write(grid_tab(name, m_v, cfg, _meta(run_v, m_v, cfg))); written.append(name)
    be.write(master_tab(master, cfg)); written.append(cfg.sheets.master_tab)
    if cfg.sheets.write_ratings:
        be.write(ratings_tab(hist, cfg)); written.append(cfg.sheets.ratings_tab)
    be.write(log_tab(store.dir, cfg)); written.append(cfg.sheets.log_tab)
    be.flush()
    return written


def sheet_status(cfg: Config, *, backend=None) -> dict:
    be = _backend(cfg, backend)
    s, out = cfg.sheets, {}
    tabs = be.tabs()
    out["tabs_found"] = {t: (t in tabs) for t in (s.lines_tab, s.initial_lines_tab, s.win_totals_tab)}
    try:
        lines = read_lines(be, s.lines_tab, cfg)
        out["lines_tab_rows"] = int(len(lines))
        out["lines_latest_update_et"] = str(max(v for v in lines["updated"] if v)) if len(lines) else None
    except InputError as exc:
        out["lines_tab_error"] = str(exc)
    store = StateStore(cfg.paths.state_dir)
    if store.exists():
        master, _, man = store.load(cfg.validation.verify_integrity)
        out["engine_version"] = man["current_version"]
        out["engine_current_week"] = man["current_week"]
        out["current_week_by_calendar_now"] = infer_current_week(master, pd.Timestamp.now(tz="UTC"),
                                                                 cfg.time.input_timezone, cfg.season.n_weeks)
        if s.lines_tab in tabs:
            _, info = prepare_refresh(be, s.lines_tab, cfg, master)
            out["lines_set_aside_if_built_now"] = info["set_aside_rows"]
    else:
        out["engine_version"] = None
        out["hint"] = "no engine state yet: run `python -m nfl_prob_grid sheets bootstrap`"
    out["output_tabs_present"] = [t for t in tabs if t in (s.grid_tab, s.master_tab, s.ratings_tab, s.log_tab)
                                  or t.startswith(s.archive_prefix + " ")]
    return out
