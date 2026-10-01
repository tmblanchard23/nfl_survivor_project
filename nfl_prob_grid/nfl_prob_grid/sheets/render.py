"""Turn engine state into TabSpecs. Everything is rendered from the committed state files, so a publish is
idempotent and can be retried after a failed sheet write without re-running the engine."""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import Config
from ..ratings import ratings_tables
from ..teams import CANONICAL_TEAMS, TEAM_TABLE
from .backends import TabSpec

FULL_NAME = {a: f"{c} {n}" for a, c, n in TEAM_TABLE}
GRID_ORDER = sorted(CANONICAL_TEAMS, key=FULL_NAME.get)          # rows alphabetical by full team name
ET = ZoneInfo("America/New_York")
GREY, FRESH, ADJ, HEAD = "#D9D9D9", "#DDEBF7", "#FCE4D6", "#1F4E79"
SOURCE_LABEL = {"VEGAS_FRESH": "Fresh line", "VEGAS_OLDER": "Older line", "VEGAS_PRESEASON": "Preseason line",
                "VEGAS_STALE_ADJUSTED": "Stale line, adjusted"}


def et_str(ts) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).tz_convert(ET).strftime("%Y-%m-%d %H:%M")


def archive_name(cfg: Config, version: int, current_week: int) -> str:
    wk = "END" if current_week > cfg.season.n_weeks else f"W{current_week:02d}"
    return f"{cfg.sheets.archive_prefix} {cfg.season.season} {wk} v{version:03d}"


def grid_tab(name: str, master: pd.DataFrame, cfg: Config, meta: dict) -> TabSpec:
    """Team x Week grid at A1 (header row 1, one row per team) - a clean block for downstream reads -
    with build information to the right of the grid."""
    n = cfg.season.n_weeks
    header = ["Team"] + [f"Week {w}" for w in range(1, n + 1)]
    rows = [header] + [[FULL_NAME[t]] + [""] * n for t in GRID_ORDER]
    ridx = {t: i + 1 for i, t in enumerate(GRID_ORDER)}
    fills = {}
    for g in master.itertuples():
        for team, p in ((g.home_team, g.home_win_prob), (g.away_team, g.away_win_prob)):
            r, c = ridx[team], int(g.week)
            rows[r][c] = round(float(p), 6)
            if g.is_frozen:
                fills[(r, c)] = GREY
            elif g.adj_active:
                fills[(r, c)] = ADJ
            elif g.prob_source in ("VEGAS_FRESH", "VEGAS_OLDER"):
                fills[(r, c)] = FRESH
    info_col = n + 2
    info = [["Build information", ""],
            ["Built (ET)", meta["built"]], ["Engine version", f"v{meta['version']:03d}"],
            ["Season", cfg.season.season], ["Current week", meta["current_week_label"]],
            ["Completed (frozen) weeks", meta["frozen_label"]],
            ["Lines applied this build", meta["applied"]], ["Stale games adjusted", meta["adjusted"]],
            ["Rejected / set-aside rows", meta["problems"]], ["", ""],
            ["Legend", ""], ["Grey", "completed week (frozen)"], ["Blue", "priced by a refreshed Vegas line"],
            ["Orange", "stale line adjusted for rating movement"], ["White", "preseason line, not adjusted"],
            ["Blank", "bye week"], ["", ""],
            ["Cell value", "probability the team wins that week's game"]]
    for i, (k, v) in enumerate(info):
        while len(rows) <= i:
            rows.append([""] * (n + 1))
        rows[i] = rows[i] + [""] * (info_col - len(rows[i])) + [k, v]
    for i, color in ((11, GREY), (12, FRESH), (13, ADJ)):
        fills[(i, info_col)] = color
    return TabSpec(name=name, values=rows, freeze_rows=1, freeze_cols=1, bold_rows={0},
                   number_formats={(1, 1, 32, n): "0.0%"}, fills=fills,
                   col_widths={0: 170, info_col: 190, info_col + 1: 290, **{c: 62 for c in range(1, n + 1)}},
                   tab_color=cfg.sheets.tab_color)


def master_tab(master: pd.DataFrame, cfg: Config) -> TabSpec:
    """Every matchup of the season with its current probabilities and where they came from."""
    m = master.sort_values(["week", "game_date", "away_team"]).reset_index(drop=True)
    header = ["Week", "Date", "Away Team", "Home Team", "Away Win %", "Home Win %", "Source",
              "Line Age (weeks)", "Latest Vegas Home %", "Original Vegas Home %", "Adjustment (pts)",
              "Last Line Update (ET)", "Line Refreshes", "Completed", "Game ID"]
    rows = [header]
    fills = {}
    for i, g in enumerate(m.itertuples(), start=1):
        rows.append([int(g.week), g.game_date, FULL_NAME[g.away_team], FULL_NAME[g.home_team],
                     float(g.away_win_prob), float(g.home_win_prob), SOURCE_LABEL.get(g.prob_source, g.prob_source),
                     "" if pd.isna(g.line_age_weeks) else round(float(g.line_age_weeks), 1),
                     float(g.latest_vegas_home_prob), float(g.original_vegas_home_prob),
                     round(float(g.adj_home_prob_delta) * 100, 2), et_str(g.last_vegas_refresh_ts),
                     int(g.refresh_count), "Yes" if g.is_frozen else "", g.game_id])
        color = GREY if g.is_frozen else ADJ if g.adj_active else FRESH if g.prob_source in ("VEGAS_FRESH", "VEGAS_OLDER") else None
        if color:
            for c in (4, 5, 6):
                fills[(i, c)] = color
    n = len(rows) - 1
    return TabSpec(name=cfg.sheets.master_tab, values=rows, freeze_rows=1, freeze_cols=0, bold_rows={0},
                   number_formats={(1, 4, n, 5): "0.0%", (1, 8, n, 9): "0.0%", (1, 10, n, 10): "+0.0;-0.0;0.0"},
                   fills=fills, col_widths={2: 170, 3: 170, 6: 140, 11: 140, 14: 170},
                   tab_color=cfg.sheets.tab_color)


def ratings_tab(hist, cfg: Config) -> TabSpec:
    t, _ = ratings_tables(hist, cfg)
    lp = cfg.output.logit_per_point
    header = ["Rank", "Team", "Rating (spread pts vs avg)", "Uncertainty (± pts)", "Change Since Preseason (pts)",
              "Preseason Win Total", "Rating (logit)"]
    rows = [header]
    for i, r in enumerate(t.itertuples(), start=1):
        rows.append([i, FULL_NAME[r.Team], round(r.CurrentRating / lp, 2), round(r.RatingSD / lp, 2),
                     round(getattr(r, "ChangeSinceBaseline", np.nan) / lp, 2) if hasattr(r, "ChangeSinceBaseline") else "",
                     getattr(r, "WinTotal", ""), round(r.CurrentRating, 4)])
    return TabSpec(name=cfg.sheets.ratings_tab, values=rows, freeze_rows=1, bold_rows={0},
                   number_formats={(1, 2, 32, 2): "+0.0;-0.0;0.0", (1, 3, 32, 3): "0.0", (1, 4, 32, 4): "+0.0;-0.0;0.0"}, col_widths={1: 170, 2: 170, 4: 190},
                   tab_color=cfg.sheets.tab_color)


def log_tab(state_dir: Path, cfg: Config) -> TabSpec:
    """One row per engine version, rebuilt from state/runs/*.json (so retries never duplicate rows)."""
    header = ["Engine Version", "Run Type", "As Of (ET)", "Current Week", "Grid Tab", "Lines In Tab",
              "Lines Applied", "Ignored (not newer)", "Skipped (completed week)", "Set Aside (not scheduled / no line)",
              "Rejected", "Stale Games Adjusted", "Skill: games scored", "Skill: error unadjusted",
              "Skill: error adjusted", "Notes"]
    rows = [header]
    for p in sorted((state_dir / "runs").glob("run_v*.json")):
        r = json.loads(p.read_text())
        c, d = r.get("counts", {}), r.get("details", {})
        sk = d.get("adjustment_skill") or {}
        bridge = d.get("bridge", {})
        notes = "; ".join(f"{i['code']}: {i['message']}" for i in r.get("issues", []) if i["level"] in ("warning", "error"))[:400]
        rows.append([f"v{r['version']:03d}", r["run_type"], et_str(pd.Timestamp(r["as_of"])),
                     r["current_week"], bridge.get("archive_tab", ""), bridge.get("rows_in_tab", c.get("rows_in", "")),
                     c.get("games", 0) if r["run_type"] == "init" else c.get("lines_refreshed", 0),
                     c.get("ignored_older", 0) + c.get("ignored_same_timestamp", 0),
                     c.get("ignored_frozen_week", 0) + bridge.get("completed_week_rows", 0), bridge.get("set_aside", 0), len(r.get("rejected_rows", [])),
                     c.get("stale_adjusted", 0), sk.get("n", ""), sk.get("mean_abs_err_stale_logit", ""),
                     sk.get("mean_abs_err_adjusted_logit", ""), notes])
    return TabSpec(name=cfg.sheets.log_tab, values=rows, freeze_rows=1, bold_rows={0},
                   col_widths={2: 130, 4: 170, 15: 400}, tab_color=cfg.sheets.tab_color)
