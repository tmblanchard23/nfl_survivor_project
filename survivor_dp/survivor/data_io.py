"""
Data I/O.

Two ways in, both funnel through the same row-parsing logic so behavior is
identical whether you're reading a live Google Sheet or a CSV export of it:

  1. Google Sheets (live) -- load_pool_from_google_sheet()
  2. CSV export           -- load_pool_from_csv_dir()

=====================================================================
SCHEMA -- matches your probability-grid engine's actual output tabs
=====================================================================

**Master Game Table** tab (one row per game -- this is the sole source of
game probabilities; there is no raw-odds parsing or de-vig math here
anymore, since your grid engine already did that, plus rating updates,
anchoring, and stale-line adjustment, upstream):
    Week | Date | Away Team | Home Team | Away Win % | Home Win % | Source |
    Line Age (weeks) | Latest Vegas Home % | Original Vegas Home % |
    Adjustment (pts) | Last Line Update (ET) | Line Refreshes | Completed |
    Game ID

    - `Week`, `Away Team`, `Home Team`, `Away Win %`, and `Home Win %` are
      read for probabilities; `Completed` (if present) auto-detects the
      current week, and `Game ID` (if present) supplies team abbreviations
      so Pools Picks can say "KC" instead of "Kansas City Chiefs". `Away Win %`/`Home Win %` are used AS-IS via
      `Game.from_probability()` -- no conversion, no staleness adjustment,
      no second-guessing. Whatever the grid engine has already decided a
      team's win probability is, that's what the DP uses.
    - The rest of the columns (`Source`, `Line Age`, the two "Vegas" columns,
      `Adjustment`, `Last Line Update`, `Line Refreshes`, `Completed`,
      `Game ID`) are metadata your grid engine tracks for its own purposes
      and aren't consumed here. If you want any of them surfaced in this
      system's output (e.g. showing `Source` in the confidence report),
      that's a small addition -- just ask, since it wasn't clear it was
      wanted given the push to trim scope.
    - A row is skipped (not an error) only if it's missing a team, a week
      number, or has no parseable probability at all in either column --
      genuinely malformed data, not "line not posted yet" (that phase of
      the pipeline is the grid engine's problem now, not this loader's).

**Pools Picks** tab (wide format -- one row per player, one column per week):
    Player | Week 1 | Week 2 | ... | Week 15
    Player 1 | Chiefs | Ravens | ...
    ...
    Me | Bills | Eagles | ...

    - The row labeled "Me" (configurable) is your own pick history.
    - A cell value of OUT / ELIMINATED / DEAD / X (case-insensitive) marks
      that player eliminated as of that week. A blank cell just means no
      pick is logged yet (e.g. a bye, or the week hasn't happened) -- it is
      NOT treated as an elimination signal on its own, to avoid false
      positives. If your pool doesn't mark eliminations this way, use
      SurvivorPool.eliminate_opponent() manually after loading.
    - Only weeks STRICTLY BEFORE `current_week` are read into history/used-
      teams. Anything in or after the current week's column is ignored on
      load, even if already filled in -- this enforces the no-look-ahead
      rule (spec sections 22/23) structurally rather than by convention.

Every run rebuilds the full state (your used teams, every opponent's
history and fitted model, eliminations) from these two tabs, so nothing
needs to be saved between runs. save_state()/load_state() (pickle) remain
only for people driving the in-memory Python API by hand.
"""
from __future__ import annotations
import csv
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

from .vegas import Game, WeekSlate
from .workflow import SurvivorPool

ELIMINATION_MARKERS = {"OUT", "ELIMINATED", "DEAD", "X"}
WEEK_COLUMN_RE = re.compile(r"week\s*#?\s*(\d+)", re.IGNORECASE)

MASTER_GAME_TABLE_COLUMNS = {
    "week": "Week",
    "away_team": "Away Team",
    "home_team": "Home Team",
    "away_win_pct": "Away Win %",
    "home_win_pct": "Home Win %",
}


# ---------------------------------------------------------------------
# Row-level adapters: turn a source (CSV file / gspread worksheet) into
# list[dict] records keyed by column header. Everything downstream is
# source-agnostic.
# ---------------------------------------------------------------------

def _rows_from_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _rows_from_gspread_worksheet(worksheet) -> list[dict]:
    """`worksheet` is a gspread Worksheet. Uses the header row as keys,
    exactly like csv.DictReader, so parse_* functions don't care which
    source they came from. Reads raw values rather than using
    get_all_records(), which raises if the sheet has any blank or duplicate
    header cells (common when a tab has unused columns off to the right)."""
    return _rows_from_value_grid(worksheet.get_all_values())


def _rows_from_value_grid(values: list[list]) -> list[dict]:
    if not values:
        return []
    header = [str(h).strip() for h in values[0]]
    keep = [i for i, h in enumerate(header) if h]
    rows = []
    for raw in values[1:]:
        if not any(str(v).strip() for v in raw):
            continue  # fully blank row
        rows.append({header[i]: (raw[i] if i < len(raw) else "") for i in keep})
    return rows


# ---------------------------------------------------------------------
# Core parsers (source-agnostic)
# ---------------------------------------------------------------------

@dataclass
class SkippedGameRow:
    week: int | None
    away_team: str
    home_team: str
    reason: str


def _parse_percentage(value) -> float | None:
    """Accepts "39.7%", "39.7", or 0.397 -- whatever shape the sheet/CSV
    export happens to hand back for a percentage-formatted cell -- and
    returns a fraction in [0, 1]. Returns None if unparseable."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    had_percent_sign = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if had_percent_sign or x > 1.0:
        x = x / 100.0
    return x


def parse_master_game_table_rows(rows: list[dict], columns: dict[str, str] = None
                                  ) -> tuple[dict[int, WeekSlate], list[SkippedGameRow]]:
    """
    Parse "Master Game Table"-shaped rows into {week: WeekSlate}, using the
    ALREADY-COMPUTED Away Win %/Home Win % columns directly -- no de-vig, no
    staleness adjustment, no odds conversion of any kind. That's all been
    done upstream by the probability-grid engine.
    """
    cols = columns or MASTER_GAME_TABLE_COLUMNS
    slates: dict[int, WeekSlate] = {}
    skipped: list[SkippedGameRow] = []

    for row in rows:
        away = str(row.get(cols["away_team"], "")).strip()
        home = str(row.get(cols["home_team"], "")).strip()
        wk_raw = row.get(cols["week"])
        wk = None
        if wk_raw not in (None, ""):
            try:
                wk = int(float(wk_raw))
            except (ValueError, TypeError):
                pass

        if not away or not home or wk is None:
            skipped.append(SkippedGameRow(wk, away, home, "missing team(s) or week number"))
            continue

        prob_away = _parse_percentage(row.get(cols["away_win_pct"]))
        prob_home = _parse_percentage(row.get(cols["home_win_pct"]))
        if prob_away is None and prob_home is None:
            skipped.append(SkippedGameRow(wk, away, home, "no probability found in this row"))
            continue
        if prob_away is None:
            prob_away = 1.0 - prob_home
        if prob_home is None:
            prob_home = 1.0 - prob_away

        g = Game.from_probability(wk, away, home, prob_away, prob_home)
        slates.setdefault(wk, WeekSlate(week=wk)).games.append(g)

    return slates, skipped


class TeamNameResolver:
    """
    Maps whatever someone typed into Pools Picks onto the exact team names
    the Master Game Table uses. Accepts, in order: the exact name, a
    case-insensitive match, an abbreviation parsed from the Game ID column
    (e.g. "KC" from "2026-W01-DEN@KC"), or a unique nickname suffix
    ("Chiefs", "49ers"). Anything ambiguous ("New York") or unknown returns
    None so it can be reported rather than silently guessed.
    """

    def __init__(self, game_rows: list[dict]):
        self.teams: set[str] = set()
        self.abbrevs: dict[str, str] = {}
        for row in game_rows:
            away = str(row.get("Away Team", "")).strip()
            home = str(row.get("Home Team", "")).strip()
            if away:
                self.teams.add(away)
            if home:
                self.teams.add(home)
            gid = str(row.get("Game ID", "")).strip()
            if gid and "@" in gid and away and home:
                matchup = gid.split("-")[-1]
                a_abbr, h_abbr = matchup.split("@", 1)
                self.abbrevs.setdefault(a_abbr.strip().upper(), away)
                self.abbrevs.setdefault(h_abbr.strip().upper(), home)
        self._folded = {t.casefold(): t for t in self.teams}

    def resolve(self, raw: str) -> str | None:
        s = str(raw).strip()
        if not s:
            return None
        if s in self.teams:
            return s
        if s.casefold() in self._folded:
            return self._folded[s.casefold()]
        if s.upper() in self.abbrevs:
            return self.abbrevs[s.upper()]
        suffix_matches = [t for t in self.teams
                          if t.casefold().endswith(" " + s.casefold())]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None


def _is_completed(value) -> bool:
    return str(value).strip().casefold() in {"yes", "y", "true", "1", "done", "final"}


def detect_current_week(game_rows: list[dict]) -> int | None:
    """
    The current decision week = the earliest week that still has at least
    one game not marked Completed in the Master Game Table (the grid engine
    maintains that column). Returns None if the table has no Completed
    column at all, in which case the week has to be given explicitly.
    If every game is completed, returns last week + 1 (season over).
    """
    if not game_rows or "Completed" not in game_rows[0]:
        return None
    by_week: dict[int, list[bool]] = {}
    for row in game_rows:
        try:
            wk = int(float(row.get("Week", "")))
        except (ValueError, TypeError):
            continue
        by_week.setdefault(wk, []).append(_is_completed(row.get("Completed", "")))
    if not by_week:
        return None
    for wk in sorted(by_week):
        if not all(by_week[wk]):
            return wk
    return max(by_week) + 1


@dataclass
class LoadReport:
    """Everything the loader noticed that a human should know about.
    `blocking` problems mean a recommendation could be wrong (e.g. it might
    suggest a team you've already used); `warnings` are worth a look but
    don't invalidate the output."""
    current_week: int
    week_source: str  # "auto-detected from Completed column" or "given explicitly"
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    my_picks: list[tuple[int, str]] = field(default_factory=list)
    opponents_alive: list[str] = field(default_factory=list)
    opponents_eliminated: list[str] = field(default_factory=list)
    skipped_game_rows: int = 0


def _week_number_from_column(col_name: str) -> int | None:
    m = WEEK_COLUMN_RE.search(col_name)
    return int(m.group(1)) if m else None


def parse_pool_picks_rows(rows: list[dict], current_week: int,
                           player_column: str = "Player",
                           my_player_label: str = "Me"
                           ) -> tuple[dict[str, list[tuple[int, str]]],
                                      list[tuple[int, str]],
                                      dict[str, int]]:
    """
    Parse "Pools Picks"-shaped wide rows (one row per player, one column per
    week) into:
      - opponent_history: {opponent_id: [(week, team), ...]}, excluding
        `my_player_label` and excluding any week >= current_week
      - my_history: [(week, team), ...] for the `my_player_label` row,
        excluding any week >= current_week
      - eliminated_from_week: {opponent_id: week} for players whose cell
        contained an elimination marker (OUT/ELIMINATED/DEAD/X), recording
        the earliest such week

    Weeks >= current_week are dropped even if populated, so a sheet that
    already has this week's picks filled in can't leak look-ahead info.
    """
    if not rows:
        return {}, [], {}

    week_cols = {col: _week_number_from_column(col) for col in rows[0].keys()
                 if _week_number_from_column(col) is not None}

    opponent_history: dict[str, list[tuple[int, str]]] = {}
    my_history: list[tuple[int, str]] = []
    eliminated_from_week: dict[str, int] = {}

    for row in rows:
        player = str(row.get(player_column, "")).strip()
        if not player:
            continue
        is_me = player == my_player_label

        for col, wk in week_cols.items():
            if wk >= current_week:
                continue
            raw = str(row.get(col, "")).strip()
            if raw == "":
                continue
            if raw.upper() in ELIMINATION_MARKERS:
                if not is_me:
                    eliminated_from_week[player] = min(
                        wk, eliminated_from_week.get(player, wk))
                continue
            if is_me:
                my_history.append((wk, raw))
            else:
                opponent_history.setdefault(player, []).append((wk, raw))

    for oid in opponent_history:
        opponent_history[oid].sort(key=lambda x: x[0])
    my_history.sort(key=lambda x: x[0])

    return opponent_history, my_history, eliminated_from_week


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Building a SurvivorPool from parsed rows (shared by both sources)
# ---------------------------------------------------------------------

def build_pool_from_rows(game_line_rows: list[dict], picks_rows: list[dict],
                          current_week: int | None = None,
                          my_player_label: str = "Me",
                          verbose: bool = False) -> SurvivorPool:
    """
    Rebuild the entire pool state from the two tabs. There is no need to
    "replay" or advance anything week by week: Pools Picks IS the record of
    what happened, so every run reconstructs your used teams, every
    opponent's history/model, and eliminations from it directly.

    `current_week=None` auto-detects it from the Master Game Table's
    Completed column. The returned pool carries a `load_report`
    (LoadReport) describing anything that needs your attention.
    """
    if current_week is None:
        current_week = detect_current_week(game_line_rows)
        if current_week is None:
            raise ValueError(
                "Couldn't auto-detect the current week: the Master Game Table "
                "has no 'Completed' column. Pass the week explicitly "
                "(e.g. --week 4 on the command line)."
            )
        week_source = "auto-detected from the Completed column"
    else:
        week_source = "given explicitly"

    report = LoadReport(current_week=current_week, week_source=week_source)
    pool = SurvivorPool()
    pool.current_week = current_week

    slates, skipped = parse_master_game_table_rows(game_line_rows)
    for wk, slate in slates.items():
        pool.set_week_slate(slate)
    report.skipped_game_rows = len(skipped)
    if skipped:
        report.warnings.append(
            f"{len(skipped)} Master Game Table row(s) skipped as malformed "
            f"(missing team, week, or probability).")
    if current_week not in slates:
        report.blocking.append(
            f"The Master Game Table has no games for week {current_week}.")

    resolver = TeamNameResolver(game_line_rows)

    def _resolve_pick(player: str, wk: int, raw: str) -> str | None:
        team = resolver.resolve(raw)
        if team is None:
            return None
        if wk in slates and team not in slates[wk].win_probs():
            return None
        return team

    opponent_history, my_history, eliminated_from_week = parse_pool_picks_rows(
        picks_rows, current_week, my_player_label=my_player_label)

    # ---- my own history: any problem here can cause a reused-team pick ----
    my_resolved: list[tuple[int, str]] = []
    for wk, raw in my_history:
        team = _resolve_pick(my_player_label, wk, raw)
        if team is None:
            report.blocking.append(
                f"Your week {wk} pick '{raw}' doesn't match any team playing "
                f"that week in the Master Game Table. Fix the spelling in "
                f"Pools Picks (full names like 'Kansas City Chiefs', nicknames "
                f"like 'Chiefs', or abbreviations like 'KC' all work).")
        else:
            my_resolved.append((wk, team))
    logged_weeks = {wk for wk, _ in my_history}
    my_eliminated = _player_marked_out(picks_rows, my_player_label, current_week)
    first_week = min(slates) if slates else 1
    if not my_eliminated:
        missing = [wk for wk in range(first_week, current_week) if wk not in logged_weeks]
        if missing:
            report.blocking.append(
                f"No pick logged for you ('{my_player_label}' row) in week(s) "
                f"{', '.join(map(str, missing))}. Without it, the optimizer "
                f"could recommend a team you've already used.")
    else:
        report.warnings.append(
            f"The '{my_player_label}' row is marked OUT -- you're recorded as "
            f"eliminated. Recommendations are still produced, but won't matter.")
    teams_used = [t for _, t in my_resolved]
    dupes = sorted({t for t in teams_used if teams_used.count(t) > 1})
    if dupes:
        report.warnings.append(
            f"Your pick history uses the same team more than once: {', '.join(dupes)}.")
    pool.set_my_used_teams(teams_used)
    report.my_picks = my_resolved

    # ---- opponents ----
    for oid, picks in opponent_history.items():
        opp = pool.add_opponent(oid)
        for wk, raw in picks:
            team = _resolve_pick(oid, wk, raw)
            if team is None:
                report.warnings.append(
                    f"{oid}'s week {wk} pick '{raw}' doesn't match a team playing "
                    f"that week; ignored for opponent modeling.")
                continue
            slate = slates[wk]
            win_probs = slate.win_probs()
            avail = [t for t in slate.teams() if not pool.registry.is_used(opp.used_mask, t)]
            if team not in avail:
                avail = list(set(avail) | {team})
            opp.observe_pick(wk, team, avail, win_probs, refit=False)
            opp.used_mask |= pool.registry.bit(team)
        opp.fit()  # once, after all of this opponent's history is loaded

    for oid in eliminated_from_week:
        pool.eliminate_opponent(oid)

    # Players who are listed but have never logged a pick still count as alive.
    for row in picks_rows:
        name = str(row.get("Player", "")).strip()
        if name and name != my_player_label:
            pool.add_opponent(name)

    for oid, opp in pool.opponents.items():
        if opp.eliminated:
            continue
        logged = {wk for wk, _ in opponent_history.get(oid, [])}
        gaps = [wk for wk in range(first_week, current_week) if wk not in logged]
        if gaps:
            report.warnings.append(
                f"{oid} has no pick logged for week(s) {', '.join(map(str, gaps))}. "
                f"If they were eliminated, put OUT in that week's cell so they "
                f"stop counting as a live opponent.")

    report.opponents_alive = sorted(pool.alive_opponents())
    report.opponents_eliminated = sorted(o for o, m in pool.opponents.items() if m.eliminated)
    pool.load_report = report

    if verbose:
        for msg in report.blocking:
            print(f"[data_io] PROBLEM: {msg}")
        for msg in report.warnings:
            print(f"[data_io] note: {msg}")
    return pool


def _player_marked_out(picks_rows: list[dict], player: str, current_week: int) -> bool:
    for row in picks_rows:
        if str(row.get("Player", "")).strip() != player:
            continue
        for col, val in row.items():
            wk = _week_number_from_column(col)
            if wk is not None and wk < current_week and \
                    str(val).strip().upper() in ELIMINATION_MARKERS:
                return True
    return False


# ---------------------------------------------------------------------
# CSV entry point
# ---------------------------------------------------------------------

def load_pool_from_csv_dir(csv_dir: str | Path, current_week: int | None = None,
                            game_lines_filename: str = "master_game_table.csv",
                            picks_filename: str = "pool_picks.csv",
                            my_player_label: str = "Me",
                            verbose: bool = False) -> SurvivorPool:
    """
    Build a SurvivorPool from CSV exports of the two tabs. Export each tab
    as-is (File > Download > CSV) -- the column headers should match your
    sheet exactly (see module docstring). `current_week=None` auto-detects.
    """
    csv_dir = Path(csv_dir)
    game_line_rows = _rows_from_csv(csv_dir / game_lines_filename)
    picks_rows = _rows_from_csv(csv_dir / picks_filename)
    return build_pool_from_rows(game_line_rows, picks_rows, current_week,
                                 my_player_label, verbose=verbose)


# ---------------------------------------------------------------------
# Live Google Sheets entry point
# ---------------------------------------------------------------------

def spreadsheet_id_from(url_or_id: str) -> str:
    """Accepts either the full sheet URL or just the id."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()


def read_google_sheet_tabs(spreadsheet: str, credentials_path: str | Path,
                            tabs: list[str]) -> dict[str, list[dict]]:
    """Read several tabs from one spreadsheet (read-only) as row dicts."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as e:
        raise ImportError(
            "Reading Google Sheets requires gspread and google-auth: "
            "pip3 install gspread google-auth"
        ) from e

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly",
              "https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_file(str(credentials_path), scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(spreadsheet_id_from(spreadsheet))
    return {tab: _rows_from_gspread_worksheet(sheet.worksheet(tab)) for tab in tabs}


def load_pool_from_google_sheet(spreadsheet_id: str, current_week: int | None = None,
                                 credentials_path: str = "service_account.json",
                                 game_lines_tab: str = "Master Game Table",
                                 picks_tab: str = "Pools Picks",
                                 my_player_label: str = "Me",
                                 verbose: bool = False) -> SurvivorPool:
    """
    Build a SurvivorPool by reading directly from a live Google Sheet.
    `spreadsheet_id` may be the full URL or just the id.
    `current_week=None` auto-detects from the Completed column.
    Most people should use the command line instead (see README):
        python3 -m survivor recommend
    """
    data = read_google_sheet_tabs(spreadsheet_id, credentials_path,
                                  [game_lines_tab, picks_tab])
    return build_pool_from_rows(data[game_lines_tab], data[picks_tab], current_week,
                                 my_player_label, verbose=verbose)



# State persistence between weekly runs
# ---------------------------------------------------------------------

def save_state(pool: SurvivorPool, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(pool, f)


def load_state(path: str | Path) -> SurvivorPool:
    with open(path, "rb") as f:
        return pickle.load(f)
