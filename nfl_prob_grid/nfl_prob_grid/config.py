"""Central configuration.

Every modelling assumption lives in ``config.toml`` and is materialised here as typed
dataclasses.  Unknown keys raise an error (typo protection).  Nothing else in the package
hard-codes an assumption that appears in this file.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError


@dataclass
class SeasonCfg:
    season: int = 2026
    n_weeks: int = 18
    n_teams: int = 32
    games_per_team: int = 17          # expected games per team (0 disables the check)
    current_week: int = 1             # first week NOT yet completed; weeks < this are frozen


@dataclass
class TimeCfg:
    input_timezone: str = "UTC"       # zone assumed for naive timestamps in the inputs
    timestamp_format: str = ""        # strptime-style format; "" = auto-detect
    age_unit_days: float = 7.0        # line age is measured in units of this many days
    max_future_ts_hours: float = 24.0 # incoming timestamps further ahead of as_of are rejected


@dataclass
class PathsCfg:
    state_dir: str = "state"
    output_dir: str = "output"
    initial_lines: str = "data/initial_lines.csv"     # Input A
    refresh_lines: str = "data/refresh_lines.csv"     # Input B (local file OR Google Sheets URL)
    win_totals: str = "data/win_totals.csv"           # Input C: PRESEASON win totals (init only)
    sheet_name: str = ""                              # only for .xlsx inputs


@dataclass
class LinesColumns:
    """Column names of a table with ONE ROW PER GAME (both teams' odds on the same row)."""
    game_id: str = ""                 # "" = not provided
    week: str = "week"
    date: str = ""                    # "" = not provided
    home_team: str = "home_team"
    away_team: str = "away_team"
    home_odds: str = "home_odds"
    away_odds: str = "away_odds"
    timestamp: str = "timestamp"      # "" = use the run's as_of time


@dataclass
class WinTotalsColumns:
    team: str = "team"
    total: str = "win_total"          # the posted season win total (e.g. 10.5)
    over_odds: str = ""               # optional decimal odds on the over  ("" = not provided)
    under_odds: str = ""              # optional decimal odds on the under ("" = not provided)
    timestamp: str = ""               # optional capture time ("" = derived from the preseason lines)


@dataclass
class MatchingCfg:
    prefer_explicit_game_id: bool = True
    use_date_in_key: bool = False           # add game date to the (week, away, home) key
    resolve_swapped_home_away: bool = True  # accept a row whose home/away are reversed (warns)


@dataclass
class DevigCfg:
    game_method: str = "proportional"       # proportional | power
    min_decimal_odds: float = 1.01
    max_decimal_odds: float = 1000.0
    min_game_overround: float = 1.0         # below this => arbitrage / data error
    max_game_overround: float = 1.25        # above this => suspicious line


@dataclass
class RatingCfg:
    """Market-perceived team strength (logit units): implied by game lines, anchored on
    preseason win totals, updated by a Kalman filter each time fresh lines arrive."""
    # -- home-field advantage (logit units) ---------------------------------------------
    hfa_mode: str = "estimate"              # estimate (regression intercept on preseason lines) | fixed
    hfa_logit: float = 0.22                 # used when hfa_mode = "fixed" (2023-25 market average)
    # -- win totals -> expected wins -----------------------------------------------------
    normalize: str = "additive"             # additive | proportional | none  (totals sum > games)
    use_over_under_price: bool = True       # shade the total by the over/under price if provided
    wins_sd: float = 2.1                    # sd of season wins, converts O/U price to mean wins
    win_total_min: float = 2.0
    win_total_max: float = 15.5
    # -- baseline (t = 0) ---------------------------------------------------------------
    prior_sd: float = 0.15                  # [PRIOR] uncertainty of the win-total anchor (logit)
    reconcile_preseason_lines: bool = True  # refine the anchor with the preseason game lines
    preseason_line_sd: float = 0.60         # [PRIOR] noise of ONE far-future preseason line (logit);
                                            #   larger => win totals dominate; 1e6 => pure win totals
    # -- dynamics ------------------------------------------------------------------------
    process_sd_per_week: float = 0.08       # [PRIOR] true-strength random-walk sd per week (logit)
    fresh_line_sd: float = 0.15             # [PRIOR] noise of ONE fresh line as a strength measure
    lookahead_sd_per_week: float = 0.02     # extra noise per week a line is beyond next week's slate
    neutral_site_games: list = field(default_factory=list)  # game_ids with NO home-field advantage


@dataclass
class StrengthCfg:
    """How rating movement (in logit units) becomes a stale-line adjustment input."""
    center: str = "median"                  # remove league-wide drift: median | mean | none
    reference: str = "line_time"            # line_time: movement since the rating snapshot that
                                            #   already contained the game's line; preseason: since t=0
    symmetric: bool = True                  # True => *_down values are forced equal to *_up
    threshold_up: float = 0.02              # rating gain (logit) that counts as meaningful (backtested)
    threshold_down: float = 0.02            # rating loss (logit); used when symmetric = false
    gating: str = "soft"                    # soft (dead-zone) | hard (all-or-nothing)
    sensitivity_up: float = 1.0             # rating logit -> game logit multiplier (1 = direct)
    sensitivity_down: float = 1.0

    def __post_init__(self) -> None:
        if self.symmetric:
            self.threshold_down = self.threshold_up
            self.sensitivity_down = self.sensitivity_up


@dataclass
class AgingCfg:
    curve: str = "hyperbolic"               # hyperbolic | linear | exponential | logistic | step
    start_age_weeks: float = 3.0            # ages <= this get NO strength adjustment
    max_weight: float = 1.00                # ceiling on the strength-movement influence (backtested)
    half_weight_weeks: float = 0.5          # hyperbolic: weeks past start where weight = max/2 (backtested)
    full_weight_weeks: float = 12.0         # linear: weeks past start to reach max
    rate: float = 0.25                      # exponential: per-week rate
    logistic_midpoint_weeks: float = 7.0
    logistic_steepness: float = 0.8
    steps: list = field(default_factory=lambda: [
        {"age": 4, "weight": 0.15}, {"age": 5, "weight": 0.30},
        {"age": 6, "weight": 0.45}, {"age": 7, "weight": 0.60}])


@dataclass
class AdjustCfg:
    link: str = "logit"                     # logit | probit
    cap: float = 1.00                       # max |adjustment| in logit units (0 disables) (backtested)
    cap_mode: str = "tanh"                  # tanh (smooth) | clip


@dataclass
class ValidationCfg:
    strict: bool = False                    # True => any rejected input row aborts the run
    prob_tol: float = 1e-9
    max_reject_fraction: float = 0.25       # abort if more than this share of rows is rejected
    verify_integrity: bool = True           # verify master hash against the manifest on load


@dataclass
class OutputCfg:
    grid_team_column: str = "Team"
    grid_week_prefix: str = "Week "
    grid_decimals: int = 6                  # -1 = no rounding
    logit_per_point: float = 0.145          # display only: converts rating logits to spread-point equivalents


@dataclass
class SheetsCfg:
    """Google Sheets bridge. The sheet is the input/output surface; engine state stays in files."""
    spreadsheet: str = ""                         # Google Sheets URL or ID (or a local .xlsx path for offline use)
    credentials: str = "service_account.json"     # service-account key file (Google backend only)
    # ---- input tabs (read only; the bridge never writes to them) ----------------------------------
    lines_tab: str = "Game Lines"                 # ONLY the pulled lines; any number of rows/weeks
    initial_lines_tab: str = "Preserved Week 1 Full Season Li"   # full-season opening lines (bootstrap only)
    win_totals_tab: str = "Preseason Win Totals"  # read once, at bootstrap
    # ---- input column headers (matched case/space-insensitively; header row is auto-detected) ------
    col_game_time: str = "Game Time"
    col_away: str = "Away Team"
    col_home: str = "Home Team"
    col_away_odds: str = "Moneyline Away"         # decimal odds
    col_home_odds: str = "Moneyline Home"
    col_updated: str = "Line Last Updated"
    col_week: str = "Week #"                      # used only if present (initial lines); otherwise inferred
    col_wt_team: str = "Team"
    col_wt_total: str = "Win Total"
    col_wt_over: str = "Over (Decimal)"           # "" to ignore over/under prices
    col_wt_under: str = "Under (Decimal)"
    # ---- output tabs (owned by the bridge; overwritten / created) -----------------------------------
    grid_tab: str = "Probability Grid (Latest)"
    master_tab: str = "Master Game Table"
    ratings_tab: str = "Team Ratings"
    log_tab: str = "Grid Run Log"
    archive_prefix: str = "Grid"                  # archive tabs: "Grid 2026 W03 v004"
    archive_grids: bool = True
    write_ratings: bool = True
    tab_color: str = "#1F4E79"                    # colour of every bridge-owned tab
    # ---- matching ----------------------------------------------------------------------------------
    week_match_max_days: float = 6.0              # a line's game date must be within this of the schedule


@dataclass
class Config:
    season: SeasonCfg = field(default_factory=SeasonCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    columns_initial: LinesColumns = field(default_factory=LinesColumns)
    columns_refresh: LinesColumns = field(default_factory=LinesColumns)
    columns_win_totals: WinTotalsColumns = field(default_factory=WinTotalsColumns)
    matching: MatchingCfg = field(default_factory=MatchingCfg)
    team_aliases: dict = field(default_factory=dict)
    devig: DevigCfg = field(default_factory=DevigCfg)
    rating: RatingCfg = field(default_factory=RatingCfg)
    strength: StrengthCfg = field(default_factory=StrengthCfg)
    aging: AgingCfg = field(default_factory=AgingCfg)
    adjust: AdjustCfg = field(default_factory=AdjustCfg)
    validation: ValidationCfg = field(default_factory=ValidationCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    sheets: SheetsCfg = field(default_factory=SheetsCfg)

    def validate(self) -> "Config":
        def choice(name: str, val: str, allowed: tuple) -> None:
            if val not in allowed:
                raise ConfigError(f"{name} = {val!r}; allowed: {allowed}")

        choice("devig.game_method", self.devig.game_method, ("proportional", "power"))
        choice("rating.hfa_mode", self.rating.hfa_mode, ("estimate", "fixed"))
        choice("rating.normalize", self.rating.normalize, ("additive", "proportional", "none"))
        choice("strength.center", self.strength.center, ("median", "mean", "none"))
        choice("strength.reference", self.strength.reference, ("line_time", "preseason"))
        choice("strength.gating", self.strength.gating, ("soft", "hard"))
        choice("aging.curve", self.aging.curve,
               ("hyperbolic", "linear", "exponential", "logistic", "step"))
        choice("adjust.link", self.adjust.link, ("logit", "probit"))
        choice("adjust.cap_mode", self.adjust.cap_mode, ("tanh", "clip"))
        if self.devig.min_decimal_odds <= 1.0:
            raise ConfigError("devig.min_decimal_odds must be > 1.0")
        if not (0.0 <= self.aging.max_weight <= 1.0):
            raise ConfigError("aging.max_weight must be in [0, 1]")
        if self.aging.half_weight_weeks <= 0 or self.aging.full_weight_weeks <= 0:
            raise ConfigError("aging.half_weight_weeks / full_weight_weeks must be > 0")
        if self.adjust.cap < 0:
            raise ConfigError("adjust.cap must be >= 0")
        if self.strength.threshold_up < 0 or self.strength.threshold_down < 0:
            raise ConfigError("strength thresholds must be >= 0")
        r = self.rating
        for name in ("prior_sd", "preseason_line_sd", "fresh_line_sd", "wins_sd"):
            if getattr(r, name) <= 0:
                raise ConfigError(f"rating.{name} must be > 0")
        if r.process_sd_per_week < 0:
            raise ConfigError("rating.process_sd_per_week must be >= 0")
        if not (0 < r.win_total_min < r.win_total_max):
            raise ConfigError("rating.win_total_min/max invalid")
        if self.time.age_unit_days <= 0:
            raise ConfigError("time.age_unit_days must be > 0")
        if not (1 <= self.season.current_week <= self.season.n_weeks + 1):
            raise ConfigError("season.current_week must be in 1..n_weeks+1")
        return self


_SECTION_MAP = {  # TOML path -> attribute on Config
    ("season",): "season", ("time",): "time", ("paths",): "paths",
    ("columns", "initial"): "columns_initial", ("columns", "refresh"): "columns_refresh",
    ("columns", "win_totals"): "columns_win_totals", ("matching",): "matching",
    ("devig",): "devig", ("rating",): "rating", ("strength",): "strength", ("aging",): "aging",
    ("adjust",): "adjust", ("validation",): "validation", ("output",): "output",
    ("sheets",): "sheets",
}


def _build(cls, data: dict, where: str):
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{where}]: {sorted(unknown)}. Valid: {sorted(valid)}")
    return cls(**data)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load ``config.toml`` (or defaults if ``path`` is None).

    ``overrides`` may set ``section__key`` values, e.g. ``season__current_week=5``.
    """
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        with p.open("rb") as fh:
            raw = tomllib.load(fh)

    kwargs: dict[str, Any] = {}
    consumed: set = set()
    for tpath, attr in _SECTION_MAP.items():
        node: Any = raw
        for part in tpath:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        cls = type(getattr(Config(), attr))
        kwargs[attr] = _build(cls, dict(node), ".".join(tpath))
        if node:
            consumed.add(tpath[0])
    if "team_aliases" in raw:
        kwargs["team_aliases"] = dict(raw["team_aliases"])
        consumed.add("team_aliases")
    leftovers = set(raw) - consumed - {"columns"}
    if leftovers:
        raise ConfigError(f"Unknown top-level section(s): {sorted(leftovers)}")
    if "columns" in raw:
        bad = set(raw["columns"]) - {"initial", "refresh", "win_totals"}
        if bad:
            raise ConfigError(f"Unknown [columns.*] section(s): {sorted(bad)}")

    cfg = Config(**kwargs)
    for key, val in overrides.items():
        if val is None:
            continue
        section, _, name = key.partition("__")
        obj = getattr(cfg, section)
        if not is_dataclass(obj) or not hasattr(obj, name):
            raise ConfigError(f"Bad override {key!r}")
        setattr(obj, name, val)
    # re-run __post_init__ semantics for strength symmetry after overrides
    cfg.strength.__post_init__()
    return cfg.validate()
