"""Master game table: schema, construction, and the versioned on-disk state store.

State layout (all under ``paths.state_dir``):

    master_current.csv            latest master table  (what the next run loads)
    versions/master_v0001.csv     immutable copy of every version ever produced
    ratings_history.csv           append-only team-rating snapshots (snapshot 0 = baseline)
    rating_state.json             Kalman covariance, home-field advantage, baseline audit table
    changes/changes_v0002.csv     per-game diff between consecutive versions
    runs/run_v0002.json           run report (counts, issues, rejected rows)
    manifest.json                 version list with a sha256 hash chain

A commit writes everything else first and the manifest LAST, so an interrupted run leaves the
previous version authoritative.  Loading verifies hashes and refuses tampered state.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config
from .errors import StateError
from .ratings import RatingHistory
from .tabular import atomic_write_csv, atomic_write_text, iso, sha256_file

PROB_SOURCES = ("VEGAS_PRESEASON", "VEGAS_FRESH", "VEGAS_OLDER", "VEGAS_STALE_ADJUSTED")

# (column, kind).  kinds: str | int | Int (nullable) | float | bool | ts
MASTER_SCHEMA: list[tuple[str, str]] = [
    ("game_id", "str"), ("source_game_id", "str"), ("season", "int"), ("week", "int"),
    ("game_date", "str"), ("away_team", "str"), ("home_team", "str"),
    # --- current state -------------------------------------------------------------------
    ("home_win_prob", "float"), ("away_win_prob", "float"), ("prob_source", "str"),
    # --- Vegas lineage --------------------------------------------------------------------
    ("original_vegas_home_prob", "float"), ("original_vegas_ts", "ts"),
    ("latest_vegas_home_prob", "float"), ("latest_home_odds", "float"), ("latest_away_odds", "float"),
    ("last_vegas_refresh_ts", "ts"), ("refresh_count", "int"), ("last_refresh_version", "int"),
    ("anchor_rating_snapshot_id", "int"),   # rating snapshot that already contained this line
    # --- strength-adjustment lineage --------------------------------------------------------
    ("adj_active", "bool"), ("adj_logit", "float"), ("adj_home_prob_delta", "float"),
    ("line_age_weeks", "float"), ("age_weight", "float"),
    ("home_rating_move", "float"), ("away_rating_move", "float"),
    ("home_strength_shift", "float"), ("away_strength_shift", "float"), ("matchup_shift", "float"),
    ("rating_ref_snapshot_id", "Int"), ("rating_now_snapshot_id", "Int"),
    # --- state ------------------------------------------------------------------------------
    ("is_frozen", "bool"), ("frozen_at_version", "Int"), ("last_updated_version", "int"),
]
MASTER_COLUMNS = [c for c, _ in MASTER_SCHEMA]
TS_COLS = [c for c, k in MASTER_SCHEMA if k == "ts"]


def coerce_master_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, kind in MASTER_SCHEMA:
        if col not in df.columns:
            raise StateError(f"master table is missing column {col!r}")
        s = df[col]
        if kind == "str":
            df[col] = s.fillna("").astype(str).astype(object)
        elif kind == "int":
            df[col] = pd.to_numeric(s).astype("int64")
        elif kind == "Int":
            df[col] = pd.to_numeric(s).astype("Int64")
        elif kind == "float":
            df[col] = pd.to_numeric(s).astype("float64")
        elif kind == "bool":
            df[col] = s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)
        elif kind == "ts":
            df[col] = pd.to_datetime(s, utc=True)
    return df[MASTER_COLUMNS]


def master_to_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[MASTER_COLUMNS].copy()
    for col in TS_COLS:
        out[col] = out[col].map(iso)
    return out


def build_initial_master(lines: pd.DataFrame, cfg: Config, *, version: int,
                         current_week: int) -> pd.DataFrame:
    """Master v1 from de-vigged Input A rows (one row per game, already resolved/deduped)."""
    lines = lines.sort_values(["week", "game_date", "away_team", "home_team"]).reset_index(drop=True)
    season = cfg.season.season
    frozen = lines["week"] < current_week
    m = pd.DataFrame({
        "game_id": [f"{season}-W{w:02d}-{a}@{h}" for w, a, h in
                    zip(lines["week"], lines["away_team"], lines["home_team"])],
        "source_game_id": lines["source_game_id"].fillna(""),
        "season": season, "week": lines["week"], "game_date": lines["game_date"],
        "away_team": lines["away_team"], "home_team": lines["home_team"],
        "home_win_prob": lines["home_prob"], "away_win_prob": 1.0 - lines["home_prob"],
        "prob_source": "VEGAS_PRESEASON",
        "original_vegas_home_prob": lines["home_prob"], "original_vegas_ts": lines["ts"],
        "latest_vegas_home_prob": lines["home_prob"],
        "latest_home_odds": lines["home_odds"], "latest_away_odds": lines["away_odds"],
        "last_vegas_refresh_ts": lines["ts"], "refresh_count": 0, "last_refresh_version": version,
        "anchor_rating_snapshot_id": 0,
        "adj_active": False, "adj_logit": 0.0, "adj_home_prob_delta": 0.0,
        "line_age_weeks": float("nan"), "age_weight": float("nan"),
        "home_rating_move": float("nan"), "away_rating_move": float("nan"),
        "home_strength_shift": float("nan"), "away_strength_shift": float("nan"),
        "matchup_shift": float("nan"),
        "rating_ref_snapshot_id": pd.NA, "rating_now_snapshot_id": pd.NA,
        "is_frozen": frozen, "frozen_at_version": pd.Series(
            [version if f else pd.NA for f in frozen], dtype="Int64"),
        "last_updated_version": version,
    })
    return coerce_master_dtypes(m)


class StateStore:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.manifest_path = self.dir / "manifest.json"
        self.current_path = self.dir / "master_current.csv"
        self.ratings_path = self.dir / "ratings_history.csv"
        self.rating_state_path = self.dir / "rating_state.json"
        self.versions_dir = self.dir / "versions"
        self.runs_dir = self.dir / "runs"
        self.changes_dir = self.dir / "changes"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def read_manifest(self) -> dict:
        if not self.exists():
            raise StateError(f"No state found in {self.dir}. Run `init` first.")
        return json.loads(self.manifest_path.read_text())

    def next_version(self) -> int:
        return self.read_manifest()["current_version"] + 1 if self.exists() else 1

    def archive_existing(self) -> Path | None:
        """For ``init --force``: move existing state aside (never deleted)."""
        if not self.dir.exists():
            return None
        dest = self.dir.with_name(f"{self.dir.name}.bak_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}")
        shutil.move(str(self.dir), str(dest))
        return dest

    def load(self, verify: bool = True) -> tuple[pd.DataFrame, RatingHistory, dict]:
        man = self.read_manifest()
        last = man["versions"][-1]
        if verify:
            if not self.current_path.exists():
                raise StateError(f"{self.current_path} is missing; restore it from "
                                 f"{self.versions_dir / last['file']}")
            if sha256_file(self.current_path) != last["sha256"]:
                raise StateError(
                    f"{self.current_path.name} does not match manifest version {last['version']} "
                    "(edited outside the system or an interrupted write). To recover, copy "
                    f"{self.versions_dir / last['file']} over it, or set validation.verify_integrity"
                    " = false to accept manual edits.")
            for path, key in ((self.ratings_path, "ratings_sha256"),
                              (self.rating_state_path, "rating_state_sha256")):
                if not path.exists() or sha256_file(path) != last[key]:
                    raise StateError(f"{path.name} does not match the manifest")
        master = coerce_master_dtypes(pd.read_csv(
            self.current_path, dtype={"source_game_id": str, "game_date": str},
            float_precision="round_trip"))   # default parser can be 1 ULP off -> would drift frozen values
        return master, RatingHistory.load(self.ratings_path, self.rating_state_path), man

    def commit(self, master: pd.DataFrame, hist: RatingHistory, *, version: int, run_type: str,
               as_of: pd.Timestamp, current_week: int, report: dict, changes: pd.DataFrame,
               cfg: Config) -> dict:
        """Persist a new version.  Manifest is written last."""
        vfile = f"master_v{version:04d}.csv"
        csv_frame = master_to_csv_frame(master)
        atomic_write_csv(csv_frame, self.versions_dir / vfile)
        hist.save(self.ratings_path, self.rating_state_path)
        atomic_write_text(self.runs_dir / f"run_v{version:04d}.json",
                          json.dumps(report, indent=2, default=str))
        if len(changes):
            atomic_write_csv(changes, self.changes_dir / f"changes_v{version:04d}.csv")
        atomic_write_csv(csv_frame, self.current_path)

        man = self.read_manifest() if self.exists() else {
            "schema_version": 1, "season": cfg.season.season, "versions": []}
        parent = man["versions"][-1]["sha256"] if man["versions"] else ""
        man["versions"].append({
            "version": version, "file": vfile, "run_type": run_type,
            "sha256": sha256_file(self.versions_dir / vfile), "parent_sha256": parent,
            "ratings_sha256": sha256_file(self.ratings_path),
            "rating_state_sha256": sha256_file(self.rating_state_path),
            "as_of": iso(as_of), "current_week": current_week, "n_games": int(len(master)),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        man["current_version"], man["current_week"] = version, current_week
        atomic_write_text(self.manifest_path, json.dumps(man, indent=2))
        return man
