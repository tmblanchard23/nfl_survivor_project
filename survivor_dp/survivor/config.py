"""
Reads config.toml (see the copy at the project root for the annotated
template). Relative paths inside it are resolved relative to the config
file's own folder -- NOT the folder you happen to run the command from --
so `credentials = "../nfl_prob_grid/service_account.json"` works no matter
where you are in the terminal.
"""
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER = "PASTE_YOUR_SHEET_URL_HERE"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """A problem with config.toml, phrased for a human to fix."""


@dataclass
class Config:
    path: Path
    spreadsheet: str
    credentials: Path
    master_game_table_tab: str
    pools_picks_tab: str
    my_player_label: str
    top_n: int


def find_config(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        return p
    for candidate in (Path.cwd() / "config.toml", PROJECT_ROOT / "config.toml"):
        if candidate.exists():
            return candidate.resolve()
    raise ConfigError(
        f"No config.toml found (looked in {Path.cwd()} and {PROJECT_ROOT}). "
        f"The project ships with one at the top level of the survivor_dp folder.")


def load_config(explicit_path: str | None = None) -> Config:
    path = find_config(explicit_path)
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} isn't valid TOML: {e}. A common cause is a "
                          f"missing quote around a value.") from e

    sheet = data.get("sheet", {})
    pool = data.get("pool", {})

    spreadsheet = str(sheet.get("spreadsheet", "")).strip()
    if not spreadsheet or spreadsheet == PLACEHOLDER:
        raise ConfigError(
            f"Open {path} and paste your Google Sheet's URL into the "
            f"`spreadsheet = \"...\"` line (the same sheet your grid engine writes to).")

    cred_raw = str(sheet.get("credentials", "service_account.json")).strip()
    cred = Path(cred_raw).expanduser()
    if not cred.is_absolute():
        cred = (path.parent / cred).resolve()

    return Config(
        path=path,
        spreadsheet=spreadsheet,
        credentials=cred,
        master_game_table_tab=str(sheet.get("master_game_table_tab", "Master Game Table")),
        pools_picks_tab=str(sheet.get("pools_picks_tab", "Pools Picks")),
        my_player_label=str(pool.get("my_player_label", "Me")),
        top_n=int(pool.get("top_n", 5)),
    )
