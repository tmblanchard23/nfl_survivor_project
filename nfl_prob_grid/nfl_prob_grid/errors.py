"""Exception and issue types shared across the package."""
from __future__ import annotations

from dataclasses import dataclass


class ConfigError(ValueError):
    """The configuration file is invalid."""


class RowError(ValueError):
    """A single input row is unusable. Carries a machine-readable ``code``."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class OddsError(RowError):
    """Odds on a row are missing / invalid / implausible."""


class InputError(RuntimeError):
    """A whole input table is unusable (missing columns, unreadable, ...)."""


class StateError(RuntimeError):
    """Master table / manifest integrity problem."""


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str
    game_id: str = ""

    def __str__(self) -> str:
        gid = f" [{self.game_id}]" if self.game_id else ""
        return f"{self.level.upper()} {self.code}{gid}: {self.message}"


class ValidationError(RuntimeError):
    """Validation found one or more errors. Nothing is written when this is raised."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        errs = [i for i in issues if i.level == "error"] or issues
        shown = "\n  ".join(str(i) for i in errs[:25])
        more = f"\n  ... and {len(errs) - 25} more" if len(errs) > 25 else ""
        super().__init__(f"{len(errs)} validation error(s):\n  {shown}{more}")
