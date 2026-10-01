"""Calendar helpers for the Sheets bridge.

* ``assign_weeks``  - the lines tab carries no week column: each row is matched to the scheduled game with the
  same two teams (either orientation) whose date is nearest to the row's game time. Division rivals meet twice;
  the date separates the two meetings. Rows more than ``max_days`` from any scheduled meeting (preseason,
  playoffs, a wrong date) are set aside and reported, never guessed.
* ``infer_current_week`` - the first week whose last game has not been played yet (Eastern dates). A week stays
  active through its Monday game and is frozen from the following day.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from .teams import TeamNormalizer


def _et_date(ts, tz: str) -> pd.Timestamp | None:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)) or str(ts).strip() == "":
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(ZoneInfo(tz))
    return t.tz_convert(ZoneInfo(tz)).normalize().tz_localize(None)


def infer_current_week(master: pd.DataFrame, as_of: pd.Timestamp, tz: str, n_weeks: int) -> int:
    today = _et_date(as_of, tz)
    last = master.assign(d=pd.to_datetime(master["game_date"])).groupby("week")["d"].max()
    open_weeks = last[last >= today]
    return int(open_weeks.index.min()) if len(open_weeks) else n_weeks + 1


def assign_weeks(lines: pd.DataFrame, master: pd.DataFrame, nz: TeamNormalizer, tz: str,
                 max_days: float) -> tuple[pd.DataFrame, list[dict]]:
    """Add a ``week`` column to ``lines`` (columns away, home, game_time). Returns (kept rows, set-aside rows).

    Rows whose teams cannot be resolved are KEPT with week = "" so the engine rejects them loudly as
    unknown teams (a naming problem must never be silently skipped)."""
    sched = master.assign(d=pd.to_datetime(master["game_date"]))
    by_pair: dict[frozenset, list[tuple[int, pd.Timestamp]]] = {}
    for w, a, h, d in zip(sched["week"], sched["away_team"], sched["home_team"], sched["d"]):
        by_pair.setdefault(frozenset((a, h)), []).append((int(w), d))
    weeks, keep, aside = [], [], []
    for i, r in enumerate(lines.itertuples(index=False)):
        a, h = nz(r.away), nz(r.home)
        if a is None or h is None:
            weeks.append(""); keep.append(True)
            continue
        cands = by_pair.get(frozenset((a, h)), [])
        d = _et_date(r.game_time, tz)
        if not cands:
            aside.append({"row": i, "reason": "not_scheduled", "detail": f"{r.away} @ {r.home} is not on the regular-season schedule"})
            weeks.append(""); keep.append(False)
            continue
        if d is None:
            if len(cands) == 1:
                weeks.append(cands[0][0]); keep.append(True)
            else:
                aside.append({"row": i, "reason": "ambiguous_no_date", "detail": f"{r.away} @ {r.home}: two meetings and no game time"})
                weeks.append(""); keep.append(False)
            continue
        wk, gd = min(cands, key=lambda c: abs((c[1] - d).days))
        if abs((gd - d).days) > max_days:
            aside.append({"row": i, "reason": "date_far_from_schedule",
                          "detail": f"{r.away} @ {r.home} on {d:%Y-%m-%d}: nearest scheduled meeting is week {wk} ({gd:%Y-%m-%d})"})
            weeks.append(""); keep.append(False)
            continue
        weeks.append(wk); keep.append(True)
    out = lines.assign(week=weeks)
    return out[keep].reset_index(drop=True), aside
