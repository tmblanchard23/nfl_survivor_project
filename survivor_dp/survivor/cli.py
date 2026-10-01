"""
Command line:

    python3 -m survivor check        # is everything wired up and logged correctly?
    python3 -m survivor recommend    # this week's Top-N recommendation

Both read the live Google Sheet described in config.toml. Your pick only
"counts" once YOU type it into the Pools Picks tab -- the program never
records it for you, which is the whole point of the recommend/decide
separation in the design.

Options (both commands):
    --week N         Override the auto-detected current week.
    --config PATH    Use a config file other than ./config.toml.
    --csv-dir DIR    Read master_game_table.csv + pool_picks.csv from a
                     folder instead of Google Sheets (offline/testing).
recommend only:
    --top N          How many options per ranking (default from config).
    --save FILE      Also write the report to a Markdown file.
    --force          Produce a recommendation even if there are problems
                     that could make it wrong (not recommended).
    --no-sheet       Don't write the "DP Recommendation" tab this run.

`recommend` also writes its output to a "DP Recommendation" tab in your
sheet (created automatically the first time), overwriting it each run.
That is the ONLY thing this program ever writes.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from . import data_io, report
from .config import ConfigError, load_config


def _load_pool(args):
    """Returns (pool, top_n). Raises SystemExit with a readable message on failure."""
    if args.csv_dir:
        pool = data_io.load_pool_from_csv_dir(args.csv_dir, current_week=args.week)
        return pool, (getattr(args, "top", None) or 5)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.exit(f"Config problem: {e}")

    if not cfg.credentials.exists():
        sys.exit(
            f"Can't find your service account key at:\n  {cfg.credentials}\n"
            f"Edit the `credentials = ...` line in {cfg.path} to point at the "
            f"service_account.json you made for the grid engine.")

    try:
        data = data_io.read_google_sheet_tabs(
            cfg.spreadsheet, cfg.credentials,
            [cfg.master_game_table_tab, cfg.pools_picks_tab])
    except ImportError as e:
        sys.exit(str(e))
    except Exception as e:  # gspread raises several exception types; explain the common ones
        name = type(e).__name__
        if name == "SpreadsheetNotFound":
            sys.exit("Google says that spreadsheet doesn't exist or the service account "
                     "can't see it. Check the URL in config.toml, and that the sheet is "
                     "shared with the service account's email (the `client_email` in "
                     "service_account.json).")
        if name == "WorksheetNotFound":
            sys.exit(f"A tab wasn't found: {e}. Check the tab names in config.toml "
                     f"match the sheet exactly (capitalization and spaces count).")
        sys.exit(f"Couldn't read the Google Sheet ({name}): {e}")

    try:
        pool = data_io.build_pool_from_rows(
            data[cfg.master_game_table_tab], data[cfg.pools_picks_tab],
            current_week=args.week, my_player_label=cfg.my_player_label)
    except ValueError as e:
        sys.exit(str(e))
    args._cfg = cfg
    return pool, (getattr(args, "top", None) or cfg.top_n)


def _summary(pool) -> str:
    r = pool.load_report
    lines = [f"Current week: {r.current_week} ({r.week_source})"]
    if r.my_picks:
        picks = ", ".join(f"W{wk} {team}" for wk, team in r.my_picks)
        lines.append(f"Your picks so far: {picks}")
    else:
        lines.append("Your picks so far: none logged")
    lines.append(f"Opponents still alive: {len(r.opponents_alive)}"
                 + (f" ({', '.join(r.opponents_alive)})" if r.opponents_alive else ""))
    if r.opponents_eliminated:
        lines.append(f"Opponents eliminated: {len(r.opponents_eliminated)} "
                     f"({', '.join(r.opponents_eliminated)})")
    return "\n".join(lines)


def _problems(pool) -> str:
    r = pool.load_report
    out = []
    if r.blocking:
        out.append("PROBLEMS (these could make a recommendation wrong):")
        out += [f"  - {m}" for m in r.blocking]
    if r.warnings:
        out.append("Notes (worth a look, won't break anything):")
        out += [f"  - {m}" for m in r.warnings]
    return "\n".join(out)


# ---------------------------------------------------------------------
# Writing the recommendation to the "DP Recommendation" tab
# ---------------------------------------------------------------------
#
# Only ever writes to this one tab (creating it if it doesn't exist) and
# fully overwrites it each run -- same pattern as the grid engine's output
# tabs. It never touches Pools Picks, Master Game Table, or any other tab,
# and it never records your pick: you still do that yourself.

OUTPUT_TAB = "DP Recommendation"


def _explanation_parts(rec, cand, pool_size):
    """Split report.candidate_explanation()'s text into its four labeled
    parts so each can go in its own spreadsheet column."""
    text = report.candidate_explanation(rec, cand, pool_size)
    parts = {}
    for line in text.split("\n")[1:]:
        label, _, body = line.lstrip("- ").partition(": ")
        parts[label] = body
    return [parts.get("Why DP likes this pick", ""),
            parts.get("Opponent interaction", ""),
            parts.get("Future opportunity cost", ""),
            parts.get("What could make this pick fail strategically", "")]


def build_sheet_rows(pool, rec=None, refused=False) -> list[list[str]]:
    """The whole tab as a grid of cell values. Percentages are written as
    "78.7%" strings, which Google Sheets turns into real percentage cells."""
    from datetime import datetime
    r = pool.load_report
    pct = lambda x: f"{x * 100:.1f}%"
    rows = [
        [f"DP Recommendation -- Week {r.current_week}"],
        ["Generated", datetime.now().strftime("%Y-%m-%d %I:%M %p")],
        ["Week", f"{r.current_week} ({r.week_source})"],
        ["Your picks so far", ", ".join(f"W{w} {t}" for w, t in r.my_picks) or "none logged"],
        ["Opponents alive", f"{len(r.opponents_alive)}: {', '.join(r.opponents_alive)}"],
        ["Opponents eliminated", f"{len(r.opponents_eliminated)}: {', '.join(r.opponents_eliminated)}"],
    ]
    if r.blocking:
        rows += [[], ["PROBLEMS (these could make a recommendation wrong -- fix in Pools Picks)"]]
        rows += [["", m] for m in r.blocking]
    if r.warnings:
        rows += [[], ["Notes (worth a look, won't break anything)"]]
        rows += [["", m] for m in r.warnings]
    if refused or rec is None:
        rows += [[], ["NO RECOMMENDATION THIS RUN. Fix the PROBLEMS above, then run "
                      "python3 -m survivor recommend again."]]
        return rows

    size = len(pool.alive_opponents())
    rows += [[], ["Ranking A -- Pure Survival"],
             ["Rank", "Team", "Win %", "P(Survive)", "Future Opportunity Cost"]]
    for i, c in enumerate(rec.pure_survival_top, 1):
        rows.append([str(i), c.team, pct(c.vegas_win_prob), pct(c.p_survive_this_week),
                     f"{c.future_opportunity_cost:.4f}"])

    rows += [[], ["Ranking B -- Pool Equity   (P(Win Pool) is an approximation)"],
             ["Rank", "Team", "Win %", "P(Survive)", "P(Win Pool)",
              "Exp. Opponents Eliminated", "Exp. Remaining Pool"]]
    for i, c in enumerate(rec.pool_equity_top, 1):
        rows.append([str(i), c.team, pct(c.vegas_win_prob), pct(c.p_survive_this_week),
                     pct(c.p_win_pool_approx), f"{c.expected_opponents_eliminated:.2f}",
                     f"{c.expected_remaining_pool:.2f}"])

    rows += [[], ["Why each Pool Equity option is here"],
             ["Team", "Why DP likes it", "Opponent interaction",
              "Future opportunity cost", "What could go wrong"]]
    for c in rec.pool_equity_top:
        rows.append([c.team] + _explanation_parts(rec, c, size))

    conf = rec.confidence
    rows += [[], ["Confidence"],
             ["Vegas information", conf["vegas_information"]],
             ["Opponent model", conf["opponent_model"]],
             ["Overall", conf["overall_strategic_confidence"]],
             [], [f"This is a recommendation only. Nothing was recorded. Log your actual "
                  f"week {r.current_week} pick in Pools Picks yourself."]]
    return rows


def publish_to_sheet(spreadsheet, rows, tab: str = OUTPUT_TAB) -> None:
    """Overwrite `tab` in an open gspread Spreadsheet with `rows`,
    creating the tab if it doesn't exist yet."""
    width = max(len(r) for r in rows)
    grid = [[str(v) for v in r] + [""] * (width - len(r)) for r in rows]
    try:
        ws = spreadsheet.worksheet(tab)
    except Exception as e:
        if type(e).__name__ != "WorksheetNotFound":
            raise
        ws = spreadsheet.add_worksheet(title=tab, rows=len(grid) + 20, cols=max(width, 10))
    if ws.row_count < len(grid) or ws.col_count < width:
        ws.resize(rows=max(ws.row_count, len(grid) + 20), cols=max(ws.col_count, width))
    ws.clear()
    ws.update(values=grid, range_name="A1", value_input_option="USER_ENTERED")


def _open_sheet_for_writing(cfg):
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_file(
        str(cfg.credentials), scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return gspread.authorize(creds).open_by_key(data_io.spreadsheet_id_from(cfg.spreadsheet))


def _write_to_sheet(args, pool, rec=None, refused=False) -> None:
    """Best-effort: a failure here is reported but never hides the
    recommendation you already got in the terminal."""
    if args.csv_dir or args.no_sheet or not hasattr(args, "_cfg"):
        return
    try:
        publish_to_sheet(_open_sheet_for_writing(args._cfg),
                         build_sheet_rows(pool, rec, refused))
        print(f"\n(Written to the '{OUTPUT_TAB}' tab in your Google Sheet.)")
    except Exception as e:
        print(f"\n(Couldn't write to the '{OUTPUT_TAB}' tab: {type(e).__name__}: {e}. "
              f"The recommendation above is still valid.)")


def cmd_check(args) -> int:
    pool, _ = _load_pool(args)
    print("Connected and loaded successfully.\n")
    print(_summary(pool))
    problems = _problems(pool)
    print()
    print(problems if problems else "No problems found. You're ready to run:  python3 -m survivor recommend")
    return 1 if pool.load_report.blocking else 0


def cmd_recommend(args) -> int:
    pool, top_n = _load_pool(args)
    r = pool.load_report
    print(_summary(pool))
    problems = _problems(pool)
    if problems:
        print()
        print(problems)
    if r.blocking and not args.force:
        print("\nNot producing a recommendation until the PROBLEMS above are fixed "
              "in the sheet (or re-run with --force if you understand the risk).")
        _write_to_sheet(args, pool, refused=True)  # so the tab never shows a stale week
        return 1

    rec = pool.get_recommendations(top_n=top_n)
    text = report.full_report(rec, len(pool.alive_opponents()))
    print("\n" + text)
    print(
        f"\nNext steps: make your week {r.current_week} pick on your pool's site, then "
        f"type it into Pools Picks (your row, 'Week {r.current_week}' column). After the "
        f"week, log everyone else's picks and mark anyone eliminated with OUT. "
        f"Nothing was recorded by running this.")
    if args.save:
        Path(args.save).write_text(text)
        print(f"(Report also saved to {args.save})")
    _write_to_sheet(args, pool, rec)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m survivor",
                                     description="NFL Survivor DP optimizer")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (("check", "verify the sheet connection and your logged picks"),
                           ("recommend", "produce this week's recommendation")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--week", type=int, default=None)
        p.add_argument("--config", default=None)
        p.add_argument("--csv-dir", default=None)
        if name == "recommend":
            p.add_argument("--top", type=int, default=None)
            p.add_argument("--save", default=None)
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-sheet", action="store_true",
                           help="don't write the DP Recommendation tab")
    args = parser.parse_args(argv)
    return {"check": cmd_check, "recommend": cmd_recommend}[args.command](args)
