"""
Command line:

    python3 -m survivor check        # is everything wired up and logged correctly?
    python3 -m survivor recommend    # this week's Top-N recommendation

Both read the live Google Sheet described in config.toml. Your pick only
"counts" once YOU type it into the Pools Picks tab -- the program never
records it for you.

Options (both commands):
    --week N         Override the auto-detected current week.
    --config PATH    Use a config file other than ./config.toml.
    --csv-dir DIR    Read master_game_table.csv + pool_picks.csv from a
                     folder instead of Google Sheets (offline/testing).
recommend only:
    --top N              How many options per ranking (default from config).
    --save FILE          Also write the report to a Markdown file.
    --force              Produce a recommendation even if there are problems
                         that could make it wrong (not recommended).
    --no-sheet            Don't write the "DP Recommendation" tab this run.
    --market-check        Fetch survivorgrid.com's real pick-percentage data
                         and show it next to this program's own estimate, for
                         comparison only -- does NOT change the rankings.
    --market-blend [W]   Like --market-check, but also blends the real data
                         into the pool-equity math. Optional weight 0-1 (how
                         much to trust THIS program's own fitted models vs.
                         the broader market); default auto-scales with how
                         much pick history you've actually logged.

`recommend` also writes its output to a "DP Recommendation" tab in your
sheet (created automatically the first time), overwriting it each run.
That is the ONLY thing this program ever writes.
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

from . import data_io, report, survivorgrid
from .config import ConfigError, load_config


def _load_pool(args):
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
    except Exception as e:
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


def cmd_check(args) -> int:
    pool, _ = _load_pool(args)
    print("Connected and loaded successfully.\n")
    print(_summary(pool))
    problems = _problems(pool)
    print()
    print(problems if problems else "No problems found. You're ready to run:  python3 -m survivor recommend")
    return 1 if pool.load_report.blocking else 0


# ---------------------------------------------------------------------
# survivorgrid.com integration
# ---------------------------------------------------------------------

def _infer_season(reference: datetime | None = None) -> int:
    """NFL season year = the year it kicks off in. Jan-Jul of year Y is
    still "season Y-1" (the season that started the previous fall)."""
    d = reference or datetime.now()
    return d.year if d.month >= 8 else d.year - 1


def _fetch_market_pick_pct(args, pool) -> dict | None:
    """Returns {team: pick_pct}, fetching survivorgrid.com for whatever
    season/week applies. Prints a clear note (never a silent failure) if
    the fetch doesn't work, and the caller continues without it."""
    cfg = getattr(args, "_cfg", None)
    season = cfg.survivorgrid_season if (cfg and cfg.survivorgrid_season) else _infer_season()
    try:
        pct = survivorgrid.fetch_week(season=season, week=pool.current_week)
        print(f"(Fetched survivorgrid.com pick percentages for {season} week "
             f"{pool.current_week}.)")
        return pct
    except survivorgrid.SurvivorGridError as e:
        print(f"(Couldn't get survivorgrid.com data: {e}. Continuing with this "
             f"program's own estimates only.)")
        return None


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
        _write_to_sheet(args, pool, refused=True)
        return 1

    market_pct = None
    blend_weight = None
    want_market = args.market_check or (args.market_blend is not False)
    if want_market:
        market_pct = _fetch_market_pick_pct(args, pool)
        if args.market_blend is not False and market_pct:
            blend_weight = args.market_blend if isinstance(args.market_blend, float) else None

    rec = pool.get_recommendations(top_n=top_n, market_pick_pct=market_pct,
                                   market_blend_weight=blend_weight)
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


# ---------------------------------------------------------------------
# Writing the recommendation to the "DP Recommendation" tab
# ---------------------------------------------------------------------

OUTPUT_TAB = "DP Recommendation"


def _explanation_parts(rec, cand, pool_size):
    text = report.candidate_explanation(rec, cand, pool_size)
    parts = {}
    for line in text.split("\n")[1:]:
        label, _, body = line.lstrip("- ").partition(": ")
        parts[label] = body
    return [parts.get("Why DP likes this pick", ""),
           parts.get("Opponent interaction", ""),
           parts.get("Future opportunity cost", ""),
           parts.get("What could make this pick fail strategically", "")]


def build_sheet_rows(pool, rec=None, refused=False) -> list:
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

    if rec.market_comparison:
        rows += [[], ["Market Check (survivorgrid.com vs. this program's own model)"],
                ["Team", "Your Model", "Market", "Used in rankings above"]]
        for team, v in sorted(rec.market_comparison.items(), key=lambda kv: -kv[1]["market"]):
            if v["model"] < 0.005 and v["market"] < 0.005:
                continue
            rows.append([team, pct(v["model"]), pct(v["market"]), pct(v["blended"])])

    conf = rec.confidence
    rows += [[], ["Confidence"],
            ["Vegas information", conf["vegas_information"]],
            ["Opponent model", conf["opponent_model"]],
            ["Overall", conf["overall_strategic_confidence"]],
            [], [f"This is a recommendation only. Nothing was recorded. Log your actual "
                f"week {r.current_week} pick in Pools Picks yourself."]]
    return rows


def publish_to_sheet(spreadsheet, rows, tab: str = OUTPUT_TAB) -> None:
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
    if args.csv_dir or args.no_sheet or not hasattr(args, "_cfg"):
        return
    try:
        publish_to_sheet(_open_sheet_for_writing(args._cfg),
                         build_sheet_rows(pool, rec, refused))
        print(f"\n(Written to the '{OUTPUT_TAB}' tab in your Google Sheet.)")
    except Exception as e:
        print(f"\n(Couldn't write to the '{OUTPUT_TAB}' tab: {type(e).__name__}: {e}. "
             f"The recommendation above is still valid.)")


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

class _OptionalFloat(argparse.Action):
    """--market-blend alone -> True (auto weight). --market-blend 0.7 -> 0.7."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, float(values) if values is not None else True)


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
            p.add_argument("--market-check", action="store_true",
                          help="show survivorgrid.com's real pick %% next to this "
                               "program's own estimate (comparison only)")
            p.add_argument("--market-blend", nargs="?", default=False,
                          action=_OptionalFloat, metavar="WEIGHT",
                          help="like --market-check, but also use the real data "
                               "in the pool-equity rankings (optional 0-1 weight)")
    args = parser.parse_args(argv)
    return {"check": cmd_check, "recommend": cmd_recommend}[args.command](args)
