"""Command line interface:  python -m nfl_prob_grid <init|refresh|grid|status|validate>"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .engine import regenerate_grid, resolve_as_of, run_init, run_refresh, status
from .errors import ConfigError, InputError, StateError, ValidationError
from .master import StateStore
from .validate import validate_master


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nfl_prob_grid", description=__doc__)
    p.add_argument("--config", default="config.toml", help="path to config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, lines_help):
        sp.add_argument("--current-week", type=int, help="first NOT-yet-completed week "
                        "(weeks before it are frozen); default from config")
        sp.add_argument("--as-of", help="run timestamp (ISO); default = now. Set explicitly for "
                        "reproducible re-runs")
        sp.add_argument("--lines", help=lines_help)
        sp.add_argument("--dry-run", action="store_true", help="run everything, write nothing")

    si = sub.add_parser("init", help="build master v1 from the preseason lines + preseason win totals")
    common(si, "initial full-season lines (path or URL)")
    si.add_argument("--win-totals", help="preseason win-total table (path or URL)")
    si.add_argument("--force", action="store_true", help="archive existing state and start over")

    sr = sub.add_parser("refresh", help="apply newly refreshed lines to the latest master")
    common(sr, "refreshed lines table (path or Google Sheets URL)")
    sr.add_argument("--no-lines", action="store_true", help="do not read a lines table")

    sub.add_parser("grid", help="regenerate the team x week grid from the current master")

    ss = sub.add_parser("sheets", help="Google Sheets bridge: bootstrap | build | publish | status")
    ss.add_argument("action", choices=("bootstrap", "build", "publish", "status"))
    ss.add_argument("--spreadsheet", help="override [sheets] spreadsheet (URL, ID, or local .xlsx)")
    ss.add_argument("--as-of", help="run timestamp (ISO, UTC if no offset); default = now")
    ss.add_argument("--current-week", type=int, help="override the calendar-inferred current week")
    ss.add_argument("--catch-up", action="append", default=[], metavar="TAB",
                    help="bootstrap: extra line-export tab(s) to replay in time order (repeatable)")
    ss.add_argument("--force", action="store_true", help="bootstrap: archive existing state and start over")
    ss.add_argument("--dry-run", action="store_true", help="build: run everything, write nothing")
    sub.add_parser("status", help="show the state of the current master")
    sub.add_parser("validate", help="validate the current master table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.cmd == "init":
            rep = run_init(cfg, as_of=args.as_of, current_week=args.current_week, force=args.force,
                           lines_source=args.lines, win_totals_source=args.win_totals,
                           dry_run=args.dry_run)
            print(rep.summary())
        elif args.cmd == "refresh":
            rep = run_refresh(cfg, as_of=args.as_of, current_week=args.current_week,
                              lines_source=args.lines, use_lines=not args.no_lines,
                              dry_run=args.dry_run)
            print(rep.summary())
            if args.dry_run:
                print("(dry run: nothing written)")
        elif args.cmd == "grid":
            g = regenerate_grid(cfg)
            print(f"wrote {cfg.paths.output_dir}/prob_grid_latest.csv  ({g.shape[0]} teams x "
                  f"{g.shape[1]} weeks)")
        elif args.cmd == "sheets":
            from .sheets import bootstrap, build, publish, sheet_status
            if args.spreadsheet:
                cfg.sheets.spreadsheet = args.spreadsheet
            if args.action == "bootstrap":
                for rep in bootstrap(cfg, catch_up=args.catch_up, as_of=args.as_of, force=args.force):
                    print(rep.summary(), "\n")
                print("Sheet updated. Downstream work should read the tab:", cfg.sheets.grid_tab)
            elif args.action == "build":
                rep = build(cfg, as_of=args.as_of, current_week=args.current_week, dry_run=args.dry_run)
                print(rep.summary())
                b = rep.details.get("bridge", {})
                for r in b.get("set_aside_rows", [])[:10]:
                    print(f"  set aside: {r['detail']} ({r['reason']})")
                print("(dry run: nothing written)" if args.dry_run else f"Sheet updated: {cfg.sheets.grid_tab}"
                      + (f" + archive tab {b['archive_tab']!r}" if b.get("archive_tab") else ""))
            elif args.action == "publish":
                print("Wrote tabs:", ", ".join(publish(cfg)))
            else:
                print(json.dumps(sheet_status(cfg), indent=2, default=str))
        elif args.cmd == "status":
            print(json.dumps(status(cfg), indent=2))
        elif args.cmd == "validate":
            master, _, man = StateStore(cfg.paths.state_dir).load(cfg.validation.verify_integrity)
            issues = validate_master(master, cfg, prev=None, current_week=man["current_week"])
            for i in issues:
                print(i)
            errs = [i for i in issues if i.level == "error"]
            print(f"{len(errs)} error(s), {len(issues) - len(errs)} warning(s)")
            return 1 if errs else 0
        return 0
    except (ConfigError, InputError, StateError, ValidationError) as exc:
        print(f"ABORTED - master table was NOT modified.\n{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
