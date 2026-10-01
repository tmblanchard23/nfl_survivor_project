# NFL Survivor DP Optimizer

Reads the probabilities your grid engine publishes to **Master Game Table**,
reads everyone's picks from **Pools Picks**, and tells you which team to
pick this week. There are two rankings: best for pure survival, and best
for actually winning the pool given what your opponents are likely to do.

It only ever **reads** your sheet. It never writes to it, and it never
assumes you took its advice. Your pick only counts once you type it into
Pools Picks yourself.

---

## One-time setup (about 10 minutes)

### 1. Put the folder in the right place

Unzip `survivor_dp` so it sits **right next to** your existing grid-engine
folder:

```
NFL Survivor Project/
├── nfl_prob_grid/        <- your existing grid engine (has service_account.json)
└── survivor_dp/          <- this project
```

This matters for two reasons. The config file finds your existing
`service_account.json` at `../nfl_prob_grid/`. It also avoids the
duplicate-folder confusion you hit on the grid project. Delete any older
`survivor_dp` zips or folders in Downloads or Desktop so there's exactly
one copy.

### 2. Install the libraries

Open Terminal, go into the folder, and install:

```bash
cd "path/to/NFL Survivor Project/survivor_dp"
pip3 install -r requirements.txt
```

(Tip: type `cd ` with a space, then drag the `survivor_dp` folder from
Finder into the Terminal window to paste its path.)

### 3. Edit `config.toml`

Open `config.toml` (inside `survivor_dp`) in any text editor and change:

- **`spreadsheet = "..."`**: paste your Google Sheet's full URL, the same
  sheet the grid engine writes to.
- **`credentials = "..."`**: already set to
  `../nfl_prob_grid/service_account.json`. Leave it alone if you did step
  1. Otherwise, point it at wherever that file lives.

No new Google setup is needed. The service account from the grid project
already has access to that sheet.

### 4. Catch up Pools Picks to the present day

**You don't replay or "advance" anything.** Every run rebuilds the whole
picture from Pools Picks: your used teams, each opponent's history, and who's
eliminated. Catching up just means filling in the tab:

- One row per player, plus your row labeled **`Me`**.
- Columns `Week 1`, `Week 2`, ... Add `Week 16` through `Week 18` headers
  if your tab stops at 15.
- Fill in every pick for every week that's been **played**. Full names
  ("Kansas City Chiefs"), nicknames ("Chiefs"), or abbreviations ("KC")
  all work.
- When someone is eliminated, type **`OUT`** in the week they lost. Leave
  their later weeks blank.
- Leave the current week's cells blank until the week is over.

### 5. Check that everything is wired up

```bash
python3 -m survivor check
```

This connects to your sheet, figures out the current week, and lists
anything that needs fixing: a week missing from your own row, a team name
it can't match, an opponent with a gap that might mean they're eliminated.
Fix anything it flags in the sheet and run it again until it says
**"No problems found."**

---

## Every week

1. **Refresh the probabilities** the way you already do, in the grid-engine
   folder:
   ```bash
   python3 -m nfl_prob_grid sheets build
   ```
2. **Get the recommendation**, in the `survivor_dp` folder:
   ```bash
   python3 -m survivor recommend
   ```
   It prints both Top-5 rankings, a short explanation for each option,
   and a confidence readout. Add `--save week4.md` to also save the
   report to a file.
3. **Decide and submit** your pick on your pool's website, as normal.
4. **Log it**: type your pick into your `Me` row under this week's column.
5. **After the games**: log everyone else's picks for the week, and put
   `OUT` for anyone who lost.

That's the whole loop.

### How it knows what week it is

It uses the `Completed` column in Master Game Table (the grid engine fills
that in). The current week is the earliest week that still has an
unfinished game. So it keeps saying "week 3" until every week-3 game is
marked completed, and only then moves to week 4.

To look ahead before that happens (say, planning week 4 on Sunday
afternoon), force it:

```bash
python3 -m survivor recommend --week 4
```

It will insist your week-3 pick is logged first. Otherwise it could
recommend a team you just used.

---

## Troubleshooting

| You see | What it means |
|---|---|
| `command not found: python` | Use `python3`, not `python` (same as the grid project). |
| `No module named survivor` | You're not in the `survivor_dp` folder. `cd` into it first. |
| `No module named scipy` (or numpy/gspread) | Re-run the `pip3 install ...` line from setup step 2. |
| "paste your Google Sheet's URL" | You haven't edited `config.toml` yet (setup step 3). |
| "Can't find your service account key" | The `credentials` path in `config.toml` is wrong. Point it at your `service_account.json`. |
| "spreadsheet doesn't exist or the service account can't see it" | Double-check the URL. The sheet must be shared with the `client_email` in `service_account.json` (it already is if the grid engine works). |
| "A tab wasn't found" | A tab name in `config.toml` doesn't match the sheet exactly (capitals and spaces count). |
| "Not producing a recommendation until the PROBLEMS above are fixed" | Your own pick history has a gap or an unrecognized name. Fix it in Pools Picks. `--force` overrides this, but don't. |

## Working offline (optional)

Download both tabs as CSVs (File → Download → CSV), name them
`master_game_table.csv` and `pool_picks.csv`, put them in a folder, and
run either command with `--csv-dir that_folder`. There's a working
example in `example_data/`:

```bash
python3 -m survivor recommend --csv-dir example_data
```

## Using it from Python (optional)

The commands above cover normal use. If you want to script it yourself,
`example_usage.py` shows the Python API: `load_pool_from_google_sheet()`,
`get_recommendations()`, and the in-memory `record_my_pick()` /
`advance_week()` workflow. When you use the sheet, you never need those
record/advance calls, because the sheet is the record.

---

# Technical reference

## What's exact vs. approximate (read this before trusting the output)

| Component | Status | Why |
|---|---|---|
| Game win probabilities | **Taken as-is from the grid engine** | This system does not compute, adjust, or second-guess them. See "Where the probabilities come from" below. |
| Pure Survival (Ranking A) | **Exact** | Maximize the product of win probabilities across remaining weeks, one team per week, no reuse. Solved as an assignment problem on log-probabilities (mathematically identical, and it solves a full 18-week season in milliseconds; an earlier recursive version was also exact but never finished on a real season). Verified identical to brute-force enumeration on hundreds of random small seasons. |
| Future Opportunity Cost | **Exact**, given the DP above | `V(best alternative this week, preserving the team) - V(using the team now)`. |
| Opponent pick prediction | **Data-driven, not exact** | A conditional logit (discrete-choice) model per opponent, fit by MAP/gradient ascent, shrunk toward a population prior in proportion to how little data that opponent has. With ~18 picks a season, treat single-opponent predictions as informed guesses, not ground truth — the system reports a confidence level (LOW/MEDIUM/HIGH) per opponent for exactly this reason. |
| Opponent elimination correlation (this week) | **Exact**, given the predicted pick distribution | If several opponents are predicted on the same team, their fates are tied to that one game, not modeled as independent coin flips. Computed via generating-function convolution across the week's games. |
| Pool Equity ranking (Ranking B), this week's mechanics | **Exact**, given the same predicted distribution | Elimination correlation is conditioned on *my own* candidate pick's outcome too, so a heavily-owned pick correctly shows less differentiation value than a lightly-owned one. |
| P(win pool) | **Approximate, clearly labeled** | A proportional-survival-share heuristic: `P(survive this week) × [my continuation value / (my continuation value + expected competing survival value)]`. Exact multi-week joint opponent simulation across a full season is combinatorially explosive; this is the standard fallback used by real survivor-equity calculators. Treat it as directionally useful for comparing this week's candidates, not as a calibrated probability. |
| Opponents' future survival (weeks beyond this one) | **Approximate** | Each opponent's future survival is proxied as `(their historical average win-probability-of-pick) ^ (remaining weeks)`, shrunk to a population default (65%) with little data. Only feeds the P(win pool) approximation above, never the current week's exact mechanics. |

Nothing here silently blends exact and approximate numbers into a single
opaque score — the report always shows Ranking A (fully exact) and Ranking B
(exact this-week mechanics + one clearly-labeled approximate figure)
side by side.

## Where the probabilities come from

Earlier iterations of this system read raw Vegas odds directly (moneylines,
decimal odds) and did its own de-vig math, and at one point also tried to
detect and flag stale lines itself. Both of those are gone now. A separate,
already-completed project (the probability-grid engine) owns that entire
problem: it ingests raw lines, updates Bradley-Terry-style team ratings via
a Kalman filter, anchors to preseason win totals, applies a bounded and
backtested stale-line adjustment when a game's line hasn't refreshed
recently, and publishes one finished win probability per game, every week,
for the entire season (real historical backtesting improved its stale-line
error 6.6%→11.7% on a held-out sample). Redoing any part of that here would
be duplicating work that's already done, and done better, upstream.

So: this system reads `Away Win %`/`Home Win %` straight from that engine's
**Master Game Table** output and uses them as-is via `Game.from_probability()`
(`survivor/vegas.py`). No conversion, no staleness adjustment, no confidence
penalty based on how old a line is -- if you want that kind of signal, it
lives in the grid engine's own `Source`/`Line Age`/`Adjustment` columns,
which this system deliberately doesn't read. Only `Week`, `Away Team`, `Home Team`,
`Away Win %`, `Home Win %` (plus `Completed` for week detection and `Game ID`
for abbreviation matching) are used.

The old raw-odds path (American moneylines, decimal odds, `no_vig_probs()`)
still exists in `vegas.py` as generic utility code, in case you ever want to
feed this system from some other odds source directly. It's just no longer
the primary path, and nothing in `data_io.py` calls it by default.

## Package layout

```
survivor/
  vegas.py          Game/WeekSlate representation; probability passthrough
                    is primary, raw-odds (American/decimal) conversion kept
                    as unused-by-default utility code
  team_registry.py  Bitmask team-availability representation
  opponents.py       Conditional-logit opponent model, persona features, shrinkage
  pool_math.py       Convolution-based joint elimination distribution
  dp.py               Pure-survival DP, opportunity cost, pool-equity evaluation
  workflow.py         SurvivorPool: the decision/update-phase state machine
  data_io.py          Google Sheets (live) + CSV loading of Master Game Table
                      + Pools Picks, team-name matching, current-week
                      detection, problem reporting
  config.py           Reads config.toml
  cli.py              The `check` / `recommend` commands
  report.py           Formats a recommendation into the Top-5 report structure
tests/                66 tests covering DP correctness, opponent
                      modeling/shrinkage, correlated elimination,
                      game-theoretic ranking shifts, workflow invariants,
                      the Master-Game-Table/Pools-Picks parsers, name
                      matching, week detection, guardrails, config, CLI,
                      and full-season performance
example_data/         Real 2026 week 1-3 data transcribed from the actual
                      Master Game Table / Pools Picks tabs
config.toml           The two settings you edit
example_usage.py       Python-API example (optional; the CLI is the normal path)
```

## Running the tests

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

66 tests, covering (mapped to spec section 24, plus usability):
- Basic DP: single/multi-week, no-reuse, used-team removal, horizon stop
- Future value: DP prefers a lower-probability team today when it unlocks a much better future state; opportunity cost is positive/near-zero as appropriate
- Opponent modeling: single/multiple opponents, count-vector aggregation, chalk detection, small-sample shrinkage toward the population prior, persona shift after unexpected picks
- Correlated elimination: hand-verified joint distribution, "7 opponents on one team live or die together" (never partial), and the conditional-on-my-pick distribution
- Game theory: the pool-equity-optimal pick changes with opponent ownership concentration, while the pure-survival ranking does not
- Workflow: recommending never mutates state, advancing requires a recorded pick, refuses only on genuinely-missing data, opponent updates only happen on explicit calls, probability conservation throughout
- Vegas: probability passthrough (the primary path) plus the legacy American/decimal odds conversion (kept as utility code)
- Data I/O: Master Game Table parsing (percentage-string and numeric-fraction formats, missing-side inference, malformed-row tolerance), wide-format pick-sheet parsing, elimination markers, and the no-look-ahead week cutoff
- Usability: nickname/abbreviation matching (and refusing to guess ambiguous names), current-week auto-detection, blocking vs. warning problem reports, blank-header-tolerant sheet reading, config path resolution, both CLI commands, and full-18-week solve time

## Known limitations (be aware of these)

- **Opponent model needs data to be useful.** Early season (or for any
  opponent with <3 observed picks), predictions lean heavily on the
  population prior and confidence is reported as LOW — by design, per the
  spec's caution against overfitting ~18 observations a season.
- **P(win pool) is a heuristic**, not a calibrated probability — see the
  table above. It's most useful for *relative* comparison between this
  week's candidates, less so as a literal percentage.
- **Multi-week lookahead beyond the current week is approximate** for the
  pool-equity objective (though exact for the pure-survival objective,
  which has no such limitation). Since the workflow recomputes every week
  with fresh data anyway, this only affects the diagnostic view of the
  future, not the actionable current-week recommendation.
- **This system trusts the grid engine's probabilities completely.** If
  something upstream is wrong (a bad rating update, a data-entry error in
  the sheet), it will flow straight through with no independent check —
  by design, since re-validating an already-validated system was explicitly
  cut from this project's scope. If that ever feels risky, relaying the
  grid's own `Source`/`Adjustment` columns into the confidence report would be a lightweight way to add some
  visibility back, without rebuilding any of the staleness logic itself.
