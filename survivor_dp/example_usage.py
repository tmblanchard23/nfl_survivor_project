"""
Runs through the full weekly workflow (spec section 16) once, end to end,
using the synthetic data in example_data/ -- which is a faithful excerpt of
the real "Master Game Table" tab your probability-grid engine produces
(real 2026 week 1-3 games and probabilities).

Note what ISN'T happening here, compared to earlier versions of this
system: no moneyline-to-probability conversion, no de-vig math, no line-
staleness computation. The grid engine already did all of that upstream --
rating updates, anchoring, stale-line adjustment, backtested separately on
real historical data. This script just reads the probabilities it produced
and runs the DP on them.

  1. Load current state from a CSV export of Master Game Table + Pools Picks
     (or, in real use, straight from the live Google Sheet -- see below).
  2. Get this week's recommendations (Ranking A + Ranking B, Top 5 each).
  3. Print the formatted report.
  4. STOP -- nothing is recorded automatically.
  5. Simulate the user making a pick and opponents' actual picks/results
     coming in later, to show the post-decision update phase.
  6. Confirm get_recommendations() refuses to run again until next week's
     probabilities are loaded.

Run with:  PYTHONPATH=. python3 example_usage.py

To point this at your LIVE Google Sheet instead of CSVs, replace the
data_io.load_pool_from_csv_dir(...) call below with:

    pool = data_io.load_pool_from_google_sheet(
        spreadsheet_id="<your sheet id from the URL>",
        current_week=CURRENT_WEEK,
        credentials_path="service_account.json",
    )

(see README.md for the one-time Google service-account setup)
"""
from survivor import data_io, report

CURRENT_WEEK = 3


def main():
    pool = data_io.load_pool_from_csv_dir("example_data", current_week=CURRENT_WEEK)

    print(f"Loaded pool. Current week: {pool.current_week}")
    print(f"My used teams: {pool.registry.teams_from_mask(pool.my_used_mask)}")
    print(f"Alive opponents: {list(pool.alive_opponents().keys())}")
    print(f"Eliminated opponents: "
          f"{[oid for oid, o in pool.opponents.items() if o.eliminated]}")
    for oid, opp in pool.alive_opponents().items():
        print(f"  {oid}: persona='{opp.persona_label()}', "
              f"confidence={opp.confidence()}, features={opp.descriptive_features()}")
    print()

    games_this_week = pool.future_slates[CURRENT_WEEK].teams()
    print(f"Teams in week {CURRENT_WEEK}'s Master Game Table rows: {games_this_week}")
    print("(Only 2 of week 3's games have been played/priced as of this "
          "snapshot -- the grid engine will add the rest as its own weekly "
          "run progresses; this script just uses whatever's there.)\n")

    # ---- DECISION PHASE ----
    rec = pool.get_recommendations(top_n=5)
    pool_size_before = len(pool.alive_opponents())
    print(report.full_report(rec, pool_size_before))

    # ---- STOP. Nothing above changed any state. ----
    assert pool.current_week == CURRENT_WEEK

    # ---- POST-DECISION UPDATE PHASE (simulated) ----
    print("\n\n--- Simulating the post-decision update phase ---\n")
    my_actual_pick = rec.pool_equity_top[0].team
    print(f"I actually picked: {my_actual_pick}")
    pool.record_my_pick(my_actual_pick)

    actual_opponent_picks = {"Player 1": "Green Bay Packers"}
    pool.record_opponent_picks(actual_opponent_picks)
    print(f"Recorded opponent picks: {actual_opponent_picks}")

    results = {"Green Bay Packers": True, "Atlanta Falcons": False}
    pool.record_results(results)
    print(f"Recorded results: {results}")

    pool.advance_week()
    print(f"\nAdvanced to week {pool.current_week}. "
          f"My used teams now: {pool.registry.teams_from_mask(pool.my_used_mask)}")

    # ---- Confirm it refuses to compute the next week with no data at all ----
    try:
        pool.get_recommendations()
        print("ERROR: should not have been able to generate a recommendation "
              "for a week with no Master Game Table rows loaded!")
    except ValueError as e:
        print(f"\nFor week {pool.current_week}, which has no rows in this "
              f"example's CSV, it correctly refuses:\n  {e}")

    data_io.save_state(pool, "pool_state.pkl")
    print("\nSaved state to pool_state.pkl for next week's run.")


if __name__ == "__main__":
    main()
