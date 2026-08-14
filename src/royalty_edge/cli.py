"""royalty_edge command line.

    python -m royalty_edge.cli probe --listing-id 6792
    python -m royalty_edge.cli discover --page-size 100
    python -m royalty_edge.cli harvest
    python -m royalty_edge.cli rebuild
    python -m royalty_edge.cli report

Harvest and rebuild are separate commands because DuckDB holds an exclusive
file lock. A two-hour crawl holding that lock would block every notebook you
have open; instead the crawl writes payloads and short transactions, and the
rebuild takes the lock for seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .db.load import connect, rebuild_analytic
from .fetch.client import FetchConfig, PoliteFetcher
from .fetch.discover import discover, queue_details_from_index
from .fetch.harvest import failure_report, harvest, pending_count, queue_report
from .fetch.probe import run_probe
from .fetch.session import build_session, load_cookie
from .model.features import FrameSpec, load_frame, FUNDAMENTALS, SELLER_ASK, ANCHOR
from .model.pricing import (anchor_analysis, fit_ladder, ladder_table,
                            residual_screen, spike_check, temporal_holdout)
from .model.decay import (decompose_composition, diagnose_reporting_lag, fit_all,
                         load_panels, market_drift, pooled_age_profile,
                         select_best, shrink_estimates, to_quarterly,
                         trim_partial_tail)
from .model.valuation import (compare_to_market, rate_sensitivity,
                             scenario_spread, value_all)

DEFAULT_DB = "data/royalty_edge.duckdb"
DEFAULT_LANDING = "landing"
CONFIG_PATH = Path("config/endpoints.json")


def _fetcher(args) -> PoliteFetcher:
    cookie = load_cookie(args.cookie_file)
    if not cookie:
        logging.warning("no session cookie found; authenticated endpoints will "
                        "return HTML shells. See fetch/session.py for how to export one.")
    cfg = FetchConfig(
        landing_dir=Path(args.landing),
        min_interval_s=args.interval,
        respect_robots=not args.ignore_robots,
    )
    return PoliteFetcher(cfg, session=build_session(cookie))


def _start_run(con, note: str) -> str:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    con.execute(
        "INSERT INTO scrape_run (run_id, started_at, code_version, status, notes) "
        "VALUES (?,?,?,?,?)",
        [run_id, datetime.now(timezone.utc), _git_sha(), "running", note])
    return run_id


def _finish_run(con, run_id: str, status: str) -> None:
    con.execute(
        "UPDATE scrape_run SET finished_at = ?, status = ?, "
        "n_requests = (SELECT count(*) FROM raw_fetch WHERE run_id = ?) "
        "WHERE run_id = ?",
        [datetime.now(timezone.utc), status, run_id, run_id])


def _git_sha() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")


# --------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------

def cmd_probe(args) -> int:
    fetcher = _fetcher(args)
    result = run_probe(fetcher, listing_id=args.listing_id)
    for n in result.notes:
        print("  " + n)
    print()
    if not result.ok:
        print("Detail endpoint NOT resolved. None of the candidates returned a "
              "payload that parses and reconciles.")
        print("Next step: open a listing in the browser with the network tab on "
              "Fetch/XHR, find the request that returns the offers array, and add "
              "its URL to DETAIL_URL_CANDIDATES in fetch/probe.py.")
        return 1
    cfg = _load_config()
    cfg.update({
        "detail_url_template": result.detail_url_template,
        "max_page_size": result.max_page_size,
        "total_results": result.total_results,
        "probed_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_config(cfg)
    print(f"detail endpoint : {result.detail_url_template}")
    print(f"max page_size   : {result.max_page_size}")
    print(f"total listings  : {result.total_results}")
    est = (result.total_results or 0) / max(result.max_page_size, 1)
    print(f"index pages     : ~{est:.0f}  "
          f"(~{est * args.interval / 60:.1f} min at {args.interval}s/req)")
    print(f"detail requests : ~{result.total_results}  "
          f"(~{(result.total_results or 0) * args.interval / 3600:.1f} h)")
    print(f"\nwrote {CONFIG_PATH}")
    return 0


def cmd_discover(args) -> int:
    cfg = _load_config()
    tmpl = args.detail_url_template or cfg.get("detail_url_template")
    page_size = args.page_size or cfg.get("max_page_size", 15)
    if not tmpl:
        logging.warning("no detail_url_template; index rows will be collected but "
                        "no detail fetches queued. Run `probe` first.")
    con = connect(args.db)
    run_id = _start_run(con, "discover")
    try:
        stats = discover(con, _fetcher(args), run_id=run_id, page_size=page_size,
                         detail_url_template=tmpl, max_pages=args.max_pages)
        _finish_run(con, run_id, stats.stopped_reason)
        print(f"pages={stats.pages} rows={stats.rows} queued={stats.queued} "
              f"total_results={stats.total_results} ({stats.stopped_reason})")
        return 0 if stats.stopped_reason in ("complete", "max_pages") else 1
    finally:
        con.close()


def cmd_queue(args) -> int:
    cfg = _load_config()
    tmpl = args.detail_url_template or cfg.get("detail_url_template")
    if not tmpl:
        print("no detail_url_template; run `probe` first")
        return 1
    con = connect(args.db)
    try:
        n = queue_details_from_index(con, detail_url_template=tmpl,
                                     only_missing=not args.all)
        print(f"queued {n} detail fetches; {pending_count(con)} pending total")
        return 0
    finally:
        con.close()


def cmd_harvest(args) -> int:
    con = connect(args.db)
    run_id = _start_run(con, "harvest")
    try:
        stats = harvest(con, _fetcher(args), run_id=run_id, limit=args.limit,
                        max_runtime_s=args.max_runtime * 60 if args.max_runtime else None,
                        retry_failed=args.retry_failed)
        _finish_run(con, run_id, stats.stopped_reason)
        print(f"ok={stats.ok} failed={stats.failed} "
              f"reconcile_warnings={stats.reconcile_warnings} "
              f"pending={pending_count(con)} ({stats.stopped_reason})")
        return 0 if stats.stopped_reason != "auth_failure" else 1
    finally:
        con.close()


def cmd_rebuild(args) -> int:
    con = connect(args.db)
    try:
        rebuild_analytic(con, all_access=args.all_access)
        n = con.execute("SELECT count(*) FROM dim_listing").fetchone()[0]
        sold = con.execute("SELECT count(*) FROM fact_outcome WHERE sold").fetchone()[0]
        print(f"rebuilt: {n} listings, {sold} sold")
        return 0
    finally:
        con.close()


def cmd_report(args) -> int:
    con = connect(args.db, read_only=True)
    try:
        def show(title, sql):
            print(f"\n=== {title} ===")
            df = con.execute(sql).fetchdf()
            print(df.to_string(index=False) if len(df) else "(empty)")

        print("=== queue ===")
        for r in queue_report(con):
            print(f"  {r[0]:10s} {r[1]:>6}")
        fails = failure_report(con)
        if fails:
            print("\n=== recent failures ===")
            for lid, att, err in fails:
                print(f"  {lid}  attempts={att}  {err}")

        show("coverage by year", "SELECT * FROM v_field_coverage")
        show("dollar age normalization", "SELECT * FROM v_dollar_age_check")
        show("quality flags",
             "SELECT flag, severity, count(*) n FROM data_quality_flag "
             "GROUP BY 1,2 ORDER BY 3 DESC")
        show("repeat sales", "SELECT count(*) n_pairs, "
             "round(median(years_held),2) med_years, round(median(ltm_ratio),3) med_ltm_ratio, "
             "round(median(price_ratio),3) med_price_ratio FROM v_repeat_sales")
        show("clearing multiple by term family",
             "SELECT term_family, count(*) n, "
             "round(median(multiple_gross),2) med_mult, "
             "round(median(multiple_vs_median),3) med_vs_anchor, "
             "round(median(n_unique_bidders),1) med_bidders "
             "FROM v_pricing_frame WHERE sold GROUP BY 1 ORDER BY 2 DESC")
        return 0
    finally:
        con.close()



def cmd_pricing(args) -> int:
    """Phase 2: fit the clearing-multiple model and report what the market prices."""
    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    con = connect(args.db, read_only=True)
    try:
        spec = FrameSpec(min_deal_year=args.min_year,
                         include_legacy_auctions=args.include_legacy)
        df, attrition = load_frame(con, spec)
        print(f"=== sample: {spec.describe()} ===")
        print(attrition.to_string(index=False))
        if len(df) < 100:
            print(f"\nonly {len(df)} rows survive; loosen the spec before fitting")
            return 1

        print(f"\n=== model ladder (n={len(df)}) ===")
        results = fit_ladder(df)
        print(ladder_table(results).to_string(index=False))

        print("\n=== M3 coefficients (the pre-bid model) ===")
        print(results[3].params.to_string())

        print("\n=== temporal holdout ===")
        for name, feats in [("M1 fundamentals", FUNDAMENTALS),
                            ("M3 + reserve + anchor", FUNDAMENTALS + SELLER_ASK + ANCHOR)]:
            h = temporal_holdout(df, feats, cutoff=args.cutoff)
            if "error" in h:
                print(f"  {name}: {h['error']}")
                continue
            print(f"  {name}: test R2={h['r2_test']:.3f} "
                  f"skill_vs_naive={h['skill_vs_naive']:.3f} "
                  f"median_abs_err={h['median_abs_pct_err']:.1%} "
                  f"(train {h['n_train']}, test {h['n_test']})")

        print("\n=== anchoring ===")
        for k, v in anchor_analysis(df).items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        print("\n=== sync-spike check: does the market normalize LTM? ===")
        print(spike_check(df).to_string(index=False))

        print(f"\n=== most underpriced vs M3 (in-sample, hypothesis generator) ===")
        scr = residual_screen(df, results[3])
        cols = ["listing_id", "title", "deal_date", "ltm", "dollar_age",
                "multiple_gross", "predicted_multiple", "mispricing_pct"]
        print(scr.head(args.top_n)[cols].to_string(index=False))
        print(f"\n=== most overpriced vs M3 ===")
        print(scr.tail(args.top_n)[cols].to_string(index=False))

        if args.out:
            df.to_parquet(args.out)
            scr.to_parquet(args.out.replace(".parquet", "_residuals.parquet"))
            print(f"\nwrote {args.out} and residuals")
        return 0
    finally:
        con.close()



def cmd_decay(args) -> int:
    """Phase 3: fit decay curves and produce fair-value multiples."""
    import pandas as pd
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    con = connect(args.db, read_only=True)
    try:
        raw = load_panels(con)
        if raw.empty:
            print("no earnings panels; run harvest and rebuild first")
            return 1
        print(f"=== panels: {raw['listing_id'].nunique()} catalogs, {len(raw)} rows ===")
        q = to_quarterly(raw)
        print(q.groupby("grain_source")["listing_id"].nunique()
                .rename("catalogs").to_frame().to_string())

        print("\n=== reporting-lag diagnostic ===")
        print("(ratio of each trailing quarter to that panel's own prior-year median;")
        print(" values well below the deeper plateau mean the statement is incomplete)")
        diag = diagnose_reporting_lag(q)
        print(diag.to_string(index=False) if not diag.empty else "(insufficient data)")
        q_trim, n_trim = trim_partial_tail(q, n_trim=args.trim)
        print(f"-> trimming {n_trim} trailing quarter(s) from every panel")

        print(f"\n=== fitting {len(q_trim.groupby('listing_id'))} catalogs x 4 forms ===")
        fits = fit_all(q_trim, min_obs=args.min_obs)
        if fits.empty:
            print("no catalog had enough observations to fit")
            return 1
        best = select_best(fits)
        print(best["form"].value_counts().rename("catalogs").to_frame().to_string())
        print("\nfit quality by chosen form:")
        print(best.groupby("form").agg(
            n=("listing_id", "size"),
            med_rmse_log=("rmse_log", "median"),
            med_cv_rmse=("cv_rmse", "median"),
            med_n_obs=("n_obs", "median"),
            med_5y_retention=("implied_5y_retention", "median"),
        ).round(3).to_string())

        print("\n=== pooled age profile (net of market drift) ===")
        prof = pooled_age_profile(q_trim)
        print(prof.round(4).to_string(index=False) if not prof.empty
              else "(too few catalogs)")

        print("\n=== market-wide calendar drift ===")
        print("(common component; decline here is streaming economics, not your catalog)")
        drift = market_drift(q_trim)
        if not drift.empty:
            print(drift.tail(12).round(4).to_string(index=False))

        meta = (q_trim.groupby("listing_id").first().reset_index()
                [["listing_id", "ltm", "three_years_average", "dollar_age",
                  "track_count", "term_family", "term_years"]])
        import numpy as _np
        meta["log_track_count"] = _np.log(meta["track_count"].fillna(1).clip(lower=1))
        meta["share_streaming"] = 0.0
        shrunk = shrink_estimates(best, meta)
        if not shrunk.empty:
            print("\n=== 5-year retention: raw vs partially pooled ===")
            print(shrunk[["retention_5y_raw", "retention_5y_shrunk",
                          "shrink_weight"]].describe().round(3).to_string())

        print("\n=== valuing every fitted catalog ===")
        vals = value_all(best, meta, rates=(0.08, 0.12, 0.18))
        if vals.empty:
            print("no valuations produced")
            return 1
        print(scenario_spread(vals, rate=args.rate).to_string(index=False))
        print("\nrate sensitivity (fair multiple at 8% vs 18%):")
        rs = rate_sensitivity(vals)
        if not rs.empty:
            print(rs.to_string())

        spec = FrameSpec(min_deal_year=args.min_year)
        frame, _ = load_frame(con, spec)
        cmp_ = compare_to_market(vals, frame, rate=args.rate, scenario="base")
        if not cmp_.empty:
            cols = [c for c in ["listing_id", "title", "deal_date", "ltm",
                                "dollar_age", "multiple_gross", "fair_multiple_ltm",
                                "walk_away_multiple", "edge_pct"] if c in cmp_.columns]
            print(f"\n=== model says cheapest vs realized clearing (rate={args.rate:.0%}) ===")
            print(cmp_.head(args.top_n)[cols].to_string(index=False))
            print("\n=== model says most expensive ===")
            print(cmp_.tail(args.top_n)[cols].to_string(index=False))
            print(f"\nmedian edge_pct across {len(cmp_)} valued lots: "
                  f"{cmp_['edge_pct'].median():.1%}")
            print("A median far from zero means the model and the market disagree")
            print("systematically -- suspect the discount rate or the horizon before")
            print("concluding the whole market is mispriced.")

        if args.out:
            best.to_parquet(args.out)
            vals.to_parquet(args.out.replace(".parquet", "_valuations.parquet"))
            cmp_.to_parquet(args.out.replace(".parquet", "_vs_market.parquet"))
            print(f"\nwrote {args.out} + valuations + vs_market")
        return 0
    finally:
        con.close()


# --------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="royalty_edge")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--landing", default=DEFAULT_LANDING)
    p.add_argument("--cookie-file", default=None)
    p.add_argument("--interval", type=float, default=3.0,
                   help="seconds between requests (default 3)")
    p.add_argument("--ignore-robots", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="resolve detail URL and max page_size")
    sp.add_argument("--listing-id", type=int, default=6792)
    sp.set_defaults(func=cmd_probe)

    sd = sub.add_parser("discover", help="walk the index, queue detail fetches")
    sd.add_argument("--page-size", type=int, default=500,
                    help="listings per index request (default 500; server accepts up to 500)")
    sd.add_argument("--max-pages", type=int, default=None)
    sd.add_argument("--detail-url-template", default=None)
    sd.set_defaults(func=cmd_discover)

    sq = sub.add_parser("queue", help="backfill detail queue from stored index rows")
    sq.add_argument("--detail-url-template", default=None)
    sq.add_argument("--all", action="store_true", help="requeue listings already fetched")
    sq.set_defaults(func=cmd_queue)

    sh = sub.add_parser("harvest", help="drain the detail queue")
    sh.add_argument("--limit", type=int, default=None)
    sh.add_argument("--max-runtime", type=float, default=None, help="minutes")
    sh.add_argument("--retry-failed", action="store_true")
    sh.set_defaults(func=cmd_harvest)

    sr = sub.add_parser("rebuild", help="rebuild the analytic layer")
    sr.add_argument("--all-access", action="store_true",
                    help="compute fee-loaded multiples assuming fees are waived")
    sr.set_defaults(func=cmd_rebuild)

    sp2 = sub.add_parser("pricing", help="Phase 2: clearing-multiple regression")
    sp2.add_argument("--min-year", type=int, default=2020)
    sp2.add_argument("--cutoff", default="2025-01-01",
                     help="temporal holdout split date")
    sp2.add_argument("--include-legacy", action="store_true",
                     help="include pre-2022 auction-mechanism listings")
    sp2.add_argument("--top-n", type=int, default=15)
    sp2.add_argument("--out", default=None, help="write frame to parquet")
    sp2.set_defaults(func=cmd_pricing)

    sp3 = sub.add_parser("decay", help="Phase 3: decay curves and fair value")
    sp3.add_argument("--min-obs", type=int, default=8,
                     help="minimum quarters required to fit a catalog")
    sp3.add_argument("--trim", type=int, default=None,
                     help="trailing quarters to drop; default auto-detect")
    sp3.add_argument("--rate", type=float, default=0.12, help="discount rate")
    sp3.add_argument("--min-year", type=int, default=2020)
    sp3.add_argument("--top-n", type=int, default=15)
    sp3.add_argument("--out", default=None, help="write fits to parquet")
    sp3.set_defaults(func=cmd_decay)

    srep = sub.add_parser("report", help="coverage, quality and queue status")
    srep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
