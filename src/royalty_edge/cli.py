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
    sd.add_argument("--page-size", type=int, default=None)
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
