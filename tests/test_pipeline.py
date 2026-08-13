"""Pipeline tests against a fixture-backed fake fetcher.

There is no network here, so the transport is stubbed and everything above it
-- pagination via cursor, queueing, resumability, auth detection, reconcile
gating, the L2 rebuild -- is exercised for real.
"""
import gzip
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from royalty_edge.db.load import connect, rebuild_analytic
from royalty_edge.fetch.client import FetchResult
from royalty_edge.fetch.discover import discover, queue_details_from_index
from royalty_edge.fetch.harvest import harvest, pending_count
from royalty_edge.fetch.probe import probe_detail_url, probe_page_size

FIX = Path(__file__).resolve().parents[1] / "fixtures"
DETAIL_TMPL = "https://example.test/inventory/listings/{id}/"


class FakeFetcher:
    """Serves fixtures. Page 1 -> page 2 via the real `next` cursor rewritten
    to a local sentinel; every detail id returns the 6792 payload."""

    def __init__(self, tmp: Path, *, detail_ok=True, auth_fail_after=None,
                 page_size_cap=100):
        self.tmp = tmp
        self.detail_ok = detail_ok
        self.auth_fail_after = auth_fail_after
        self.page_size_cap = page_size_cap
        self.calls = []
        self._p1 = (FIX / "index_page1.clean.json").read_bytes()
        self._p2 = (FIX / "index_page2.clean.json").read_bytes()
        self._detail = (FIX / "listing_detail.clean.json").read_bytes()

    def _store(self, url, body, status=200):
        sha = hashlib.sha256(body).hexdigest()
        p = self.tmp / f"{sha}.gz"
        p.write_bytes(gzip.compress(body))
        return FetchResult(url=url, fetched_at=datetime.now(timezone.utc),
                           http_status=status, content_type="application/json",
                           content_sha256=sha, content_bytes=len(body),
                           payload_path=str(p), from_cache=False)

    def read_payload(self, path):
        return gzip.decompress(Path(path).read_bytes())

    def fetch(self, url, *, params=None, method="GET", json_body=None):
        self.calls.append((url, dict(params) if params else None))
        n = len(self.calls)
        if self.auth_fail_after and n > self.auth_fail_after:
            return self._store(url, b"<!DOCTYPE html><html>login</html>", 200)
        if "/listings/" in url and url.rstrip("/").split("/")[-1].isdigit():
            if not self.detail_ok:
                return self._store(url, b'{"detail":"Not found."}', 404)
            return self._store(url, self._detail)
        # index
        if params and dict(params).get("page") == "2" or "page=2" in url:
            body = self._p2
        else:
            body = self._p1
        return self._store(url, body)


@pytest.fixture
def tmpfetch(tmp_path):
    return FakeFetcher(tmp_path)


@pytest.fixture
def con():
    c = connect(":memory:")
    yield c
    c.close()


def _run(con, name="r1"):
    con.execute("INSERT INTO scrape_run (run_id, started_at, status) VALUES (?, now(), 'running')", [name])
    return name


def test_probe_detects_detail_endpoint(tmpfetch):
    tmpl, notes = probe_detail_url(tmpfetch, listing_id=6792,
                                   candidates=[DETAIL_TMPL])
    assert tmpl == DETAIL_TMPL
    assert any("OK" in n for n in notes)


def test_probe_rejects_wrong_listing(tmp_path):
    """A 200 that parses but returns a different listing must be rejected."""
    f = FakeFetcher(tmp_path)
    tmpl, notes = probe_detail_url(f, listing_id=9999, candidates=[DETAIL_TMPL])
    assert tmpl is None
    assert any("expected 9999" in n for n in notes)


def test_probe_page_size_reports_cap(tmpfetch):
    best, total, notes = probe_page_size(tmpfetch, candidates=(15,))
    assert total == 2528
    assert notes


def test_discover_follows_cursor_and_queues(con, tmpfetch):
    run = _run(con)
    stats = discover(con, tmpfetch, run_id=run, page_size=15,
                     detail_url_template=DETAIL_TMPL, max_pages=2)
    assert stats.pages == 2
    assert stats.rows == 30
    assert stats.total_results == 2528
    assert stats.queued == 30
    assert pending_count(con) == 30
    # second page was fetched via the cursor URL, not rebuilt params
    assert tmpfetch.calls[1][1] is None


def test_discover_is_idempotent_on_queue(con, tmpfetch):
    run = _run(con)
    discover(con, tmpfetch, run_id=run, page_size=15,
             detail_url_template=DETAIL_TMPL, max_pages=2)
    before = pending_count(con)
    discover(con, tmpfetch, run_id=run, page_size=15,
             detail_url_template=DETAIL_TMPL, max_pages=2)
    assert pending_count(con) == before   # re-running does not duplicate work


def test_discover_stops_on_auth_failure(con, tmp_path):
    f = FakeFetcher(tmp_path, auth_fail_after=1)
    run = _run(con)
    stats = discover(con, f, run_id=run, page_size=15,
                     detail_url_template=DETAIL_TMPL, max_pages=5)
    assert stats.stopped_reason == "auth_failure"
    assert stats.pages == 1


def test_harvest_drains_queue(con, tmpfetch):
    run = _run(con)
    discover(con, tmpfetch, run_id=run, page_size=15,
             detail_url_template=DETAIL_TMPL, max_pages=2)
    stats = harvest(con, tmpfetch, run_id=run, limit=5)
    assert stats.ok == 5
    assert stats.failed == 0
    assert pending_count(con) == 25
    assert con.execute("SELECT count(*) FROM obs_listing").fetchone()[0] == 5


def test_harvest_resumes_where_it_stopped(con, tmpfetch):
    run = _run(con)
    discover(con, tmpfetch, run_id=run, page_size=15,
             detail_url_template=DETAIL_TMPL, max_pages=2)
    harvest(con, tmpfetch, run_id=run, limit=5)
    harvest(con, tmpfetch, run_id=run, limit=5)
    assert pending_count(con) == 20
    assert con.execute("SELECT count(*) FROM obs_listing").fetchone()[0] == 10


def test_harvest_aborts_run_on_auth_expiry(con, tmp_path):
    """A cookie dying mid-crawl must not burn the rest of the queue."""
    f = FakeFetcher(tmp_path)
    run = _run(con)
    discover(con, f, run_id=run, page_size=15,
             detail_url_template=DETAIL_TMPL, max_pages=2)
    f.auth_fail_after = len(f.calls) + 3
    stats = harvest(con, f, run_id=run)
    assert stats.stopped_reason == "auth_failure"
    assert stats.ok == 3
    assert pending_count(con) >= 26   # queue preserved, not marked failed


def test_harvest_marks_http_errors_failed(con, tmp_path):
    f = FakeFetcher(tmp_path, detail_ok=False)
    run = _run(con)
    discover(con, f, run_id=run, page_size=15,
             detail_url_template=DETAIL_TMPL, max_pages=1)
    stats = harvest(con, f, run_id=run, limit=3)
    assert stats.ok == 0 and stats.failed == 3
    assert con.execute(
        "SELECT count(*) FROM fetch_queue WHERE state='failed'").fetchone()[0] == 3


def test_queue_backfill_from_stored_index(con, tmpfetch):
    run = _run(con)
    discover(con, tmpfetch, run_id=run, page_size=15,
             detail_url_template=None, max_pages=2)   # endpoint unknown at the time
    assert pending_count(con) == 0
    n = queue_details_from_index(con, detail_url_template=DETAIL_TMPL)
    assert n == 30


def test_full_pipeline_to_analytic_layer(con, tmpfetch):
    run = _run(con)
    discover(con, tmpfetch, run_id=run, page_size=15,
             detail_url_template=DETAIL_TMPL, max_pages=2)
    harvest(con, tmpfetch, run_id=run, limit=3)
    rebuild_analytic(con)
    assert con.execute("SELECT count(*) FROM dim_listing").fetchone()[0] == 30
    sold = con.execute("SELECT count(*) FROM fact_outcome WHERE sold").fetchone()[0]
    assert sold > 0
    # fixed_return lots are excluded from the modeling sample
    assert con.execute("SELECT count(*) FROM data_quality_flag "
                       "WHERE flag='fixed_return_instrument'").fetchone()[0] == 6
    # detail rows carry an offer ladder
    assert con.execute("SELECT count(*) FROM fact_offer").fetchone()[0] > 0
