"""Regression tests pinned to the payloads captured 2026-08-11.

These are the contract with the platform's API. When the site changes shape,
these fail first and loudly, which is the point.
"""
import json
from pathlib import Path

import pytest

from royalty_edge.parse.listing import (
    parse_index_payload, parse_listing_payload, reconcile,
    term_family, term_years, is_partial_share, buyer_fee,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def index_raw():
    return (FIX / "index_page1.clean.json").read_bytes()


@pytest.fixture(scope="module")
def detail_raw():
    return (FIX / "listing_detail.clean.json").read_bytes()


def test_index_shape(index_raw):
    rows, nxt, meta = parse_index_payload(index_raw)
    assert len(rows) == 15
    assert meta["total_results"] == 2528
    assert nxt and "page=2" in nxt
    assert rows[0].listing_id == 6242
    assert rows[0].fields["kind"] == "secondary_listing"
    assert rows[0].fields["ltm"] == pytest.approx(1774.58)


def test_detail_scalars(detail_raw):
    o = parse_listing_payload(detail_raw)
    assert o.listing_id == 6792
    assert o.fields["asset_id"] == 39967
    assert o.fields["ltm"] == pytest.approx(2899.08)
    assert o.fields["three_years_average"] == pytest.approx(2838.3)
    assert o.fields["minimum_price"] == pytest.approx(3050.0)
    assert o.fields["unique_bidder_count"] == 13
    assert o.fields["deal_amount"] == pytest.approx(19900.0)


def test_offer_ladder(detail_raw):
    o = parse_listing_payload(detail_raw)
    assert len(o.offers) == 38
    assert len({x.buyer_id for x in o.offers}) == 13
    winners = [x for x in o.offers if x.state == "awaiting_transfer"]
    assert len(winners) == 1 and winners[0].amount == pytest.approx(19900.0)
    losing = [x.amount for x in o.offers if x.state != "awaiting_transfer"]
    assert max(losing) == pytest.approx(19700.0)   # second price


def test_quarterly_panel_reconciles(detail_raw):
    """The panel must sum to lifetime and its last four quarters to LTM.
    If this ever fails the decay sample is silently corrupted."""
    o = parse_listing_payload(detail_raw)
    assert len(o.quarters) == 27
    assert reconcile(o) == []


def test_annual_panel_dimensions(detail_raw):
    o = parse_listing_payload(detail_raw)
    dims = {y.dimension for y in o.years}
    assert dims == {"song", "income_type", "source", "music_user"}
    # year_5 income_type total must equal LTM
    y5 = sum(y.amount for y in o.years
             if y.dimension == "income_type" and y.year_index == 5)
    assert y5 == pytest.approx(o.fields["ltm"], rel=1e-4)


def test_term_taxonomy():
    assert term_family("life_of_rights") == "perpetual"
    assert term_family("fixed_return") == "fixed_return"
    assert term_family("30_year") == "fixed_term"
    assert term_family("partial_10_year") == "fixed_term"
    assert term_years("partial_10_year") == 10.0
    assert is_partial_share("partial_10_year")
    assert not is_partial_share("10_year")


def test_buyer_fee_waivers():
    assert buyer_fee(10_000) == 500.0          # minimum binds, 5% of capital
    assert buyer_fee(200_000) == 2_000.0
    assert buyer_fee(200_000, all_access=True) == 0.0
    assert buyer_fee(200_000, opened_bidding=True) == 0.0
