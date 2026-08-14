"""Phase 3 tests.

The load-bearing ones are test_recovers_* (does the fitter find a known
curve) and test_trimming_removes_lag_bias (does the reporting-lag correction
actually correct anything). If those two pass, the rest is plumbing.
"""
import numpy as np
import pandas as pd
import pytest

from royalty_edge.model.decay import (diagnose_reporting_lag, fit_all,
                                      fit_catalog, select_best, to_quarterly,
                                      trim_partial_tail)
from royalty_edge.model.valuation import (SCENARIOS, npv, project_income,
                                          scenario_spread, value_catalog,
                                          walk_away_multiple)

T = np.arange(0, 10, 0.25)


def _noisy(y, sd=0.06, seed=0):
    return y * np.exp(np.random.default_rng(seed).normal(0, sd, len(y)))


def test_recovers_exponential_lambda():
    lam = 0.18
    f = fit_catalog(T, _noisy(1000 * np.exp(-lam * T)), "exponential")
    assert f is not None
    assert abs(f.params[1] - lam) < 0.02
    assert abs(f.half_life_years - np.log(2) / lam) < 0.5


def test_recovers_exp_floor_asymptote():
    """The asymptote is where a perpetuity's value lives, so recovering it is
    the single most important estimation property in Phase 3."""
    A, lam, C = 1000.0, 0.55, 220.0
    y = _noisy((A - C) * np.exp(-lam * T) + C, sd=0.05)
    f = fit_catalog(T, y, "exp_floor")
    assert f is not None
    assert abs(np.exp(f.params[2]) - C) / C < 0.15
    assert abs(f.params[1] - lam) < 0.12


def test_recovers_power_alpha():
    a = 0.8
    f = fit_catalog(T, _noisy(1000 * (1 + T) ** -a), "power")
    assert f is not None
    assert abs(f.params[1] - a) < 0.1


def test_fit_returns_none_on_short_panel():
    assert fit_catalog(T[:3], np.array([100.0, 90, 80]), "exp_floor") is None


def _panel_bank(n_cat=200, lag=(0.45, 0.80), seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for lid in range(n_cat):
        n = int(rng.integers(16, 40))
        inc = 800 * np.exp(-rng.uniform(0.05, 0.5) * (np.arange(n) * 0.25))
        inc = inc * np.exp(rng.normal(0, 0.10, n))
        inc[-1] *= lag[0]
        inc[-2] *= lag[1]
        out.append(pd.DataFrame({
            "listing_id": lid,
            "period_start": pd.date_range("2015-01-01", periods=n, freq="QS"),
            "total": inc, "domestic": inc, "intl": 0.0, "unreported": 0.0,
            "deal_date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(n * 91)),
            "published_date": pd.Timestamp("2015-01-01"),
            "first_earnings_date": pd.Timestamp("2015-01-01"),
            "ltm": inc[-4:].sum(), "three_years_average": inc[-12:].sum() / 3,
            "dollar_age": rng.uniform(1, 15), "track_count": int(rng.integers(1, 30)),
            "term_family": "perpetual", "term_years": np.nan,
            "kind": "direct_listing"}))
    return to_quarterly(pd.concat(out, ignore_index=True))


@pytest.fixture(scope="module")
def panels():
    return _panel_bank()


def test_lag_diagnostic_detects_partial_quarters(panels):
    d = diagnose_reporting_lag(panels)
    r1 = d.loc[d["position_from_end"] == 1, "median_ratio"].iloc[0]
    r4 = d.loc[d["position_from_end"] == 4, "median_ratio"].iloc[0]
    assert r1 < 0.6, "final quarter should read as badly incomplete"
    assert r1 < r4, "the incompleteness must be concentrated at the very end"


def test_trimming_removes_lag_bias(panels):
    """Untrimmed panels must overstate decay; trimming must reduce it."""
    def med_lambda(p):
        vals = []
        for _, g in p.groupby("listing_id"):
            g = g.sort_values("period_start")
            f = fit_catalog(g["age_years"].to_numpy(float),
                            g["total"].to_numpy(float), "exponential")
            if f:
                vals.append(f.params[1])
        return float(np.median(vals))

    trimmed, n_trim = trim_partial_tail(panels)
    assert 1 <= n_trim <= 3, f"auto-trim chose {n_trim}, expected 1-3"
    assert med_lambda(trimmed) < med_lambda(panels)


def test_no_trim_when_panels_are_clean():
    clean = _panel_bank(n_cat=120, lag=(1.0, 1.0), seed=11)
    _, n_trim = trim_partial_tail(clean)
    assert n_trim == 0, "clean panels must not be trimmed"


def test_select_best_returns_one_row_per_catalog(panels):
    fits = fit_all(panels.groupby("listing_id").head(30))
    best = select_best(fits)
    assert len(best) == best["listing_id"].nunique()


# --- valuation -------------------------------------------------------

def test_npv_discounts_correctly():
    # four quarterly payments of 100 over one year at 0% is exactly 400
    assert npv(np.full(4, 100.0), 0.0) == pytest.approx(400.0)
    assert npv(np.full(4, 100.0), 0.12) < 400.0


def test_flat_scenario_beats_base():
    """The naive bidder assumes royalties are annuities. That assumption must
    produce a strictly higher valuation than any decaying one -- this is the
    magnitude of the error the whole strategy is trying to harvest."""
    f = fit_catalog(T, _noisy(1000 * np.exp(-0.25 * T)), "exponential")
    flat = value_catalog(listing_id=1, form="exponential", params=f.params,
                         t_end=10.0, ltm=4000.0,
                         scenario=[s for s in SCENARIOS if s.name == "flat_income"][0])
    base = value_catalog(listing_id=1, form="exponential", params=f.params,
                         t_end=10.0, ltm=4000.0,
                         scenario=[s for s in SCENARIOS if s.name == "base"][0])
    assert flat.fair_multiple_ltm > base.fair_multiple_ltm


def test_scenarios_separate_on_floor_catalog():
    """For a mature exp_floor catalog nearly all value is the asymptote, so a
    scenario set that only scales lambda would collapse to one number."""
    A, lam, C = 1000.0, 0.55, 220.0
    f = fit_catalog(T, _noisy((A - C) * np.exp(-lam * T) + C, sd=0.05), "exp_floor")
    got = {s.name: value_catalog(listing_id=1, form="exp_floor", params=f.params,
                                 t_end=10.0, ltm=4000.0, scenario=s).fair_multiple_ltm
           for s in SCENARIOS}
    assert got["bull"] > got["base"] > got["bear"] > got["floor_collapse"]
    assert got["bull"] / got["floor_collapse"] > 1.8, "scenarios must actually spread"


def test_higher_rate_lowers_value():
    f = fit_catalog(T, _noisy(1000 * np.exp(-0.2 * T)), "exponential")
    v = [value_catalog(listing_id=1, form="exponential", params=f.params,
                       t_end=10.0, ltm=4000.0, rate=r).fair_multiple_ltm
         for r in (0.08, 0.12, 0.18)]
    assert v[0] > v[1] > v[2]


def test_term_listing_worth_less_than_perpetuity():
    f = fit_catalog(T, _noisy(1000 * np.exp(-0.1 * T)), "exponential")
    kw = dict(listing_id=1, form="exponential", params=f.params, t_end=10.0,
              ltm=4000.0)
    perp = value_catalog(term_family="perpetual", **kw)
    term = value_catalog(term_family="fixed_term", term_years=10.0, **kw)
    assert term.fair_multiple_ltm < perp.fair_multiple_ltm


def test_walk_away_below_fair_and_fee_floor_binds():
    """On a small lot the $500 minimum fee is the binding constraint, and the
    walk-away multiple must fall further below fair value than on a large one."""
    f = fit_catalog(T, _noisy(1000 * np.exp(-0.2 * T)), "exponential")
    small = value_catalog(listing_id=1, form="exponential", params=f.params,
                          t_end=10.0, ltm=800.0)
    big = value_catalog(listing_id=2, form="exponential", params=f.params,
                        t_end=10.0, ltm=400000.0)
    gap_small = 1 - walk_away_multiple(small) / small.fair_multiple_ltm
    gap_big = 1 - walk_away_multiple(big) / big.fair_multiple_ltm
    assert walk_away_multiple(small) < small.fair_multiple_ltm
    assert gap_small > gap_big


# --- regression guards for the projection blowup ---------------------

def test_decay_rate_cannot_be_negative():
    """A rising catalog must fit as flat, never as compounding growth.

    This is the guard on the bug that produced fair multiples of 1e15: with
    lambda unbounded below, a catalog whose observed window happened to rise
    fitted as exponential growth and was then projected forty years forward.
    """
    rising = 1000 * np.exp(+0.30 * T)          # genuinely growing history
    for form in ("exponential", "power", "exp_floor", "two_phase"):
        f = fit_catalog(T, _noisy(rising, seed=5), form)
        if f is None:
            continue
        rate_idx = {"exponential": [1], "power": [1], "exp_floor": [1],
                    "two_phase": [1, 2]}[form]
        for i in rate_idx:
            assert f.params[i] >= -1e-9, f"{form} fitted a negative decay rate"


def test_rising_catalog_values_finitely():
    """The end-to-end consequence: no projection may diverge."""
    f = fit_catalog(T, _noisy(1000 * np.exp(+0.30 * T), seed=6), "exponential")
    v = value_catalog(listing_id=1, form="exponential", params=f.params,
                      t_end=10.0, ltm=4000.0, term_family="perpetual")
    assert np.isfinite(v.fair_multiple_ltm)
    assert v.fair_multiple_ltm < 60.0


def test_bear_never_exceeds_base():
    """With negative lambda allowed, scaling it for the bear case made the
    catalog grow faster and bear came out ABOVE base. It must not."""
    for gen, seed in [(lambda t: 1000 * np.exp(-0.2 * t), 7),
                      (lambda t: 1000 * np.exp(+0.2 * t), 8),
                      (lambda t: np.full_like(t, 1000.0), 9)]:
        f = fit_catalog(T, _noisy(gen(T), seed=seed), "exponential")
        got = {s.name: value_catalog(listing_id=1, form="exponential",
                                     params=f.params, t_end=10.0, ltm=4000.0,
                                     scenario=s).fair_multiple_ltm
               for s in SCENARIOS if s.name in ("base", "bear", "bull")}
        assert got["bear"] <= got["base"] + 1e-9
        assert got["bull"] >= got["base"] - 1e-9


def test_annual_grain_converted_to_quarterly_rate():
    """Annual rows carry a year of income; left as-is they overstate the run
    rate fourfold and the NPV with it."""
    n = 20
    annual = pd.DataFrame({
        "listing_id": 1,
        "period_start": pd.date_range("2005-01-01", periods=n, freq="YS"),
        "total": np.full(n, 4000.0), "domestic": 4000.0, "intl": 0.0,
        "unreported": 0.0, "deal_date": pd.Timestamp("2025-01-01"),
        "published_date": pd.Timestamp("2025-01-01"),
        "first_earnings_date": pd.Timestamp("2005-01-01"),
        "ltm": 4000.0, "three_years_average": 4000.0, "dollar_age": 12.0,
        "track_count": 5, "term_family": "perpetual", "term_years": np.nan,
        "kind": "direct_listing"})
    q = to_quarterly(annual)
    assert q["grain_source"].iloc[0] == "year"
    assert q["total"].iloc[0] == pytest.approx(1000.0)


def test_unknown_grain_panels_are_dropped():
    two_points = pd.DataFrame({
        "listing_id": 9, "period_start": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "total": [100.0, 90.0], "domestic": [100.0, 90.0], "intl": 0.0,
        "unreported": 0.0, "deal_date": pd.Timestamp("2022-01-01"),
        "published_date": pd.Timestamp("2022-01-01"),
        "first_earnings_date": pd.Timestamp("2020-01-01"),
        "ltm": 90.0, "three_years_average": 95.0, "dollar_age": 2.0,
        "track_count": 1, "term_family": "perpetual", "term_years": np.nan,
        "kind": "direct_listing"})
    assert to_quarterly(two_points).empty
