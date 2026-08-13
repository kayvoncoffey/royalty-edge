"""Phase 2 tests.

The important one is test_recovers_known_coefficients: synthetic data with a
known DGP, and the estimator must recover it. Everything else in this file is
plumbing; that test is the one that would catch a sign error or a botched
design matrix.
"""
import numpy as np
import pandas as pd
import pytest

from royalty_edge.model.features import (ANCHOR, FUNDAMENTALS, SELLER_ASK,
                                         FrameSpec, _engineer)
from royalty_edge.model.pricing import (anchor_analysis, fit_ladder, fit_model,
                                        ladder_table, residual_screen,
                                        spike_check, temporal_holdout)

TRUE = dict(dollar_age=0.030, dollar_age_sq=-0.0006, log_ltm=-0.045,
            is_perpetual=0.30, log_ltm_vs_3yr=-0.22, share_streaming=-0.18)


def _synth(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    ltm = np.exp(rng.normal(9, 1.3, n))
    da = np.abs(rng.gamma(3, 2.2, n))
    three = ltm / rng.lognormal(0.0, 0.28, n)
    share = np.clip(rng.beta(6, 2, n), 0, 1)
    perp = rng.binomial(1, 0.62, n)
    dates = pd.to_datetime("2020-01-01", utc=True) + pd.to_timedelta(
        rng.integers(0, 2200, n), "D")
    log_mult = (1.55 + TRUE["dollar_age"] * da + TRUE["dollar_age_sq"] * da ** 2
                + TRUE["log_ltm"] * np.log(ltm) + TRUE["is_perpetual"] * perp
                + TRUE["log_ltm_vs_3yr"] * np.log(ltm / three)
                + TRUE["share_streaming"] * share + rng.normal(0, 0.22, n))
    mult = np.exp(log_mult)
    df = pd.DataFrame({
        "listing_id": np.arange(n), "asset_id": np.arange(n),
        "title": "x", "kind": "direct_listing",
        "term": np.where(perp == 1, "life_of_rights", "10_year"),
        "term_family": np.where(perp == 1, "perpetual", "fixed_term"),
        "term_years": np.where(perp == 1, np.nan, 10.0),
        "is_partial_share": False, "seller_id": rng.integers(1, 180, n),
        "ltm": ltm, "three_years_average": three, "lifetime_amount": ltm * 5,
        # catalog_age must vary independently of dollar_age or the two are
        # collinear and the coefficient splits between them
        "dollar_age": da, "catalog_age_years": da * rng.uniform(0.9, 1.6, n),
        "track_count": rng.integers(1, 60, n),
        "minimum_price": mult * ltm * 0.3, "reserve_multiple": mult * 0.3,
        "marketplace_median_multiplier": mult * rng.lognormal(0, .12, n),
        "is_marketplace_median_disabled": rng.binomial(1, .28, n).astype(bool),
        "deal_date": dates, "published_date": dates, "sold": True,
        "clearing_price": mult * ltm, "multiple_gross": mult,
        "multiple_all_in": mult * 1.01, "multiple_vs_median": 1.0,
        "n_offers": 10, "n_unique_bidders": rng.integers(1, 15, n),
        "second_price": mult * ltm * .97, "opening_price": mult * ltm * .4,
        "buyer_fee_standard": np.maximum(500, .01 * mult * ltm),
        "share_streaming": share, "share_radio": 1 - share, "share_live": 0.0,
        "share_satellite": 0.0, "share_retail": 0.0, "share_sync_src": 0.0,
        "n_source_members": 3, "share_performance": 0.5, "share_mechanical": 0.0,
        "share_sync_inc": 0.05, "top_platform_share": 0.4, "platform_hhi": 0.3,
        "n_periods": 20, "panel_ltm": ltm, "panel_prior_ltm": ltm,
        "n_exclude_flags": 0,
    })
    df = _engineer(df)
    lo, hi = df["multiple_gross"].quantile([.01, .99])
    df["multiple_winsor"] = df["multiple_gross"].clip(lo, hi)
    df["log_multiple"] = np.log(df["multiple_winsor"])
    return df


@pytest.fixture(scope="module")
def synth():
    return _synth()


def test_recovers_known_coefficients(synth):
    """Every true coefficient must land within 3 SE of its estimate."""
    fit = fit_model(synth, FUNDAMENTALS, name="M1")
    for var, truth in TRUE.items():
        assert var in fit.params.index, f"{var} missing from design"
        est, se = fit.params.loc[var, "coef"], fit.params.loc[var, "se"]
        assert abs(est - truth) < 3 * se, \
            f"{var}: true={truth:.4f} est={est:.4f} se={se:.4f}"


def test_ladder_r2_is_monotone(synth):
    res = fit_ladder(synth)
    r2 = [r.r2 for r in res]
    assert all(b >= a - 1e-9 for a, b in zip(r2, r2[1:])), \
        "nested models must not lose explanatory power"


def test_holdout_beats_naive(synth):
    h = temporal_holdout(synth, FUNDAMENTALS + SELLER_ASK + ANCHOR,
                         cutoff="2024-06-01")
    assert h["skill_vs_naive"] > 0.3
    assert h["n_test"] > 50


def test_holdout_aligns_unseen_quarters(synth):
    """Test quarters never seen in training must not blow up the design."""
    h = temporal_holdout(synth, FUNDAMENTALS, cutoff="2024-06-01")
    assert "error" not in h and np.isfinite(h["rmse_test"])


def test_spike_check_orders_correctly(synth):
    """Normalized premium must rise monotonically with the spike ratio --
    that IS the sync-distortion thesis, stated as a testable claim."""
    g = spike_check(synth)
    prem = g["normalized_premium"].to_numpy()
    assert prem[0] < prem[-1]
    assert (np.diff(prem) > -0.05).all(), "normalized premium should be increasing"


def test_residual_screen_is_signed_correctly(synth):
    res = fit_ladder(synth)
    scr = residual_screen(synth, res[3])
    assert scr["mispricing_pct"].iloc[0] < 0     # cheapest first
    assert scr["mispricing_pct"].iloc[-1] > 0    # priciest last
    assert (scr["predicted_multiple"] > 0).all()


def test_anchor_analysis_runs(synth):
    a = anchor_analysis(synth)
    assert a["n_anchor_shown"] > 0 and a["n_anchor_hidden"] > 0
    assert "anchor_elasticity" in a


def test_frame_spec_describe():
    assert "2020" in FrameSpec().describe()
