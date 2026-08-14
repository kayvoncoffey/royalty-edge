"""Phase 3b: from a decay curve to a bid.

Takes a fitted curve and produces the number the bid engine needs: the
multiple of LTM at which this catalog is worth buying, and how fast that
number falls apart when the assumptions move.

Three decisions here are judgement calls rather than mathematics, and each is
made explicitly so it can be argued with:

VALUATION BASE. The multiple is quoted against LTM because that is the
platform's convention and what every bidder sees. But LTM is a noisy base --
a catalog whose trailing year ran hot is being quoted against an inflated
denominator. Every output carries both `fair_multiple_ltm` (comparable to the
screen) and `fair_multiple_normalized` (against the three-year average). When
those two disagree sharply, the disagreement IS the trade.

DISCOUNT RATE. There is no market rate for an illiquid, non-marginable,
unhedgeable royalty interest, so the rate is an input, not an output.
Default 12% reflects: no leverage available, no secondary market, single-
administrator counterparty risk, and a real option value forgone by locking
capital up. Anyone using 8% here is implicitly assuming this asset is as safe
as a BDC, which it is not. Sensitivity across 8-18% is reported for every
valuation because the answer moves a great deal.

TERMINAL HORIZON. A perpetuity is integrated to 40 years, not to infinity.
Beyond four decades the discounted contribution is negligible at any sane
rate, and the modelling error on a 40-year-out streaming payout dwarfs the
present value. Quoting a true perpetuity implies a confidence about 2066
listener behaviour that nobody has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .decay import FORMS, _eval

DEFAULT_RATE = 0.12
PERPETUITY_HORIZON_Y = 40.0
BUYER_FEE_PCT = 0.01
BUYER_FEE_MIN = 500.0


@dataclass
class Scenario:
    name: str
    decay_multiplier: float           # scales the fitted decay rate
    label: str
    floor_multiplier: float = 1.0     # scales the fitted long-tail asymptote


# Scenarios must stress the parameter that actually carries the value.
# For a mature catalog fitted with exp_floor, almost all the NPV is the
# asymptote, not the decay rate -- by the time it is listed the steep phase is
# already history. Scaling only lambda there produces three nearly identical
# numbers and a false sense of robustness. The floor is the exposure, so the
# bear case cuts the floor, and that is also the honest statement of the risk:
# the question is not "how fast does it fall" but "what does it settle at".
SCENARIOS = [
    Scenario("bull", 0.60, "slower decay, floor holds 20% higher", 1.20),
    Scenario("base", 1.00, "fitted decay and floor", 1.00),
    Scenario("bear", 1.75, "faster decay, floor 35% lower", 0.65),
    Scenario("floor_collapse", 1.00, "fitted decay, floor cut in half", 0.50),
    Scenario("flat_income", 0.0, "income never declines (the naive bidder's model)"),
    Scenario("minus_10pct", None, "fixed -10%/yr regardless of fit"),
]


def project_income(form: str, params, t_start: float, horizon_years: float,
                   *, decay_multiplier: float = 1.0,
                   floor_multiplier: float = 1.0,
                   fixed_annual_decline: float | None = None,
                   steps_per_year: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Project quarterly income forward from t_start.

    decay_multiplier scales the rate parameters only, never the level -- so a
    bear case is 'this decays faster', not 'this is smaller', which are
    different claims and only the first is a decay assumption.
    """
    n = int(round(horizon_years * steps_per_year))
    t = t_start + np.arange(1, n + 1) / steps_per_year

    if fixed_annual_decline is not None:
        y0 = float(np.exp(_eval(form, np.array([t_start]), np.asarray(params)))[0])
        yrs = np.arange(1, n + 1) / steps_per_year
        return t, y0 * (1 + fixed_annual_decline) ** yrs / steps_per_year * steps_per_year

    p = np.asarray(params, dtype=float).copy()
    if decay_multiplier != 1.0:
        if form in ("exponential", "power", "exp_floor"):
            p[1] *= decay_multiplier
        elif form == "two_phase":
            p[1] *= decay_multiplier
            p[2] *= decay_multiplier
    if floor_multiplier != 1.0 and form == "exp_floor":
        # log-parameterized asymptote
        p[2] = p[2] + np.log(floor_multiplier)
    return t, np.exp(_eval(form, t, p))


def npv(quarterly_income: np.ndarray, rate: float, *,
        steps_per_year: int = 4) -> float:
    """Discount a quarterly stream. Income here is a quarterly *rate*, so each
    element is already a quarter's cash."""
    k = np.arange(1, len(quarterly_income) + 1) / steps_per_year
    return float(np.sum(quarterly_income / (1 + rate) ** k))


@dataclass
class Valuation:
    listing_id: int
    form: str
    scenario: str
    rate: float
    horizon_years: float
    npv_income: float
    ltm: float
    normalized_base: float
    fair_price: float
    fair_multiple_ltm: float
    fair_multiple_normalized: float
    current_run_rate: float


def value_catalog(*, listing_id: int, form: str, params, t_end: float,
                  ltm: float, normalized_base: float | None = None,
                  term_family: str = "perpetual", term_years: float | None = None,
                  rate: float = DEFAULT_RATE,
                  scenario: Scenario | None = None) -> Valuation:
    scenario = scenario or SCENARIOS[1]
    horizon = (PERPETUITY_HORIZON_Y if term_family == "perpetual"
               else float(term_years or 10.0))

    if scenario.name == "minus_10pct":
        _, q = project_income(form, params, t_end, horizon,
                              fixed_annual_decline=-0.10)
    elif scenario.name == "flat_income":
        y0 = float(np.exp(_eval(form, np.array([t_end]), np.asarray(params)))[0])
        q = np.full(int(horizon * 4), y0)
    else:
        _, q = project_income(form, params, t_end, horizon,
                              decay_multiplier=scenario.decay_multiplier,
                              floor_multiplier=scenario.floor_multiplier)

    pv = npv(q, rate)
    base = normalized_base if normalized_base and normalized_base > 0 else ltm
    run_rate = float(np.exp(_eval(form, np.array([t_end]), np.asarray(params)))[0]) * 4
    return Valuation(
        listing_id=listing_id, form=form, scenario=scenario.name, rate=rate,
        horizon_years=horizon, npv_income=pv, ltm=ltm, normalized_base=base,
        fair_price=pv,
        fair_multiple_ltm=pv / ltm if ltm > 0 else np.nan,
        fair_multiple_normalized=pv / base if base > 0 else np.nan,
        current_run_rate=run_rate,
    )


def buyer_fee(price: float, *, all_access: bool = False,
              opened_bidding: bool = False) -> float:
    if all_access or opened_bidding:
        return 0.0
    return max(BUYER_FEE_MIN, BUYER_FEE_PCT * price)


def walk_away_multiple(v: Valuation, *, all_access: bool = False) -> float:
    """The highest multiple of LTM that still clears the hurdle rate once the
    buyer fee is loaded on. This is the number to take to an auction.

    Solving p + fee(p) = NPV, where fee has a $500 floor. On a small lot the
    floor binds and the walk-away multiple sits well below the headline fair
    multiple -- which is precisely the case where bidders ignore it.
    """
    target = v.npv_income
    if all_access:
        return target / v.ltm if v.ltm > 0 else np.nan
    # try the percentage regime first, then check the floor regime
    p_pct = target / (1 + BUYER_FEE_PCT)
    if BUYER_FEE_PCT * p_pct >= BUYER_FEE_MIN:
        p = p_pct
    else:
        p = target - BUYER_FEE_MIN
    return max(p, 0.0) / v.ltm if v.ltm > 0 else np.nan


def value_all(best_fits: pd.DataFrame, meta: pd.DataFrame, *,
              rates: tuple[float, ...] = (0.08, 0.12, 0.18),
              scenarios: list[Scenario] | None = None) -> pd.DataFrame:
    """Value every catalog with a fitted curve, across rates and scenarios."""
    scenarios = scenarios or SCENARIOS
    d = best_fits.merge(meta, on="listing_id", how="left", suffixes=("", "_m"))
    rows = []
    for _, r in d.iterrows():
        params = r["params"]
        if isinstance(params, str):
            params = eval(params)  # noqa: S307 - our own serialization
        ltm = r.get("ltm")
        if not ltm or not np.isfinite(ltm) or ltm <= 0:
            continue
        t_end = float(r.get("span_years") or 0.0)
        base = r.get("three_years_average")
        base = base if base and np.isfinite(base) and base > 0 else ltm
        for rate in rates:
            for sc in scenarios:
                try:
                    v = value_catalog(
                        listing_id=int(r["listing_id"]), form=r["form"],
                        params=params, t_end=t_end, ltm=float(ltm),
                        normalized_base=float(base),
                        term_family=r.get("term_family") or "perpetual",
                        term_years=r.get("term_years"), rate=rate, scenario=sc)
                except Exception:
                    continue
                rows.append({
                    "listing_id": v.listing_id, "form": v.form,
                    "scenario": v.scenario, "rate": v.rate,
                    "horizon_years": v.horizon_years,
                    "fair_multiple_ltm": round(v.fair_multiple_ltm, 3),
                    "fair_multiple_normalized": round(v.fair_multiple_normalized, 3),
                    "walk_away_multiple": round(walk_away_multiple(v), 3),
                    "npv_income": round(v.npv_income, 2),
                })
    return pd.DataFrame(rows)


def compare_to_market(valuations: pd.DataFrame, frame: pd.DataFrame, *,
                      rate: float = DEFAULT_RATE,
                      scenario: str = "base") -> pd.DataFrame:
    """Join model fair value against what the lot actually cleared at.

    `edge_pct` is the quantity the whole project exists to produce: how far
    below model fair value the market cleared. Positive means the market
    underpaid relative to the model -- which is either an opportunity or a
    sign the model is missing something the bidders could see. Early on,
    assume the second.
    """
    v = valuations[(valuations["rate"] == rate) &
                   (valuations["scenario"] == scenario)]
    cols = ["listing_id", "title", "kind", "term_family", "deal_date", "ltm",
            "dollar_age", "multiple_gross", "ltm_vs_3yr", "n_unique_bidders"]
    cols = [c for c in cols if c in frame.columns]
    out = v.merge(frame[cols], on="listing_id", how="inner")
    out["edge_pct"] = out["fair_multiple_ltm"] / out["multiple_gross"] - 1
    out["cleared_vs_walkaway"] = out["multiple_gross"] / out["walk_away_multiple"]
    return out.sort_values("edge_pct", ascending=False)


def rate_sensitivity(valuations: pd.DataFrame) -> pd.DataFrame:
    """How much does the fair multiple move with the discount rate?

    If a modest rate change flips the bid/pass decision on most of the book,
    the model is not producing a valuation, it is producing an opinion about
    rates wearing a valuation's clothes.
    """
    if valuations.empty:
        return valuations
    base = valuations[valuations["scenario"] == "base"]
    piv = base.pivot_table(index="listing_id", columns="rate",
                           values="fair_multiple_ltm")
    if piv.shape[1] < 2:
        return pd.DataFrame()
    lo, hi = piv.columns.min(), piv.columns.max()
    out = pd.DataFrame({
        f"fair_at_{lo:.0%}": piv[lo],
        f"fair_at_{hi:.0%}": piv[hi],
    })
    out["ratio"] = piv[hi] / piv[lo]
    return out.describe().round(3)


def scenario_spread(valuations: pd.DataFrame, *, rate: float = DEFAULT_RATE
                    ) -> pd.DataFrame:
    """Fair multiple by scenario. The gap between `flat_income` and `base` is
    the size of the error a bidder makes by assuming royalties are annuities --
    which is exactly the assumption the platform's own multiple display
    encourages."""
    v = valuations[valuations["rate"] == rate]
    if v.empty:
        return v
    return (v.groupby("scenario")["fair_multiple_ltm"]
             .agg(n="size", p25=lambda s: s.quantile(.25), median="median",
                  p75=lambda s: s.quantile(.75))
             .round(2).reset_index())
