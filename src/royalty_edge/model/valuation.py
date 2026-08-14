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
from scipy import optimize

from .decay import FORMS, _eval

DEFAULT_RATE = 0.12
# Above this, a "valuation" is a divergent projection rather than a price.
# Real catalogs clear between roughly 1x and 25x; anything past 60x is a fit
# artifact and is excluded from summaries rather than winsorized, because the
# right response is to distrust the curve, not to shrink its output.
SANITY_CAP_MULTIPLE = 60.0
PERPETUITY_HORIZON_Y = 40.0
BUYER_FEE_PCT = 0.01
BUYER_FEE_MIN = 500.0


@dataclass
class Scenario:
    """A scenario is a statement about the FUTURE, relative to today.

    Earlier versions scaled the fitted parameters. That stops working once
    the projection is anchored to observed income: scaling a floor parameter
    moves the asymptote and the anchor point together, and after re-anchoring
    the trajectory is nearly unchanged. Expressing scenarios as a forward
    decay rate and a terminal share of TODAY's income makes them independent
    of the fitted parameterization, comparable across functional forms, and
    -- more importantly -- directly arguable. "Settles at 40% of current
    income" is a claim someone can disagree with. "floor_multiplier = 0.65"
    is not.
    """
    name: str
    decay_multiplier: float | None      # scales the forward decay rate
    label: str
    terminal_multiplier: float = 1.0    # scales the terminal share of today
    fixed_annual_decline: float | None = None


SCENARIOS = [
    Scenario("bull", 0.60, "slower decay, 20% higher long-run floor", 1.20),
    Scenario("base", 1.00, "decay and floor as fitted", 1.00),
    Scenario("bear", 1.75, "faster decay, 35% lower floor", 0.65),
    Scenario("floor_collapse", 1.00, "as fitted but the long tail halves", 0.50),
    Scenario("flat_income", 0.0, "income never declines (the naive bidder's model)", 1.0),
    Scenario("minus_10pct", None, "fixed -10%/yr regardless of fit",
             fixed_annual_decline=-0.10),
]


def forward_params(form: str, params, t_end: float) -> tuple[float, float]:
    """Reduce any fitted form to the two numbers that drive a valuation:

        lam_fwd  -- the local decay rate at the end of the observed history
        s        -- the long-run floor as a share of income TODAY

    Everything a buyer is actually purchasing is in these two numbers plus
    the current run rate. The functional form's only job is to estimate them
    from history; once estimated, the form itself is irrelevant to the NPV,
    which is what makes valuations comparable across catalogs fitted with
    different forms.
    """
    p = np.asarray(params, dtype=float)
    t = max(float(t_end), 1e-6)

    if form == "exponential":
        return max(float(p[1]), 0.0), 0.0

    if form == "power":
        # local log-slope of A*(1+t)^-alpha is alpha/(1+t)
        return max(float(p[1]) / (1.0 + t), 0.0), 0.0

    if form == "exp_floor":
        lam = max(float(p[1]), 0.0)
        f = 1.0 / (1.0 + np.exp(-float(p[2])))
        # level today relative to A, and the floor relative to that level
        level_now = f + (1.0 - f) * np.exp(-lam * t)
        s = float(np.clip(f / level_now, 0.0, 1.0)) if level_now > 0 else 0.0
        return lam, s

    if form == "two_phase":
        lam1, lam2, tau = float(p[1]), float(p[2]), float(p[3])
        return max(lam2 if t > tau else lam1, 0.0), 0.0

    return 0.0, 0.0


def project_forward(anchor_level: float, lam_fwd: float, terminal_share: float,
                    horizon_years: float, *, steps_per_year: int = 4
                    ) -> np.ndarray:
    """Quarterly income from today forward.

        y(u) = anchor * (s + (1 - s) * exp(-lam * u))

    Monotonically non-increasing for lam >= 0 and s in [0, 1], so it cannot
    diverge for any input the callers can produce.
    """
    n = int(round(horizon_years * steps_per_year))
    if n <= 0:
        return np.zeros(0)
    u = np.arange(1, n + 1) / steps_per_year
    s = float(np.clip(terminal_share, 0.0, 1.0))
    lam = max(float(lam_fwd), 0.0)
    return anchor_level * (s + (1.0 - s) * np.exp(-lam * u))


def project_income(form: str, params, t_start: float, horizon_years: float, *,
                   decay_multiplier: float | None = 1.0,
                   terminal_multiplier: float = 1.0,
                   anchor_level: float | None = None,
                   fixed_annual_decline: float | None = None,
                   steps_per_year: int = 4) -> tuple[np.ndarray, np.ndarray]:
    lam_fwd, s = forward_params(form, params, t_start)
    if anchor_level is None:
        anchor_level = float(np.exp(_eval(form, np.array([t_start]),
                                          np.asarray(params)))[0])
    n = int(round(horizon_years * steps_per_year))
    t = t_start + np.arange(1, n + 1) / steps_per_year

    if fixed_annual_decline is not None:
        u = np.arange(1, n + 1) / steps_per_year
        return t, anchor_level * (1 + fixed_annual_decline) ** u

    lam = 0.0 if decay_multiplier is None else lam_fwd * decay_multiplier
    s_adj = float(np.clip(s * terminal_multiplier, 0.0, 1.0))
    return t, project_forward(anchor_level, lam, s_adj, horizon_years,
                              steps_per_year=steps_per_year)


def npv(quarterly_income: np.ndarray, rate: float, *,
        steps_per_year: int = 4) -> float:
    """Discount a quarterly stream; each element is one quarter's cash."""
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
                  scenario: Scenario | None = None,
                  anchor_level: float | None = None) -> Valuation:
    """anchor_level: observed current QUARTERLY income; defaults to ltm/4.

    LTM is measured; the fitted curve's endpoint is an extrapolation of a
    noisy regression to its own boundary. On these panels the two differ by
    tens of percent and the error multiplies through the whole NPV, so the
    measured quantity is used for level and the fit is used only for shape.
    """
    scenario = scenario or SCENARIOS[1]
    if anchor_level is None:
        anchor_level = ltm / 4.0 if ltm and ltm > 0 else None
    horizon = (PERPETUITY_HORIZON_Y if term_family == "perpetual"
               else float(term_years or 10.0))

    _, q = project_income(
        form, params, t_end, horizon, anchor_level=anchor_level,
        decay_multiplier=scenario.decay_multiplier,
        terminal_multiplier=scenario.terminal_multiplier,
        fixed_annual_decline=scenario.fixed_annual_decline)

    pv = npv(q, rate)
    base = normalized_base if normalized_base and normalized_base > 0 else ltm
    run_rate = (anchor_level * 4 if anchor_level is not None else np.nan)
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


def blend_forward(form: str, params, t_end: float,
                  pooled_params, age: float, weight: float) -> tuple[float, float]:
    """Combine a catalog's own forward parameters with the pooled curve's.

    weight is the confidence in the catalog's own history -- n/(n+k) from the
    shrinkage step. With two thirds of individual fits pinned at flat, a pure
    per-catalog model is a constant and a pure pooled model ignores real
    catalog-level information where it exists. Blending in the FORWARD
    parameters rather than in the fitted coefficients means the two sources
    are on the same scale and comparable regardless of functional form.
    """
    lam_i, s_i = forward_params(form, params, t_end)
    if pooled_params is None:
        return lam_i, s_i
    lam_p, s_p = forward_params("exp_floor", pooled_params, max(age, 0.0))
    w = float(np.clip(weight, 0.0, 1.0))
    return w * lam_i + (1 - w) * lam_p, w * s_i + (1 - w) * s_p


def value_all(best_fits: pd.DataFrame, meta: pd.DataFrame, *,
              rates: tuple[float, ...] = (0.08, 0.12, 0.18),
              scenarios: list[Scenario] | None = None,
              pooled_params=None, decay_source: str = "blend",
              k_prior: float = 12.0) -> pd.DataFrame:
    """Value every catalog with a fitted curve, across rates and scenarios.

    decay_source:
      'fit'    -- each catalog's own curve. On these panels that means flat
                  for most of the book and a fair multiple that is really an
                  annuity constant.
      'pooled' -- the cross-sectional age profile applied at each catalog's
                  age. Ignores catalog-specific information but is estimated
                  on 1,600+ panels instead of 25 noisy quarters.
      'blend'  -- per-catalog weighted n/(n+k), pooled otherwise. Default.
    """
    scenarios = scenarios or SCENARIOS
    dropped: list[dict] = []
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
        n_obs = float(r.get("n_obs") or 0.0)

        if decay_source == "fit" or pooled_params is None:
            eff_form, eff_params, eff_t = r["form"], params, t_end
        elif decay_source == "pooled":
            eff_form, eff_params, eff_t = "exp_floor", pooled_params, t_end
        else:
            w = n_obs / (n_obs + k_prior) if n_obs > 0 else 0.0
            lam_b, s_b = blend_forward(r["form"], params, t_end,
                                       pooled_params, t_end, w)
            # re-express the blended pair as an exp_floor curve so the rest of
            # the machinery is untouched
            s_c = float(np.clip(s_b, 1e-6, 1 - 1e-6))
            eff_form = "exp_floor"
            eff_params = np.array([0.0, max(lam_b, 0.0),
                                   float(np.log(s_c / (1 - s_c)))])
            eff_t = 0.0

        for rate in rates:
            for sc in scenarios:
                try:
                    v = value_catalog(
                        listing_id=int(r["listing_id"]), form=eff_form,
                        params=eff_params, t_end=eff_t, ltm=float(ltm),
                        normalized_base=float(base),
                        term_family=r.get("term_family") or "perpetual",
                        term_years=r.get("term_years"), rate=rate, scenario=sc)
                except Exception:
                    continue
                fm = v.fair_multiple_ltm
                if not np.isfinite(fm) or fm <= 0 or fm > SANITY_CAP_MULTIPLE:
                    # A fair multiple above the cap is a fit artifact, not a
                    # find. Recording it would let a handful of divergent
                    # projections dominate every summary statistic downstream.
                    dropped.append({"listing_id": int(r["listing_id"]),
                                    "form": r["form"], "scenario": sc.name,
                                    "rate": rate, "fair_multiple_ltm": fm})
                    continue
                rows.append({
                    "listing_id": v.listing_id, "form": v.form,
                    "scenario": v.scenario, "rate": v.rate,
                    "horizon_years": v.horizon_years,
                    "fair_multiple_ltm": round(fm, 3),
                    "fair_multiple_normalized": round(v.fair_multiple_normalized, 3),
                    "walk_away_multiple": round(walk_away_multiple(v), 3),
                    "npv_income": round(v.npv_income, 2),
                })
    out = pd.DataFrame(rows)
    out.attrs["dropped"] = pd.DataFrame(dropped)
    return out


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


# --------------------------------------------------------------------
# What rate is the market actually paying?
# --------------------------------------------------------------------

def implied_discount_rate(form, params, t_end: float, price: float, *,
                          term_family: str = "perpetual",
                          term_years: float | None = None,
                          anchor_level: float | None = None,
                          lo: float = 1e-4, hi: float = 2.0) -> float:
    """The discount rate at which the model's projection exactly justifies the
    price actually paid. This is the buyer's implied IRR on the projected
    cash flows.

    This is the number that makes `edge_pct` interpretable. A raw edge of
    -18% could mean the market is overpaying, or it could mean the 12%
    discount rate is simply higher than the rate clearing buyers demand -- and
    those are completely different conclusions. Converting every clearing
    price into the rate that rationalizes it turns the question into one that
    can be answered: is the market pricing these at 7% or at 20%? The bid rule
    then follows directly. Bid where the implied rate exceeds the hurdle;
    pass otherwise.

    It also relocates the discount-rate assumption. Instead of a number
    imposed on the analysis, the rate becomes an output measured from
    observed transactions, and the only judgement left is what return the
    illiquidity and administrator risk deserve.
    """
    horizon = (PERPETUITY_HORIZON_Y if term_family == "perpetual"
               else float(term_years or 10.0))

    _, q = project_income(form, params, t_end, horizon,
                          anchor_level=anchor_level)

    def excess(r: float) -> float:
        return npv(q, r) - price

    try:
        if excess(lo) < 0:
            return np.nan          # price above PV even at ~0%: no solution
        if excess(hi) > 0:
            return np.inf          # justified at any rate: income dwarfs price
        return float(optimize.brentq(excess, lo, hi, maxiter=200))
    except Exception:
        return np.nan


def implied_rate_table(best_fits: pd.DataFrame, frame: pd.DataFrame, *,
                       pooled_params=None, decay_source: str = "blend",
                       k_prior: float = 12.0) -> pd.DataFrame:
    """Implied rate for every catalog with both a fitted curve and a price."""
    cols = [c for c in ["listing_id", "title", "kind", "term_family", "term_years",
                        "deal_date", "ltm", "dollar_age", "clearing_price",
                        "multiple_gross"] if c in frame.columns]
    d = best_fits.merge(frame[cols], on="listing_id", how="inner",
                        suffixes=("", "_f"))
    rows = []
    for _, r in d.iterrows():
        params = r["params"]
        if isinstance(params, str):
            params = eval(params)  # noqa: S307
        price = r.get("clearing_price")
        if not price or not np.isfinite(price) or price <= 0:
            continue
        ltm_v = r.get("ltm")
        anchor = (float(ltm_v) / 4.0
                  if ltm_v and np.isfinite(ltm_v) and ltm_v > 0 else None)
        t_end = float(r.get("span_years") or 0.0)
        n_obs = float(r.get("n_obs") or 0.0)
        if decay_source == "fit" or pooled_params is None:
            ef, ep, et = r["form"], params, t_end
        elif decay_source == "pooled":
            ef, ep, et = "exp_floor", pooled_params, t_end
        else:
            w = n_obs / (n_obs + k_prior) if n_obs > 0 else 0.0
            lam_b, s_b = blend_forward(r["form"], params, t_end,
                                       pooled_params, t_end, w)
            s_c = float(np.clip(s_b, 1e-6, 1 - 1e-6))
            ef, ep, et = "exp_floor", np.array(
                [0.0, max(lam_b, 0.0), float(np.log(s_c / (1 - s_c)))]), 0.0
        rate = implied_discount_rate(
            ef, ep, et, float(price),
            term_family=r.get("term_family") or "perpetual",
            term_years=r.get("term_years"), anchor_level=anchor)
        rows.append({"listing_id": int(r["listing_id"]),
                     "title": r.get("title"),
                     "term_family": r.get("term_family"),
                     "deal_date": r.get("deal_date"),
                     "ltm": r.get("ltm"), "dollar_age": r.get("dollar_age"),
                     "multiple_gross": r.get("multiple_gross"),
                     "implied_rate": rate})
    out = pd.DataFrame(rows)
    return out.sort_values("implied_rate", ascending=False)


def implied_rate_summary(tbl: pd.DataFrame) -> pd.DataFrame:
    """Distribution of implied rates, and how many lots are unsolvable.

    `n_no_solution` counts lots whose price exceeds the projected income's PV
    even at a rate near zero -- the model says they can never repay, at any
    hurdle. A large count there is a statement about the model as much as
    about the market, and should be read that way first.
    """
    if tbl.empty:
        return tbl
    finite = tbl[np.isfinite(tbl["implied_rate"])]
    return pd.DataFrame([{
        "n_total": len(tbl),
        "n_solved": len(finite),
        "n_no_solution": int(tbl["implied_rate"].isna().sum()),
        "n_any_rate": int(np.isinf(tbl["implied_rate"]).sum()),
        "p10": finite["implied_rate"].quantile(0.10),
        "p25": finite["implied_rate"].quantile(0.25),
        "median": finite["implied_rate"].median(),
        "p75": finite["implied_rate"].quantile(0.75),
        "p90": finite["implied_rate"].quantile(0.90),
    }]).round(4)
