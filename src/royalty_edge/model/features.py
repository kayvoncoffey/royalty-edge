"""Phase 2 regression frame.

Turns the analytic layer into a modeling matrix. Every decision that could
bias the estimate is made explicitly here rather than implicitly in a filter
somewhere downstream.

Three decisions worth arguing about, made here:

1. SAMPLE. `auction` and `buy_it_now_auction` (all pre-2022) have zero
   coverage on bidder count, three-year average, and the published anchor,
   and they used a different price-formation mechanism. Pooling them with
   post-2020 offer-based listings would identify the term coefficients off a
   mechanism change rather than off catalog characteristics. They are
   excluded by default and available via include_legacy_auctions=True for
   anyone who wants to fit them separately.

2. DEPENDENT VARIABLE. log(clearing multiple), not the multiple. Multiples
   are a ratio, bounded below at zero, and right-skewed; the log makes the
   error roughly symmetric and turns coefficients into percentage effects,
   which is also how a bid is actually adjusted ("pay 12% less than the
   anchor" not "pay 0.8x less").

3. MISSINGNESS. three_years_average is absent on 12-27% of rows depending on
   vintage. Dropping those rows would select on data availability, which is
   correlated with listing age and therefore with the outcome. Instead the
   ratio is set to 1.0 (no detected spike) and a missing indicator is carried
   so the model can price the absence of information separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Members of fact_source_mix we carry as shares. Two distinct taxonomies live
# in that table -- `source` (STREAMING/RADIO/LIVE...) and `income_type`
# (PERFORMANCE/MECHANICAL...). They are not alternatives; a catalog has both.
SOURCE_MEMBERS = ["STREAMING", "RADIO", "LIVE", "SATELLITE RADIO",
                  "RETAIL MUSIC SERVICE", "SYNC"]
INCOME_TYPE_MEMBERS = ["PERFORMANCE", "MECHANICAL", "SYNC", "OTHER"]


@dataclass
class FrameSpec:
    """Sample definition. Printed with the results so a fit is reproducible."""
    kinds: tuple[str, ...] = ("direct_listing", "secondary_listing")
    min_deal_year: int = 2020
    sold_only: bool = True
    drop_exclude_flags: bool = True
    term_families: tuple[str, ...] = ("perpetual", "fixed_term")
    min_ltm: float = 100.0          # sub-$100 LTM lots are noise, not catalogs
    include_legacy_auctions: bool = False

    def describe(self) -> str:
        return (f"kinds={list(self.kinds)} year>={self.min_deal_year} "
                f"sold_only={self.sold_only} terms={list(self.term_families)} "
                f"min_ltm={self.min_ltm} legacy={self.include_legacy_auctions}")


BASE_SQL = """
WITH src AS (
    SELECT listing_id,
           max(CASE WHEN member = 'STREAMING'            THEN ltm_share END) AS share_streaming,
           max(CASE WHEN member = 'RADIO'                THEN ltm_share END) AS share_radio,
           max(CASE WHEN member = 'LIVE'                 THEN ltm_share END) AS share_live,
           max(CASE WHEN member = 'SATELLITE RADIO'      THEN ltm_share END) AS share_satellite,
           max(CASE WHEN member = 'RETAIL MUSIC SERVICE' THEN ltm_share END) AS share_retail,
           max(CASE WHEN member = 'SYNC'                 THEN ltm_share END) AS share_sync_src,
           count(*)                                                          AS n_source_members
    FROM fact_source_mix WHERE dimension = 'source' GROUP BY listing_id
), inc AS (
    SELECT listing_id,
           max(CASE WHEN member = 'PERFORMANCE' THEN ltm_share END) AS share_performance,
           max(CASE WHEN member = 'MECHANICAL'  THEN ltm_share END) AS share_mechanical,
           max(CASE WHEN member = 'SYNC'        THEN ltm_share END) AS share_sync_inc
    FROM fact_source_mix WHERE dimension = 'income_type' GROUP BY listing_id
), usr AS (
    -- platform concentration: a catalog riding one DSP is a different risk
    SELECT listing_id,
           max(ltm_share)              AS top_platform_share,
           sum(ltm_share * ltm_share)  AS platform_hhi
    FROM fact_source_mix WHERE dimension = 'music_user' GROUP BY listing_id
), panel AS (
    -- realized trailing growth from the earnings panel, independent of the
    -- platform's own three_years_average
    SELECT listing_id,
           count(*)                                          AS n_periods,
           min(period_start)                                 AS panel_start,
           max(period_start)                                 AS panel_end,
           sum(CASE WHEN quarters_before_listing BETWEEN -4 AND 0
                    THEN total END)                          AS panel_ltm,
           sum(CASE WHEN quarters_before_listing BETWEEN -8 AND -4
                    THEN total END)                          AS panel_prior_ltm
    FROM fact_earnings_panel GROUP BY listing_id
)
SELECT
    l.listing_id, l.asset_id, l.title, l.kind, l.term, l.term_family,
    l.term_years, l.is_partial_share, l.seller_id,
    l.ltm, l.three_years_average, l.lifetime_amount, l.dollar_age,
    l.catalog_age_years, l.track_count,
    l.minimum_price, l.reserve_multiple,
    l.marketplace_median_multiplier, l.is_marketplace_median_disabled,
    l.deal_date, l.published_date,
    o.sold, o.clearing_price, o.multiple_gross, o.multiple_all_in,
    o.multiple_vs_median, o.n_offers, o.n_unique_bidders,
    o.second_price, o.opening_price, o.buyer_fee_standard,
    src.share_streaming, src.share_radio, src.share_live, src.share_satellite,
    src.share_retail, src.share_sync_src, src.n_source_members,
    inc.share_performance, inc.share_mechanical, inc.share_sync_inc,
    usr.top_platform_share, usr.platform_hhi,
    panel.n_periods, panel.panel_ltm, panel.panel_prior_ltm,
    (SELECT count(*) FROM data_quality_flag q
      WHERE q.listing_id = l.listing_id AND q.severity = 'exclude') AS n_exclude_flags
FROM dim_listing l
JOIN fact_outcome o USING (listing_id)
LEFT JOIN src   USING (listing_id)
LEFT JOIN inc   USING (listing_id)
LEFT JOIN usr   USING (listing_id)
LEFT JOIN panel USING (listing_id)
"""


def load_frame(con, spec: FrameSpec | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (frame, attrition) -- the modeling matrix and a row-by-row
    account of what each filter removed. Always look at the attrition table;
    a filter that drops 40% of the sample is a modeling decision, not
    housekeeping."""
    spec = spec or FrameSpec()
    df = con.execute(BASE_SQL).fetchdf()
    steps: list[dict] = [{"step": "all listings with an outcome", "n": len(df)}]

    def cut(mask: pd.Series, label: str) -> None:
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        steps.append({"step": label, "n": len(df), "dropped": before - len(df)})

    kinds = list(spec.kinds)
    if spec.include_legacy_auctions:
        kinds += ["auction", "buy_it_now_auction"]
    cut(df["kind"].isin(kinds), f"kind in {kinds}")

    df["deal_year"] = pd.to_datetime(df["deal_date"], utc=True).dt.year
    cut(df["deal_year"] >= spec.min_deal_year, f"deal_year >= {spec.min_deal_year}")

    if spec.sold_only:
        cut(df["sold"].fillna(False).astype(bool), "sold")
    cut(df["term_family"].isin(spec.term_families), f"term_family in {list(spec.term_families)}")
    if spec.drop_exclude_flags:
        cut(df["n_exclude_flags"].fillna(0) == 0, "no exclude-severity quality flags")
    cut(df["ltm"].fillna(0) >= spec.min_ltm, f"ltm >= {spec.min_ltm}")
    cut(df["multiple_gross"].notna() & (df["multiple_gross"] > 0), "positive clearing multiple")

    df = _engineer(df)

    # Winsorize the dependent variable at 1/99 rather than dropping. A 40x
    # multiple is real -- some tiny catalogs clear at absurd multiples because
    # the $500 fee floor and bidder attention do not scale down -- but it
    # should not be allowed to set the slope by itself.
    lo, hi = df["multiple_gross"].quantile([0.01, 0.99])
    df["multiple_winsor"] = df["multiple_gross"].clip(lo, hi)
    df["log_multiple"] = np.log(df["multiple_winsor"])
    steps.append({"step": f"winsorized multiple to [{lo:.2f}, {hi:.2f}]", "n": len(df)})

    return df.reset_index(drop=True), pd.DataFrame(steps)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_ltm"] = np.log(df["ltm"].clip(lower=1))
    df["log_track_count"] = np.log(df["track_count"].fillna(1).clip(lower=1))

    # --- sync / spike normalization -------------------------------------
    # ltm_vs_3yr > 1 means the trailing year ran hot relative to the 3-year
    # average. That is the sync-spike signal, and it is the single most
    # actionable feature here: if the market prices off raw LTM, a hot LTM is
    # mechanically overpriced.
    r = df["ltm"] / df["three_years_average"]
    df["ltm_vs_3yr_missing"] = df["three_years_average"].isna().astype(int)
    df["ltm_vs_3yr"] = r.fillna(1.0).clip(0.2, 5.0)
    df["log_ltm_vs_3yr"] = np.log(df["ltm_vs_3yr"])
    df["spike_flag"] = (df["ltm_vs_3yr"] > 1.5).astype(int)

    # Normalized valuation base: what the multiple would be on trailing
    # 3-year average earnings instead of raw LTM.
    df["normalized_base"] = df["three_years_average"].fillna(df["ltm"])
    df["multiple_normalized"] = df["clearing_price"] / df["normalized_base"]

    # --- independent growth measure from the panel -----------------------
    g = df["panel_ltm"] / df["panel_prior_ltm"]
    df["panel_growth_missing"] = (~np.isfinite(g)).astype(int)
    df["panel_growth"] = g.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.2, 5.0)
    df["log_panel_growth"] = np.log(df["panel_growth"])

    # --- shares ----------------------------------------------------------
    for c in ["share_streaming", "share_radio", "share_live", "share_satellite",
              "share_retail", "share_sync_src", "share_performance",
              "share_mechanical", "share_sync_inc", "top_platform_share",
              "platform_hhi"]:
        df[c + "_missing"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(0.0)
    df["streaming_heavy"] = (df["share_streaming"] > 0.75).astype(int)

    # --- age -------------------------------------------------------------
    df["dollar_age"] = df["dollar_age"].fillna(df["dollar_age"].median())
    df["catalog_age_years"] = df["catalog_age_years"].clip(0, 60)
    df["catalog_age_missing"] = df["catalog_age_years"].isna().astype(int)
    df["catalog_age_years"] = df["catalog_age_years"].fillna(
        df["catalog_age_years"].median())
    # dollar_age enters with a square term: the stability premium should be
    # concave -- the difference between a 1-year and a 5-year catalog is much
    # larger than between 20 and 24 years.
    df["dollar_age_sq"] = df["dollar_age"] ** 2

    # --- anchor ----------------------------------------------------------
    df["anchor_disabled"] = df["is_marketplace_median_disabled"].fillna(False).astype(int)
    df["anchor_multiple"] = df["marketplace_median_multiplier"]
    df["anchor_missing"] = df["anchor_multiple"].isna().astype(int)
    df["log_anchor"] = np.log(df["anchor_multiple"].clip(lower=0.5)).fillna(0.0)

    # --- seller ask ------------------------------------------------------
    df["reserve_multiple"] = df["reserve_multiple"].clip(0, 30)
    df["reserve_missing"] = df["reserve_multiple"].isna().astype(int)
    df["log_reserve"] = np.log(df["reserve_multiple"].clip(lower=0.1)).fillna(0.0)

    # --- competition (POST-outcome; see pricing.py) -----------------------
    df["n_unique_bidders"] = df["n_unique_bidders"].fillna(0)
    df["log_bidders"] = np.log(df["n_unique_bidders"].clip(lower=1))
    df["single_bidder"] = (df["n_unique_bidders"] <= 1).astype(int)

    # --- time ------------------------------------------------------------
    dd = pd.to_datetime(df["deal_date"], utc=True)
    df["deal_quarter"] = dd.dt.tz_localize(None).dt.to_period("Q").astype(str)
    df["deal_year"] = dd.dt.year
    df["t_index"] = (dd - dd.min()).dt.days / 365.25

    # --- fee drag --------------------------------------------------------
    # The $500 floor is a real wedge on small lots and it is knowable pre-bid.
    df["fee_drag"] = df["buyer_fee_standard"] / df["clearing_price"]
    df["all_in_premium"] = df["multiple_all_in"] / df["multiple_gross"] - 1

    df["is_secondary"] = (df["kind"] == "secondary_listing").astype(int)
    df["is_perpetual"] = (df["term_family"] == "perpetual").astype(int)
    return df


# Feature blocks. Kept as named lists so a model spec is a composition of
# blocks rather than a hand-maintained formula string.
FUNDAMENTALS = [
    "log_ltm", "dollar_age", "dollar_age_sq", "catalog_age_years",
    "log_track_count", "log_ltm_vs_3yr", "ltm_vs_3yr_missing", "spike_flag",
    "share_streaming", "share_performance", "share_sync_inc",
    "top_platform_share", "log_panel_growth", "panel_growth_missing",
    "is_perpetual", "is_partial_share",
]
SELLER_ASK = ["log_reserve", "reserve_missing"]
ANCHOR = ["log_anchor", "anchor_missing", "anchor_disabled"]
COMPETITION = ["log_bidders", "single_bidder"]
