"""Dollar Age recomputation and LTM normalization.

Two things here, both of which need to happen before any regression.

1. Dollar Age is the platform's own metric, and its published description is
   ambiguous about normalization: one page describes multiplying each song's
   LTM by its age to get a "weighted total", another describes it as a
   time-weighted measure of stability. Those are different quantities. The
   weighted total is in dollar-years and scales with catalog size, which
   would make it strongly collinear with LTM and would mean bidders
   comparing "Dollar Age" across catalogs of different sizes are comparing
   nothing at all. The weighted average is in years and is size-invariant.

   Do not assume. Recompute both from the song table and check which one
   reproduces the published figure. If it is the unnormalized total, that is
   itself a finding: the platform's headline quality metric is contaminated
   by size, and the mispricing it induces is directly tradeable.

2. LTM is the denominator of every multiple on the platform, so a sync
   placement inside the trailing window inflates the denominator and makes
   an expensive catalog look cheap. Normalize before valuing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def dollar_age_weighted_average(songs: pd.DataFrame) -> float | None:
    """sum(age_i * ltm_i) / sum(ltm_i). Units: years. Size-invariant."""
    s = songs.dropna(subset=["song_age_years", "ltm_earnings"])
    total = s["ltm_earnings"].sum()
    if total <= 0:
        return None
    return float((s["song_age_years"] * s["ltm_earnings"]).sum() / total)


def dollar_age_weighted_total(songs: pd.DataFrame) -> float | None:
    """sum(age_i * ltm_i). Units: dollar-years. Scales with catalog size."""
    s = songs.dropna(subset=["song_age_years", "ltm_earnings"])
    if s.empty:
        return None
    return float((s["song_age_years"] * s["ltm_earnings"]).sum())


def diagnose_reported_dollar_age(df: pd.DataFrame) -> pd.DataFrame:
    """df: one row per listing with reported, recomputed_avg, recomputed_total,
    ltm_earnings. Returns which construction the platform is publishing."""
    out = {}
    for name, col in [("weighted_average", "recomputed_avg"),
                      ("weighted_total", "recomputed_total")]:
        sub = df.dropna(subset=["dollar_age_reported", col])
        if len(sub) < 10:
            out[name] = {"n": len(sub), "median_ratio": np.nan, "corr": np.nan}
            continue
        ratio = sub["dollar_age_reported"] / sub[col].replace(0, np.nan)
        out[name] = {
            "n": len(sub),
            "median_ratio": float(ratio.median()),   # ~1.0 => this is the construction
            "iqr_ratio": float(ratio.quantile(0.75) - ratio.quantile(0.25)),
            "corr": float(sub["dollar_age_reported"].corr(sub[col])),
        }
    sub = df.dropna(subset=["dollar_age_reported", "ltm_earnings"])
    out["_size_contamination"] = {
        "corr_reported_vs_log_ltm": float(
            sub["dollar_age_reported"].corr(np.log(sub["ltm_earnings"].clip(lower=1)))
        ),
        "n": len(sub),
    }
    return pd.DataFrame(out).T


# --------------------------------------------------------------------
# Sync normalization
# --------------------------------------------------------------------


@dataclass
class NormalizedLTM:
    raw_ltm: float
    normalized_ltm: float
    excess_attributed_to_sync: float
    method: str
    confidence: str          # high | medium | low
    detail: str


def normalize_ltm(
    monthly: pd.DataFrame | None,
    *,
    raw_ltm: float,
    source_mix: dict[str, float] | None = None,
    prior_year_total: float | None = None,
    spike_z: float = 3.0,
) -> NormalizedLTM:
    """Strip suspected one-off sync income from the valuation base.

    Preference order, best evidence first:

    high   monthly panel by source: identify sync months whose amount exceeds
           the median non-sync-adjusted month by spike_z robust deviations,
           and replace them with the trailing median.
    medium source mix only: cap the sync share at its cross-sectional median
           and treat the excess as non-recurring.
    low    LTM vs prior-year total only: if LTM is far above the prior year
           and the mix is unknown, haircut toward the prior year. This is the
           weakest inference and is flagged as such -- a genuinely growing
           streaming catalog looks identical on this evidence.

    The confidence field is not decoration. A "low" normalization should
    widen the bid engine's uncertainty band, not silently shift the point
    estimate.
    """
    if monthly is not None and not monthly.empty and "source_type" in monthly:
        m = monthly.copy()
        sync = m[m["source_type"] == "sync"]
        non_sync = m[m["source_type"] != "sync"]["amount"].sum()
        if not sync.empty:
            med = sync["amount"].median()
            mad = (sync["amount"] - med).abs().median() or 1.0
            spikes = sync[sync["amount"] > med + spike_z * 1.4826 * mad]
            excess = float((spikes["amount"] - med).sum())
            norm = float(non_sync + sync["amount"].sum() - excess)
            return NormalizedLTM(
                raw_ltm, norm, excess, "monthly_sync_spike", "high",
                f"{len(spikes)} spike month(s) trimmed to sync median {med:,.0f}",
            )

    if source_mix and "sync" in source_mix and sum(source_mix.values()) > 0:
        share = source_mix["sync"] / sum(source_mix.values())
        cap = 0.10                      # replace with the cross-sectional median once populated
        if share > cap:
            excess = raw_ltm * (share - cap)
            return NormalizedLTM(
                raw_ltm, raw_ltm - excess, excess, "sync_share_cap", "medium",
                f"sync share {share:.0%} capped at {cap:.0%}",
            )
        return NormalizedLTM(raw_ltm, raw_ltm, 0.0, "sync_share_cap", "medium",
                             f"sync share {share:.0%} within cap")

    if prior_year_total and prior_year_total > 0:
        lift = raw_ltm / prior_year_total - 1
        if lift > 0.50:
            norm = float(np.sqrt(raw_ltm * prior_year_total))   # geometric mean
            return NormalizedLTM(
                raw_ltm, norm, raw_ltm - norm, "geometric_mean_vs_prior", "low",
                f"LTM {lift:+.0%} vs prior year; could be sync or genuine growth",
            )

    return NormalizedLTM(raw_ltm, raw_ltm, 0.0, "none", "low",
                         "insufficient detail to normalize")
