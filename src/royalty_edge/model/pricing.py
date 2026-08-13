"""Phase 2: what does the market actually price?

Four nested models, fitted in order, because the interesting number is not
any single R-squared but how much each block adds over the one before.

    M0  quarter fixed effects only        -- how much is just the vintage?
    M1  + catalog fundamentals            -- the pre-bid fair-value model
    M2  + seller's reserve                -- does the ask move the clearing price?
    M3  + published anchor                -- does the platform's number move it?
    M4  + realized bidder count           -- descriptive only, NOT for the bid engine

The bid-engine model is M3, not M4. Bidder count is realized at the same time
as the price and is partly caused by it; conditioning on it to forecast a
clearing price is conditioning on the outcome. M4 exists to decompose what
happened, not to predict what will happen.

Two methodological points that matter more than the choice of regressors:

VALIDATION IS TEMPORAL, NOT RANDOM. K-fold cross-validation on this panel
leaks: multiples move with the rate environment and with platform-level
pricing drift, so a random fold trains on 2025 to predict 2023. Every
out-of-sample number here comes from training on everything before a cutoff
and testing after it, which is the only split that matches how the model
would actually be used.

STANDARD ERRORS ARE CLUSTERED BY SELLER. Sellers list repeatedly and their
catalogs share unobserved quality, so residuals are correlated within seller.
Unclustered SEs on this frame would be roughly half their honest size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import ANCHOR, COMPETITION, FUNDAMENTALS, SELLER_ASK

DEPENDENT = "log_multiple"


@dataclass
class FitResult:
    name: str
    features: list[str]
    n: int
    r2: float
    adj_r2: float
    rmse_in: float
    params: pd.DataFrame
    model: object = field(repr=False, default=None)
    design_cols: list[str] = field(default_factory=list, repr=False)


def _design(df: pd.DataFrame, features: list[str], *, quarter_fe: bool,
            drop_first_quarter: bool = True,
            columns: list[str] | None = None) -> pd.DataFrame:
    X = df[features].astype(float).copy() if features else pd.DataFrame(index=df.index)
    if quarter_fe:
        d = pd.get_dummies(df["deal_quarter"], prefix="q",
                           drop_first=drop_first_quarter, dtype=float)
        X = pd.concat([X, d], axis=1)
    X = sm.add_constant(X, has_constant="add")
    if columns is not None:
        # align a holdout design to the training design: unseen quarters
        # become all-zero columns rather than silently reindexing the matrix
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def fit_model(df: pd.DataFrame, features: list[str], *, name: str,
              quarter_fe: bool = True, cluster: str | None = "seller_id") -> FitResult:
    X = _design(df, features, quarter_fe=quarter_fe)
    y = df[DEPENDENT].astype(float)
    ok = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
    X, y = X[ok], y[ok]

    model = sm.OLS(y, X)
    if cluster and cluster in df.columns:
        groups = df.loc[X.index, cluster].fillna(-1).astype(int)
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = model.fit(cov_type="HC1")

    params = pd.DataFrame({
        "coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues,
    })
    # Quarter dummies are nuisance parameters; keep them out of the printed
    # table but leave them in the fit.
    params = params[~params.index.str.startswith("q_")]

    resid = y - res.predict(X)
    return FitResult(name=name, features=features, n=int(ok.sum()),
                     r2=res.rsquared, adj_r2=res.rsquared_adj,
                     rmse_in=float(np.sqrt((resid ** 2).mean())),
                     params=params.round(4), model=res,
                     design_cols=list(X.columns))


MODEL_LADDER = [
    ("M0 quarter FE only",        []),
    ("M1 + fundamentals",         FUNDAMENTALS),
    ("M2 + seller reserve",       FUNDAMENTALS + SELLER_ASK),
    ("M3 + published anchor",     FUNDAMENTALS + SELLER_ASK + ANCHOR),
    ("M4 + realized bidders",     FUNDAMENTALS + SELLER_ASK + ANCHOR + COMPETITION),
]


def fit_ladder(df: pd.DataFrame, *, cluster: str | None = "seller_id") -> list[FitResult]:
    return [fit_model(df, feats, name=name, cluster=cluster)
            for name, feats in MODEL_LADDER]


def ladder_table(results: list[FitResult]) -> pd.DataFrame:
    rows = []
    prev = None
    for r in results:
        rows.append({
            "model": r.name, "n": r.n, "k": len(r.features),
            "R2": round(r.r2, 4), "adj_R2": round(r.adj_r2, 4),
            "dR2": None if prev is None else round(r.r2 - prev, 4),
            "rmse_log": round(r.rmse_in, 4),
        })
        prev = r.r2
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# Temporal validation
# --------------------------------------------------------------------

def temporal_holdout(df: pd.DataFrame, features: list[str], *,
                     cutoff: str = "2025-01-01", quarter_fe: bool = True,
                     cluster: str | None = "seller_id") -> dict:
    """Train before cutoff, test after. The only honest split for this panel."""
    dd = pd.to_datetime(df["deal_date"], utc=True)
    cut = pd.Timestamp(cutoff, tz="UTC")
    tr, te = df[dd < cut], df[dd >= cut]
    if len(tr) < 50 or len(te) < 20:
        return {"error": f"insufficient split: train={len(tr)} test={len(te)}"}

    fit = fit_model(tr, features, name="holdout", quarter_fe=quarter_fe, cluster=cluster)
    Xte = _design(te, features, quarter_fe=quarter_fe, columns=fit.design_cols)
    yte = te[DEPENDENT].astype(float)
    ok = np.isfinite(Xte.to_numpy()).all(axis=1) & np.isfinite(yte.to_numpy())
    pred = fit.model.predict(Xte[ok])
    err = yte[ok] - pred

    # Naive benchmark: predict the training median multiple for everything.
    # A model that cannot beat this is not a model.
    naive = np.full(ok.sum(), tr[DEPENDENT].median())
    naive_err = yte[ok] - naive

    return {
        "n_train": len(tr), "n_test": int(ok.sum()), "cutoff": cutoff,
        "rmse_test": float(np.sqrt((err ** 2).mean())),
        "mae_test": float(err.abs().mean()),
        "rmse_naive": float(np.sqrt((naive_err ** 2).mean())),
        "r2_test": float(1 - (err ** 2).sum() / ((yte[ok] - yte[ok].mean()) ** 2).sum()),
        "skill_vs_naive": float(1 - (err ** 2).sum() / (naive_err ** 2).sum()),
        # in multiple terms, not log terms, for interpretation
        "median_abs_pct_err": float((np.exp(err.abs()) - 1).median()),
    }


# --------------------------------------------------------------------
# The anchoring question
# --------------------------------------------------------------------

def anchor_analysis(df: pd.DataFrame) -> dict:
    """Does the platform's published median multiple move the clearing price?

    IMPORTANT CAVEAT, and it is not a footnote: anchor_disabled is almost
    certainly NOT randomly assigned. The platform most plausibly disables the
    median when it has no good comparables -- unusual catalogs, thin genres,
    odd terms. Those catalogs differ from the rest in ways the covariates only
    partly capture, so the disabled-vs-enabled contrast is a correlation with
    a selection story attached, not a causal effect. Treat the number below as
    an upper bound on the anchoring effect and be suspicious of it.

    The within-enabled elasticity (how much clearing multiple moves per 1%
    move in the anchor, holding fundamentals fixed) is the more defensible
    quantity and is reported alongside.
    """
    out: dict = {}
    on = df[df["anchor_disabled"] == 0]
    off = df[df["anchor_disabled"] == 1]
    out["n_anchor_shown"] = len(on)
    out["n_anchor_hidden"] = len(off)
    if len(off) < 30 or len(on) < 30:
        out["note"] = "too few in one arm for a meaningful contrast"
        return out

    out["median_mult_shown"] = float(on["multiple_gross"].median())
    out["median_mult_hidden"] = float(off["multiple_gross"].median())
    out["raw_gap_pct"] = float(out["median_mult_hidden"] / out["median_mult_shown"] - 1)

    # dispersion: if the anchor coordinates bidders, prices should be tighter
    # when it is shown. This is the cleaner test -- it does not require the
    # levels to be comparable across the two groups.
    out["iqr_log_mult_shown"] = float(on["log_multiple"].quantile(0.75)
                                      - on["log_multiple"].quantile(0.25))
    out["iqr_log_mult_hidden"] = float(off["log_multiple"].quantile(0.75)
                                       - off["log_multiple"].quantile(0.25))
    out["dispersion_ratio"] = out["iqr_log_mult_hidden"] / out["iqr_log_mult_shown"]

    # conditional gap, controlling for fundamentals
    adj = fit_model(df, FUNDAMENTALS + ["anchor_disabled"],
                    name="anchor_adjusted")
    if "anchor_disabled" in adj.params.index:
        row = adj.params.loc["anchor_disabled"]
        out["adjusted_gap_log"] = float(row["coef"])
        out["adjusted_gap_pct"] = float(np.exp(row["coef"]) - 1)
        out["adjusted_gap_p"] = float(row["p"])

    # elasticity within the shown group
    shown = df[(df["anchor_disabled"] == 0) & (df["anchor_missing"] == 0)]
    if len(shown) > 100:
        el = fit_model(shown, FUNDAMENTALS + ["log_anchor"], name="anchor_elasticity")
        if "log_anchor" in el.params.index:
            r = el.params.loc["log_anchor"]
            out["anchor_elasticity"] = float(r["coef"])
            out["anchor_elasticity_se"] = float(r["se"])
            out["anchor_elasticity_p"] = float(r["p"])
    return out


# --------------------------------------------------------------------
# The mispricing screen -- the actual Phase 2 deliverable
# --------------------------------------------------------------------

def residual_screen(df: pd.DataFrame, fit: FitResult, *, top_n: int = 25
                    ) -> pd.DataFrame:
    """Residuals from the fair-value model, in percentage terms.

    A large negative residual means the lot cleared well below what its
    characteristics predict -- that is where a bidder got a bargain, and by
    extension where the model says to look. A large positive residual is a lot
    the market overpaid for.

    This is in-sample and therefore descriptive. It is a hypothesis generator
    for Phase 3, not a signal. The out-of-sample version is what the bid
    engine will use.
    """
    X = _design(df, fit.features, quarter_fe=True, columns=fit.design_cols)
    ok = np.isfinite(X.to_numpy()).all(axis=1)
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    pred[ok] = fit.model.predict(X[ok])

    out = df.loc[ok, ["listing_id", "title", "kind", "term_family", "deal_date",
                      "ltm", "dollar_age", "multiple_gross", "anchor_multiple",
                      "n_unique_bidders", "spike_flag"]].copy()
    out["predicted_multiple"] = np.exp(pred[ok])
    out["residual_log"] = df.loc[ok, DEPENDENT] - pred[ok]
    out["mispricing_pct"] = np.exp(out["residual_log"]) - 1
    return out.sort_values("mispricing_pct")


def spike_check(df: pd.DataFrame) -> pd.DataFrame:
    """The central thesis test: does the market pay for a hot LTM?

    If bidders anchor on raw LTM without normalizing, catalogs whose trailing
    year ran above their three-year average should clear at a similar multiple
    to everyone else -- which means they are overpaying in normalized terms.
    Comparing multiple_gross (on raw LTM) against multiple_normalized (on the
    3-year base) across spike buckets shows whether the market is doing this
    normalization or not.
    """
    d = df[df["ltm_vs_3yr_missing"] == 0].copy()
    d["spike_bucket"] = pd.cut(
        d["ltm_vs_3yr"], [0, 0.8, 0.95, 1.05, 1.25, 1.5, 10],
        labels=["<0.8 declining", "0.8-0.95", "0.95-1.05 flat",
                "1.05-1.25", "1.25-1.5", ">1.5 hot"])
    g = d.groupby("spike_bucket", observed=True).agg(
        n=("listing_id", "count"),
        med_mult_raw=("multiple_gross", "median"),
        med_mult_normalized=("multiple_normalized", "median"),
        med_dollar_age=("dollar_age", "median"),
        med_bidders=("n_unique_bidders", "median"),
    ).round(3)
    g["normalized_premium"] = (g["med_mult_normalized"] / g["med_mult_raw"] - 1).round(3)
    return g.reset_index()
