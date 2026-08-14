"""Phase 3: decay curve estimation.

The valuation question is: given a catalog's earnings history, what is the
present value of its future income? That requires a decay curve, and fitting
one to this data has four traps. Each is handled explicitly below because
each of them, left alone, biases the answer in the same direction --
optimistically for the seller, pessimistically for the buyer, or both.

TRAP 1: REPORTING LAG.
Royalties arrive at the collecting society months after the play. The final
one or two quarters of every panel are therefore partial, and a curve fitted
through them sees a cliff at the end of every catalog's life. That cliff is an
artifact of the statement calendar, not of listener behaviour. `trim_partial_tail`
detects it empirically -- comparing each panel's last quarters against its own
trailing average, pooled across catalogs -- rather than assuming a fixed lag.

TRAP 2: AGE-PERIOD CONFOUNDING.
Observed income decline mixes two things: the catalog getting older, and
streaming payout rates drifting across calendar time for everyone. A per-catalog
fit attributes all of it to age. With ~1,600 panels overlapping in calendar
time, the two are separately identified by a two-way fixed effects model, and
`pooled_age_profile` extracts the age component net of market drift. Skipping
this means every catalog's decay estimate absorbs whatever the streaming
market did during its observation window.

TRAP 3: COMPOSITION MASQUERADING AS DYNAMICS.
A catalog is a portfolio of songs with different ages. One new hit plus five
old stable songs produces an aggregate curve that falls fast then flattens --
which fits a two-phase model beautifully, while no individual song behaves
that way. The two-phase fit is capturing the mix, not song-level physics.
That matters for extrapolation: composition effects do not persist the way
song-level decay does. `decompose_composition` uses the song-level annual
panel to flag catalogs where this is happening.

TRAP 4: LENGTH-BIASED SAMPLING.
Catalogs in the steep phase are young and therefore have short panels; the
catalogs with long panels have already survived into the flat tail. Fitting
each independently and pooling the results describes the age distribution of
listings, not the decay process. `shrink_estimates` partially pools each
catalog's estimate toward a prior conditioned on its characteristics, with
weight rising in panel length.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

PANEL_SQL = """
SELECT p.listing_id, p.period_start, p.total, p.domestic, p.intl, p.unreported,
       l.deal_date, l.published_date, l.first_earnings_date,
       l.ltm, l.three_years_average, l.dollar_age, l.track_count,
       l.term_family, l.term_years, l.kind
FROM fact_earnings_panel p
JOIN dim_listing l USING (listing_id)
WHERE p.total IS NOT NULL
ORDER BY p.listing_id, p.period_start
"""


def load_panels(con) -> pd.DataFrame:
    df = con.execute(PANEL_SQL).fetchdf()
    df["period_start"] = pd.to_datetime(df["period_start"])
    for c in ("deal_date", "published_date", "first_earnings_date"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_localize(None)
    return df


def detect_grain(g: pd.DataFrame) -> str:
    """Panels arrive at mixed grain -- older catalogs report monthly, newer
    ones quarterly. Detected from the modal gap, not assumed."""
    if len(g) < 3:
        return "unknown"
    gap = g["period_start"].diff().dt.days.dropna()
    if gap.empty:
        return "unknown"
    med = gap.median()
    if med <= 45:
        return "month"
    if med <= 135:
        return "quarter"
    return "year"


def to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample every panel to a common quarterly grain.

    Quarterly is the target because it is what the platform's own LTM
    reconciles against, and because monthly royalty data is dominated by
    statement-timing noise that has nothing to do with the underlying decay.
    """
    out = []
    for lid, g in df.groupby("listing_id", sort=False):
        grain = detect_grain(g)
        if grain == "unknown":
            continue          # too few points to establish a period; not fittable
        g = g.sort_values("period_start")
        if grain == "month":
            q = (g.set_index("period_start")[["total", "domestic", "intl", "unreported"]]
                   .resample("QS").sum(min_count=1).reset_index())
            # a quarter assembled from fewer than 3 months is incomplete
            counts = (g.set_index("period_start")["total"]
                        .resample("QS").count().reset_index(name="n_months"))
            q = q.merge(counts, on="period_start", how="left")
            q["partial_period"] = q["n_months"] < 3
        elif grain == "year":
            # Annual rows carry a full year of income. The curve and the
            # valuation both treat y(t) as a QUARTERLY rate, so an annual
            # total left as-is overstates the run rate fourfold and the NPV
            # with it. Convert to a quarterly-equivalent rate.
            q = g[["period_start", "total", "domestic", "intl", "unreported"]].copy()
            for c in ("total", "domestic", "intl", "unreported"):
                q[c] = q[c] / 4.0
            q["n_months"] = 12
            q["partial_period"] = False
        else:
            q = g[["period_start", "total", "domestic", "intl", "unreported"]].copy()
            q["n_months"] = 3
            q["partial_period"] = False
        q["listing_id"] = lid
        q["grain_source"] = grain
        out.append(q)
    if not out:
        return pd.DataFrame()
    q = pd.concat(out, ignore_index=True)
    meta = df.groupby("listing_id").first().reset_index()[
        ["listing_id", "deal_date", "published_date", "first_earnings_date",
         "ltm", "three_years_average", "dollar_age", "track_count",
         "term_family", "term_years", "kind"]]
    q = q.merge(meta, on="listing_id", how="left")
    q["age_years"] = ((q["period_start"] - q.groupby("listing_id")["period_start"]
                       .transform("min")).dt.days / 365.25)
    q["quarters_to_deal"] = ((q["period_start"] - q["deal_date"]).dt.days / 91.31)
    return q


def diagnose_reporting_lag(q: pd.DataFrame, max_tail: int = 4) -> pd.DataFrame:
    """Is the end of the typical panel incomplete?

    For each panel, compare each of the last `max_tail` quarters against that
    panel's own median over the preceding year. If the final quarters are
    systematically below their own recent baseline across hundreds of
    unrelated catalogs, the cause is the statement calendar, not the music.
    A ratio near 1.0 means no lag and nothing needs trimming.
    """
    rows = []
    for lid, g in q.groupby("listing_id", sort=False):
        g = g.sort_values("period_start")
        if len(g) < 8:
            continue
        for k in range(1, max_tail + 1):
            tail = g["total"].iloc[-k]
            base = g["total"].iloc[-(k + 5):-(k + 1)].median()
            if base and base > 0:
                rows.append({"listing_id": lid, "position_from_end": k,
                             "ratio": tail / base})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return (d.groupby("position_from_end")
             .agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                  q25=("ratio", lambda s: s.quantile(0.25)),
                  q75=("ratio", lambda s: s.quantile(0.75)),
                  pct_below_80=("ratio", lambda s: (s < 0.8).mean()))
             .round(3).reset_index())


def trim_partial_tail(q: pd.DataFrame, n_trim: int | None = None,
                      threshold: float = 0.85) -> tuple[pd.DataFrame, int]:
    """Drop trailing quarters that the lag diagnostic says are incomplete.

    If n_trim is None the count is chosen from the data: trim while the
    median tail ratio at that position is below `threshold`. Returns the
    trimmed panel and the number of quarters removed, so the choice is
    visible rather than buried.
    """
    if n_trim is None:
        # Compare each tail position against the *deepest* one measured rather
        # than against 1.0. A decaying catalog is legitimately below its own
        # trailing median at every position, so an absolute threshold trims
        # real decay along with the incomplete statements. What identifies a
        # partial quarter is the position being anomalous relative to the
        # plateau the ratio settles into once reporting is complete.
        diag = diagnose_reporting_lag(q, max_tail=6)
        n_trim = 0
        if len(diag) >= 3:
            plateau = diag["median_ratio"].iloc[2:].median()
            for k in range(1, len(diag) + 1):
                row = diag[diag["position_from_end"] == k]
                if row.empty:
                    break
                if row["median_ratio"].iloc[0] < threshold * plateau:
                    n_trim = k
                else:
                    break
    if n_trim == 0:
        return q.copy(), 0
    keep = []
    for lid, g in q.groupby("listing_id", sort=False):
        g = g.sort_values("period_start")
        keep.append(g.iloc[:-n_trim] if len(g) > n_trim + 3 else g)
    return pd.concat(keep, ignore_index=True), n_trim


# --------------------------------------------------------------------
# Functional forms
# --------------------------------------------------------------------
# All fitted in log space: minimizing squared error on log(income) is
# minimizing relative error on income, which is right for a quantity whose
# variance scales with its level.

def f_exponential(t, log_a, lam):
    return log_a - lam * t


def f_power(t, log_a, alpha):
    return log_a - alpha * np.log1p(t)


def f_exp_floor(t, log_a, lam, log_c):
    """Exponential decay onto a stable floor.

    This is the form that matches the stylized fact everyone describes: a
    steep fall from the promotional peak, settling into a long tail that does
    not go to zero. Pure exponential sends income to zero and undervalues a
    perpetuity; power law decays too slowly at long horizons and overvalues
    it. The floor parameter is exactly the quantity a perpetuity buyer is
    actually purchasing.
    """
    a, c = np.exp(log_a), np.exp(log_c)
    return np.log(np.maximum((a - c) * np.exp(-lam * t) + c, 1e-9))


def f_two_phase(t, log_a, lam1, lam2, tau):
    tau = np.clip(tau, 0.5, 12.0)
    t = np.asarray(t, dtype=float)
    early = log_a - lam1 * t
    late = log_a - lam1 * tau - lam2 * (t - tau)
    return np.where(t <= tau, early, late)


FORMS = {
    "exponential": (f_exponential, 2),
    "power":       (f_power, 2),
    "exp_floor":   (f_exp_floor, 3),
    "two_phase":   (f_two_phase, 4),
}


def _p0_bounds(form: str, t: np.ndarray, y: np.ndarray):
    """Parameter bounds. Decay rates are constrained NON-NEGATIVE, deliberately.

    A catalog can genuinely have grown over its observed window -- a track
    catching a playlist, a sync landing, an artist breaking. Fitting that as a
    negative decay rate describes the history accurately and then, projected
    forty years forward, prices the catalog as a compounding perpetuity. That
    is not a valuation, it is a divergent series: unbounded growth extrapolated
    from a handful of noisy quarters.

    Constraining lambda >= 0 means the most optimistic projectable case is
    FLAT income. A catalog whose history is genuinely rising fits at lambda
    close to zero and is valued as a level perpetuity, which is already an
    aggressive assumption for a decaying asset. Observed growth belongs in the
    level parameter and in the decision to bid, not in an extrapolated trend.
    """
    log_a0 = float(np.log(max(y[0], 1e-6)))
    log_ymax = float(np.log(max(y.max(), 1e-6)))
    if form == "exponential":
        return [log_a0, 0.15], ([-30, 0.0], [30, 3.0])
    if form == "power":
        return [log_a0, 0.5], ([-30, 0.0], [30, 5.0])
    if form == "exp_floor":
        floor0 = float(np.log(max(np.percentile(y, 25), 1e-6)))
        floor0 = min(floor0, log_ymax)
        # the asymptote cannot exceed the highest level ever observed
        return [log_a0, 0.4, floor0], ([-30, 0.0, -30], [30, 5.0, log_ymax])
    if form == "two_phase":
        return [log_a0, 0.35, 0.05, 4.0], ([-30, 0.0, 0.0, 0.5], [30, 5, 5, 12])
    raise ValueError(form)


@dataclass
class CatalogFit:
    listing_id: int
    form: str
    params: np.ndarray
    n_obs: int
    span_years: float
    rmse_log: float
    aic: float
    bic: float
    cv_rmse: float | None = None
    converged: bool = True
    # interpretable summaries
    half_life_years: float | None = None
    floor_share: float | None = None      # asymptote as a share of current run rate
    implied_5y_retention: float | None = None


def _summarize(form: str, p: np.ndarray, t_end: float) -> dict:
    out: dict = {}
    y_now = np.exp(_eval(form, np.array([t_end]), p))[0]
    y_5 = np.exp(_eval(form, np.array([t_end + 5.0]), p))[0]
    out["implied_5y_retention"] = float(y_5 / y_now) if y_now > 0 else None
    if form == "exponential":
        out["half_life_years"] = float(np.log(2) / p[1]) if p[1] > 1e-6 else np.inf
    elif form == "exp_floor":
        out["half_life_years"] = float(np.log(2) / p[1]) if p[1] > 1e-6 else np.inf
        out["floor_share"] = float(np.exp(p[2]) / y_now) if y_now > 0 else None
    elif form == "two_phase":
        lam = p[2] if t_end > p[3] else p[1]
        out["half_life_years"] = float(np.log(2) / lam) if lam > 1e-6 else np.inf
    return out


def _eval(form: str, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return FORMS[form][0](t, *p)


def fit_catalog(t: np.ndarray, y: np.ndarray, form: str, *,
                listing_id: int = -1, cv_holdout: int = 4) -> CatalogFit | None:
    """Fit one form to one catalog. Returns None if there is not enough data.

    Model selection uses a within-catalog temporal holdout (the last
    `cv_holdout` quarters) rather than AIC alone. With 10-30 noisy points AIC
    reliably prefers the most flexible form; a forward holdout asks the
    question that actually matters, which is whether the curve extrapolates.
    """
    k = FORMS[form][1]
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[mask], y[mask]
    if len(t) < k + 3:
        return None
    ly = np.log(y)

    def _fit(tt, ll):
        p0, bounds = _p0_bounds(form, tt, np.exp(ll))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(
                FORMS[form][0], tt, ll, p0=p0, bounds=bounds, maxfev=20000)
        return popt

    try:
        popt = _fit(t, ly)
    except Exception:
        return None

    resid = ly - _eval(form, t, popt)
    n = len(t)
    rss = float(np.sum(resid ** 2))
    sigma2 = max(rss / n, 1e-12)
    aic = n * np.log(sigma2) + 2 * k
    bic = n * np.log(sigma2) + k * np.log(n)

    cv = None
    if n >= k + 3 + cv_holdout:
        try:
            popt_tr = _fit(t[:-cv_holdout], ly[:-cv_holdout])
            e = ly[-cv_holdout:] - _eval(form, t[-cv_holdout:], popt_tr)
            cv = float(np.sqrt(np.mean(e ** 2)))
        except Exception:
            cv = None

    fit = CatalogFit(listing_id=listing_id, form=form, params=popt, n_obs=n,
                     span_years=float(t.max() - t.min()),
                     rmse_log=float(np.sqrt(rss / n)), aic=aic, bic=bic,
                     cv_rmse=cv)
    for key, val in _summarize(form, popt, float(t.max())).items():
        setattr(fit, key, val)
    return fit


def fit_all(q: pd.DataFrame, forms: tuple[str, ...] = tuple(FORMS),
            min_obs: int = 8) -> pd.DataFrame:
    rows = []
    for lid, g in q.groupby("listing_id", sort=False):
        g = g.sort_values("period_start")
        if len(g) < min_obs:
            continue
        t = g["age_years"].to_numpy(float)
        y = g["total"].to_numpy(float)
        for form in forms:
            f = fit_catalog(t, y, form, listing_id=int(lid))
            if f is None:
                continue
            rows.append({
                "listing_id": f.listing_id, "form": f.form, "n_obs": f.n_obs,
                "span_years": round(f.span_years, 2),
                "rmse_log": round(f.rmse_log, 4), "aic": round(f.aic, 2),
                "bic": round(f.bic, 2),
                "cv_rmse": None if f.cv_rmse is None else round(f.cv_rmse, 4),
                "half_life_years": f.half_life_years,
                "floor_share": f.floor_share,
                "implied_5y_retention": f.implied_5y_retention,
                "params": f.params.tolist(),
            })
    return pd.DataFrame(rows)


def select_best(fits: pd.DataFrame, criterion: str = "cv_rmse") -> pd.DataFrame:
    """Pick one form per catalog. Prefer the forward holdout; fall back to BIC
    where the panel was too short to hold anything out."""
    if fits.empty:
        return fits
    d = fits.copy()
    d["_crit"] = d[criterion] if criterion in d else np.nan
    d["_fallback"] = d["bic"]
    # Tie-break toward parsimony. A 4-parameter two_phase will edge out a
    # 2-parameter exponential on holdout RMSE by a rounding error on noisy
    # panels; requiring a material improvement stops the flexible form
    # winning by accident, which matters because the forms extrapolate very
    # differently even when they fit the observed window equally well.
    d["_penalty"] = d["form"].map(
        {"exponential": 1.00, "power": 1.00, "exp_floor": 1.02, "two_phase": 1.08}
    ).fillna(1.0)
    d["_crit"] = d["_crit"] * d["_penalty"]
    d["_key"] = d["_crit"].fillna(d["_fallback"] + 1e6)
    best = d.sort_values("_key").groupby("listing_id", as_index=False).first()
    return best.drop(columns=["_crit", "_fallback", "_key"])


# --------------------------------------------------------------------
# Age vs calendar-time decomposition
# --------------------------------------------------------------------

def pooled_age_profile(q: pd.DataFrame, *, min_obs: int = 8,
                       max_age: float = 25.0) -> pd.DataFrame:
    """Separate catalog ageing from market-wide drift.

    log(income) ~ catalog FE + calendar-quarter FE + age bin FE

    The catalog fixed effect absorbs scale, the calendar effect absorbs
    everything that moved for all catalogs at once -- streaming payout rate
    changes, platform mix shifts, statement policy -- and what remains on the
    age bins is the decay attributable to the catalog getting older.

    Note on identification: age, period and cohort are linearly dependent
    (cohort = period - age), so all three cannot enter together. Cohort is
    omitted here, which means any genuine vintage effect (2015 catalogs being
    different animals from 2022 catalogs) is absorbed into the age profile.
    That is the standard restriction and it is a real assumption, not a
    technicality.
    """
    d = q[q["total"] > 0].copy()
    counts = d.groupby("listing_id")["total"].transform("size")
    d = d[counts >= min_obs]
    d = d[d["age_years"] <= max_age]
    if d["listing_id"].nunique() < 30:
        return pd.DataFrame()

    d["ly"] = np.log(d["total"])
    d["age_bin"] = pd.cut(d["age_years"], np.arange(0, max_age + 1, 1.0),
                          right=False)
    d["cal_q"] = d["period_start"].dt.to_period("Q").astype(str)

    # within-transform by catalog, then OLS on age + calendar dummies. Doing
    # the catalog demeaning by hand keeps the design matrix at a few hundred
    # columns instead of a few thousand.
    d["ly_dm"] = d["ly"] - d.groupby("listing_id")["ly"].transform("mean")
    X = pd.get_dummies(d[["age_bin", "cal_q"]].astype(str), drop_first=True,
                       dtype=float)
    X = X.sub(X.groupby(d["listing_id"].values).transform("mean"))
    import statsmodels.api as sm
    res = sm.OLS(d["ly_dm"].to_numpy(float), sm.add_constant(X.to_numpy(float)),
                 ).fit(cov_type="cluster",
                       cov_kwds={"groups": d["listing_id"].to_numpy()})
    names = ["const"] + list(X.columns)
    coefs = pd.Series(res.params, index=names)

    age_rows = []
    for name in names:
        if name.startswith("age_bin_"):
            lo = float(name.split("[")[1].split(",")[0])
            age_rows.append({"age_start": lo, "log_effect": coefs[name],
                             "level": float(np.exp(coefs[name]))})
    prof = pd.DataFrame(age_rows).sort_values("age_start").reset_index(drop=True)
    if not prof.empty:
        prof["yoy_decay"] = prof["level"].pct_change()
    return prof


def market_drift(q: pd.DataFrame, *, min_obs: int = 8) -> pd.DataFrame:
    """The calendar-time component alone: what streaming did to everyone.

    This is worth looking at on its own. If it is large and negative, the
    "decay" most bidders think they are seeing in a catalog is substantially
    a market-wide payout phenomenon, and it applies to whatever they buy next
    as well.
    """
    d = q[q["total"] > 0].copy()
    counts = d.groupby("listing_id")["total"].transform("size")
    d = d[counts >= min_obs]
    if d["listing_id"].nunique() < 30:
        return pd.DataFrame()
    d["ly"] = np.log(d["total"])
    d["ly_dm"] = d["ly"] - d.groupby("listing_id")["ly"].transform("mean")
    d["age_bin"] = pd.cut(d["age_years"], np.arange(0, 26, 1.0), right=False)
    d["cal_q"] = d["period_start"].dt.to_period("Q").astype(str)
    X = pd.get_dummies(d[["age_bin", "cal_q"]].astype(str), drop_first=True,
                       dtype=float)
    X = X.sub(X.groupby(d["listing_id"].values).transform("mean"))
    import statsmodels.api as sm
    res = sm.OLS(d["ly_dm"].to_numpy(float),
                 sm.add_constant(X.to_numpy(float))).fit()
    names = ["const"] + list(X.columns)
    coefs = pd.Series(res.params, index=names)
    counts = d.groupby("cal_q")["listing_id"].nunique()
    rows = [{"quarter": n.replace("cal_q_", ""), "log_effect": coefs[n]}
            for n in names if n.startswith("cal_q_")]
    out = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)
    if out.empty:
        return out
    # Coefficients are relative to an arbitrary omitted quarter, which makes
    # the raw levels uninterpretable. Rebase to the first quarter shown so the
    # series reads as an index, and carry the catalog count -- early and very
    # recent quarters rest on few panels and should not be over-read.
    out["log_effect"] = out["log_effect"] - out["log_effect"].iloc[0]
    out["index_level"] = np.exp(out["log_effect"])
    out["n_catalogs"] = out["quarter"].map(counts).fillna(0).astype(int)
    return out[out["n_catalogs"] >= 20].reset_index(drop=True)


# --------------------------------------------------------------------
# Partial pooling
# --------------------------------------------------------------------

def shrink_estimates(best: pd.DataFrame, meta: pd.DataFrame, *,
                     k_prior: float = 12.0) -> pd.DataFrame:
    """Shrink each catalog's retention estimate toward a characteristics prior.

    A catalog with 9 quarters of history has a decay estimate with enormous
    standard error, and the length-bias problem means those short panels are
    disproportionately young, steep-decay catalogs. Taking their point
    estimates at face value and averaging produces a decay curve that
    describes the listing pipeline rather than the asset class.

    weight = n_obs / (n_obs + k_prior). At the default k, a catalog needs 12
    quarters to put equal weight on its own history and the prior.
    """
    if best.empty:
        return best
    d = best.merge(meta, on="listing_id", how="left")
    d = d[np.isfinite(d["implied_5y_retention"])]
    d["implied_5y_retention"] = d["implied_5y_retention"].clip(0.01, 3.0)
    d["log_ret"] = np.log(d["implied_5y_retention"])

    feats = ["dollar_age", "log_track_count", "share_streaming"]
    for c in feats:
        if c not in d:
            d[c] = 0.0
        d[c] = d[c].fillna(d[c].median() if d[c].notna().any() else 0.0)

    import statsmodels.api as sm
    X = sm.add_constant(d[feats].astype(float))
    ok = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(d["log_ret"])
    prior_model = sm.OLS(d.loc[ok, "log_ret"].astype(float), X[ok]).fit()
    d["prior_log_ret"] = np.nan
    d.loc[ok, "prior_log_ret"] = prior_model.predict(X[ok])
    d["prior_log_ret"] = d["prior_log_ret"].fillna(d["log_ret"].median())

    w = d["n_obs"] / (d["n_obs"] + k_prior)
    d["shrink_weight"] = w
    d["retention_5y_raw"] = d["implied_5y_retention"]
    d["retention_5y_shrunk"] = np.exp(w * d["log_ret"] + (1 - w) * d["prior_log_ret"])
    return d


# --------------------------------------------------------------------
# Composition check
# --------------------------------------------------------------------

COMPOSITION_SQL = """
SELECT o.listing_id, y.year_index, y.member, y.amount
FROM obs_earnings_year y
JOIN obs_listing o USING (obs_id)
WHERE y.dimension = 'song' AND y.amount IS NOT NULL
"""


def decompose_composition(con) -> pd.DataFrame:
    """Flag catalogs whose aggregate curve is driven by song mix.

    Using the five-year song-level panel: if the top song's share of income
    has moved sharply, the catalog's aggregate decay is a composition effect.
    Those catalogs should not have their aggregate curve extrapolated, because
    the mix shift does not continue -- once the new hit has decayed into the
    pack, the portfolio behaves like the pack.
    """
    d = con.execute(COMPOSITION_SQL).fetchdf()
    if d.empty:
        return d
    tot = d.groupby(["listing_id", "year_index"])["amount"].transform("sum")
    d["share"] = d["amount"] / tot.replace(0, np.nan)
    top = (d.sort_values("share", ascending=False)
             .groupby(["listing_id", "year_index"], as_index=False).first())
    piv = top.pivot(index="listing_id", columns="year_index", values="share")
    out = pd.DataFrame(index=piv.index)
    first_col, last_col = piv.columns.min(), piv.columns.max()
    out["top_share_first"] = piv[first_col]
    out["top_share_last"] = piv[last_col]
    out["top_share_change"] = out["top_share_last"] - out["top_share_first"]
    hhi = (d.assign(sq=d["share"] ** 2)
             .groupby(["listing_id", "year_index"])["sq"].sum()
             .unstack())
    out["hhi_first"] = hhi[first_col]
    out["hhi_last"] = hhi[last_col]
    out["composition_shift"] = (out["top_share_change"].abs() > 0.20)
    return out.reset_index()
