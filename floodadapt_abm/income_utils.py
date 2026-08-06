"""
income_utils.py
===============
Helpers to derive per-agent **income percentiles** for a case study.

The engine's ``income_mode="synthetic_lognormal"`` mirrors native DYNAMO-M:
each household reads its income off a regional lognormal distribution at
its *income percentile* (natively supplied by ``household_incomes.npy``,
``coastal_nodes.py:598-603``, with a uniform fallback).  These helpers
build that percentile array from real data — e.g. ACS 5-year median
household income per census block group joined to the FIAT building
footprints — so household economic heterogeneity reflects the study area
instead of the uniform fallback.

Typical Charleston workflow (see notebook 2):

1. fetch ACS block-group median household income for the study counties,
2. spatially join building footprints to block groups
   (:func:`join_buildings_to_regions`),
3. convert each building's block-group income into its percentile within
   the regional income distribution
   (:func:`percentiles_from_income_values`), with a small jitter so
   households within one block group are not identical,
4. save with ``np.save("income_percentiles_<site>.npy", pct)`` and pass to
   ``SimulationEngine(..., income_percentile_per_agent=pct)``.

When the lookup table carries no building locations (the shipped ``.nc``
schema does not), the spatial join above is impossible.  For that case
:func:`percentiles_from_value_proxy` derives percentiles from the building
value (``max_pot_dmg``) through a *noisy* rank link (Gaussian copula) — an
explicit, calibratable assumption rather than data.

Two independent inputs, calibrate them separately
-------------------------------------------------
The synthetic income model takes **two** inputs, and they need different
data:

1. the **marginal distribution** — how much a household at percentile *p*
   earns, set by ``DecisionConfig.median_income`` and
   ``mean_median_inc_ratio`` (``mu = ln(median)``,
   ``sd = sqrt(2 ln(mean/median))``);
2. the **rank assignment** — which *building* sits at percentile *p*,
   supplied by ``income_percentile_per_agent``.

Only the second needs building locations.  The first is a property of the
study area as a whole and is measurable from published county totals with
no geometry at all: :func:`fetch_acs_county_income` reads the county median
(ACS table B19013) and the mean (aggregate income B19025 over households
B11001), which is exactly the ``(median, mean/median)`` pair the model
wants.  :func:`fetch_acs_income_brackets` and
:func:`lognormal_bracket_shares` then check whether the fitted lognormal
actually reproduces the observed income histogram (ACS table B19001).

So the percentile tiers (A/B/C) describe *rank* quality only.  Calibrating
the marginal is orthogonal and worth doing at every tier, including the
value-proxy fallback.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "percentiles_from_income_values",
    "percentiles_from_value_proxy",
    "join_buildings_to_regions",
    "derive_income_percentiles",
    "ACS_B19001_EDGES",
    "ACS_B19001_LABELS",
    "fetch_acs_county_income",
    "fetch_acs_income_brackets",
    "lognormal_bracket_shares",
    "bracket_fit_distance",
]


def percentiles_from_income_values(
    income_values: np.ndarray,
    jitter: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """
    Convert per-building income values into income percentiles in [1, 99].

    Each building's percentile is the empirical rank of its (block-group)
    income within the distribution of all supplied values — buildings in
    richer areas land on higher percentiles of the regional income
    distribution.  A uniform integer jitter of ``+/- jitter`` percentile
    points (seeded, reproducible) breaks the ties inside a block group so
    households are not economically identical.

    Parameters
    ----------
    income_values : np.ndarray
        Income value per building (e.g. its block group's ACS median
        household income), shape ``(n_buildings,)``.  ``NaN`` entries
        (buildings without data) receive the median percentile ``50``
        before jitter.
    jitter : int
        Half-width of the percentile jitter band; ``0`` disables it.
        Default ``5``.
    seed : int
        Seed for the jitter draw.  Default ``0``.

    Returns
    -------
    percentiles : np.ndarray[int64]
        Income percentile per building, clipped to ``[1, 99]``.
    """
    values = np.asarray(income_values, dtype=np.float64)
    n = values.shape[0]
    valid = ~np.isnan(values)

    percentiles = np.full(n, 50.0)
    if valid.sum() > 1:
        # Empirical percentile: average mid-rank among the valid values
        # (ties share one rank, so a whole block group pins to a single
        # percentile before jitter instead of spreading arbitrarily).
        vals = values[valid]
        sorted_vals = np.sort(vals)
        lo = np.searchsorted(sorted_vals, vals, side="left")
        hi = np.searchsorted(sorted_vals, vals, side="right")
        avg_rank = (lo + hi - 1) / 2.0
        percentiles[valid] = (avg_rank + 0.5) / vals.shape[0] * 100.0

    if jitter > 0:
        rng = np.random.default_rng(seed)
        percentiles = percentiles + rng.integers(-jitter, jitter + 1, n)

    return np.clip(np.round(percentiles), 1, 99).astype(np.int64)


def percentiles_from_value_proxy(
    values: np.ndarray,
    rank_correlation: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """
    Derive income percentiles from a building-value proxy (no geometry).

    Fallback for lookup tables that carry no building locations, so real
    income data cannot be joined spatially.  The only per-agent covariate in
    the table is the building value (``max_pot_dmg``); household income and
    housing value are correlated, but far from perfectly (rank correlations
    of roughly 0.4 to 0.6 in household-finance data).  This helper encodes
    exactly that: a *noisy* rank link between building value and income
    percentile via a Gaussian copula:

    ``z = rho * zscore(rank(values)) + sqrt(1 - rho^2) * noise``

    and the income percentile is the rank of ``z``.  ``rank_correlation``
    (Spearman's rho, i.e. the correlation between the *orderings* rather
    than the values) is the single knob:

    * ``rho = 0`` — independence: identical to the engine's uniform
      percentile fallback (zero-assumption option);
    * ``rho = 0.5`` (default) — plausible income-to-value sorting without
      recreating the degenerate ``income == value`` identity;
    * ``rho = 1`` — percentile strictly follows building value.

    This is an assumption, not data.  Replace with a spatial join of real
    income data (:func:`derive_income_percentiles`) when building
    footprints are available, and treat ``rank_correlation`` as a
    calibration target (bounds ~[0.3, 0.7]) until then.

    Parameters
    ----------
    values : np.ndarray
        Per-agent building value proxy (e.g. ``max_pot_dmg``), shape
        ``(n_agents,)``.  ``NaN`` entries receive the median rank before
        the copula blend.
    rank_correlation : float
        Target Spearman rank correlation ``rho`` in ``[0, 1]`` between
        ``values`` and the returned percentiles.  Default ``0.5``.
    seed : int
        Seed for the dedicated noise RNG (offline helper; never touches
        the engine's RNG streams).  Default ``0``.

    Returns
    -------
    percentiles : np.ndarray[int64]
        Income percentile per agent, clipped to ``[1, 99]``.
    """
    if not 0.0 <= rank_correlation <= 1.0:
        raise ValueError(
            f"rank_correlation must be in [0, 1]; got {rank_correlation}."
        )
    vals = np.asarray(values, dtype=np.float64)
    n = vals.shape[0]

    # Normal scores of the value ranks (ties -> average mid-rank; NaN ->
    # median rank), mapped through the standard-normal quantile function.
    order = np.full(n, 0.5)
    valid = ~np.isnan(vals)
    if valid.sum() > 1:
        v = vals[valid]
        sorted_v = np.sort(v)
        lo = np.searchsorted(sorted_v, v, side="left")
        hi = np.searchsorted(sorted_v, v, side="right")
        avg_rank = (lo + hi - 1) / 2.0
        order[valid] = (avg_rank + 0.5) / v.shape[0]
    z_rank = _norm_ppf(order)

    rho = float(rank_correlation)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    z = rho * z_rank + np.sqrt(1.0 - rho * rho) * noise

    # Percentile = empirical rank of the blended score.
    pct = (np.argsort(np.argsort(z)) + 0.5) / n * 100.0
    return np.clip(np.round(pct), 1, 99).astype(np.int64)


def _norm_ppf(q: np.ndarray) -> np.ndarray:
    """
    Standard-normal quantile function (inverse CDF) via scipy.

    Parameters
    ----------
    q : np.ndarray
        Probabilities strictly inside ``(0, 1)``.

    Returns
    -------
    np.ndarray
        Standard-normal quantiles of ``q``.
    """
    from scipy.special import ndtri  # scipy is a core dependency

    return ndtri(np.clip(q, 1e-9, 1.0 - 1e-9))


def join_buildings_to_regions(
    buildings,
    regions,
    income_column: str,
    predicate: str = "intersects",
):
    """
    Spatially join building footprints to income regions (block groups).

    Thin wrapper around ``geopandas.sjoin`` returning the region income per
    building, aligned with the ``buildings`` row order (first match wins for
    buildings straddling a boundary; ``NaN`` where no region matches).

    Parameters
    ----------
    buildings : geopandas.GeoDataFrame
        Building footprints in the order of the lookup table's residential
        ``object_id``s (one row per agent).
    regions : geopandas.GeoDataFrame
        Income regions (e.g. census block groups) carrying
        ``income_column``.
    income_column : str
        Column of ``regions`` holding the income value (e.g. ACS median
        household income).
    predicate : str
        Spatial predicate for the join.  Default ``"intersects"``.

    Returns
    -------
    income_per_building : np.ndarray[float64]
        Region income per building (``NaN`` where unmatched).
    """
    try:
        import geopandas as gpd  # deferred: keep module import light
    except ImportError as exc:
        raise ImportError(
            "join_buildings_to_regions requires geopandas, which is not a "
            "core dependency of floodadapt_abm. Install it with "
            "'pip install geopandas' (only needed for the spatial income "
            "join; percentiles_from_income_values and "
            "percentiles_from_value_proxy are numpy-only)."
        ) from exc

    if regions.crs is not None and buildings.crs is not None:
        regions = regions.to_crs(buildings.crs)
    joined = gpd.sjoin(
        buildings[["geometry"]].reset_index(drop=True),
        regions[[income_column, "geometry"]],
        how="left",
        predicate=predicate,
    )
    # A building straddling a boundary matches several regions: keep the first.
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined[income_column].to_numpy(dtype=np.float64)


def derive_income_percentiles(
    buildings,
    regions,
    income_column: str,
    jitter: int = 5,
    seed: int = 0,
    predicate: str = "intersects",
) -> np.ndarray:
    """
    End-to-end helper: buildings + income regions -> income percentiles.

    Composes :func:`join_buildings_to_regions` and
    :func:`percentiles_from_income_values`; see both for parameter details.

    Returns
    -------
    percentiles : np.ndarray[int64]
        Income percentile per building in ``[1, 99]``, aligned with the
        ``buildings`` row order — ready for
        ``SimulationEngine(..., income_percentile_per_agent=...)``.
    """
    income_per_building = join_buildings_to_regions(
        buildings, regions, income_column, predicate=predicate
    )
    return percentiles_from_income_values(
        income_per_building, jitter=jitter, seed=seed
    )


# ---------------------------------------------------------------------------
# Marginal income distribution: measuring it from published county totals
# ---------------------------------------------------------------------------

#: Upper edges of the 16 household-income brackets of ACS table B19001,
#: in dollars.  The first entry is the lower edge of the first bracket, so
#: ``len(ACS_B19001_EDGES) == 17`` and bracket ``i`` spans
#: ``[edges[i], edges[i + 1])``.
ACS_B19001_EDGES: tuple[float, ...] = (
    0.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0, 35_000.0,
    40_000.0, 45_000.0, 50_000.0, 60_000.0, 75_000.0, 100_000.0,
    125_000.0, 150_000.0, 200_000.0, float("inf"),
)

#: Human-readable labels for :data:`ACS_B19001_EDGES`, same order.
ACS_B19001_LABELS: tuple[str, ...] = (
    "<10k", "10-15k", "15-20k", "20-25k", "25-30k", "30-35k", "35-40k",
    "40-45k", "45-50k", "50-60k", "60-75k", "75-100k", "100-125k",
    "125-150k", "150-200k", "200k+",
)

#: Base URL of the Census Data API 5-year ACS endpoint.
_ACS_BASE = "https://api.census.gov/data/{year}/acs/acs5"


def _acs_resolve_key(api_key: str | None = None) -> str:
    """
    Resolve a Census Data API key from the argument or the environment.

    Parameters
    ----------
    api_key : str or None
        Explicit key.  When ``None``, the ``CENSUS_API_KEY`` environment
        variable is used.

    Returns
    -------
    key : str
        The resolved API key.

    Raises
    ------
    RuntimeError
        If no key is available, with instructions for obtaining one.
    """
    import os

    key = api_key or os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError(
            "No Census API key found. Request a free key at "
            "https://api.census.gov/data/key_signup.html, then set the "
            "CENSUS_API_KEY environment variable "
            '(PowerShell: setx CENSUS_API_KEY "your-key-here").'
        )
    return key


def _acs_get_json(url: str, timeout: float = 120.0) -> list[list[str]]:
    """
    GET a Census Data API URL and return the parsed JSON table.

    The API answers a missing or invalid key with HTTP 200 and an HTML
    error page rather than an error status, so a bare ``json.loads`` fails
    with a misleading parse error.  That case is detected and reported for
    what it is.

    Parameters
    ----------
    url : str
        Fully-formed Census Data API request URL, key included.
    timeout : float
        Socket timeout in seconds.  Default ``120.0``.

    Returns
    -------
    rows : list[list[str]]
        The decoded response: a header row followed by data rows.

    Raises
    ------
    RuntimeError
        If the API returns an HTML page instead of JSON.
    """
    import json
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()

    head = body[:200].lstrip().lower()
    if (
        "html" in content_type.lower()
        or head.startswith(b"<html")
        or head.startswith(b"<!doctype")
    ):
        raise RuntimeError(
            "The Census API returned an HTML page instead of data, which "
            "means the API key is missing or invalid (the API answers "
            "HTTP 200 with a 'Missing Key' page rather than an error "
            "status). Request a free key at "
            "https://api.census.gov/data/key_signup.html and set "
            "CENSUS_API_KEY."
        )
    return json.loads(body)


def fetch_acs_county_income(
    state_fips: str,
    county_fips: str,
    api_key: str | None = None,
    year: int = 2023,
    timeout: float = 120.0,
) -> dict[str, float]:
    """
    Measure a county's household-income marginal from published ACS totals.

    Returns exactly the pair the synthetic income model is parameterised
    by, so ``DecisionConfig`` can be calibrated to the study area instead
    of carrying generic defaults::

        stats = fetch_acs_county_income("45", "019")     # Charleston Co., SC
        cfg = replace(
            config.decision,
            median_income=stats["median_income"],
            mean_median_inc_ratio=stats["mean_median_ratio"],
        )

    **No building locations are required.**  These are county-wide totals,
    so this works even when the lookup table carries no geometry and the
    percentiles come from :func:`percentiles_from_value_proxy`.

    Three ACS 5-year variables are combined:

    * ``B19013_001E`` — median household income;
    * ``B19025_001E`` — aggregate household income;
    * ``B11001_001E`` — number of households.

    The mean is ``aggregate / households``; ACS publishes no mean directly.

    Parameters
    ----------
    state_fips : str
        Two-digit state FIPS code, e.g. ``"45"`` for South Carolina.
    county_fips : str
        Three-digit county FIPS code, e.g. ``"019"`` for Charleston County.
    api_key : str or None
        Census Data API key; falls back to ``CENSUS_API_KEY``.
    year : int
        ACS 5-year vintage (the final year of the five). Default ``2023``.
    timeout : float
        Socket timeout in seconds. Default ``120.0``.

    Returns
    -------
    stats : dict[str, float]
        ``median_income``, ``mean_income``, ``mean_median_ratio``,
        ``n_households`` and ``year``.

    Raises
    ------
    RuntimeError
        If no API key is available or the API returns an HTML error page.
    """
    key = _acs_resolve_key(api_key)
    url = (
        f"{_ACS_BASE.format(year=year)}"
        f"?get=NAME,B19013_001E,B19025_001E,B11001_001E"
        f"&for=county:{county_fips}&in=state:{state_fips}&key={key}"
    )
    rows = _acs_get_json(url, timeout=timeout)
    record = dict(zip(rows[0], rows[1]))

    median = float(record["B19013_001E"])
    aggregate = float(record["B19025_001E"])
    households = float(record["B11001_001E"])
    mean = aggregate / households
    return {
        "median_income": median,
        "mean_income": mean,
        "mean_median_ratio": mean / median,
        "n_households": households,
        "year": float(year),
    }


def fetch_acs_income_brackets(
    state_fips: str,
    county_fips: str,
    api_key: str | None = None,
    year: int = 2023,
    timeout: float = 120.0,
) -> dict[str, np.ndarray]:
    """
    Fetch the observed household-income histogram (ACS table B19001).

    Used to *validate* the fitted lognormal: two numbers are enough to fit
    it, but only the histogram shows whether the resulting shape matches
    reality.  Pair with :func:`lognormal_bracket_shares` and
    :func:`bracket_fit_distance`.

    Parameters
    ----------
    state_fips : str
        Two-digit state FIPS code.
    county_fips : str
        Three-digit county FIPS code.
    api_key : str or None
        Census Data API key; falls back to ``CENSUS_API_KEY``.
    year : int
        ACS 5-year vintage. Default ``2023``.
    timeout : float
        Socket timeout in seconds. Default ``120.0``.

    Returns
    -------
    brackets : dict[str, np.ndarray]
        ``counts`` (households per bracket, shape ``(16,)``) and ``shares``
        (the same normalised to sum to 1).

    Raises
    ------
    RuntimeError
        If no API key is available or the API returns an HTML error page.
    """
    key = _acs_resolve_key(api_key)
    variables = ",".join(f"B19001_{i:03d}E" for i in range(2, 18))
    url = (
        f"{_ACS_BASE.format(year=year)}?get=NAME,{variables}"
        f"&for=county:{county_fips}&in=state:{state_fips}&key={key}"
    )
    rows = _acs_get_json(url, timeout=timeout)
    record = dict(zip(rows[0], rows[1]))

    counts = np.array(
        [float(record[f"B19001_{i:03d}E"]) for i in range(2, 18)],
        dtype=np.float64,
    )
    return {"counts": counts, "shares": counts / counts.sum()}


def lognormal_bracket_shares(
    median_income: float,
    mean_median_ratio: float,
    edges: tuple[float, ...] | np.ndarray = ACS_B19001_EDGES,
) -> np.ndarray:
    """
    Household shares per income bracket implied by the fitted lognormal.

    Applies the engine's own parameterisation (``mu = ln(median)``,
    ``sd = sqrt(2 ln(mean/median))``, see
    ``_DynamoDecisionBridge._synthesize_income_wealth``) and integrates it
    over the bracket edges, so the result is directly comparable with
    :func:`fetch_acs_income_brackets`.

    Pure function: no network, no RNG, deterministic.

    Parameters
    ----------
    median_income : float
        Median household income, strictly positive.
    mean_median_ratio : float
        Mean-to-median income ratio, strictly greater than 1 (a lognormal
        always has mean > median; a ratio <= 1 has no lognormal solution).
    edges : tuple[float, ...] or np.ndarray
        Bracket edges in dollars, ascending, length ``n_brackets + 1``.
        Default :data:`ACS_B19001_EDGES`.

    Returns
    -------
    shares : np.ndarray[float64]
        Fraction of households in each bracket, shape ``(len(edges) - 1,)``,
        summing to 1 when the edges span ``[0, inf)``.

    Raises
    ------
    ValueError
        If ``median_income <= 0`` or ``mean_median_ratio <= 1``.
    """
    from scipy.special import ndtr  # scipy is a core dependency

    if median_income <= 0.0:
        raise ValueError(
            f"median_income must be positive; got {median_income}."
        )
    if mean_median_ratio <= 1.0:
        raise ValueError(
            "mean_median_ratio must be > 1 for a lognormal fit "
            f"(mean always exceeds median); got {mean_median_ratio}."
        )

    mu = np.log(float(median_income))
    sd = np.sqrt(2.0 * np.log(float(mean_median_ratio)))

    bounds = np.asarray(edges, dtype=np.float64)
    # CDF on the log scale; 0 -> 0, +inf -> 1, handled without warnings.
    cdf = np.empty_like(bounds)
    finite_positive = np.isfinite(bounds) & (bounds > 0.0)
    cdf[finite_positive] = ndtr(
        (np.log(bounds[finite_positive]) - mu) / sd
    )
    cdf[bounds <= 0.0] = 0.0
    cdf[~np.isfinite(bounds)] = 1.0
    return np.diff(cdf)


def bracket_fit_distance(
    observed_shares: np.ndarray,
    predicted_shares: np.ndarray,
) -> float:
    """
    Total-variation distance between two bracket-share vectors.

    Reported as a single goodness-of-fit number for the marginal income
    distribution.  It reads directly as *the fraction of households placed
    in the wrong income bracket*: ``0`` is a perfect match, ``1`` is
    disjoint.

    Parameters
    ----------
    observed_shares : np.ndarray
        Observed fraction per bracket (e.g. from
        :func:`fetch_acs_income_brackets`).
    predicted_shares : np.ndarray
        Model-implied fraction per bracket (e.g. from
        :func:`lognormal_bracket_shares`), same shape.

    Returns
    -------
    distance : float
        Total-variation distance in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the two inputs have different shapes.
    """
    obs = np.asarray(observed_shares, dtype=np.float64)
    pred = np.asarray(predicted_shares, dtype=np.float64)
    if obs.shape != pred.shape:
        raise ValueError(
            f"shape mismatch: observed {obs.shape} vs predicted "
            f"{pred.shape}."
        )
    return float(0.5 * np.abs(obs - pred).sum())
