"""
analysis.py
===========
Post-processing of the results dictionary returned by
:meth:`~floodadapt_abm.simulation_engine.SimulationEngine.run`,
:func:`~floodadapt_abm.mesa_native.run_mesa_native` and
:func:`~floodadapt_abm.mesa_native_full.run_mesa_native_full`.

Everything here is a pure NumPy/pandas function over the result arrays.
Nothing touches the engine, the RNG streams or the arrays in place, so the
bit-parity contract between the drivers is unaffected.

Stock versus cumulative uptake
------------------------------
``adoption_fraction`` and ``insured_fraction`` are per-year **stocks**: the
share of households holding the status in year ``t``
(``history.mean(axis=1)``).  The two stocks behave differently:

* floodproofing persists until the lifespan reset (``lifespan_dryproof``,
  75 years by default), so over a shorter horizon the stock only rises;
* insurance is a one-year contract re-decided every year (``is_insured``
  is reassigned in every step), so its stock rises and falls with premiums
  and affordability.

:func:`cumulative_uptake` gives the comparable view for both: the share of
households that have held the status **at least once** by year ``t``.  It is
monotone non-decreasing and never below the stock.  Uptake figures show the
cumulative share as the main line and the stock as a thin secondary line
(see :mod:`floodadapt_abm.plotting`).

Array conventions
-----------------
``history`` arrays are boolean with shape ``(no_seq, n_agents, n_years)``;
``damage_history`` has the same shape (gross damage, before any insurance
payout).  Per-year aggregates returned here have shape ``(no_seq, n_years)``
and are ``float64``; money is expressed in units of ``scale`` (``1e6`` gives
millions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Result keys that exist only when ``include_insurance=True``.
INSURANCE_KEYS: tuple[str, ...] = (
    "insured_history",
    "out_of_pocket_history",
    "premium_history",
    "premium_paid_history",
    "insured_fraction",
)

#: Pricing labels used by the affordability diagnostics.
PRICING_COMMUNITY: str = "community (flat)"
PRICING_RISK_BASED: str = "risk-based"

#: Base of the risk-perception decay, hard-coded in native DYNAMO-M
#: (``flood_risk.py:650-651``) and mirrored by the engine.
RISK_PERC_BASE: float = 1.6


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _as_history(history: np.ndarray) -> np.ndarray:
    """
    Validate a ``(no_seq, n_agents, n_years)`` status history.

    Parameters
    ----------
    history : np.ndarray
        Boolean (or bool-convertible) per-agent status array.

    Returns
    -------
    np.ndarray[bool]
        The same array (no copy when already boolean).
    """
    arr = np.asarray(history)
    if arr.ndim != 3:
        raise ValueError(
            "history must have shape (no_seq, n_agents, n_years); "
            f"got ndim={arr.ndim} with shape {arr.shape}."
        )
    return arr if arr.dtype == bool else arr.astype(bool)


def _as_damage(damage_history: np.ndarray) -> np.ndarray:
    """
    Validate a damage cube and return it as ``float64``.

    Parameters
    ----------
    damage_history : np.ndarray
        ``(no_seq, n_agents, n_years)`` gross damage.

    Returns
    -------
    np.ndarray[float64]
    """
    arr = np.asarray(damage_history, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            "damage_history must have shape (no_seq, n_agents, n_years); "
            f"got ndim={arr.ndim} with shape {arr.shape}."
        )
    return arr


def money_unit(scale: float) -> str:
    """
    Human-readable unit label for a money ``scale``.

    Parameters
    ----------
    scale : float
        Divisor applied to dollar amounts (``1e6`` -> ``"M$"``).

    Returns
    -------
    str
        ``"$"``, ``"k$"``, ``"M$"`` or ``"B$"``; otherwise ``"$/<scale:g>"``.
    """
    labels = {1.0: "$", 1e3: "k$", 1e6: "M$", 1e9: "B$"}
    return labels.get(float(scale), f"$/{scale:g}")


def has_insurance(results: dict) -> bool:
    """
    Whether a results dict carries the insurance arrays.

    Parameters
    ----------
    results : dict
        Output of ``SimulationEngine.run`` or a native driver.

    Returns
    -------
    bool
    """
    return all(key in results for key in INSURANCE_KEYS)


# ---------------------------------------------------------------------------
# Uptake: stock and cumulative views
# ---------------------------------------------------------------------------

def current_uptake(history: np.ndarray) -> np.ndarray:
    """
    Share of agents holding the status in year ``t`` (the per-year stock).

    Identical to ``results["adoption_fraction"]`` for ``adapted_history``
    and to ``results["insured_fraction"]`` for ``insured_history``.

    Parameters
    ----------
    history : np.ndarray[bool]
        ``(no_seq, n_agents, n_years)`` status history.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.
    """
    return _as_history(history).mean(axis=1)


def cumulative_uptake(history: np.ndarray) -> np.ndarray:
    """
    Share of agents that have held the status **at least once** by year ``t``.

    Monotone non-decreasing along the year axis and never below
    :func:`current_uptake`.  Equal to it whenever the status never clears
    (floodproofing with ``lifespan_dryproof`` beyond the horizon).

    Parameters
    ----------
    history : np.ndarray[bool]
        ``(no_seq, n_agents, n_years)`` status history.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.
    """
    ever = np.logical_or.accumulate(_as_history(history), axis=2)
    return ever.mean(axis=1)


def count_history(history: np.ndarray) -> np.ndarray:
    """
    Number of agents holding the status per year.

    Parameters
    ----------
    history : np.ndarray[bool]
        ``(no_seq, n_agents, n_years)`` status history.

    Returns
    -------
    np.ndarray[int64]
        Shape ``(no_seq, n_years)``.
    """
    return _as_history(history).sum(axis=1)


def uptake_statistics(history: np.ndarray) -> dict[str, float]:
    """
    Headline uptake numbers, averaged over sequences.

    Parameters
    ----------
    history : np.ndarray[bool]
        ``(no_seq, n_agents, n_years)`` status history.

    Returns
    -------
    dict[str, float]
        ``final_current`` (stock in the final year), ``final_cumulative``
        (ever-held share in the final year) and ``peak_current`` (highest
        stock over the horizon), all as fractions in ``[0, 1]``.
    """
    current = current_uptake(history).mean(axis=0)
    cumulative = cumulative_uptake(history).mean(axis=0)
    return {
        "final_current": float(current[-1]),
        "final_cumulative": float(cumulative[-1]),
        "peak_current": float(current.max()),
    }


# ---------------------------------------------------------------------------
# Damage, avoided damage and household cost
# ---------------------------------------------------------------------------

def damage_per_year(damage_history: np.ndarray, scale: float = 1e6) -> np.ndarray:
    """
    Total gross damage per year, summed over agents.

    Parameters
    ----------
    damage_history : np.ndarray
        ``(no_seq, n_agents, n_years)`` gross damage.
    scale : float
        Money divisor (``1e6`` -> millions).  Default ``1e6``.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.
    """
    return _as_damage(damage_history).sum(axis=1) / scale


def cumulative_damage(damage_history: np.ndarray, scale: float = 1e6) -> np.ndarray:
    """
    Running total of :func:`damage_per_year` along the year axis.

    Parameters
    ----------
    damage_history : np.ndarray
        ``(no_seq, n_agents, n_years)`` gross damage.
    scale : float
        Money divisor.  Default ``1e6``.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.
    """
    return np.cumsum(damage_per_year(damage_history, scale), axis=1)


def avoided_damage_per_year(
    damage_history: np.ndarray,
    baseline_damage_history: np.ndarray,
    scale: float = 1e6,
    clip: bool = True,
) -> np.ndarray:
    """
    Damage avoided per year relative to a never-adapt baseline.

    The baseline must have been run on the **same event draws** (same
    hazard settings and seed), so the per-agent difference isolates the
    effect of adaptation.

    Parameters
    ----------
    damage_history : np.ndarray
        ``(no_seq, n_agents, n_years)`` gross damage of the run.
    baseline_damage_history : np.ndarray
        Same shape, from the never-adapt baseline.
    scale : float
        Money divisor.  Default ``1e6``.
    clip : bool
        Clip negative per-agent differences at zero before summing (an
        agent cannot avoid more than it would have lost).  Default ``True``.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.
    """
    diff = _as_damage(baseline_damage_history) - _as_damage(damage_history)
    if clip:
        diff = np.clip(diff, 0.0, None)
    return diff.sum(axis=1) / scale


def cumulative_avoided_damage(
    damage_history: np.ndarray,
    baseline_damage_history: np.ndarray,
    scale: float = 1e6,
    clip: bool = True,
) -> np.ndarray:
    """
    Running total of :func:`avoided_damage_per_year`.

    Parameters
    ----------
    damage_history, baseline_damage_history : np.ndarray
        See :func:`avoided_damage_per_year`.
    scale : float
        Money divisor.  Default ``1e6``.
    clip : bool
        See :func:`avoided_damage_per_year`.  Default ``True``.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.
    """
    return np.cumsum(
        avoided_damage_per_year(damage_history, baseline_damage_history, scale, clip),
        axis=1,
    )


def household_cost_per_year(results: dict, scale: float = 1e6) -> np.ndarray:
    """
    What households actually bear per year: out-of-pocket damage plus
    premiums paid, summed over agents.

    Parameters
    ----------
    results : dict
        Results of an insurance-enabled run (``include_insurance=True``).
    scale : float
        Money divisor.  Default ``1e6``.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.

    Raises
    ------
    KeyError
        When the run was made without insurance.
    """
    missing = [
        key for key in ("out_of_pocket_history", "premium_paid_history")
        if key not in results
    ]
    if missing:
        raise KeyError(
            f"results lack {missing}: household cost needs an insurance run "
            "(DecisionConfig.include_insurance=True). Without insurance the "
            "household cost equals damage_per_year(results['damage_history'])."
        )
    paid = (
        _as_damage(results["out_of_pocket_history"])
        + _as_damage(results["premium_paid_history"])
    )
    return paid.sum(axis=1) / scale


def cumulative_household_cost(results: dict, scale: float = 1e6) -> np.ndarray:
    """
    Running total of :func:`household_cost_per_year`.

    Parameters
    ----------
    results : dict
        Results of an insurance-enabled run.
    scale : float
        Money divisor.  Default ``1e6``.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(no_seq, n_years)``.
    """
    return np.cumsum(household_cost_per_year(results, scale), axis=1)


def mean_and_std(per_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean and population standard deviation across the sequence axis.

    Parameters
    ----------
    per_seq : np.ndarray
        ``(no_seq, n_years)`` per-sequence series.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean, std)``, each of shape ``(n_years,)``.
    """
    arr = np.asarray(per_seq, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _summary_columns(
    results: dict,
    baseline_results: dict | None,
    scale: float,
) -> dict[str, tuple[np.ndarray, int]]:
    """
    Per-sequence summary metrics with their display precision.

    Parameters
    ----------
    results : dict
        Run results.
    baseline_results : dict or None
        Never-adapt baseline on the same event draws.
    scale : float
        Money divisor.

    Returns
    -------
    dict[str, tuple[np.ndarray, int]]
        ``column -> (values of shape (no_seq,), decimals)``.
    """
    unit = money_unit(scale)
    adapted = results["adapted_history"]
    cols: dict[str, tuple[np.ndarray, int]] = {
        "final floodproofed (#)": (count_history(adapted)[:, -1], 0),
        "final floodproofed (%)": (100.0 * current_uptake(adapted)[:, -1], 1),
        "ever floodproofed (%)": (100.0 * cumulative_uptake(adapted)[:, -1], 1),
        f"cum. damage ({unit})": (
            damage_per_year(results["damage_history"], scale).sum(axis=1), 1
        ),
    }
    if baseline_results is not None:
        cols[f"cum. avoided ({unit})"] = (
            avoided_damage_per_year(
                results["damage_history"],
                baseline_results["damage_history"],
                scale,
            ).sum(axis=1),
            1,
        )
    if has_insurance(results):
        insured = results["insured_history"]
        stock = current_uptake(insured)
        cols["final insured (%)"] = (100.0 * stock[:, -1], 2)
        cols["ever insured (%)"] = (100.0 * cumulative_uptake(insured)[:, -1], 2)
        cols["peak insured (%)"] = (100.0 * stock.max(axis=1), 2)
        cols["premium final year ($)"] = (
            np.asarray(results["premium_history"], dtype=np.float64)[:, -1], 0
        )
        cols[f"cum. household cost ({unit})"] = (
            household_cost_per_year(results, scale).sum(axis=1), 1
        )
    return cols


def create_summary_table(
    results: dict,
    baseline_results: dict | None = None,
    scale: float = 1e6,
) -> pd.DataFrame:
    """
    Per-sequence summary of a run, with ``MEAN`` and ``STD`` rows appended.

    Columns: ``final floodproofed (#)``, ``final floodproofed (%)`` (the
    stock), ``ever floodproofed (%)`` (cumulative), ``cum. damage``; plus
    ``cum. avoided`` when a baseline is given, and the insurance columns
    (``final insured (%)``, ``ever insured (%)``, ``peak insured (%)``,
    ``premium final year ($)``, ``cum. household cost``) when the run was
    made with insurance.

    Parameters
    ----------
    results : dict
        Run results.
    baseline_results : dict or None
        Never-adapt baseline on the same event draws.  Default ``None``.
    scale : float
        Money divisor for the damage and cost columns.  Default ``1e6``.

    Returns
    -------
    pd.DataFrame
        ``no_seq + 2`` rows; the ``sequence`` column holds ``seq 0`` ...
        then ``MEAN`` and ``STD``.
    """
    cols = _summary_columns(results, baseline_results, scale)
    no_seq = results["adapted_history"].shape[0]

    per_seq = {"sequence": [f"seq {s}" for s in range(no_seq)]}
    agg = {"sequence": ["MEAN", "STD"]}
    for name, (values, decimals) in cols.items():
        values = np.asarray(values, dtype=np.float64)
        per_seq[name] = (
            values.astype(np.int64) if decimals == 0 and name.endswith("(#)")
            else np.round(values, decimals)
        )
        agg_decimals = max(decimals, 1)
        agg[name] = [
            np.round(values.mean(), agg_decimals),
            np.round(values.std(), agg_decimals),
        ]
    return pd.concat(
        [pd.DataFrame(per_seq), pd.DataFrame(agg)], ignore_index=True
    )


def scenario_summary(
    scenarios: dict[str, dict],
    reference: dict | None = None,
    baselines: dict[str, dict] | None = None,
    scale: float = 1e6,
) -> pd.DataFrame:
    """
    One row per scenario with sequence-mean headline metrics.

    Parameters
    ----------
    scenarios : dict[str, dict]
        ``label -> results`` for each run to summarise.
    reference : dict or None
        A run to measure avoided damage against (for insurance experiments
        this is the no-insurance run).  Adds ``cum. avoided vs reference``
        = cumulative reference damage minus cumulative scenario damage
        (negative means more damage than the reference).  Default ``None``.
    baselines : dict[str, dict] or None
        ``label -> never-adapt results`` on the same event draws; adds the
        clipped ``cum. avoided`` column for the labels present.
        Default ``None``.
    scale : float
        Money divisor.  Default ``1e6``.

    Returns
    -------
    pd.DataFrame
        Columns: ``scenario``, ``final floodproofed (%)``,
        ``ever floodproofed (%)``, ``cum. damage``, optional avoided
        columns, and the insurance columns (``NaN`` for runs without
        insurance) when any scenario carries them.
    """
    unit = money_unit(scale)
    any_insurance = any(has_insurance(res) for res in scenarios.values())
    ref_cum = (
        None if reference is None
        else cumulative_damage(reference["damage_history"], scale)[:, -1].mean()
    )

    rows: list[dict[str, float | str]] = []
    for label, res in scenarios.items():
        adapted = res["adapted_history"]
        row: dict[str, float | str] = {
            "scenario": label,
            "final floodproofed (%)": round(
                100.0 * current_uptake(adapted)[:, -1].mean(), 2
            ),
            "ever floodproofed (%)": round(
                100.0 * cumulative_uptake(adapted)[:, -1].mean(), 2
            ),
            f"cum. damage ({unit})": round(
                cumulative_damage(res["damage_history"], scale)[:, -1].mean(), 1
            ),
        }
        if baselines is not None:
            row[f"cum. avoided ({unit})"] = (
                round(
                    cumulative_avoided_damage(
                        res["damage_history"],
                        baselines[label]["damage_history"],
                        scale,
                    )[:, -1].mean(),
                    1,
                )
                if label in baselines else np.nan
            )
        if ref_cum is not None:
            row[f"cum. avoided vs reference ({unit})"] = round(
                ref_cum
                - cumulative_damage(res["damage_history"], scale)[:, -1].mean(),
                1,
            )
        if any_insurance:
            if has_insurance(res):
                stats = uptake_statistics(res["insured_history"])
                row["peak insured (%)"] = round(100.0 * stats["peak_current"], 2)
                row["final insured (%)"] = round(100.0 * stats["final_current"], 2)
                row["ever insured (%)"] = round(100.0 * stats["final_cumulative"], 2)
                row["premium final year ($)"] = round(
                    float(np.asarray(res["premium_history"])[:, -1].mean()), 0
                )
                row[f"cum. household cost ({unit})"] = round(
                    cumulative_household_cost(res, scale)[:, -1].mean(), 1
                )
            else:
                for name in (
                    "peak insured (%)", "final insured (%)", "ever insured (%)",
                    "premium final year ($)", f"cum. household cost ({unit})",
                ):
                    row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Risk perception (no simulation needed)
# ---------------------------------------------------------------------------

def perception_curve(
    decision,
    severity: float | np.ndarray,
    years: int = 10,
) -> np.ndarray:
    """
    Risk perception of a household after a single flood, year by year.

    Mirrors :meth:`SimulationEngine.update_flood_experience`: the flood
    resets the timer to 0, and perception then decays as
    ``peak * 1.6 ** (risk_perc_coef * tau) + risk_perc_min``.  Under
    ``perception_mode="severity"`` the peak is
    ``risk_perc_max * s ** perception_severity_exponent``; under
    ``"binary"`` it is ``risk_perc_max`` for any flood.

    Parameters
    ----------
    decision : DecisionConfig
        Behaviour settings (perception parameters and mode).
    severity : float or np.ndarray
        Damage severity of the flood, realised damage over building value,
        in ``[0, 1]``.  An array gives one curve per value.
    years : int
        Years after the flood to evaluate.  Default ``10``.

    Returns
    -------
    np.ndarray[float64]
        Shape ``(n_severity, years + 1)``; column ``tau`` is the perception
        ``tau`` years after the flood (column 0 is the flood year).
    """
    s = np.clip(np.atleast_1d(np.asarray(severity, dtype=np.float64)), 0.0, 1.0)
    if decision.perception_mode == "severity":
        gamma = decision.perception_severity_exponent
        if gamma <= 0.0:
            raise ValueError(
                f"perception_severity_exponent must be > 0; got {gamma}."
            )
        peak = decision.risk_perc_max * s ** gamma
    elif decision.perception_mode == "binary":
        peak = np.full_like(s, decision.risk_perc_max)
    else:
        raise ValueError(
            f"Unknown perception_mode {decision.perception_mode!r}; "
            "expected 'binary' or 'severity'."
        )
    tau = np.arange(years + 1, dtype=np.float64)
    decay = RISK_PERC_BASE ** (decision.risk_perc_coef * tau)
    return peak[:, None] * decay[None, :] + decision.risk_perc_min


# ---------------------------------------------------------------------------
# Insurance affordability diagnostics (no simulation needed)
# ---------------------------------------------------------------------------

def premium_offers(ead: np.ndarray, deductible: float) -> dict[str, np.ndarray]:
    """
    Per-household premiums under both rating rules, mirroring the engine.

    Community rating charges the **full** mean EAD of the pool to everyone
    (native DYNAMO-M's insurer), although the policy only covers
    ``1 - deductible`` of each loss.  Risk-based rating charges each
    household its own expected payout ``(1 - deductible) * EAD_i``.

    Parameters
    ----------
    ead : np.ndarray
        Expected annual damage per household, shape ``(n_agents,)``.
    deductible : float
        Share of each loss the insured household still pays.

    Returns
    -------
    dict[str, np.ndarray]
        ``{PRICING_COMMUNITY: flat, PRICING_RISK_BASED: per-household}``.
    """
    ead = np.asarray(ead, dtype=np.float64)
    return {
        PRICING_COMMUNITY: np.full_like(ead, ead.mean()),
        PRICING_RISK_BASED: (1.0 - deductible) * ead,
    }


def ead_summary(ead: np.ndarray) -> dict[str, float]:
    """
    Skewness snapshot of the risk pool.

    Parameters
    ----------
    ead : np.ndarray
        Expected annual damage per household.

    Returns
    -------
    dict[str, float]
        ``mean``, ``median``, ``p99`` and ``share_zero`` (fraction of
        households with nothing to insure).
    """
    ead = np.asarray(ead, dtype=np.float64)
    return {
        "mean": float(ead.mean()),
        "median": float(np.median(ead)),
        "p99": float(np.percentile(ead, 99)),
        "share_zero": float(np.mean(ead <= 0.0)),
    }


def affordability_table(
    ead: np.ndarray,
    income: np.ndarray,
    expenditure_cap: float,
    deductible: float,
    high_risk_quantile: float = 0.9,
) -> pd.DataFrame:
    """
    Who can afford cover, and who needs it, under each rating rule.

    A premium is affordable when it is strictly below
    ``expenditure_cap * income`` (the same gate the SEU kernels apply).

    Parameters
    ----------
    ead : np.ndarray
        Expected annual damage per household, shape ``(n_agents,)``.
    income : np.ndarray
        Annual income per household, same shape.
    expenditure_cap : float
        Maximum share of income spent on a premium.
    deductible : float
        Share of each loss the insured household still pays.
    high_risk_quantile : float
        Quantile of EAD defining the "needs cover" tail.  Default ``0.9``
        (the top decile).

    Returns
    -------
    pd.DataFrame
        One row per pricing rule: ``pricing``, ``median premium ($/yr)``,
        ``affordable - all households (%)`` and
        ``affordable - top-<x>% risk (%)``.
    """
    ead = np.asarray(ead, dtype=np.float64)
    budget = np.asarray(income, dtype=np.float64) * expenditure_cap
    tail = ead >= np.percentile(ead, 100.0 * high_risk_quantile)
    tail_label = f"affordable - top-{100 * (1 - high_risk_quantile):.0f}% risk (%)"

    rows = []
    for label, premium in premium_offers(ead, deductible).items():
        affordable = budget > premium
        rows.append({
            "pricing": label,
            "median premium ($/yr)": float(np.median(premium)),
            "affordable - all households (%)": 100.0 * float(affordable.mean()),
            tail_label: 100.0 * float(affordable[tail].mean()),
        })
    return pd.DataFrame(rows)


def affordability_vs_subsidy(
    ead: np.ndarray,
    income: np.ndarray,
    expenditure_cap: float,
    deductible: float,
    subsidies: np.ndarray,
    high_risk_quantile: float = 0.9,
) -> dict[str, np.ndarray]:
    """
    Share of the high-risk tail that can afford cover at each subsidy depth.

    The subsidy is the public share of the premium, applied after pricing:
    the household pays ``(1 - subsidy) * premium``.

    Parameters
    ----------
    ead, income : np.ndarray
        Per-household expected annual damage and income.
    expenditure_cap : float
        Maximum share of income spent on a premium.
    deductible : float
        Share of each loss the insured household still pays.
    subsidies : np.ndarray
        Subsidy depths in ``[0, 1)``, shape ``(n_sub,)``.
    high_risk_quantile : float
        Quantile of EAD defining the tail.  Default ``0.9``.

    Returns
    -------
    dict[str, np.ndarray]
        ``pricing label -> affordable share of the tail`` (fractions in
        ``[0, 1]``, shape ``(n_sub,)``), monotone non-decreasing in the
        subsidy.
    """
    ead = np.asarray(ead, dtype=np.float64)
    budget = np.asarray(income, dtype=np.float64) * expenditure_cap
    subs = np.asarray(subsidies, dtype=np.float64)
    tail = ead >= np.percentile(ead, 100.0 * high_risk_quantile)

    shares: dict[str, np.ndarray] = {}
    for label, premium in premium_offers(ead, deductible).items():
        household_pays = (1.0 - subs)[:, None] * premium[tail][None, :]
        shares[label] = (budget[tail][None, :] > household_pays).mean(axis=1)
    return shares
