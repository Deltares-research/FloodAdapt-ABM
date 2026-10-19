"""
plotting.py
===========
Matplotlib figures for the results of a coupled run.  Every function draws
into caller-supplied axes when given, otherwise creates a figure, and
returns the :class:`~matplotlib.figure.Figure`.  Nothing here calls
``plt.show()`` or selects a backend, so the same functions serve notebooks,
batch scripts (``matplotlib.use("Agg")``) and tests.

Uptake convention
-----------------
Two views exist for every status (see :mod:`floodadapt_abm.analysis`): the
**cumulative** share of households that have ever taken the action, and the
per-year **stock** currently holding it.  Insurance figures show them in two
separate panels, one line per scenario in each.  Floodproofing panels show
one line per scenario; the stock is added as a thin dotted line only where it
differs from the cumulative share (that is, when a measure expired), so the
legend never repeats a coinciding pair.

Styles
------
Multi-scenario figures take ``styles: dict[label, (color, linestyle)]``.  The
convention used by notebook 2 and the batch driver is one colour per scenario
and the line style for the rating rule (solid community, dashed risk-based).
A label without an entry gets the next colour of the default cycle and a
solid line.  The reference run (no insurance offered) is drawn in
``reference_color`` (black by default) so it never clashes with a scenario.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from floodadapt_abm import analysis as an

#: Colour of the sea-level-rise overlay.
SLR_COLOR: str = "#6c757d"

#: Axes label marking the SLR twin axis (lets callers tell it from panels).
SLR_TWIN_LABEL: str = "slr_twin"

#: Line width of a scenario line; the secondary stock line is half.
LINE_LW: float = 2.2

#: Allowed values of the ``which`` argument of :func:`plot_uptake`.
UPTAKE_VIEWS: tuple[str, ...] = ("auto", "both", "ever", "current")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_styles(
    labels: list[str],
    styles: dict[str, tuple[str, str]] | None,
) -> dict[str, tuple[str, str]]:
    """
    Fill in ``(color, linestyle)`` for every label.

    Parameters
    ----------
    labels : list[str]
        Scenario labels in plotting order.
    styles : dict or None
        Partial style map supplied by the caller.

    Returns
    -------
    dict[str, tuple[str, str]]
    """
    styles = dict(styles or {})
    next_color = 0
    for label in labels:
        if label not in styles:
            styles[label] = (f"C{next_color % 10}", "-")
            next_color += 1
    return styles


def _panels(
    axes: np.ndarray | list[Axes] | None,
    n_panels: int,
    figsize: tuple[float, float],
    ncols: int | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Return ``(figure, axes)`` with at least ``n_panels`` axes, flattened.

    Parameters
    ----------
    axes : array-like of Axes or None
        Caller-supplied axes; ``None`` creates a new grid.
    n_panels : int
        Number of panels the figure needs.
    figsize : tuple[float, float]
        Figure size used when creating the grid.
    ncols : int or None
        Columns of the created grid; ``None`` puts all panels in one row.

    Returns
    -------
    tuple[Figure, np.ndarray]
    """
    if axes is None:
        ncols = ncols or n_panels
        nrows = -(-n_panels // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, np.atleast_1d(axes).ravel()
    axes = np.asarray(axes, dtype=object).ravel()
    if axes.size < n_panels:
        raise ValueError(f"need {n_panels} axes, got {axes.size}.")
    return axes[0].figure, axes


def _finish(fig: Figure, save_path: str | Path | None) -> Figure:
    """
    Tighten the layout and optionally save.

    Parameters
    ----------
    fig : Figure
        Figure to finish.
    save_path : str, Path or None
        File to write (``dpi=150``, tight bounding box).

    Returns
    -------
    Figure
    """
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _money_label(prefix: str, scale: float) -> str:
    """Axis label such as ``"Damage (M$)"``."""
    return f"{prefix} ({an.money_unit(scale)})"


def _label_panels(axes: np.ndarray, titles: list[str], ylabels: list[str]) -> None:
    """Set title, x-label, y-label and a small legend on each panel."""
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def overlay_slr(
    ax: Axes,
    years: np.ndarray,
    slr_values: np.ndarray,
    label: str = "SLR trajectory",
    color: str = SLR_COLOR,
) -> Axes:
    """
    Add the sea-level trajectory on a twin right-hand axis.

    Parameters
    ----------
    ax : Axes
        Panel to overlay.
    years : np.ndarray
        Calendar years, shape ``(n_years,)``.
    slr_values : np.ndarray
        Sea level per year, same shape (table units).
    label : str
        Legend label.  Default ``"SLR trajectory"``.
    color : str
        Line and tick colour.  Default :data:`SLR_COLOR`.

    Returns
    -------
    Axes
        The twin axis (labelled :data:`SLR_TWIN_LABEL`).
    """
    axr = ax.twinx()
    axr.set_label(SLR_TWIN_LABEL)
    axr.plot(years, slr_values, color=color, ls="--", lw=1.8, label=label)
    axr.set_ylabel("Sea-level rise (table units)", color=color)
    axr.tick_params(axis="y", labelcolor=color)
    axr.set_ylim(bottom=min(0.0, float(np.min(slr_values))))
    return axr


def plot_band(
    ax: Axes,
    years: np.ndarray,
    per_seq: np.ndarray,
    color: str,
    label: str,
    show_sequences: bool = True,
    clip_zero: bool = True,
) -> Axes:
    """
    Thin per-sequence lines, the sequence mean and a ±1 s.d. band.

    Parameters
    ----------
    ax : Axes
        Target panel.
    years : np.ndarray
        Calendar years, shape ``(n_years,)``.
    per_seq : np.ndarray
        ``(no_seq, n_years)`` series.
    color : str
        Colour for all three layers.
    label : str
        Legend label of the mean line.
    show_sequences : bool
        Draw the individual sequences.  Default ``True``.
    clip_zero : bool
        Clip the lower edge of the band at zero.  Default ``True``.

    Returns
    -------
    Axes
    """
    per_seq = np.asarray(per_seq, dtype=np.float64)
    if show_sequences:
        for s in range(per_seq.shape[0]):
            ax.plot(
                years, per_seq[s], color=color, alpha=0.30, lw=1,
                label="individual sequence" if s == 0 else None,
            )
    mean, std = an.mean_and_std(per_seq)
    lower = np.maximum(0.0, mean - std) if clip_zero else mean - std
    ax.plot(years, mean, color=color, lw=2.4, label=label)
    ax.fill_between(
        years, lower, mean + std, color=color, alpha=0.20,
        label="±1 s.d. across sequences",
    )
    return ax


def plot_uptake(
    ax: Axes,
    years: np.ndarray,
    history: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    which: str = "auto",
    as_percent: bool = True,
) -> Axes:
    """
    Uptake of a status over time: cumulative ("ever"), stock ("current"),
    or both.

    Parameters
    ----------
    ax : Axes
        Target panel.
    years : np.ndarray
        Calendar years, shape ``(n_years,)``.
    history : np.ndarray[bool]
        ``(no_seq, n_agents, n_years)`` status history
        (``adapted_history`` or ``insured_history``).
    color : str
        Line colour.
    label : str
        Legend label.  When both views are drawn in this panel they are
        labelled ``"<label> (ever)"`` and ``"<label> (current)"``; a single
        line keeps the bare label.
    linestyle : str
        Line style of the main line.  Default ``"-"``.
    which : str
        ``"ever"`` or ``"current"`` draws that view alone; ``"both"`` draws
        the cumulative share as the main line and the stock as a thin
        dotted line; ``"auto"`` (default) behaves like ``"both"`` only where
        the two differ somewhere, otherwise like ``"ever"``.
    as_percent : bool
        Plot percentages instead of fractions.  Default ``True``.

    Returns
    -------
    Axes
    """
    if which not in UPTAKE_VIEWS:
        raise ValueError(f"which must be one of {UPTAKE_VIEWS}; got {which!r}.")
    factor = 100.0 if as_percent else 1.0
    ever = an.cumulative_uptake(history).mean(axis=0)
    current = an.current_uptake(history).mean(axis=0)
    if which == "auto":
        which = "both" if np.any(np.abs(ever - current) > 1e-12) else "ever"

    if which == "current":
        ax.plot(years, factor * current, color=color, ls=linestyle,
                lw=LINE_LW, label=label)
        return ax
    both = which == "both"
    ax.plot(
        years, factor * ever, color=color, ls=linestyle, lw=LINE_LW,
        label=f"{label} (ever)" if both else label,
    )
    if both:
        ax.plot(
            years, factor * current, color=color, ls=":", lw=LINE_LW / 2,
            alpha=0.85, label=f"{label} (current)",
        )
    return ax


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_adaptation_overview(
    years: np.ndarray,
    slr_values: np.ndarray,
    results: dict,
    baseline_results: dict,
    scale: float = 1e6,
    axes: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Four-panel overview of one run against its never-adapt baseline.

    Panels: (a) households floodproofed per year (the stock, with sequences,
    mean and ±1 s.d. band; the cumulative ever-floodproofed mean is added as
    a thin line only where it differs), (b) total damage per year,
    (c) avoided damage per year, (d) cumulative avoided damage.  Each panel
    carries the SLR trajectory on a twin axis.

    Parameters
    ----------
    years, slr_values : np.ndarray
        Calendar years and sea level per year.
    results : dict
        Run results.
    baseline_results : dict
        Never-adapt baseline on the same event draws.
    scale : float
        Money divisor.  Default ``1e6``.
    axes : array-like of Axes or None
        Four axes to draw into; ``None`` creates them.  Default ``None``.
    title : str or None
        Figure title; ``None`` uses a default naming the sequence count.
    save_path : str, Path or None
        File to save the figure to.  Default ``None``.

    Returns
    -------
    Figure
    """
    adapted = results["adapted_history"]
    no_seq, n_agents, _ = adapted.shape
    damage = an.damage_per_year(results["damage_history"], scale)
    avoided = an.avoided_damage_per_year(
        results["damage_history"], baseline_results["damage_history"], scale
    )
    counts = an.count_history(adapted)
    ever = an.cumulative_uptake(adapted).mean(axis=0)
    current = an.current_uptake(adapted).mean(axis=0)

    fig, ax = _panels(axes, 4, (23, 4.6))
    panels = [
        (counts, "(a) Households floodproofed per year",
         "# households floodproofed", "C0"),
        (damage, "(b) Total flood damage per year",
         _money_label("Damage", scale), "C1"),
        (avoided, "(c) Avoided damage per year (vs never-adapt)",
         _money_label("Avoided damage", scale), "C2"),
        (np.cumsum(avoided, axis=1),
         "(d) Cumulative avoided damage (vs never-adapt)",
         _money_label("Cumulative avoided damage", scale), "C3"),
    ]
    for i, (data, panel_title, ylabel, color) in enumerate(panels):
        plot_band(ax[i], years, data, color, "mean (across sequences)")
        if i == 0 and np.any(np.abs(ever - current) > 1e-12):
            ax[i].plot(
                years, ever * n_agents, color="k", ls=":", lw=1.2,
                label="ever floodproofed (mean)",
            )
        ax[i].set_title(panel_title)
        ax[i].set_xlabel("Year")
        ax[i].set_ylabel(ylabel)
        axr = overlay_slr(ax[i], years, slr_values)
        handles, labels = ax[i].get_legend_handles_labels()
        handles_r, labels_r = axr.get_legend_handles_labels()
        ax[i].legend(handles + handles_r, labels + labels_r, loc="upper left",
                     fontsize=8)

    fig.suptitle(
        title if title is not None else
        f"Coupled ABM outcomes under rising SLR - {no_seq} sequences, "
        "mean ±1 s.d.",
        y=1.03,
    )
    return _finish(fig, save_path)


def plot_rule_hazard_comparison(
    years: np.ndarray,
    runs: dict[str, dict],
    baseline_damage: dict[str, np.ndarray],
    styles: dict[str, tuple[str, str]] | None = None,
    scale: float = 1e6,
    axes: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Three-panel sequence-mean comparison of several runs.

    Panels: (a) households floodproofed (one line per run; the stock is
    added only where it differs from the cumulative share), (b) total damage
    per year, (c) cumulative avoided damage, each run against its own
    never-adapt baseline.

    Parameters
    ----------
    years : np.ndarray
        Calendar years.
    runs : dict[str, dict]
        ``label -> results`` in plotting order.
    baseline_damage : dict[str, np.ndarray]
        ``label -> never-adapt damage_history`` on the same event draws as
        that run.
    styles : dict or None
        ``label -> (color, linestyle)``.  Default cycle.
    scale : float
        Money divisor.  Default ``1e6``.
    axes : array-like of Axes or None
        Three axes to draw into.  Default ``None``.
    title : str or None
        Figure title.  Default ``None`` (no title).
    save_path : str, Path or None
        File to save the figure to.  Default ``None``.

    Returns
    -------
    Figure
    """
    styles = _resolve_styles(list(runs), styles)
    fig, ax = _panels(axes, 3, (18, 4.4))
    for label, res in runs.items():
        color, ls = styles[label]
        plot_uptake(ax[0], years, res["adapted_history"], color, label, ls)
        ax[1].plot(
            years, an.damage_per_year(res["damage_history"], scale).mean(axis=0),
            color=color, ls=ls, lw=LINE_LW, label=label,
        )
        ax[2].plot(
            years,
            an.cumulative_avoided_damage(
                res["damage_history"], baseline_damage[label], scale
            ).mean(axis=0),
            color=color, ls=ls, lw=LINE_LW, label=label,
        )
    _label_panels(
        ax[:3],
        ["(a) Households floodproofed (mean)",
         "(b) Total flood damage (mean)",
         "(c) Cumulative avoided damage (mean)"],
        ["% of households",
         _money_label("Damage", scale),
         _money_label("Cumulative avoided", scale)],
    )
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def _insurance_uptake_panels(
    ax: np.ndarray,
    years: np.ndarray,
    scenarios: dict[str, dict],
    reference: dict,
    styles: dict[str, tuple[str, str]],
    reference_label: str,
    reference_color: str,
) -> None:
    """
    The three uptake panels shared by the insurance figures:
    (a) ever insured, (b) currently insured, (c) floodproofed.

    Parameters
    ----------
    ax : np.ndarray
        At least three axes.
    years : np.ndarray
        Calendar years.
    scenarios : dict[str, dict]
        ``label -> insurance-enabled results``.
    reference : dict
        The matching run without insurance.
    styles : dict
        Resolved ``label -> (color, linestyle)``.
    reference_label, reference_color : str
        Legend name and colour of the reference run.
    """
    for label, res in scenarios.items():
        color, ls = styles[label]
        plot_uptake(ax[0], years, res["insured_history"], color, label, ls,
                    which="ever")
        plot_uptake(ax[1], years, res["insured_history"], color, label, ls,
                    which="current")
        plot_uptake(ax[2], years, res["adapted_history"], color, label, ls)
    plot_uptake(
        ax[2], years, reference["adapted_history"], reference_color,
        f"{reference_label} (no insurance offered)", "--",
    )


def plot_insurance_overview(
    years: np.ndarray,
    scenarios: dict[str, dict],
    reference: dict,
    cap_budget: float,
    styles: dict[str, tuple[str, str]] | None = None,
    reference_label: str = "R1",
    reference_color: str = "k",
    scale: float = 1e6,
    axes: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Six-panel insurance overview against a no-insurance reference run.

    Panels: (a) ever insured, (b) currently insured, (c) floodproofed
    (crowding-out check, reference dashed), (d) mean offered premium against
    the affordability budget, (e) household cost per year (out-of-pocket
    plus premiums) against the reference's gross damage, (f) cumulative
    household cost against the reference's cumulative gross damage.

    Parameters
    ----------
    years : np.ndarray
        Calendar years.
    scenarios : dict[str, dict]
        ``label -> insurance-enabled results``.
    reference : dict
        The matching run without insurance.
    cap_budget : float
        Affordability line, e.g. ``expenditure_cap * median income``.
    styles : dict or None
        ``label -> (color, linestyle)``; one colour per scenario, line style
        for the rating rule.  Default cycle.
    reference_label : str
        Legend name of the reference run.  Default ``"R1"``.
    reference_color : str
        Colour of the reference lines.  Default black.
    scale : float
        Money divisor.  Default ``1e6``.
    axes : array-like of Axes or None
        Six axes to draw into.  Default ``None`` (a ``2 x 3`` grid).
    title : str or None
        Figure title.  Default ``None``.
    save_path : str, Path or None
        File to save the figure to.  Default ``None``.

    Returns
    -------
    Figure
    """
    styles = _resolve_styles(list(scenarios), styles)
    fig, ax = _panels(axes, 6, (23, 8.6), ncols=3)
    _insurance_uptake_panels(ax, years, scenarios, reference, styles,
                             reference_label, reference_color)
    for label, res in scenarios.items():
        color, ls = styles[label]
        ax[3].plot(years, np.asarray(res["premium_history"]).mean(axis=0),
                   color=color, ls=ls, lw=LINE_LW, label=label)
        ax[4].plot(years, an.household_cost_per_year(res, scale).mean(axis=0),
                   color=color, ls=ls, lw=LINE_LW, label=label)
        ax[5].plot(years, an.cumulative_household_cost(res, scale).mean(axis=0),
                   color=color, ls=ls, lw=LINE_LW, label=label)
    ax[3].axhline(cap_budget, color="k", ls=":", lw=1.6,
                  label="affordability budget (cap x median income)")
    ax[4].plot(years, an.damage_per_year(reference["damage_history"], scale).mean(axis=0),
               color=reference_color, ls="--", lw=LINE_LW,
               label=f"{reference_label} gross damage")
    ax[5].plot(years, an.cumulative_damage(reference["damage_history"], scale).mean(axis=0),
               color=reference_color, ls="--", lw=LINE_LW,
               label=f"{reference_label} gross damage")
    _label_panels(
        ax[:6],
        ["(a) Ever insured (cumulative)",
         "(b) Currently insured (per-year stock)",
         "(c) Floodproofed: crowding-out check",
         "(d) Mean offered premium vs affordability budget",
         "(e) Household cost per year: out-of-pocket + premiums",
         "(f) Cumulative household cost"],
        ["% of households", "% of households", "% of households",
         "$ per household per year",
         _money_label("Cost per year", scale),
         _money_label("Cumulative cost", scale)],
    )
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def plot_subsidy_experiment(
    years: np.ndarray,
    scenarios: dict[str, dict],
    reference: dict,
    styles: dict[str, tuple[str, str]] | None = None,
    reference_label: str = "R1",
    reference_color: str = "k",
    cap_budget: float | None = None,
    scale: float = 1e6,
    axes: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Six-panel policy comparison against a no-insurance reference run.

    Panels: (a) ever insured, (b) currently insured, (c) floodproofed
    (reference dashed), (d) mean offered premium (with the affordability
    budget when given), (e) cumulative avoided damage relative to the
    reference (negative means more damage), (f) cumulative household cost
    against the reference's cumulative gross damage.

    Parameters
    ----------
    years : np.ndarray
        Calendar years.
    scenarios : dict[str, dict]
        ``label -> insurance-enabled results``.
    reference : dict
        The matching run without insurance.
    styles : dict or None
        ``label -> (color, linestyle)``; one colour per scenario, line style
        for the rating rule.  Default cycle.
    reference_label : str
        Legend name of the reference run.  Default ``"R1"``.
    reference_color : str
        Colour of the reference lines.  Default black.
    cap_budget : float or None
        Affordability line for panel (d).  Default ``None`` (not drawn).
    scale : float
        Money divisor.  Default ``1e6``.
    axes : array-like of Axes or None
        Six axes to draw into.  Default ``None`` (a ``2 x 3`` grid).
    title : str or None
        Figure title.  Default ``None``.
    save_path : str, Path or None
        File to save the figure to.  Default ``None``.

    Returns
    -------
    Figure
    """
    styles = _resolve_styles(list(scenarios), styles)
    ref_cum = an.cumulative_damage(reference["damage_history"], scale).mean(axis=0)
    fig, ax = _panels(axes, 6, (23, 8.6), ncols=3)
    _insurance_uptake_panels(ax, years, scenarios, reference, styles,
                             reference_label, reference_color)
    for label, res in scenarios.items():
        color, ls = styles[label]
        ax[3].plot(years, np.asarray(res["premium_history"]).mean(axis=0),
                   color=color, ls=ls, lw=LINE_LW, label=label)
        run_cum = an.cumulative_damage(res["damage_history"], scale).mean(axis=0)
        ax[4].plot(years, ref_cum - run_cum, color=color, ls=ls, lw=LINE_LW,
                   label=label)
        ax[5].plot(years, an.cumulative_household_cost(res, scale).mean(axis=0),
                   color=color, ls=ls, lw=LINE_LW, label=label)
    if cap_budget is not None:
        ax[3].axhline(cap_budget, color="k", ls=":", lw=1.6,
                      label="affordability budget (cap x median income)")
    ax[4].axhline(0.0, color="k", lw=1.0, ls=":")
    ax[5].plot(years, ref_cum, color=reference_color, ls="--", lw=LINE_LW,
               label=f"{reference_label} gross damage")
    _label_panels(
        ax[:6],
        ["(a) Ever insured (cumulative)",
         "(b) Currently insured (per-year stock)",
         "(c) Floodproofed: crowded out?",
         "(d) Mean offered premium",
         f"(e) Cumulative avoided damage vs {reference_label}",
         "(f) Cumulative household cost"],
        ["% of households", "% of households", "% of households",
         "$ per household per year",
         f"{an.money_unit(scale)}  (negative = more damage)",
         f"{an.money_unit(scale)} out-of-pocket + premiums"],
    )
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def plot_agent_profile(
    max_pot_dmg: np.ndarray,
    income: np.ndarray,
    median_income: float | None = None,
    axes: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    Who the agents are: the income distribution, and how income relates to
    building value.

    Panels: (a) histogram of household income on a log axis, with the
    median marked, (b) building value against income as a log-log hexbin,
    titled with the Spearman rank correlation between the two.

    Parameters
    ----------
    max_pot_dmg : np.ndarray
        Building value (maximum potential damage) per agent.
    income : np.ndarray
        Annual income per agent, same shape.
    median_income : float or None
        Regional median to mark in panel (a); ``None`` marks the agents'
        own median.  Default ``None``.
    axes : array-like of Axes or None
        Two axes to draw into.  Default ``None``.
    title : str or None
        Figure title.  Default ``None``.
    save_path : str, Path or None
        File to save the figure to.  Default ``None``.

    Returns
    -------
    Figure
    """
    value = np.asarray(max_pot_dmg, dtype=np.float64)
    inc = np.asarray(income, dtype=np.float64)
    keep = (value > 0) & (inc > 0)
    value, inc = value[keep], inc[keep]
    median = float(np.median(inc)) if median_income is None else float(median_income)

    fig, ax = _panels(axes, 2, (12, 4.4))
    # Coarse bins: synthetic incomes sit on 99 percentile steps, so fine bins
    # would show spikes that are an artefact of the steps.
    bins = np.logspace(np.log10(inc.min()), np.log10(inc.max()), 25)
    ax[0].hist(inc, bins=bins, color="C0", alpha=0.8)
    ax[0].set_xscale("log")
    ax[0].axvline(median, color="k", ls="--", lw=1.4,
                  label=f"median ${median:,.0f}")
    ax[0].set_title("(a) Household income")
    ax[0].set_xlabel("Annual income ($, log scale)")
    ax[0].set_ylabel("Number of households")
    ax[0].legend(fontsize=8)

    # Spearman rank correlation: Pearson correlation of the ranks.
    ranks = np.argsort(np.argsort(np.stack([value, inc])), axis=1)
    rho = float(np.corrcoef(ranks)[0, 1])
    hb = ax[1].hexbin(value, inc, xscale="log", yscale="log", gridsize=40,
                      mincnt=1, cmap="viridis", bins="log")
    cbar = fig.colorbar(hb, ax=ax[1], label="Households per cell (log scale)")
    # Plain counts (1, 10, 100) instead of the default 10^n labels.
    plain = FuncFormatter(lambda v, _pos: f"{v:,.0f}")
    cbar.ax.yaxis.set_major_formatter(plain)
    cbar.ax.yaxis.set_minor_formatter(FuncFormatter(lambda _v, _pos: ""))
    ax[1].set_title(f"(b) Building value vs income (rank correlation {rho:.2f})")
    ax[1].set_xlabel("Building value ($, log scale)")
    ax[1].set_ylabel("Annual income ($, log scale)")
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def plot_perception_response(
    decision,
    severities: tuple[float, ...] = (0.05, 0.2, 1.0),
    years: int = 10,
    ax: Axes | None = None,
    title: str = "Risk perception after a single flood",
    save_path: str | Path | None = None,
) -> Figure:
    """
    How risk perception jumps after a flood and fades again.

    One line per flood severity (realised damage over building value),
    from :func:`floodadapt_abm.analysis.perception_curve`, plus the floor a
    household returns to without floods.

    Parameters
    ----------
    decision : DecisionConfig
        Behaviour settings.
    severities : tuple[float, ...]
        Flood severities to draw.  Default 5 %, 20 % and a total loss.
    years : int
        Years after the flood.  Default ``10``.
    ax : Axes or None
        Panel to draw into.  Default ``None`` (new ``7 x 4.2`` figure).
    title : str
        Panel title.
    save_path : str, Path or None
        File to save the figure to.  Default ``None``.

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig = ax.figure
    curves = an.perception_curve(decision, np.asarray(severities), years)
    tau = np.arange(years + 1)
    # Severity is ordered, so use a sequential palette (light = mild flood)
    # rather than the categorical cycle that marks runs elsewhere.
    shades = plt.get_cmap("plasma_r")(np.linspace(0.25, 0.85, len(severities)))
    for shade, s, curve in zip(shades, severities, curves):
        label = ("a total loss" if s >= 1.0
                 else f"flood costing {100 * s:.0f}% of the building")
        ax.plot(tau, curve, color=shade, lw=LINE_LW, marker="o", ms=4,
                label=label)
    ax.axhline(decision.risk_perc_min, color="k", ls=":", lw=1.2,
               label="floor without floods")
    ax.set_xlabel("Years since the flood")
    ax.set_ylabel("Risk perception (multiplier on probability)")
    ax.set_title(title)
    ax.set_xticks(tau)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)


def plot_affordability_curve(
    subsidies: np.ndarray,
    shares: dict[str, np.ndarray],
    ax: Axes | None = None,
    styles: dict[str, tuple[str, str]] | None = None,
    title: str = "Affordability of cover for the households that need it",
    save_path: str | Path | None = None,
) -> Figure:
    """
    Share of the high-risk tail able to afford cover versus subsidy depth.

    Parameters
    ----------
    subsidies : np.ndarray
        Subsidy depths in ``[0, 1)``.
    shares : dict[str, np.ndarray]
        Output of :func:`floodadapt_abm.analysis.affordability_vs_subsidy`.
    ax : Axes or None
        Panel to draw into.  Default ``None`` (new ``7 x 4.2`` figure).
    styles : dict or None
        ``pricing label -> (color, linestyle)``.  Default: community solid
        ``C0``, risk-based dashed ``C1``.
    title : str
        Panel title.
    save_path : str, Path or None
        File to save the figure to.  Default ``None``.

    Returns
    -------
    Figure
    """
    default = {an.PRICING_COMMUNITY: ("C0", "-"), an.PRICING_RISK_BASED: ("C1", "--")}
    styles = _resolve_styles(list(shares), {**default, **(styles or {})})
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig = ax.figure
    subs = 100.0 * np.asarray(subsidies, dtype=np.float64)
    for label, share in shares.items():
        color, ls = styles[label]
        ax.plot(subs, 100.0 * np.asarray(share), lw=2.4, color=color, ls=ls,
                label=label)
    ax.set_xlabel("Public subsidy on the premium (%)")
    ax.set_ylabel("% of high-risk households\nthat can afford cover")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save_path)
