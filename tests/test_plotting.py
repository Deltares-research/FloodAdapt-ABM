"""
test_plotting.py
================
Tests for :mod:`floodadapt_abm.plotting`: every figure function returns a
``Figure`` with the expected panels, the uptake views are drawn as requested
(and coinciding pairs are never duplicated), caller-supplied axes are
honoured, ``plt.show`` is never called and figures can be saved.  Runs
headless on the Agg backend with the mock dataset.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from floodadapt_abm import SimulationEngine, ThresholdRule  # noqa: E402
from floodadapt_abm import analysis as an  # noqa: E402
from floodadapt_abm import plotting as pl  # noqa: E402
from tests.conftest import historical_modes_config, make_mock_dataset  # noqa: E402

SLR = np.linspace(0.0, 1.0, 8)
YEARS = 2020 + np.arange(SLR.size)
SEED = 123
NO_SEQ = 3


def _run(config, rule=None) -> dict:
    """Run the mock table under ``config``."""
    engine = SimulationEngine(ds=make_mock_dataset(), config=config,
                              decision_rule=rule)
    return engine.run(SLR, no_seq=NO_SEQ, seed=SEED)


def _churn_history() -> np.ndarray:
    """A status history whose stock falls, so ever and current differ."""
    history = np.zeros((2, 6, YEARS.size), dtype=bool)
    history[:, 0, 1:3] = True          # agent 0 holds the status in years 1-2
    history[:, 1, 4:] = True           # agent 1 from year 4 on
    return history


@pytest.fixture(scope="module")
def adapt_results() -> dict:
    cfg = historical_modes_config(lifespan_dryproof=None)
    return _run(cfg, ThresholdRule(cfg.decision, damage_threshold=0.05))


@pytest.fixture(scope="module")
def baseline_results() -> dict:
    cfg = historical_modes_config(lifespan_dryproof=None)
    return _run(cfg, ThresholdRule(cfg.decision, damage_threshold=1.01))


@pytest.fixture(scope="module")
def insurance_results() -> dict:
    return _run(historical_modes_config(include_insurance=True))


@pytest.fixture(autouse=True)
def _close_figures():
    """Keep the figure registry empty between tests."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _forbid_show(monkeypatch):
    """The package must never call ``plt.show``."""
    def _raise(*_args, **_kwargs):
        raise AssertionError("plt.show() must not be called by plotting.py")
    monkeypatch.setattr(plt, "show", _raise)


def _primary_axes(fig: Figure) -> list:
    """Axes of ``fig`` that are not SLR twin axes."""
    return [a for a in fig.axes if a.get_label() != pl.SLR_TWIN_LABEL]


def _labels(ax) -> list[str]:
    return [line.get_label() for line in ax.lines]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def test_plot_uptake_both_views():
    fig, ax = plt.subplots()
    pl.plot_uptake(ax, YEARS, _churn_history(), "C0", "run", which="both")
    assert _labels(ax) == ["run (ever)", "run (current)"]
    ever, current = ax.lines
    assert (np.diff(ever.get_ydata()) >= -1e-12).all()
    assert current.get_ydata().max() <= ever.get_ydata().max() + 1e-12
    np.testing.assert_allclose(
        ever.get_ydata(), 100 * an.cumulative_uptake(_churn_history()).mean(0)
    )


def test_plot_uptake_auto_draws_pair_only_when_views_differ(adapt_results):
    fig, (ax_churn, ax_sticky) = plt.subplots(1, 2)
    pl.plot_uptake(ax_churn, YEARS, _churn_history(), "C0", "churn")
    assert _labels(ax_churn) == ["churn (ever)", "churn (current)"]
    # Floodproofing without a lifespan reset never clears: one bare line.
    pl.plot_uptake(ax_sticky, YEARS, adapt_results["adapted_history"], "C1", "sticky")
    assert _labels(ax_sticky) == ["sticky"]


def test_plot_uptake_single_views():
    fig, (ax_ever, ax_cur) = plt.subplots(1, 2)
    pl.plot_uptake(ax_ever, YEARS, _churn_history(), "C0", "run", which="ever")
    pl.plot_uptake(ax_cur, YEARS, _churn_history(), "C0", "run", which="current",
                   as_percent=False)
    assert _labels(ax_ever) == ["run"] and _labels(ax_cur) == ["run"]
    assert ax_cur.lines[0].get_ydata().max() <= 1.0
    assert ax_ever.lines[0].get_ydata()[-1] > ax_cur.lines[0].get_ydata()[-1]
    with pytest.raises(ValueError, match="which must be one of"):
        pl.plot_uptake(ax_ever, YEARS, _churn_history(), "C0", "x", which="bogus")


def test_plot_band_layers_and_overlay(adapt_results):
    fig, ax = plt.subplots()
    per_seq = an.damage_per_year(adapt_results["damage_history"])
    pl.plot_band(ax, YEARS, per_seq, "C1", "mean")
    assert len(ax.lines) == NO_SEQ + 1          # sequences + mean
    assert len(ax.collections) == 1             # the band
    axr = pl.overlay_slr(ax, YEARS, SLR)
    assert axr.get_label() == pl.SLR_TWIN_LABEL
    assert axr in fig.axes


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def test_adaptation_overview_panels(adapt_results, baseline_results):
    fig = pl.plot_adaptation_overview(YEARS, SLR, adapt_results, baseline_results)
    assert isinstance(fig, Figure)
    assert len(_primary_axes(fig)) == 4
    assert len(fig.axes) == 8                    # four SLR twins
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "SLR trajectory" in labels
    # No lifespan reset: the ever-floodproofed line would duplicate the stock.
    assert "ever floodproofed (mean)" not in labels


def test_rule_hazard_comparison_panels(adapt_results, baseline_results):
    runs = {"A": adapt_results, "B": baseline_results}
    base = {"A": baseline_results["damage_history"],
            "B": baseline_results["damage_history"]}
    fig = pl.plot_rule_hazard_comparison(YEARS, runs, base,
                                         styles={"A": ("C3", "--")})
    axes = _primary_axes(fig)
    assert len(axes) == 3
    assert _labels(axes[0]) == ["A", "B"]        # one line per run
    b_line = [l for l in axes[2].lines if l.get_label() == "B"][0]
    np.testing.assert_allclose(b_line.get_ydata(), 0.0)


def test_insurance_overview_panels(insurance_results, adapt_results):
    fig = pl.plot_insurance_overview(
        YEARS, {"ins": insurance_results}, reference=adapt_results,
        cap_budget=1_000.0,
    )
    axes = _primary_axes(fig)
    assert len(axes) == 6
    assert _labels(axes[0]) == ["ins"]                     # ever insured
    assert _labels(axes[1]) == ["ins"]                     # currently insured
    assert axes[0].get_title().startswith("(a) Ever insured")
    assert axes[1].get_title().startswith("(b) Currently insured")
    assert "R1 (no insurance offered)" in _labels(axes[2])
    assert any("affordability budget" in l for l in _labels(axes[3]))
    assert "R1 gross damage" in _labels(axes[4])
    assert "R1 gross damage" in _labels(axes[5])


def test_subsidy_experiment_panels(insurance_results, adapt_results):
    fig = pl.plot_subsidy_experiment(
        YEARS, {"s60": insurance_results, "s90": insurance_results},
        reference=adapt_results,
        styles={"s60": ("C2", "-"), "s90": ("C4", "--")},
        reference_label="REF", cap_budget=500.0,
    )
    axes = _primary_axes(fig)
    assert len(axes) == 6
    assert _labels(axes[0]) == ["s60", "s90"]
    assert axes[4].get_title().endswith("vs REF")
    assert any("affordability budget" in l for l in _labels(axes[3]))
    assert "REF gross damage" in _labels(axes[5])
    colors = {l.get_label(): l.get_color() for l in axes[0].lines}
    assert colors["s60"] != colors["s90"]


def test_affordability_curve(tmp_path):
    rng = np.random.default_rng(0)
    ead = np.concatenate([np.zeros(40), rng.lognormal(5.0, 1.5, 60)])
    income = rng.lognormal(11.3, 0.9, ead.size)
    subs = np.linspace(0.0, 0.99, 10)
    shares = an.affordability_vs_subsidy(ead, income, 0.06, 0.1, subs)
    out = tmp_path / "curve.png"
    fig = pl.plot_affordability_curve(subs, shares, save_path=out)
    assert len(_primary_axes(fig)) == 1
    assert len(fig.axes[0].lines) == 2
    assert out.exists() and out.stat().st_size > 0


def test_agent_profile_panels():
    rng = np.random.default_rng(1)
    value = rng.lognormal(12.4, 0.5, 400)
    income = rng.lognormal(11.4, 0.9, 400)
    fig = pl.plot_agent_profile(value, income, median_income=88_494.0)
    axes = [a for a in _primary_axes(fig) if a.get_label() != "<colorbar>"]
    assert len(axes) == 2
    assert axes[0].get_xscale() == "log"
    assert "rank correlation" in axes[1].get_title()
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert labels == ["median $88,494"]


def test_perception_response_lines():
    cfg = historical_modes_config(perception_mode="severity")
    fig = pl.plot_perception_response(cfg.decision, severities=(0.1, 1.0), years=5)
    ax = _primary_axes(fig)[0]
    assert len(ax.lines) == 3                   # two severities + floor
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), an.perception_curve(cfg.decision, 0.1, 5)[0]
    )


def test_caller_supplied_axes_are_used(adapt_results, baseline_results):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    out = pl.plot_adaptation_overview(YEARS, SLR, adapt_results, baseline_results,
                                      axes=axes)
    assert out is fig
    assert all(len(ax.lines) > 0 for ax in axes)
    with pytest.raises(ValueError, match="need 4 axes"):
        pl.plot_adaptation_overview(YEARS, SLR, adapt_results, baseline_results,
                                    axes=axes[:2])


def test_save_path_writes_png(tmp_path, adapt_results, baseline_results):
    out = tmp_path / "overview.png"
    pl.plot_adaptation_overview(YEARS, SLR, adapt_results, baseline_results,
                                save_path=out)
    assert out.exists() and out.stat().st_size > 0
