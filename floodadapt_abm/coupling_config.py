"""
coupling_config.py
==================
Configuration dataclasses for the coupling of FloodAdapt-ABM with the
DYNAMO-M Subjective Expected Utility (SEU) decision framework.

All field defaults are calibrated against the Charleston probabilistic
lookup table (lookup_table_charleston_beta_release_ABM_probabilistic_set.nc)
and the DYNAMO-M settings.yml.  Override any field when constructing the
dataclass to adapt to a different site or parameterisation.

Reference
--------------------
Tierolf, L., Haer, T., Botzen, W. J. W., de Bruijn, J. A., Ton, M. J.,
Reimann, L., & Aerts, J. C. J. H. (2023). A coupled agent-based model for
France for simulating adaptation and migration decisions under future coastal
flood risk. Scientific Reports, 13(1), 4176.
https://doi.org/10.1038/s41598-023-31351-y
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# NetCDF dimension / variable / attribute name mapping
# ---------------------------------------------------------------------------

@dataclass
class NetCDFMappingConfig:
    """
    Maps logical names used throughout the bridge to the actual dimension,
    variable, and attribute names stored in the xarray.Dataset produced by
    ``setup_lookup_table.create_lookup_table``.

    Changing these strings is the *only* action required when the lookup table
    schema changes (e.g., a column is renamed in a future FloodAdapt release).

    Attributes
    ----------
    dimension_object_id : str
        Name of the building-level dimension in the dataset.
    dimension_event : str
        Name of the event dimension.
    dimension_slr : str
        Name of the sea-level-rise dimension.
    dimension_strategy : str
        Name of the strategy dimension.
    var_total_damage : str
        Name of the total-damage data variable.
    var_inun_depth : str
        Name of the inundation-depth data variable.
    attr_max_pot_dmg : str
        Key for the maximum-potential-damage array stored as a coordinate
        attribute on ``object_id``.
    attr_event_freq : str
        Key for the event-frequency array stored as an attribute on
        ``event``.
    attr_building_type : str
        Key for the primary-object-type list stored as an attribute on
        ``object_id``.
    residential_substring : str
        Substring match used to identify residential buildings inside
        ``attr_building_type``.  Matching is case-sensitive and uses
        ``np.char.find``.  The default ``"RES"`` matches ``"RES"``,
        ``"COM_RES"``, etc.
    strategy_no_measures : str
        Strategy label for the baseline (no adaptation) case.
    strategy_floodproof : str
        Strategy label for the adapted (floodproofed) case.
    """

    # Dimension names
    dimension_object_id: str = "object_id"
    dimension_event: str = "event"
    dimension_slr: str = "slr"
    dimension_strategy: str = "strategy"

    # Variable names
    var_total_damage: str = "total_damage"
    var_inun_depth: str = "inun_depth"

    # Coordinate attribute keys
    attr_max_pot_dmg: str = "max_pot_dmg"
    attr_event_freq: str = "freq"
    attr_building_type: str = "primary_object_type"

    # Filtering & strategy
    residential_substring: str = "RES"
    strategy_no_measures: str = "no_measures"
    strategy_floodproof: str = "floodproof_all_0"


# ---------------------------------------------------------------------------
# SEU decision-model parameters
# ---------------------------------------------------------------------------

@dataclass
class DecisionConfig:
    """
    Parameters that govern the Subjective Expected Utility (SEU) decision
    model ported from DYNAMO-M.

    All default values match the DYNAMO-M ``settings.yml`` calibration for
    the France coastal study (Tierolf et al., 2023) and the Charleston
    test site.

    Attributes
    ----------
    risk_aversion : float
        CRRA risk-aversion coefficient (sigma).  ``sigma == 1`` activates
        log-utility; ``sigma != 1`` uses power-utility
        ``U(x) = x^(1-sigma) / (1-sigma)``.
        Default: ``1.0`` (log-utility; from DYNAMO-M ``settings.yml``).
    discount_rate : float
        Annual time-discounting rate ``r`` used in NPV calculations.
        Default: ``0.032`` (3.2 %, from DYNAMO-M ``decisions.time_discounting``).
    decision_horizon : int
        Planning horizon ``T`` in years over which households discount future
        flood damages.
        Default: ``15`` years (from DYNAMO-M ``decisions.decision_horizon``).
    risk_perc_min : float
        Minimum flood risk perception multiplier.  Households that have not
        experienced a flood for a long time converge to this value.
        Default: ``0.01`` (from DYNAMO-M ``risk_perception.min``).
    risk_perc_max : float
        Maximum flood risk perception multiplier, applied immediately after
        a flood.
        Default: ``2.0`` (from DYNAMO-M ``risk_perception.max``).
    risk_perc_coef : float
        Exponential decay coefficient for risk perception.  Negative values
        produce decay over time since last flood.  Formula:
        ``risk_perc = risk_perc_max * 1.6^(coef * flood_timer) + risk_perc_min``.
        Default: ``-3.6`` (from DYNAMO-M ``risk_perception.coef``).
    loan_duration : int
        Duration of the adaptation loan in years.  Used to annualise and
        time-discount the one-off floodproofing cost.
        Default: ``16`` years (from DYNAMO-M settings).
    interest_rate : float
        Interest rate ``r_loan`` applied to the annualised adaptation loan.
        Default: ``0.04`` (4 %, from DYNAMO-M ``adaptation.interest_rate``).
    adaptation_cost_fraction : float
        Fraction of ``max_pot_dmg`` used as the total (one-off) adaptation
        cost per building when external cost data are unavailable.
        Default: ``0.10`` (10 % of maximum potential damage).
    expenditure_cap : float
        Maximum fraction of annual income a household is willing to spend on
        adaptation per year.  Households where
        ``income * expenditure_cap <= annual_adaptation_cost`` are set to
        ``EU_adapt = -inf`` (cannot afford).
        Default: ``0.06`` (from DYNAMO-M settings).
    amenity_weight : float
        Scalar weight applied to the amenity value when computing NPV.
        Default: ``1.0`` (neutral).
    error_interval : float
        Half-width of the uniform error term applied to each EU outcome to
        introduce stochastic choice.  ``0.0`` disables stochastic errors.
        Default: ``0.0``.
    income_to_wealth_ratio : float
        Multiplier converting annual income to household wealth when wealth
        data are not available from the dataset.
        Default: ``4.14`` (median ratio from DYNAMO-M income-wealth table,
        corresponding to the 40th percentile).  Source:
        ``decision_module.py`` lines 27-30, percentile table
        ``[0, 20, 40, 60, 80, 100]`` → ratio ``[0, 1.06, 4.14, 4.19, 5.24, 6]``.
    event_draw_mode : str
        Stochastic model for the yearly flood-event draw.

        * ``"poisson"`` — each event occurs ``n_i ~ Poisson(freq_i * dt)``
          times per year.  Statistically exact for occurrence *rates*: the
          realised long-run rate of every event (including sub-annual
          ``freq > 1`` events and rare extremes) equals its nominal
          frequency, and one event may occur several times in a year.
        * ``"bernoulli_clip"`` — legacy behaviour: one Bernoulli trial per
          event with ``p = min(freq_i * dt, 1)``.  Rates above ``1/dt`` are
          clipped to certainty, so sub-annual events occur every year and
          (combined with ``max_events_per_year``) crowd out extremes.
          Retained for reproducing historical results only.
        Default: ``"poisson"`` (``legacy()`` pins ``"bernoulli_clip"``).
    nuisance_freq_threshold : float or None
        When set, events with ``freq > threshold`` (events/year) are dropped
        from the catalogue once at data load — from both the hazard draw and
        the SEU decision integral, which therefore always see the same event
        set.  ``1.0`` reproduces the historical pre-coupling setup that
        disregarded sub-annual "nuisance" events entirely, and is the
        recommended setting for event sets containing ``freq > 1`` entries
        (their near-certain probabilities otherwise collapse onto the 0.998
        perceived-probability cap inside the SEU integral).
        ``None`` keeps every event.
        Default: ``None``.
    max_events_per_year : int or None
        Maximum number of stochastic flood-event occurrences retained in a
        single simulation year; ``None`` disables the cap (recommended with
        ``"poisson"`` — realised damage is already bounded by
        ``max_pot_dmg``, and any discard biases the hazard statistics).
        Default: ``None`` (``legacy()`` pins ``4``).
    cap_policy : str
        How surplus occurrences are discarded when ``max_events_per_year``
        binds.

        * ``"largest_damage"`` — keep the most damaging occurrences
          (deterministic, no extra RNG; preserves extremes).
        * ``"random"`` — legacy: uniform random selection without
          replacement (extremes are discarded at the same rate as nuisance
          events).
        Default: ``"largest_damage"`` (``legacy()`` pins ``"random"``).
    seu_prob_mode : str
        How event frequencies are converted into the exceedance
        probabilities fed to the SEU expected-utility integral.

        * ``"exceedance"`` — ``p_i = 1 - exp(-freq_i * dt)``, the exact
          probability of at least one occurrence of a Poisson arrival with
          rate ``freq_i``; always < 1, consistent with the exceedance-curve
          semantics of the integral.
        * ``"raw_freq"`` — legacy: ``p_i = freq_i`` used directly (valid
          only while every frequency is well below 1).
        Default: ``"exceedance"`` (``legacy()`` pins ``"raw_freq"``).
    perception_mode : str
        How a realised flood feeds the risk-perception spike.

        * ``"severity"`` — the post-flood perception peak scales with the
          damage severity ``s = realised / max_pot_dmg`` through the
          functional form selected by ``perception_severity_form``
          (default: concave power law), so a nuisance flood and a
          catastrophe are no longer identical to the agent.  A deliberate
          improvement beyond native DYNAMO-M, whose trigger is binary
          (``flood_risk.py:619``).
        * ``"binary"`` — legacy/native: any positive damage produces the
          full ``risk_perc_max`` spike.
        Default: ``"severity"`` (``legacy()`` pins ``"binary"``).
    flood_significance_threshold : float
        Minimum damage severity (fraction of ``max_pot_dmg``) for a flood to
        register as experienced — resets the flood timer and spikes risk
        perception.  ``0.0`` reproduces the legacy ``realised > 0`` trigger
        (where float round-off or a residual post-adaptation trickle counts
        as a flood).
        Default: ``0.01`` (``legacy()`` pins ``0.0``).
    perception_severity_form : str
        Functional form mapping damage severity ``s = realised /
        max_pot_dmg`` (clipped to [0, 1]) to the post-flood perception peak
        under ``perception_mode="severity"``.  All three forms are
        one-parameter families (identifiable from small survey samples) and
        agree at ``s = 1`` (total loss reproduces the full legacy spike):

        * ``"power"`` — ``peak = risk_perc_max * s ** gamma`` with
          ``gamma = perception_severity_exponent``.  Concave with infinite
          slope at ``s = 0``: even small floods produce large spikes
          (availability heuristic).  **Preferred default.**
        * ``"saturating_exp"`` — ``peak = risk_perc_max *
          (1 - exp(-k*s)) / (1 - exp(-k))`` with
          ``k = perception_severity_rate``.  Concave like the power law but
          with *finite* slope at ``s = 0`` — the alternative hypothesis
          that small floods produce proportionally small responses.
        * ``"threshold_linear"`` — ``peak = risk_perc_max *
          clip((s - s0) / (1 - s0), 0, 1)`` with
          ``s0 = perception_severity_threshold``.  No perception response
          below the damage threshold ``s0``, then linear — the qualitatively
          opposite (near-miss/threshold) hypothesis.

        See ``docs/calibration_validation_guide.md`` for the recommended
        analysis order and how to discriminate the forms with survey data.
        Default: ``"power"``.
    perception_severity_exponent : float
        Exponent ``gamma`` of the concave severity scaling used by
        ``perception_severity_form="power"``:
        ``peak = risk_perc_max * clip(severity, 0, 1) ** gamma``.
        ``gamma = 0.5`` (default) encodes diminishing sensitivity — a flood
        damaging 25 % of the home already triggers ~71 % of the maximum
        spike — consistent with the availability-heuristic literature the
        perception model is built on.  ``gamma -> 0`` recovers the
        binary/native response; ``gamma = 1`` is linear.  Calibratable.
        Default: ``0.5``.
    perception_severity_rate : float
        Rate ``k > 0`` of the ``"saturating_exp"`` severity form (ignored by
        the other forms).  Larger ``k`` saturates faster (more concave);
        ``k -> 0`` approaches linear.
        Default: ``3.0``.
    perception_severity_threshold : float
        Damage-severity threshold ``s0`` in ``[0, 1)`` of the
        ``"threshold_linear"`` severity form (ignored by the other forms):
        floods below ``s0`` produce no perception spike.
        Default: ``0.1``.
    income_mode : str
        Fallback used to construct per-agent incomes when no
        ``income_per_agent`` array is supplied.

        * ``"synthetic_lognormal"`` — port of the native DYNAMO-M pipeline:
          a regional lognormal income distribution is built from
          ``median_income`` and ``mean_median_inc_ratio``, each household
          samples it at its income percentile, and wealth follows the
          percentile-varying wealth-to-income table (see
          ``income_to_wealth_ratio``).  Income is independent of building
          value, so the affordability constraint can genuinely bind.
        * ``"mpd_ratio"`` — legacy: ``income = max_pot_dmg /
          income_to_wealth_ratio`` (which makes wealth identically equal to
          the building value and the affordability gate algebraically
          inert).
        Default: ``"synthetic_lognormal"`` (``legacy()`` pins ``"mpd_ratio"``).
    median_income : float
        Regional median gross household income used by
        ``income_mode="synthetic_lognormal"`` (same role as the GDL/World-
        Bank regional income in native DYNAMO-M ``base_nodes.py:37-76``).
        Site-specific; override per case study.
        Default: ``70_000.0`` (Charleston-area order of magnitude, USD).
    mean_median_inc_ratio : float
        Mean-to-median income ratio controlling the spread (sigma) of the
        synthetic lognormal income distribution.  Native DYNAMO-M reads this
        from the UN WIID per country with fallback ``1.15``
        (``settings.yml`` ``adaptation.mean_median_inc_ratio``).
        Default: ``1.15``.
    adaptation_total_cost : float or None
        Fixed one-off dry-floodproofing cost per household (currency units
        of the lookup table).  Mirrors native DYNAMO-M, where the cost is a
        country-scaled constant (10,800 EUR France-anchored,
        ``prepare_scale_to_GDP.py``) rather than a fraction of property
        value.  ``None`` falls back to the legacy
        ``adaptation_cost_fraction * max_pot_dmg``.
        Default: ``None`` (legacy).
    include_insurance : bool
        Offer flood insurance as a third decision option (ported
        ``calcEU_insure``; flat community premium = mean expected annual
        damage, as in native ``insurer_agent.py``).  ``False`` matches the
        native DYNAMO-M default (``settings.yml`` ``include_insurance``).
        Default: ``False``.
    insurance_deductible : float
        Fraction of flood damage still borne by an insured household (the
        native module hard-codes ``deductable = 0.1``,
        ``decision_module.py:259``).
        Default: ``0.1``.
    insurance_pricing : str
        How the insurer prices the annual premium.

        * ``"community"`` — native behaviour: one **flat** community-rated
          premium for everybody, equal to the *mean* expected annual damage
          of the node (``insurer_agent.py:20-26``).  Over a skewed risk pool
          this prices the median household far above its own risk (on the
          real Charleston table the flat premium ends up above the
          expenditure cap of most households, so uptake stays near 0 %).
        * ``"risk_based"`` — each household is charged its **own** expected
          annual damage.  This is the standard actuarially-fair premium; it
          makes the premium heterogeneous, affordable for the low-risk
          majority, and lets the SEU comparison (not the affordability cap)
          drive uptake.  Beyond native, which has no risk-based insurer.
        Default: ``"community"`` (native).
    insurance_loading : float
        Multiplier applied to the actuarially-fair premium to represent the
        insurer's expenses, capital costs and margin.  ``1.0`` is a fair
        premium; ``1.3`` a 30 % loading.  Applied under both pricing modes.
        Default: ``1.0``.
    insurance_subsidy : float
        Fraction of the premium paid by a public scheme rather than the
        household (the premium analogue of native's
        ``subsidize_adaptation_costs``, ``government_agent.py:572-576``,
        which halves adaptation costs for constrained households).  ``0.0``
        disables it; ``0.5`` halves the household's premium.  Beyond native.
        Default: ``0.0``.
    lifespan_dryproof : int
        Service life of a dry-floodproofing measure in years.  Adapted
        households whose adaptation age (``time_adapted``) reaches this value
        have their floodproofing expire (``is_adapted`` reset to ``False``)
        and re-enter the decision each subsequent year.  Ported from
        DYNAMO-M's agent layer (``coastal_nodes.py`` lines 2221-2227,
        ``self.adapt[self.time_adapt == lifespan_dryproof] = 0``), where the
        default is ``settings.yml`` ``adaptation.lifespan_dryproof``.
        Default: ``75`` years.
    """

    risk_aversion: float = 1.0
    discount_rate: float = 0.032
    decision_horizon: int = 15
    risk_perc_min: float = 0.01
    risk_perc_max: float = 2.0
    risk_perc_coef: float = -3.6
    loan_duration: int = 16
    interest_rate: float = 0.04
    adaptation_cost_fraction: float = 0.10
    expenditure_cap: float = 0.06
    amenity_weight: float = 1.0
    error_interval: float = 0.0
    income_to_wealth_ratio: float = 4.14
    max_events_per_year: int | None = None
    lifespan_dryproof: int = 75

    # -- Behaviour-mode switches (see class docstring for semantics) --------
    # Defaults are the current behaviour; use ``DecisionConfig.legacy()``
    # to reproduce the historical behaviour.
    event_draw_mode: str = "poisson"
    nuisance_freq_threshold: float | None = None
    cap_policy: str = "largest_damage"
    seu_prob_mode: str = "exceedance"
    perception_mode: str = "severity"
    flood_significance_threshold: float = 0.01
    perception_severity_form: str = "power"
    perception_severity_exponent: float = 0.5
    perception_severity_rate: float = 3.0
    perception_severity_threshold: float = 0.1
    income_mode: str = "synthetic_lognormal"
    median_income: float = 70_000.0
    mean_median_inc_ratio: float = 1.15
    adaptation_total_cost: float | None = None
    include_insurance: bool = False
    insurance_deductible: float = 0.1
    insurance_pricing: str = "community"
    insurance_loading: float = 1.0
    insurance_subsidy: float = 0.0

    @classmethod
    def legacy(cls) -> "DecisionConfig":
        """
        Preset reproducing the historical (pre-refactor) behaviour **bit-exactly**.

        Pins every behaviour-mode switch to its historical value regardless
        of the current class defaults: Bernoulli-clip event draw with a
        random cap of 4, raw-frequency SEU probabilities, binary flood
        perception, ``max_pot_dmg``-derived income, fractional adaptation
        cost, and no insurance.  Used by the golden regression test
        (``tests/test_legacy_mode.py``) and the verification harnesses.

        Returns
        -------
        DecisionConfig
            A configuration whose behaviour is bit-identical to the
            pre-refactor engine.
        """
        return cls(
            event_draw_mode="bernoulli_clip",
            nuisance_freq_threshold=None,
            max_events_per_year=4,
            cap_policy="random",
            seu_prob_mode="raw_freq",
            perception_mode="binary",
            flood_significance_threshold=0.0,
            income_mode="mpd_ratio",
            adaptation_total_cost=None,
            include_insurance=False,
            insurance_pricing="community",
            insurance_loading=1.0,
            insurance_subsidy=0.0,
        )


# ---------------------------------------------------------------------------
# Composite coupling configuration
# ---------------------------------------------------------------------------

@dataclass
class CouplingConfig:
    """
    Top-level configuration container that combines the NetCDF mapping and
    the SEU decision parameters.

    Usage example
    -------------
    >>> from coupling_config import CouplingConfig
    >>> cfg = CouplingConfig()                   # all defaults
    >>> cfg.netcdf.residential_substring = "COM" # select commercial instead
    >>> cfg.decision.risk_aversion = 2.0          # higher risk aversion

    Attributes
    ----------
    netcdf : NetCDFMappingConfig
        Dataset column / dimension / attribute name mapping.
    decision : DecisionConfig
        SEU behavioural parameters.
    random_seed : int
        Global random seed for reproducibility of stochastic error terms.
        Default: ``42``.
    """

    netcdf: NetCDFMappingConfig = field(default_factory=NetCDFMappingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    random_seed: int = 42

    @classmethod
    def legacy(cls, random_seed: int = 42) -> "CouplingConfig":
        """
        Composite preset wrapping :meth:`DecisionConfig.legacy`.

        Parameters
        ----------
        random_seed : int
            Global random seed (default ``42``, the historical default).

        Returns
        -------
        CouplingConfig
            Configuration reproducing the pre-refactor behaviour bit-exactly.
        """
        return cls(decision=DecisionConfig.legacy(), random_seed=random_seed)
