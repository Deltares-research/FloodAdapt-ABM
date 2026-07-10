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
    max_events_per_year : int
        Maximum number of stochastic flood events that can occur in a
        single simulation year.  When the Bernoulli-trial draw yields
        more events than this cap, the surplus events are dropped.

        .. note::
           Cap-selection semantics are being unified (see
           ``20260707_todo_next_steps``).  Historically the retained events
           were the highest-frequency ones; the current example scripts
           retain the highest-magnitude (largest expected-damage) events;
           the agreed target is **random selection without replacement** from
           the drawn pool, which preserves the Monte-Carlo distribution.
           Whichever policy is active, the count is capped at this value.
        Default: ``4``.
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
    max_events_per_year: int = 4
    lifespan_dryproof: int = 75


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
