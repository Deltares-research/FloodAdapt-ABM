"""
dynamo_decision_bridge.py
=========================
Bridge module coupling FloodAdapt-ABM's precomputed damage lookup table with
the Subjective Expected Utility (SEU) decision framework from DYNAMO-M.

The SEU mathematical functions (``_iterate_through_flood``,
``_calc_eu_do_nothing``, ``_calc_eu_adapt``) and the risk-perception decay
formula are ported *directly* from DYNAMO-M's ``decision_module.py`` and
``hazards/flooding/flood_risk.py`` and adapted for a numpy-only, self-
contained implementation that does **not** require a running DYNAMO-M model
instance.

Scientific reference
--------------------
Tierolf, L., Haer, T., Botzen, W. J. W., de Bruijn, J. A., Ton, M. J.,
Reimann, L., & Aerts, J. C. J. H. (2023). A coupled agent-based model for
France for simulating adaptation and migration decisions under future coastal
flood risk. Scientific Reports, 13(1), 4176.
https://doi.org/10.1038/s41598-023-31351-y

Design decisions
----------------
* All heavy arrays (``damage_matrix``, NPV tensors) are ``float32`` to
  minimise memory when ``n_agents`` is large (61 k buildings in Charleston).
* ``filter_residential`` applies a substring search on
  ``primary_object_type`` so mixed types such as ``"COM_RES"`` are still
  included by default.
* Wealth and property capping bound all damage values to ``max_pot_dmg`` to
  stay consistent with the FloodAdapt-ABM lookup-table semantics.
* Risk perception is updated per-agent after each year using the
  DYNAMO-M exponential decay formula:
  ``rp = rp_max * base^(coef * flood_timer) + rp_min``
  where ``base = 1.6`` (hard-coded in the original DYNAMO-M source).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from floodadapt_abm.coupling_config import CouplingConfig, DecisionConfig, NetCDFMappingConfig
from floodadapt_abm._core.lookup_utils import (
    interpolate_damage_at_slr,
    materialize_strategy_cube,
    interpolate_cube_at_slr,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DynamoDecisionBridge:
    """
    Couples a FloodAdapt-ABM xarray lookup-table Dataset with DYNAMO-M's
    Subjective Expected Utility decision logic.

    Parameters
    ----------
    ds : xarray.Dataset
        Precomputed damage lookup table produced by
        ``setup_lookup_table.create_lookup_table``.  Must contain the
        dimensions and attributes described in ``NetCDFMappingConfig``.
    config : CouplingConfig
        Top-level coupling configuration.  Defaults to ``CouplingConfig()``
        (all DYNAMO-M defaults + Charleston NetCDF column names).
    income_per_agent : np.ndarray or None
        Annual income for each *residential* agent in the same unit as
        damages (USD by default).  Shape ``(n_res_agents,)``.
        When ``None`` the bridge synthesises income from
        ``max_pot_dmg / income_to_wealth_ratio`` (fallback for demos).
    amenity_value_per_agent : np.ndarray or None
        Amenity value per agent.  When ``None`` defaults to zero.

    Attributes
    ----------
    n_agents : int
        Number of residential buildings after filtering.
    object_ids : np.ndarray[str]
        Object IDs of the residential subset, shape ``(n_agents,)``.
    max_pot_dmg : np.ndarray[float32]
        Maximum potential damage per agent, shape ``(n_agents,)``.
    income : np.ndarray[float32]
        Annual income per agent, shape ``(n_agents,)``.
    wealth : np.ndarray[float32]
        Household wealth per agent, shape ``(n_agents,)``.
    amenity_value : np.ndarray[float32]
        Amenity value per agent, shape ``(n_agents,)``.
    risk_perception : np.ndarray[float32]
        Current risk-perception multiplier per agent, shape ``(n_agents,)``.
    flood_timer : np.ndarray[int32]
        Years since last flood per agent, shape ``(n_agents,)``.
    is_adapted : np.ndarray[bool]
        Current adaptation status per agent, shape ``(n_agents,)``.
    """

    # Risk-perception decay base (from DYNAMO-M flood_risk.py:650-651)
    # See Tierolf et al. (2023) Section 2.2 for the scientific basis.
    _RISK_PERC_BASE: float = 1.6

    def __init__(
        self,
        ds: xr.Dataset,
        config: CouplingConfig | None = None,
        income_per_agent: np.ndarray | None = None,
        amenity_value_per_agent: np.ndarray | None = None,
        wealth_per_agent: np.ndarray | None = None,
        income_percentile_per_agent: np.ndarray | None = None,
    ) -> None:
        self._ds = ds
        self.config: CouplingConfig = config if config is not None else CouplingConfig()
        self._nc: NetCDFMappingConfig = self.config.netcdf
        self._dec: DecisionConfig = self.config.decision
        self._rng = np.random.default_rng(self.config.random_seed)

        # -- Build residential mask & extract per-agent arrays ---------------
        self._res_mask: np.ndarray = self._build_residential_mask()
        all_max_pot_dmg: np.ndarray = np.asarray(
            ds[self._nc.dimension_object_id].attrs[self._nc.attr_max_pot_dmg],
            dtype=np.float32,
        )
        all_object_ids: np.ndarray = ds[self._nc.dimension_object_id].values

        self.object_ids: np.ndarray = all_object_ids[self._res_mask]
        self.max_pot_dmg: np.ndarray = all_max_pot_dmg[self._res_mask]
        self.n_agents: int = int(self._res_mask.sum())

        # -- Initialize economic and state variables ---------------------------
        self._init_economic_variables(
            income_per_agent,
            amenity_value_per_agent,
            wealth_per_agent=wealth_per_agent,
            income_percentile_per_agent=income_percentile_per_agent,
        )
        self._init_state_variables()

        # -- Annualised adaptation cost (amortised loan) ---------------------
        self._annual_adapt_cost: np.ndarray = self._compute_annual_adapt_cost()

        # -- Event metadata from the dataset ----------------------------------
        self._event_names: np.ndarray = (
            ds[self._nc.dimension_event].values
        )
        self._event_freqs: np.ndarray = np.asarray(
            ds[self._nc.dimension_event].attrs[self._nc.attr_event_freq],
            dtype=np.float64,
        )

        # -- Nuisance-event filter (single source of truth) -------------------
        # When ``nuisance_freq_threshold`` is set, events with a higher annual
        # frequency are dropped from the catalogue ONCE here, so the hazard
        # draw and the SEU decision integral always see the same event set.
        # ``self._event_keep`` also slices the event axis of the materialised
        # damage cubes in ``_interpolate_strategy``.
        self._event_keep: np.ndarray | None = None
        threshold = self._dec.nuisance_freq_threshold
        if threshold is not None:
            keep = self._event_freqs <= float(threshold)
            self._event_keep = keep
            self._event_names = self._event_names[keep]
            self._event_freqs = self._event_freqs[keep]

        # -- Pre-build damage arrays (one per strategy, shape n_agents x n_events) --
        self._damage_no_measures: np.ndarray | None = None
        self._damage_floodproof: np.ndarray | None = None

        # -- Per-SLR interpolation cache --------------------------------------
        # Repeated ticks/sequences at the same SLR value reuse the interpolated
        # (no_measures, floodproof) matrices instead of re-running the scipy
        # interpolation over the full (object x event) grid.  The SLR trajectory
        # is identical across Monte-Carlo sequences, so this removes ``no_seq x``
        # redundant interpolation.  The cached arrays are treated as read-only.
        self._interp_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._interp_cache_enabled: bool = True
        self._interp_cache_max: int = 256
        # Materialized residential damage cubes per strategy (shape
        # (n_agents, n_slr, n_events)), built lazily once.  Re-reading and
        # residential-masking these cubes from the (lazily backed) dataset is
        # the dominant cost of a single-SLR interpolation, so we cache them.
        self._strategy_cubes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -----------------------------------------------------------------------
    # Public entry points
    # -----------------------------------------------------------------------

    @property
    def p_floods_seu(self) -> np.ndarray:
        """
        Annual exceedance probabilities fed to the SEU decision kernels.

        Controlled by ``DecisionConfig.seu_prob_mode``:

        * ``"exceedance"`` — ``p_i = 1 - exp(-freq_i)``, the exact
          probability of at least one occurrence of a Poisson arrival with
          annual rate ``freq_i``.  Always < 1, consistent with the
          exceedance-curve semantics of the trapezoidal EU integral (and
          with the yearly ``dt = 1`` step of the engine).
        * ``"raw_freq"`` — legacy: the raw frequencies are used directly
          (valid only while every frequency is well below 1).

        Returns
        -------
        p_floods : np.ndarray[float64], shape (n_events,)
            Aligned with the (possibly nuisance-filtered) event catalogue.
        """
        if self._dec.seu_prob_mode == "exceedance":
            return -np.expm1(-self._event_freqs)
        if self._dec.seu_prob_mode == "raw_freq":
            return self._event_freqs
        raise ValueError(
            f"Unknown seu_prob_mode {self._dec.seu_prob_mode!r}; "
            "expected 'exceedance' or 'raw_freq'."
        )

    def prepare_damage_arrays(
        self,
        slr_value: float,
        interp_method: str = "linear",
    ) -> None:
        """
        Interpolate per-event damage arrays for a given SLR level.

        Call this **once per simulation year** before calling
        ``evaluate_decisions``, passing the SLR value for that year.

        Parameters
        ----------
        slr_value : float
            Sea-level rise in the same unit as the lookup-table SLR
            coordinate (feet for Charleston).
        interp_method : str
            scipy interpolation kind: ``'linear'``, ``'nearest'``, ``'cubic'``.

        Notes
        -----
        Results are memoised per ``(slr_value, interp_method)`` in
        :attr:`_interp_cache` (see :meth:`clear_interp_cache`); repeated calls at
        the same SLR level return the cached matrices without re-interpolating.
        """
        cache_key = (round(float(slr_value), 9), interp_method)
        if self._interp_cache_enabled and cache_key in self._interp_cache:
            self._damage_no_measures, self._damage_floodproof = (
                self._interp_cache[cache_key]
            )
            return

        slr_arr = np.asarray(
            self._ds[self._nc.dimension_slr].values, dtype=np.float64
        )
        n_events = len(self._event_names)

        self._damage_no_measures = self._interpolate_strategy(
            slr_arr, slr_value, self._nc.strategy_no_measures,
            n_events, interp_method,
        )
        self._damage_floodproof = self._interpolate_strategy(
            slr_arr, slr_value, self._nc.strategy_floodproof,
            n_events, interp_method,
        )

        if self._interp_cache_enabled:
            # Treat cached arrays as read-only shared references (downstream code
            # never mutates the damage matrices in place).
            self._damage_no_measures.setflags(write=False)
            self._damage_floodproof.setflags(write=False)
            if len(self._interp_cache) >= self._interp_cache_max:
                self._interp_cache.pop(next(iter(self._interp_cache)))
            self._interp_cache[cache_key] = (
                self._damage_no_measures,
                self._damage_floodproof,
            )

    def clear_interp_cache(self) -> None:
        """Drop all memoised
        (stored to be retrieved without repeating the computation) per-SLR interpolation results (frees memory)."""
        self._interp_cache.clear()

    def compute_expected_annual_damages(
        self,
        use_adapted_strategy: bool = False,
    ) -> np.ndarray:
        """
        Compute the Expected Annual Damage (EAD) per agent by integrating
        event damages over their exceedance-probability curve.

        Parameters
        ----------
        use_adapted_strategy : bool
            If ``True`` use the floodproof damage matrix; otherwise use
            no-measures.

        Returns
        -------
        ead : np.ndarray, shape (n_agents,), dtype float32
            EAD per residential agent (same unit as damages in the lookup
            table).
        """
        if self._damage_no_measures is None:
            raise RuntimeError(
                "Call prepare_damage_arrays() before compute_expected_annual_damages()."
            )
        dmg = (
            self._damage_floodproof
            if use_adapted_strategy
            else self._damage_no_measures
        )
        # EAD ≈ Σ damage_i * freq_i  (trapezoidal integration in freq space
        # is skipped here; simple summation is consistent with how
        # FloodAdapt-ABM uses the table)
        ead: np.ndarray = (dmg * self._event_freqs[np.newaxis, :]).sum(
            axis=1
        ).astype(np.float32)
        return ead

    def update_flood_experience(
        self,
        flooded_agents: np.ndarray,
    ) -> None:
        """
        Update per-agent flood timer and risk perception after a simulated
        year.

        This replicates the DYNAMO-M ``stochastic_flood`` risk-perception
        update formula (source: ``flood_risk.py`` lines 650-651, and
        parameters from ``settings.yml`` lines 26-30):

            ``risk_perc = rp_max * 1.6^(coef * flood_timer) + rp_min``

        See Tierolf et al. (2023) Section 2.2 for the scientific basis.

        .. note::
           Legacy binary-only duplicate kept for the deprecated
           ``evaluate_decisions`` path.  The maintained implementation —
           including the severity-aware ``perception_mode`` — lives in
           :meth:`floodadapt_abm.simulation_engine.SimulationEngine.update_flood_experience`.

        Parameters
        ----------
        flooded_agents : np.ndarray[bool], shape (n_agents,)
            Boolean array; ``True`` for agents that experienced flooding
            this year.
        """
        # Increment timer for all; reset to 0 for those who flooded
        self.flood_timer += 1
        self.flood_timer[flooded_agents] = 0

        # Update risk perception using the DYNAMO-M decay formula
        self.risk_perception = (
            self._dec.risk_perc_max
            * (self._RISK_PERC_BASE ** (self._dec.risk_perc_coef * self.flood_timer))
            + self._dec.risk_perc_min
        ).astype(np.float32)

    def evaluate_decisions(
        self,
        year_index: int,
    ) -> np.ndarray:
        """
        Apply the SEU decision model to determine which non-adapted agents
        choose to adapt this year.

        The decision rule is:
            adapt if ``EU_adapt > EU_do_nothing``
            (subject to expenditure-cap constraint).

        Adapted agents' status is updated in-place in ``self.is_adapted``.

        Parameters
        ----------
        year_index : int
            Current simulation year index (0-based).  Passed through to
            ``time_adapted`` for loan-cost computation.

        Returns
        -------
        newly_adapted : np.ndarray[bool], shape (n_agents,)
            Boolean array marking agents that adapted *this* year.
        """
        if self._damage_no_measures is None:
            raise RuntimeError(
                "Call prepare_damage_arrays() before evaluate_decisions()."
            )

        # Build expected-damage matrices shaped (n_events, n_agents)
        # (DYNAMO-M convention: events first, agents second)
        exp_dmg_no_measures: np.ndarray = self._damage_no_measures.T.astype(
            np.float32
        )
        exp_dmg_floodproof: np.ndarray = self._damage_floodproof.T.astype(
            np.float32
        )

        # Exceedance probabilities of each event (sorted ascending for trapz).
        # Honours seu_prob_mode; identical to the raw frequencies under the
        # legacy "raw_freq" mode.
        p_floods: np.ndarray = self.p_floods_seu.astype(np.float32)

        # Time horizon array (same for all agents here; extend later if needed)
        T: np.ndarray = np.full(
            self.n_agents, self._dec.decision_horizon, dtype=np.int32
        )

        # Stochastic error terms (uniform ± error_interval)
        if self._dec.error_interval > 0:
            error_terms = self._rng.uniform(
                1.0 - self._dec.error_interval,
                1.0 + self._dec.error_interval,
                size=self.n_agents,
            ).astype(np.float32)
        else:
            error_terms = np.ones(self.n_agents, dtype=np.float32)

        # -- EU do-nothing ---------------------------------------------------
        eu_do_nothing: np.ndarray = _calc_eu_do_nothing(
            n_agents=self.n_agents,
            wealth=self.wealth,
            income=self.income,
            amenity_value=self.amenity_value,
            amenity_weight=self._dec.amenity_weight,
            risk_perception=self.risk_perception,
            expected_damages=exp_dmg_no_measures,
            adapted=self.is_adapted.astype(np.int32),
            p_floods=p_floods,
            T=T,
            r=self._dec.discount_rate,
            sigma=self._dec.risk_aversion,
            error_terms=error_terms,
        )

        # -- EU adapt --------------------------------------------------------
        # time_adapted: 0 if not yet adapted (we query only non-adapted agents)
        time_adapted: np.ndarray = np.zeros(self.n_agents, dtype=np.int32)

        eu_adapt: np.ndarray = _calc_eu_adapt(
            n_agents=self.n_agents,
            wealth=self.wealth,
            income=self.income,
            expenditure_cap=self._dec.expenditure_cap,
            amenity_value=self.amenity_value,
            amenity_weight=self._dec.amenity_weight,
            risk_perception=self.risk_perception,
            expected_damages_adapt=exp_dmg_floodproof,
            adaptation_costs=self._annual_adapt_cost,
            time_adapted=time_adapted,
            loan_duration=self._dec.loan_duration,
            p_floods=p_floods,
            T=T,
            r=self._dec.discount_rate,
            sigma=self._dec.risk_aversion,
            error_terms=error_terms,
        )

        # -- Decision --------------------------------------------------------
        # Store these on the instance purely so trace scripts can inspect them
        self._eu_do_nothing = eu_do_nothing.copy()
        self._eu_adapt = eu_adapt.copy()

        eu_diff = eu_adapt - eu_do_nothing
        newly_adapted: np.ndarray = (eu_diff > 0) & (~self.is_adapted)
        self.is_adapted[newly_adapted] = True

        return newly_adapted

    def get_current_damages(
        self,
        event_name: str,
    ) -> np.ndarray:
        """
        Return the (potentially capped) damage for each residential agent
        for a single event, respecting adaptation status.

        Damages are capped at ``max_pot_dmg`` (Property Capping) to enforce
        physical feasibility.

        Parameters
        ----------
        event_name : str
            Name of the flood event (must exist in the dataset's event coord).

        Returns
        -------
        damages : np.ndarray[float32], shape (n_agents,)
            Per-agent damage for ``event_name``.
        """
        if self._damage_no_measures is None:
            raise RuntimeError(
                "Call prepare_damage_arrays() before get_current_damages()."
            )
        event_names_list: list[str] = list(self._event_names)
        if event_name not in event_names_list:
            raise ValueError(
                f"Event '{event_name}' not found in dataset.  "
                f"Known events: {event_names_list[:5]}..."
            )
        event_idx = event_names_list.index(event_name)

        # Select strategy per agent
        dmg_no: np.ndarray = self._damage_no_measures[:, event_idx]
        dmg_fp: np.ndarray = self._damage_floodproof[:, event_idx]
        damages: np.ndarray = np.where(
            self.is_adapted, dmg_fp, dmg_no
        ).astype(np.float32)

        # --- Property capping: damage cannot exceed max_pot_dmg -------------
        damages = np.clip(damages, 0.0, self.max_pot_dmg)
        return damages

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    #: Native DYNAMO-M wealth-to-income ratio by income percentile
    #: (``coastal_nodes.py:626-632`` / ``decision_module.py:27-30``): richer
    #: households hold proportionally more wealth relative to income.
    _WEALTH_RATIO_PERCENTILES: tuple[float, ...] = (0, 20, 40, 60, 80, 100)
    _WEALTH_RATIO_VALUES: tuple[float, ...] = (0.0, 1.06, 4.14, 4.19, 5.24, 6.0)

    def _synthesize_income_wealth(
        self,
        income_percentile: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Port of the native DYNAMO-M income/wealth sampling pipeline.

        Mirrors ``base_nodes.py:59-76`` and ``coastal_nodes.py:597-632``:

        1. a regional **lognormal income distribution** is fitted from the
           configured ``median_income`` and ``mean_median_inc_ratio``
           (``mu = ln(median)``, ``sd = sqrt(2 ln(mean/median))``,
           5,000 samples, ``int32`` cast, sorted) — natively the mean comes
           from the GDL sub-national income raster / World-Bank table and
           the ratio from the UN WIID;
        2. each household reads its income off that distribution at its
           **income percentile** (supplied per agent, or drawn uniformly in
           ``[1, 99]`` — the native fallback);
        3. ``wealth = ratio(percentile) * income`` with the interpolated
           native wealth-to-income table (see ``_WEALTH_RATIO_*``).

        Randomness uses a **dedicated generator** derived from the config
        seed so the one-off income synthesis never perturbs the event-draw
        or decision-rule RNG streams (bit-parity discipline).

        Parameters
        ----------
        income_percentile : np.ndarray or None
            Per-agent income percentiles in ``[1, 99]`` (e.g. derived from
            census data for the case study); ``None`` draws the uniform
            native fallback.

        Returns
        -------
        income, wealth : np.ndarray[float32]
            Per-agent income and wealth, shape ``(n_agents,)``.
        percentile : np.ndarray[int64]
            The percentile actually used per agent (stored for diagnostics
            and reused for the wealth ratio, keeping the two consistent).
        """
        # Dedicated, seed-derived stream (never shared with self._rng).
        rng = np.random.default_rng([self.config.random_seed, 0x1C0])

        median = float(self._dec.median_income)
        mean = median * float(self._dec.mean_median_inc_ratio)
        mu = np.log(median)
        sd = np.sqrt(2.0 * np.log(mean / median))
        # Native: 5,000 samples, int32 cast, sorted ascending.
        distribution = np.sort(rng.lognormal(mu, sd, 5_000).astype(np.int32))

        if income_percentile is not None:
            percentile = np.asarray(income_percentile)
            if percentile.shape != (self.n_agents,):
                raise ValueError(
                    f"income_percentile_per_agent must have shape "
                    f"({self.n_agents},), got {percentile.shape}."
                )
            percentile = np.clip(percentile.astype(np.int64), 1, 99)
        else:
            # Native fallback: uniform percentile per household
            # (coastal_nodes.py:603).
            percentile = rng.integers(1, 100, self.n_agents)

        income = (
            np.percentile(distribution, percentile)
            .astype(np.int32)
            .astype(np.float32)
        )
        ratio = np.interp(
            percentile,
            self._WEALTH_RATIO_PERCENTILES,
            self._WEALTH_RATIO_VALUES,
        )
        wealth = (ratio * income).astype(np.float32)
        return income, wealth, percentile

    def _init_economic_variables(
        self,
        income_per_agent: np.ndarray | None,
        amenity_value_per_agent: np.ndarray | None,
        wealth_per_agent: np.ndarray | None = None,
        income_percentile_per_agent: np.ndarray | None = None,
    ) -> None:
        """
        Initialize income, wealth, and amenity arrays for all agents.

        Resolution order for income/wealth:

        1. explicit ``income_per_agent`` / ``wealth_per_agent`` arrays
           (e.g. census/ACS data joined on ``object_id``),
        2. otherwise the ``income_mode`` fallback:
           ``"synthetic_lognormal"`` (native DYNAMO-M port, income
           independent of building value) or ``"mpd_ratio"`` (legacy,
           ``income = max_pot_dmg / income_to_wealth_ratio`` — which makes
           wealth identically the building value and the affordability
           constraint inert; retained for reproducibility only).
        """
        self.income_percentile: np.ndarray | None = None

        # -- Income & wealth --------------------------------------------------
        if income_per_agent is not None:
            self.income = np.asarray(income_per_agent, dtype=np.float32)
            if self.income.shape != (self.n_agents,):
                raise ValueError(
                    f"income_per_agent must have shape ({self.n_agents},), "
                    f"got {self.income.shape}."
                )
            if wealth_per_agent is not None:
                self.wealth = np.asarray(wealth_per_agent, dtype=np.float32)
                if self.wealth.shape != (self.n_agents,):
                    raise ValueError(
                        f"wealth_per_agent must have shape ({self.n_agents},), "
                        f"got {self.wealth.shape}."
                    )
            else:
                self.wealth = (
                    self.income * self._dec.income_to_wealth_ratio
                ).astype(np.float32)
        elif self._dec.income_mode == "synthetic_lognormal":
            self.income, self.wealth, self.income_percentile = (
                self._synthesize_income_wealth(income_percentile_per_agent)
            )
        elif self._dec.income_mode == "mpd_ratio":
            # Legacy fallback: income derived from max_pot_dmg (wealth then
            # cancels back to exactly max_pot_dmg — see class docstring).
            self.income = np.where(
                self._dec.income_to_wealth_ratio > 0,
                self.max_pot_dmg / self._dec.income_to_wealth_ratio,
                0.0,
            ).astype(np.float32)
            self.wealth = (
                self.income * self._dec.income_to_wealth_ratio
            ).astype(np.float32)
        else:
            raise ValueError(
                f"Unknown income_mode {self._dec.income_mode!r}; expected "
                "'synthetic_lognormal' or 'mpd_ratio'."
            )

        # -- Amenity value ---------------------------------------------------
        if amenity_value_per_agent is not None:
            self.amenity_value = np.asarray(
                amenity_value_per_agent, dtype=np.float32
            )
            if self.amenity_value.shape != (self.n_agents,):
                raise ValueError(
                    f"amenity_value_per_agent must have shape "
                    f"({self.n_agents},), got {self.amenity_value.shape}."
                )
        else:
            self.amenity_value = np.zeros(self.n_agents, dtype=np.float32)

    def _init_state_variables(self) -> None:
        """Initialize dynamic per-agent state arrays."""
        # -- Dynamic per-agent state -----------------------------------------
        self.risk_perception: np.ndarray = np.full(
            self.n_agents,
            self._dec.risk_perc_min,
            dtype=np.float32,
        )
        # Start at 99 years since last flood → minimum risk perception
        self.flood_timer: np.ndarray = np.full(
            self.n_agents, 99, dtype=np.int32
        )
        self.is_adapted: np.ndarray = np.zeros(self.n_agents, dtype=bool)

    def _build_residential_mask(self) -> np.ndarray:
        """
        Build a boolean mask selecting only residential buildings.

        Matching uses substring search (``np.char.find``) on the
        ``primary_object_type`` attribute so that mixed types such as
        ``"COM_RES"`` are also captured by the default ``residential_substring
        = "RES"``.

        Returns
        -------
        mask : np.ndarray[bool], shape (n_total_buildings,)
        """
        raw_types: list = self._ds[
            self._nc.dimension_object_id
        ].attrs[self._nc.attr_building_type]
        types_arr: np.ndarray = np.array(raw_types, dtype=str)
        mask: np.ndarray = (
            np.char.find(types_arr, self._nc.residential_substring) >= 0
        )
        return mask

    def _compute_annual_adapt_cost(self) -> np.ndarray:
        """
        Annualise the one-off adaptation cost using a standard loan formula:

            ``annual_cost = total_cost * [r*(1+r)^L / ((1+r)^L - 1)]``

        where ``r`` is the interest rate and ``L`` is the loan duration
        (identical annuity to native ``coastal_nodes.py:666-672``).

        ``total_cost`` follows ``DecisionConfig.adaptation_total_cost``:

        * a **fixed per-household amount** when set — the native DYNAMO-M
          model (country-scaled constant, ``prepare_scale_to_GDP.py``);
          cost independent of property value lets the affordability
          constraint discriminate between households;
        * legacy fallback (``None``):
          ``adaptation_cost_fraction * max_pot_dmg`` — proportional to
          building value, which (with ``income_mode="mpd_ratio"``) makes
          the affordability gate cancel algebraically.

        Returns
        -------
        annual_cost : np.ndarray[float32], shape (n_agents,)
        """
        if self._dec.adaptation_total_cost is not None:
            total_cost: np.ndarray = np.full(
                self.n_agents,
                float(self._dec.adaptation_total_cost),
                dtype=np.float64,
            )
        else:
            total_cost = self._dec.adaptation_cost_fraction * self.max_pot_dmg
        r: float = self._dec.interest_rate
        lp: int = self._dec.loan_duration
        if r == 0.0:
            annual_cost = total_cost / lp
        else:
            annual_cost = total_cost * (
                r * (1 + r) ** lp / ((1 + r) ** lp - 1)
            )
        return annual_cost.astype(np.float32)

    def _interpolate_strategy(
        self,
        slr_arr: np.ndarray,
        slr_target: float,
        strategy: str,
        n_events: int,
        method: str,
    ) -> np.ndarray:
        """
        Interpolate the damage lookup table along the SLR axis for a single
        strategy, returning an array shaped ``(n_agents, n_events)``.

        Parameters
        ----------
        slr_arr : np.ndarray
            SLR coordinate values from the dataset.
        slr_target : float
            Target SLR value to interpolate to.
        strategy : str
            Strategy name (must exist in dataset).
        n_events : int
            Number of events.
        interp_method : str
            Interpolation method (``'linear'``, ``'nearest'``, ``'cubic'``).
            Note: ``'cubic'`` requires at least 4 SLR points in the lookup table.

        Returns
        -------
        dmg : np.ndarray[float32], shape (n_agents, n_events)
        """
        # Property capping: bound damages to [0, max_pot_dmg] and no negatives
        # (cubic spline undershoot is handled inside lookup_utils).  The
        # residential strategy cube is materialized once and reused for every
        # SLR target, so only the cheap per-SLR interpolation runs per call.
        cube = self._strategy_cubes.get(strategy)
        if cube is None:
            cube = materialize_strategy_cube(
                ds=self._ds,
                strategy=strategy,
                res_mask=self._res_mask,
                dim_object_id=self._nc.dimension_object_id,
                dim_slr=self._nc.dimension_slr,
                dim_event=self._nc.dimension_event,
                dim_strategy=self._nc.dimension_strategy,
                var_total_damage=self._nc.var_total_damage,
            )
            if self._event_keep is not None:
                # Apply the nuisance-event filter to the cube's event axis so
                # the damage matrices stay aligned with the filtered catalogue.
                values, slr_arr_cube = cube
                cube = (
                    np.ascontiguousarray(values[:, :, self._event_keep]),
                    slr_arr_cube,
                )
            self._strategy_cubes[strategy] = cube
        values, slr_arr_cube = cube
        return interpolate_cube_at_slr(
            values, slr_arr_cube, slr_target,
            method=method, max_pot_dmg=self.max_pot_dmg,
        )


# ---------------------------------------------------------------------------
# Standalone SEU functions (ported from DYNAMO-M decision_module.py)
# ---------------------------------------------------------------------------

def _iterate_through_flood(
    n_floods: int,
    wealth: np.ndarray,
    income: np.ndarray,
    amenity_value: np.ndarray,
    max_T: int,
    expected_damages: np.ndarray,
    n_agents: int,
    r: float,
) -> np.ndarray:
    """
    Compute the time-discounted Net Present Value (NPV) for each flood
    scenario and agent.

    This is a pure-NumPy port of ``DecisionModule.IterateThroughFlood``
    from DYNAMO-M (Tierolf et al., 2023).

    Parameters
    ----------
    n_floods : int
        Number of flood scenarios (events).
    wealth : np.ndarray, shape (n_agents,)
        Household wealth per agent.
    income : np.ndarray, shape (n_agents,)
        Annual household income per agent.
    amenity_value : np.ndarray, shape (n_agents,)
        Amenity (location) value per agent.
    max_T : int
        Decision horizon (years).
    expected_damages : np.ndarray, shape (n_floods, n_agents)
        Expected damage per flood scenario per agent.
    n_agents : int
        Number of agents.
    r : float
        Time-discounting rate.

    Returns
    -------
    NPV_summed : np.ndarray, shape (n_floods + 3, n_agents), dtype float32
        Time-discounted NPV for each flood scenario and agent.
        Index 0 is a copy of index 1 (lower integration bound).
        Indices 1 … n_floods are flood scenarios.
        Indices n_floods+1, n_floods+2 are no-flood scenarios.
    """
    NPV_summed = np.full((n_floods + 3, n_agents), -1.0, dtype=np.float32)

    base_NPV = (wealth + income + amenity_value).astype(np.float32)

    # Discounting factors for t = 1 … max_T-1
    t_arr = np.arange(1, max_T, dtype=np.float32)
    if r == 0.0:
        discount_sum = float(len(t_arr))
    else:
        discount_sum = float(np.sum(1.0 / (1.0 + r) ** t_arr))

    for i in range(n_floods + 2):
        if i < n_floods:
            NPV_flood_i = (wealth + income + amenity_value
                           - expected_damages[i]).astype(np.float32)
        else:
            # No-flood scenarios (last two iterations)
            NPV_flood_i = (wealth + income + amenity_value).astype(np.float32)

        # Discounted NPV across time horizon + undiscounted t=0
        NPV_tx = discount_sum * NPV_flood_i + base_NPV
        NPV_summed[i + 1] = NPV_tx

    # Lower integration bound = first flood scenario
    NPV_summed[0] = NPV_summed[1]
    return NPV_summed


def _integrate_expected_utility(
    NPV_summed: np.ndarray,
    p_all: np.ndarray,
    sigma: float,
    error_terms: np.ndarray,
) -> np.ndarray:
    """
    Apply CRRA utility to time-discounted NPVs and integrate over perceived 
    probabilities using the trapezoidal rule.
    
    Parameters
    ----------
    NPV_summed : np.ndarray, shape (n_floods + 3, n_agents)
    p_all : np.ndarray, shape (n_floods + 3, n_agents)
    sigma : float
    error_terms : np.ndarray, shape (n_agents,)
    
    Returns
    -------
    eu_array : np.ndarray, shape (n_agents,)
    """
    # Clip negative NPVs to 1 (log/power-utility undefined for non-positive)
    NPV_summed = np.maximum(NPV_summed, 1.0)

    # --- Utility function ------------------------------------------------
    if sigma == 1.0:
        EU_store = np.log(NPV_summed)
    else:
        EU_store = (NPV_summed ** (1.0 - sigma)) / (1.0 - sigma)

    # Integrate EU over perceived probability (trapezoidal rule)
    # np.trapezoid is the preferred name in numpy >= 2.0; fall back to
    # the deprecated np.trapz alias for older environments. Use hasattr so
    # np.trapz is never *accessed* on numpy >= 2.0 (where it was removed and
    # accessing it raises AttributeError — a plain getattr default would still
    # evaluate np.trapz eagerly and crash).
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    eu_array: np.ndarray = _trapz(y=EU_store, x=p_all, axis=0).astype(
        np.float32
    )
    eu_array *= error_terms
    return eu_array


def _calc_eu_do_nothing(
    n_agents: int,
    wealth: np.ndarray,
    income: np.ndarray,
    amenity_value: np.ndarray,
    amenity_weight: float,
    risk_perception: np.ndarray,
    expected_damages: np.ndarray,
    adapted: np.ndarray,
    p_floods: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: float,
    error_terms: np.ndarray,
) -> np.ndarray:
    """
    Calculate the Subjective Expected Utility of doing nothing.

    Ported from ``DecisionModule.calcEU_do_nothing`` in DYNAMO-M
    (Tierolf et al., 2023).

    Parameters
    ----------
    n_agents : int
        Number of agents.
    wealth : np.ndarray, shape (n_agents,)
        Household wealth.
    income : np.ndarray, shape (n_agents,)
        Annual household income.
    amenity_value : np.ndarray, shape (n_agents,)
        Location amenity value (weighted by ``amenity_weight``).
    amenity_weight : float
        Scalar multiplier for amenity.
    risk_perception : np.ndarray, shape (n_agents,)
        Perceived-risk multiplier per agent.
    expected_damages : np.ndarray, shape (n_floods, n_agents)
        Expected damages per flood event per agent under no-measures strategy.
    adapted : np.ndarray[int32], shape (n_agents,)
        1 if agent is already adapted; forces EU to ``-inf``.
    p_floods : np.ndarray, shape (n_floods,)
        Exceedance probabilities (annual frequencies) of each event.
    T : np.ndarray[int32], shape (n_agents,)
        Decision horizon per agent.
    r : float
        Time-discounting rate.
    sigma : float
        CRRA risk-aversion coefficient.
    error_terms : np.ndarray, shape (n_agents,)
        Multiplicative stochastic error term.

    Returns
    -------
    eu_array : np.ndarray, shape (n_agents,), dtype float32
        Subjective expected utility of doing nothing per agent.
    """
    # Weigh amenities
    amenity_value = (amenity_value * amenity_weight).astype(np.float32)

    # Sort events by ascending probability (required for trapz integration)
    sort_idx = np.argsort(p_floods)
    expected_damages = expected_damages[sort_idx, :]
    p_floods = p_floods[sort_idx]

    n_floods = len(p_floods)
    max_T = int(np.max(T))

    # --- Build perceived-risk probability array ----------------------------
    # Shape: (n_floods + 3, n_agents)
    p_all = np.full((n_floods + 3, n_agents), -1.0, dtype=np.float32)
    perc_risk = (
        p_floods[:, np.newaxis]
        * risk_perception[np.newaxis, :]
    ).astype(np.float32)
    p_all[1:-2, :] = perc_risk
    # Cap perceived probability at 0.998
    p_all = np.minimum(p_all, 0.998)
    # Extend domain to [0, 1] for trapezoidal integration
    p_all[-2, :] = p_all[-3, :] + 0.001
    p_all[-1, :] = 1.0
    p_all[0, :] = 0.0

    # --- NPV for each event scenario --------------------------------------
    NPV_summed = _iterate_through_flood(
        n_floods=n_floods,
        wealth=wealth,
        income=income,
        amenity_value=amenity_value,
        max_T=max_T,
        expected_damages=expected_damages,
        n_agents=n_agents,
        r=r,
    )

    # --- Integrate Expected Utility ---------------------------------------
    eu_array = _integrate_expected_utility(
        NPV_summed=NPV_summed,
        p_all=p_all,
        sigma=sigma,
        error_terms=error_terms,
    )

    # Agents who are already adapted cannot choose do-nothing
    eu_array[adapted == 1] = -np.inf

    return eu_array


def _calc_eu_adapt(
    n_agents: int,
    wealth: np.ndarray,
    income: np.ndarray,
    expenditure_cap: float,
    amenity_value: np.ndarray,
    amenity_weight: float,
    risk_perception: np.ndarray,
    expected_damages_adapt: np.ndarray,
    adaptation_costs: np.ndarray,
    time_adapted: np.ndarray,
    loan_duration: int,
    p_floods: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: float,
    error_terms: np.ndarray,
) -> np.ndarray:
    """
    Calculate the Subjective Expected Utility of adapting (floodproofing).

    Ported from ``DecisionModule.calcEU_adapt`` in DYNAMO-M
    (Tierolf et al., 2023).

    Parameters
    ----------
    n_agents : int
        Number of agents.
    wealth : np.ndarray, shape (n_agents,)
        Household wealth.
    income : np.ndarray, shape (n_agents,)
        Annual household income.
    expenditure_cap : float
        Maximum fraction of income an agent can spend on adaptation per year.
    amenity_value : np.ndarray, shape (n_agents,)
        Location amenity value.
    amenity_weight : float
        Scalar multiplier for amenity.
    risk_perception : np.ndarray, shape (n_agents,)
        Perceived-risk multiplier per agent.
    expected_damages_adapt : np.ndarray, shape (n_floods, n_agents)
        Expected damages per flood event per agent under adapted strategy.
    adaptation_costs : np.ndarray, shape (n_agents,)
        Annual adaptation (loan) payment per agent.
    time_adapted : np.ndarray[int32], shape (n_agents,)
        Years since adaptation decision (0 = deciding this year).
    loan_duration : int
        Total loan repayment period in years.
    p_floods : np.ndarray, shape (n_floods,)
        Exceedance probabilities of each event.
    T : np.ndarray[int32], shape (n_agents,)
        Decision horizon per agent.
    r : float
        Time-discounting rate.
    sigma : float
        CRRA risk-aversion coefficient.
    error_terms : np.ndarray, shape (n_agents,)
        Multiplicative stochastic error term.

    Returns
    -------
    eu_array : np.ndarray, shape (n_agents,), dtype float32
        Subjective expected utility of adapting per agent.
        Set to ``-inf`` for agents who cannot afford the adaptation cost.
    """
    amenity_value = (amenity_value * amenity_weight).astype(np.float32)

    sort_idx = np.argsort(p_floods)
    expected_damages_adapt = expected_damages_adapt[sort_idx, :]
    p_floods = p_floods[sort_idx]

    n_floods = len(p_floods)
    max_T = int(np.max(T))

    # --- Perceived risk probabilities -------------------------------------
    p_all = np.full((n_floods + 3, n_agents), -1.0, dtype=np.float32)
    perc_risk = (
        p_floods[:, np.newaxis]
        * risk_perception[np.newaxis, :]
    ).astype(np.float32)
    p_all[1:-2, :] = perc_risk
    p_all = np.minimum(p_all, 0.998)
    p_all[-2, :] = p_all[-3, :] + 0.001
    p_all[-1, :] = 1.0
    p_all[0, :] = 0.0

    # --- NPV under adapted strategy --------------------------------------
    NPV_summed = _iterate_through_flood(
        n_floods=n_floods,
        wealth=wealth,
        income=income,
        amenity_value=amenity_value,
        max_T=max_T,
        expected_damages=expected_damages_adapt,
        n_agents=n_agents,
        r=r,
    )

    # --- Subtract time-discounted adaptation costs -----------------------
    # Build cost_array[agent, years_of_loan_left] = PV of remaining payments
    if r == 0.0:
        discounts = np.ones(loan_duration, dtype=np.float32)
    else:
        discounts = (
            1.0 / (1.0 + r) ** np.arange(loan_duration)
        ).astype(np.float32)

    years = np.arange(loan_duration + 1, dtype=np.int32)
    # cost_array shape: (n_agents, loan_duration + 1)
    cost_array = np.zeros((n_agents, len(years)), dtype=np.float32)
    for i, yr in enumerate(years):
        cost_array[:, i] = np.sum(discounts[:yr]) * adaptation_costs

    loan_left = (loan_duration - time_adapted).astype(np.int32)
    loan_left = np.maximum(loan_left, 0)
    loan_left = np.minimum(loan_left, T)

    # Gather time-discounted adaptation cost per agent
    agent_indices = np.arange(n_agents)
    time_discounted_adapt_cost = cost_array[agent_indices, loan_left]

    NPV_summed -= time_discounted_adapt_cost  # broadcast over flood dimension

    # --- Integrate Expected Utility ---------------------------------------
    eu_array = _integrate_expected_utility(
        NPV_summed=NPV_summed,
        p_all=p_all,
        sigma=sigma,
        error_terms=error_terms,
    )

    # Expenditure cap: agents who cannot afford adaptation → EU = -inf
    constrained = income * expenditure_cap <= adaptation_costs
    eu_array[constrained] = -np.inf

    return eu_array


def _calc_eu_insure(
    n_agents: int,
    wealth: np.ndarray,
    income: np.ndarray,
    expenditure_cap: float,
    amenity_value: np.ndarray,
    amenity_weight: float,
    risk_perception: np.ndarray,
    expected_damages: np.ndarray,
    premium: np.ndarray,
    p_floods: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: float,
    error_terms: np.ndarray,
    deductible: float = 0.1,
) -> np.ndarray:
    """
    Calculate the Subjective Expected Utility of taking flood insurance.

    Ported from ``DecisionModule.calcEU_insure`` in DYNAMO-M
    (``decision_module.py`` lines 243-367; Tierolf et al., 2023):

    * insured households still face ``deductible * expected_damages``
      (native hard-codes ``deductable = 0.1``),
    * the (flat, community-rated) premium is paid every year of the
      decision horizon: the time-discounted premium stream
      ``sum_{t<T} premium.mean() / (1+r)^t`` is subtracted from the NPVs,
    * the same 0.998 perceived-probability cap, ``max(1, NPV)`` floor,
      CRRA utility, and trapezoidal integration as the other two options,
    * households whose annual premium exceeds ``expenditure_cap * income``
      cannot afford insurance (``EU = -inf``).

    Parameters
    ----------
    n_agents : int
        Number of agents.
    wealth, income : np.ndarray, shape (n_agents,)
        Household wealth and annual income.
    expenditure_cap : float
        Maximum fraction of income spendable on the premium per year.
    amenity_value : np.ndarray, shape (n_agents,)
        Location amenity value (weighted by ``amenity_weight``).
    amenity_weight : float
        Scalar multiplier for amenity.
    risk_perception : np.ndarray, shape (n_agents,)
        Perceived-risk multiplier per agent.
    expected_damages : np.ndarray, shape (n_floods, n_agents)
        Expected damages per flood event per agent under the NO-MEASURES
        strategy (insurance does not change the hazard, only who pays).
    premium : np.ndarray, shape (n_agents,)
        Annual insurance premium per agent.  Flat under the ported
        community-rated insurer (``mean EAD``, ``insurer_agent.py:20-26``);
        heterogeneous under risk-based pricing (each agent's own EAD).
    p_floods : np.ndarray, shape (n_floods,)
        Exceedance probabilities of each event.
    T : np.ndarray[int32], shape (n_agents,)
        Decision horizon per agent.
    r : float
        Time-discounting rate.
    sigma : float
        CRRA risk-aversion coefficient.
    error_terms : np.ndarray, shape (n_agents,)
        Multiplicative stochastic error term (all-ones when
        ``error_interval == 0``; must always be supplied).
    deductible : float
        Fraction of damage still borne by the insured household.
        Default ``0.1`` (native hard-coded value).

    Returns
    -------
    eu_array : np.ndarray, shape (n_agents,), dtype float32
        Subjective expected utility of insuring per agent
        (``-inf`` where unaffordable).
    """
    # Insured households still bear the deductible share of every damage.
    expected_damages_insured = (expected_damages * deductible).astype(np.float32)

    amenity_value = (amenity_value * amenity_weight).astype(np.float32)

    sort_idx = np.argsort(p_floods)
    expected_damages_insured = expected_damages_insured[sort_idx, :]
    p_floods = p_floods[sort_idx]

    n_floods = len(p_floods)
    max_T = int(np.max(T))

    # --- Perceived risk probabilities (identical construction) ------------
    p_all = np.full((n_floods + 3, n_agents), -1.0, dtype=np.float32)
    perc_risk = (
        p_floods[:, np.newaxis]
        * risk_perception[np.newaxis, :]
    ).astype(np.float32)
    p_all[1:-2, :] = perc_risk
    p_all = np.minimum(p_all, 0.998)
    p_all[-2, :] = p_all[-3, :] + 0.001
    p_all[-1, :] = 1.0
    p_all[0, :] = 0.0

    # --- NPV under the insured damage profile -----------------------------
    NPV_summed = _iterate_through_flood(
        n_floods=n_floods,
        wealth=wealth,
        income=income,
        amenity_value=amenity_value,
        max_T=max_T,
        expected_damages=expected_damages_insured,
        n_agents=n_agents,
        r=r,
    )

    # --- Subtract the time-discounted premium stream ----------------------
    # Native (decision_module.py:331-343): discounts over the decision
    # horizon (NOT the loan duration), a scalar cost per horizon length
    # using premium.mean(), gathered per agent via np.take(cost_array, T).
    if r == 0.0:
        discounts = np.ones(max_T, dtype=np.float32)
    else:
        discounts = (1.0 / (1.0 + r) ** np.arange(max_T)).astype(np.float32)

    # Native (``decision_module.py:331-343``) discounts the single scalar
    # ``premium.mean()`` — exact for its flat community-rated premium, but
    # wrong under risk-based pricing.  We discount each agent's OWN premium,
    # which is identical whenever the premium array is flat (native's only
    # case) and coherent when it is not.
    years = np.arange(max_T + 1, dtype=np.int32)
    cost_array = np.zeros((n_agents, len(years)), dtype=np.float32)
    for i, yr in enumerate(years):
        cost_array[:, i] = np.sum(discounts[:yr]) * premium

    time_discounted_premium = cost_array[np.arange(n_agents), T]
    NPV_summed -= time_discounted_premium  # broadcast over flood dimension

    # --- Integrate Expected Utility ---------------------------------------
    eu_array = _integrate_expected_utility(
        NPV_summed=NPV_summed,
        p_all=p_all,
        sigma=sigma,
        error_terms=error_terms,
    )

    # Affordability: premium above the expenditure cap → EU = -inf
    constrained = income * expenditure_cap <= premium
    eu_array[constrained] = -np.inf

    return eu_array
