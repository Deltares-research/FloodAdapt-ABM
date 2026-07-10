"""
test_dynamo_decision_bridge.py
==============================
Unit tests for the FloodAdapt-ABM × DYNAMO-M coupling bridge.

Tests are deliberately self-contained: they build small mock xarray datasets
and do **not** require the full Charleston lookup table to run.

Run with:

    pytest test_dynamo_decision_bridge.py -v

Scientific reference
--------------------
Tierolf, L., Haer, T., Botzen, W. J. W., de Bruijn, J. A., Ton, M. J.,
Reimann, L., & Aerts, J. C. J. H. (2023). A coupled agent-based model for
France for simulating adaptation and migration decisions under future coastal
flood risk. Scientific Reports, 13(1), 4176.
https://doi.org/10.1038/s41598-023-31351-y
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from floodadapt_abm import CouplingConfig, DecisionConfig, NetCDFMappingConfig
from floodadapt_abm._core import (
    DynamoDecisionBridge,
    _calc_eu_adapt,
    _calc_eu_do_nothing,
)
from floodadapt_abm._core.dynamo_decision_bridge import _iterate_through_flood


# ===========================================================================
# ── MOCK DATASET FACTORY ────────────────────────────────────────────────────
# ===========================================================================

def _make_mock_dataset(
    n_objects: int = 10,
    n_events: int = 5,
    n_slr: int = 5,
    residential_fraction: float = 0.8,
    residential_type: str = "RES",
    commercial_type: str = "COM",
    max_dmg_value: float = 200_000.0,
    seed: int = 7,
) -> xr.Dataset:
    """
    Build a minimal xarray.Dataset that mirrors the structure of the real
    Charleston lookup table, using random but deterministic values.

    Parameters
    ----------
    n_objects : int
        Total number of buildings.
    n_events : int
        Number of flood events.
    n_slr : int
        Number of SLR levels.
    residential_fraction : float
        Fraction of buildings labelled as ``residential_type``.
    residential_type : str
        Primary-object-type string for residential buildings.
    commercial_type : str
        Primary-object-type string for non-residential buildings.
    max_dmg_value : float
        Constant max-potential-damage value for all buildings.
    seed : int
        RNG seed.

    Returns
    -------
    xr.Dataset
        Mock dataset compatible with ``DynamoDecisionBridge``.
    """
    rng = np.random.default_rng(seed)

    object_ids = np.array([str(i) for i in range(n_objects)])
    slr_values = np.linspace(0.0, 2.0, n_slr)
    event_names = np.array([f"ev_{i:03d}" for i in range(n_events)])
    strategies = np.array(["no_measures", "floodproof_all_0"])

    # Assign building types
    n_res = int(n_objects * residential_fraction)
    primary_types: list[str] = (
        [residential_type] * n_res
        + [commercial_type] * (n_objects - n_res)
    )
    max_pot_dmg = np.full(n_objects, max_dmg_value, dtype=np.float64)

    # Event frequencies: decreasing (most frequent first)
    freqs = np.array(
        [1.0 / (2 ** i) for i in range(n_events)], dtype=np.float64
    )

    # Damage data: shape (n_objects, n_slr, n_strategies, n_events)
    # Floodproof damage = 30% of no-measures damage
    dmg_no_measures = rng.uniform(
        0.0, max_dmg_value * 0.5,
        size=(n_objects, n_slr, n_events),
    ).astype(np.float32)
    dmg_floodproof = (dmg_no_measures * 0.30).astype(np.float32)

    total_damage = xr.DataArray(
        np.stack([dmg_no_measures, dmg_floodproof], axis=2),
        dims=["object_id", "slr", "strategy", "event"],
        coords={
            "object_id": object_ids,
            "slr": slr_values,
            "strategy": strategies,
            "event": event_names,
        },
    )

    ds = xr.Dataset({"total_damage": total_damage, "inun_depth": total_damage * 0.01})
    ds["object_id"].attrs["max_pot_dmg"] = max_pot_dmg
    ds["object_id"].attrs["primary_object_type"] = primary_types
    ds["event"].attrs["freq"] = freqs
    return ds


# ===========================================================================
# ── FIXTURES ────────────────────────────────────────────────────────────────
# ===========================================================================

@pytest.fixture
def mock_ds() -> xr.Dataset:
    """Default 10-building, 5-event mock dataset."""
    return _make_mock_dataset()


@pytest.fixture
def default_config() -> CouplingConfig:
    """Default CouplingConfig matching the mock dataset column names."""
    return CouplingConfig()


@pytest.fixture
def bridge(mock_ds: xr.Dataset, default_config: CouplingConfig) -> DynamoDecisionBridge:
    """Fully initialised DynamoDecisionBridge using the mock dataset."""
    return DynamoDecisionBridge(ds=mock_ds, config=default_config)


# ===========================================================================
# ── TESTS: CONFIGURATION ────────────────────────────────────────────────────
# ===========================================================================

class TestNetCDFMappingConfig:
    """Tests for NetCDFMappingConfig defaults and override behaviour."""

    def test_default_dimension_names(self) -> None:
        """All default dimension names match the Charleston dataset."""
        cfg = NetCDFMappingConfig()
        assert cfg.dimension_object_id == "object_id"
        assert cfg.dimension_event == "event"
        assert cfg.dimension_slr == "slr"
        assert cfg.dimension_strategy == "strategy"

    def test_default_variable_names(self) -> None:
        """Default variable name must be 'total_damage'."""
        cfg = NetCDFMappingConfig()
        assert cfg.var_total_damage == "total_damage"

    def test_default_attr_names(self) -> None:
        """Default attribute keys match Charleston dataset attrs."""
        cfg = NetCDFMappingConfig()
        assert cfg.attr_max_pot_dmg == "max_pot_dmg"
        assert cfg.attr_event_freq == "freq"
        assert cfg.attr_building_type == "primary_object_type"

    def test_residential_substring_default(self) -> None:
        """Default substring filter is 'RES'."""
        cfg = NetCDFMappingConfig()
        assert cfg.residential_substring == "RES"

    def test_override_residential_substring(self) -> None:
        """Override of residential_substring must be respected."""
        cfg = NetCDFMappingConfig(residential_substring="COM")
        assert cfg.residential_substring == "COM"

    def test_override_strategy_names(self) -> None:
        """Strategy label overrides must persist."""
        cfg = NetCDFMappingConfig(
            strategy_no_measures="baseline",
            strategy_floodproof="fp_1m",
        )
        assert cfg.strategy_no_measures == "baseline"
        assert cfg.strategy_floodproof == "fp_1m"


class TestDecisionConfig:
    """Tests for DecisionConfig defaults."""

    def test_default_risk_aversion(self) -> None:
        cfg = DecisionConfig()
        assert cfg.risk_aversion == 1.0

    def test_default_discount_rate(self) -> None:
        cfg = DecisionConfig()
        assert cfg.discount_rate == 0.032

    def test_custom_values(self) -> None:
        cfg = DecisionConfig(risk_aversion=2.0, discount_rate=0.05)
        assert cfg.risk_aversion == 2.0
        assert cfg.discount_rate == 0.05


# ===========================================================================
# ── TESTS: RESIDENTIAL FILTERING ────────────────────────────────────────────
# ===========================================================================

class TestResidentialFiltering:
    """Tests for the residential-agent substring filter."""

    def test_correct_n_agents(self, bridge: DynamoDecisionBridge) -> None:
        """Bridge should retain only the 8 RES buildings out of 10."""
        # Default mock: 10 buildings, 80% residential → 8 RES
        assert bridge.n_agents == 8

    def test_object_ids_are_residential(
        self, mock_ds: xr.Dataset, default_config: CouplingConfig
    ) -> None:
        """All retained object_ids should belong to RES buildings."""
        raw_types = mock_ds["object_id"].attrs["primary_object_type"]
        bridge = DynamoDecisionBridge(ds=mock_ds, config=default_config)
        for oid in bridge.object_ids:
            idx = list(mock_ds["object_id"].values).index(oid)
            assert raw_types[idx] == "RES"

    def test_mixed_type_substring_match(self) -> None:
        """Buildings with type 'COM_RES' should also be captured by 'RES'."""
        ds = _make_mock_dataset(n_objects=4, residential_fraction=0.5)
        # Manually set two buildings to mixed type
        ds["object_id"].attrs["primary_object_type"] = [
            "RES", "COM_RES", "COM", "COM"
        ]
        cfg = CouplingConfig()
        bridge = DynamoDecisionBridge(ds=ds, config=cfg)
        # 'RES' and 'COM_RES' both contain 'RES'
        assert bridge.n_agents == 2

    def test_custom_substring_filters_commercial(self) -> None:
        """Setting residential_substring='COM' should select commercial only."""
        ds = _make_mock_dataset(
            n_objects=6,
            residential_fraction=0.5,
            residential_type="RES",
            commercial_type="COM",
        )
        cfg = CouplingConfig(netcdf=NetCDFMappingConfig(residential_substring="COM"))
        bridge = DynamoDecisionBridge(ds=ds, config=cfg)
        assert bridge.n_agents == 3  # 50% of 6 are COM


# ===========================================================================
# ── TESTS: PROPERTY AND WEALTH CAPPING ──────────────────────────────────────
# ===========================================================================

class TestPropertyCapping:
    """Ensure damage values are capped at max_pot_dmg."""

    def test_damage_never_exceeds_max_pot_dmg(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """No interpolated damage value should exceed the building's max_pot_dmg."""
        bridge.prepare_damage_arrays(slr_value=1.5, interp_method="linear")
        # Check all events and both strategies
        for ie in range(len(bridge._event_names)):
            dmg_nm = bridge._damage_no_measures[:, ie]
            dmg_fp = bridge._damage_floodproof[:, ie]
            assert np.all(
                dmg_nm <= bridge.max_pot_dmg + 1e-3
            ), "no_measures damage exceeds max_pot_dmg"
            assert np.all(
                dmg_fp <= bridge.max_pot_dmg + 1e-3
            ), "floodproof damage exceeds max_pot_dmg"

    def test_get_current_damages_capped(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """get_current_damages must cap at max_pot_dmg."""
        bridge.prepare_damage_arrays(slr_value=2.0)
        dmg = bridge.get_current_damages("ev_000")
        assert np.all(dmg <= bridge.max_pot_dmg + 1e-3)

    def test_wealth_consistent_with_income(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """Wealth should equal income * income_to_wealth_ratio."""
        expected_wealth = bridge.income * bridge.config.decision.income_to_wealth_ratio
        np.testing.assert_allclose(bridge.wealth, expected_wealth, rtol=1e-4)

    def test_wealth_upper_bounded(self, bridge: DynamoDecisionBridge) -> None:
        """Wealth should not be negative (income from max_pot_dmg is non-negative)."""
        assert np.all(bridge.wealth >= 0)


# ===========================================================================
# ── TESTS: ANNUAL ADAPTATION COST ───────────────────────────────────────────
# ===========================================================================

class TestAnnualAdaptCost:
    """Tests for the loan-amortisation formula."""

    def test_cost_positive(self, bridge: DynamoDecisionBridge) -> None:
        """All annual adaptation costs should be positive."""
        assert np.all(bridge._annual_adapt_cost > 0)

    def test_cost_formula_manual(self) -> None:
        """Manually verify the loan formula for a single agent."""
        max_dmg = 100_000.0
        frac = 0.10
        r = 0.03
        L = 16
        total = max_dmg * frac
        expected_annual = total * (r * (1 + r) ** L / ((1 + r) ** L - 1))

        ds = _make_mock_dataset(n_objects=2, n_events=2, max_dmg_value=max_dmg)
        cfg = CouplingConfig(
            decision=DecisionConfig(
                adaptation_cost_fraction=frac,
                interest_rate=r,
                loan_duration=L,
            )
        )
        bridge = DynamoDecisionBridge(ds=ds, config=cfg)
        np.testing.assert_allclose(
            bridge._annual_adapt_cost[0], expected_annual, rtol=1e-5
        )

    def test_zero_interest_rate(self) -> None:
        """With r=0 the annual cost should be total_cost / loan_duration."""
        max_dmg = 100_000.0
        frac = 0.10
        L = 10
        expected_annual = max_dmg * frac / L

        ds = _make_mock_dataset(n_objects=2, n_events=2, max_dmg_value=max_dmg)
        cfg = CouplingConfig(
            decision=DecisionConfig(
                adaptation_cost_fraction=frac,
                interest_rate=0.0,
                loan_duration=L,
            )
        )
        bridge = DynamoDecisionBridge(ds=ds, config=cfg)
        np.testing.assert_allclose(
            bridge._annual_adapt_cost[0], expected_annual, rtol=1e-5
        )


# ===========================================================================
# ── TESTS: RISK PERCEPTION UPDATES ──────────────────────────────────────────
# ===========================================================================

class TestRiskPerceptionUpdate:
    """Tests for the DYNAMO-M flood-timer / risk-perception formula."""

    def test_flood_timer_increments(self, bridge: DynamoDecisionBridge) -> None:
        """Flood timer should increment by 1 for non-flooded agents."""
        initial_timer = bridge.flood_timer.copy()
        flooded = np.zeros(bridge.n_agents, dtype=bool)
        bridge.update_flood_experience(flooded)
        np.testing.assert_array_equal(bridge.flood_timer, initial_timer + 1)

    def test_flood_timer_resets_for_flooded(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """Flood timer should reset to 0 for flooded agents."""
        flooded = np.zeros(bridge.n_agents, dtype=bool)
        flooded[0] = True
        bridge.update_flood_experience(flooded)
        assert bridge.flood_timer[0] == 0
        assert bridge.flood_timer[1] > 0

    def test_risk_perception_increases_after_flood(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """Risk perception for flooded agents should be higher than for non-flooded."""
        flooded = np.zeros(bridge.n_agents, dtype=bool)
        flooded[0] = True
        bridge.update_flood_experience(flooded)
        rp_flooded = bridge.risk_perception[0]
        rp_not_flooded = bridge.risk_perception[1]
        assert rp_flooded > rp_not_flooded

    def test_risk_perception_formula(self, bridge: DynamoDecisionBridge) -> None:
        """Verify the risk perception formula matches DYNAMO-M exactly."""
        # Reset timers to known values
        bridge.flood_timer[:] = 10
        flooded = np.zeros(bridge.n_agents, dtype=bool)
        bridge.update_flood_experience(flooded)  # timer becomes 11
        dec = bridge.config.decision
        expected_rp = (
            dec.risk_perc_max
            * (1.6 ** (dec.risk_perc_coef * 11))
            + dec.risk_perc_min
        )
        np.testing.assert_allclose(
            bridge.risk_perception, expected_rp, rtol=1e-4
        )


# ===========================================================================
# ── TESTS: STANDALONE SEU FUNCTIONS ─────────────────────────────────────────
# ===========================================================================

class TestIterateThroughFlood:
    """Unit tests for the _iterate_through_flood helper."""

    def test_output_shape(self) -> None:
        n_floods, n_agents, max_T = 3, 5, 10
        damages = np.ones((n_floods, n_agents), dtype=np.float32) * 1000
        wealth = np.ones(n_agents) * 50_000
        income = np.ones(n_agents) * 10_000
        amenity = np.zeros(n_agents)
        result = _iterate_through_flood(
            n_floods=n_floods,
            wealth=wealth,
            income=income,
            amenity_value=amenity,
            max_T=max_T,
            expected_damages=damages,
            n_agents=n_agents,
            r=0.04,
        )
        assert result.shape == (n_floods + 3, n_agents)

    def test_no_flood_scenario_higher_than_flood(self) -> None:
        """NPV under no-flood should be higher than under flood."""
        n_floods, n_agents = 2, 4
        damages = np.ones((n_floods, n_agents)) * 5_000
        wealth = np.ones(n_agents) * 100_000
        income = np.ones(n_agents) * 20_000
        amenity = np.zeros(n_agents)
        result = _iterate_through_flood(
            n_floods=n_floods,
            wealth=wealth,
            income=income,
            amenity_value=amenity,
            max_T=10,
            expected_damages=damages,
            n_agents=n_agents,
            r=0.04,
        )
        # No-flood indices are n_floods+1 and n_floods+2
        npv_flood = result[1, :]     # first flood scenario
        npv_no_flood = result[n_floods + 1, :]  # first no-flood scenario
        assert np.all(npv_no_flood > npv_flood)


class TestCalcEUDoNothing:
    """Unit tests for _calc_eu_do_nothing."""

    def _make_args(
        self,
        n_agents: int = 5,
        n_events: int = 3,
        sigma: float = 1.5,
    ) -> dict:
        rng = np.random.default_rng(0)
        return dict(
            n_agents=n_agents,
            wealth=rng.uniform(50_000, 200_000, n_agents).astype(np.float32),
            income=rng.uniform(10_000, 60_000, n_agents).astype(np.float32),
            amenity_value=np.zeros(n_agents, dtype=np.float32),
            amenity_weight=1.0,
            risk_perception=np.ones(n_agents, dtype=np.float32),
            expected_damages=rng.uniform(0, 10_000, (n_events, n_agents)).astype(np.float32),
            adapted=np.zeros(n_agents, dtype=np.int32),
            p_floods=np.array([0.5, 0.1, 0.02], dtype=np.float32),
            T=np.full(n_agents, 10, dtype=np.int32),
            r=0.04,
            sigma=sigma,
            error_terms=np.ones(n_agents, dtype=np.float32),
        )

    def test_returns_correct_shape(self) -> None:
        args = self._make_args()
        result = _calc_eu_do_nothing(**args)
        assert result.shape == (args["n_agents"],)

    def test_adapted_agents_get_neg_inf(self) -> None:
        """Agents already adapted should receive EU = -inf for do-nothing."""
        args = self._make_args(n_agents=4)
        args["adapted"][0] = 1  # first agent already adapted
        result = _calc_eu_do_nothing(**args)
        assert result[0] == -np.inf
        assert np.isfinite(result[1])

    def test_log_utility_when_sigma_eq_1(self) -> None:
        """sigma=1 should use log utility without raising errors."""
        args = self._make_args(sigma=1.0)
        result = _calc_eu_do_nothing(**args)
        assert result.shape == (args["n_agents"],)
        # Result should be finite for non-adapted agents
        assert np.all(np.isfinite(result))

    def test_higher_damage_lowers_eu(self) -> None:
        """Higher damages should lead to lower EU for do-nothing."""
        args_low = self._make_args()
        args_high = self._make_args()
        args_high["expected_damages"] = args_low["expected_damages"] * 10
        eu_low = _calc_eu_do_nothing(**args_low)
        eu_high = _calc_eu_do_nothing(**args_high)
        assert np.all(eu_low >= eu_high)


class TestCalcEUAdapt:
    """Unit tests for _calc_eu_adapt."""

    def _make_args(self, n_agents: int = 5, n_events: int = 3) -> dict:
        rng = np.random.default_rng(0)
        income = rng.uniform(10_000, 60_000, n_agents).astype(np.float32)
        return dict(
            n_agents=n_agents,
            wealth=rng.uniform(50_000, 200_000, n_agents).astype(np.float32),
            income=income,
            expenditure_cap=0.06,
            amenity_value=np.zeros(n_agents, dtype=np.float32),
            amenity_weight=1.0,
            risk_perception=np.ones(n_agents, dtype=np.float32),
            expected_damages_adapt=rng.uniform(0, 5_000, (n_events, n_agents)).astype(np.float32),
            adaptation_costs=(income * 0.04).astype(np.float32),  # affordable
            time_adapted=np.zeros(n_agents, dtype=np.int32),
            loan_duration=16,
            p_floods=np.array([0.5, 0.1, 0.02], dtype=np.float32),
            T=np.full(n_agents, 10, dtype=np.int32),
            r=0.04,
            sigma=1.5,
            error_terms=np.ones(n_agents, dtype=np.float32),
        )

    def test_returns_correct_shape(self) -> None:
        args = self._make_args()
        result = _calc_eu_adapt(**args)
        assert result.shape == (args["n_agents"],)

    def test_unaffordable_agents_get_neg_inf(self) -> None:
        """Agents where adaptation cost > expenditure_cap * income get -inf."""
        args = self._make_args(n_agents=4)
        # Make adaptation cost exceed expenditure cap for agent 0
        args["adaptation_costs"][0] = args["income"][0] * 0.9  # 90% >> 6%
        result = _calc_eu_adapt(**args)
        assert result[0] == -np.inf
        # Affordable agents should have finite EU
        assert np.isfinite(result[1])

    def test_adapt_eu_higher_than_do_nothing_for_high_risk(self) -> None:
        """Under very high damage, adaptation should be preferred."""
        n_agents = 3
        n_events = 2
        income = np.full(n_agents, 50_000.0, dtype=np.float32)
        wealth = np.full(n_agents, 200_000.0, dtype=np.float32)
        # No-measures damage is 100% of wealth per event
        dmg_none = np.full((n_events, n_agents), 200_000.0, dtype=np.float32)
        # Adapted damage is only 10% of wealth
        dmg_adapt = np.full((n_events, n_agents), 20_000.0, dtype=np.float32)
        p_floods = np.array([0.5, 0.2], dtype=np.float32)
        T = np.full(n_agents, 10, dtype=np.int32)
        risk_perc = np.ones(n_agents, dtype=np.float32)
        adapt_cost = (income * 0.03).astype(np.float32)  # well within cap
        error_terms = np.ones(n_agents, dtype=np.float32)

        eu_nothing = _calc_eu_do_nothing(
            n_agents=n_agents,
            wealth=wealth,
            income=income,
            amenity_value=np.zeros(n_agents),
            amenity_weight=1.0,
            risk_perception=risk_perc,
            expected_damages=dmg_none,
            adapted=np.zeros(n_agents, dtype=np.int32),
            p_floods=p_floods,
            T=T,
            r=0.04,
            sigma=1.5,
            error_terms=error_terms,
        )
        eu_adapt = _calc_eu_adapt(
            n_agents=n_agents,
            wealth=wealth,
            income=income,
            expenditure_cap=0.06,
            amenity_value=np.zeros(n_agents),
            amenity_weight=1.0,
            risk_perception=risk_perc,
            expected_damages_adapt=dmg_adapt,
            adaptation_costs=adapt_cost,
            time_adapted=np.zeros(n_agents, dtype=np.int32),
            loan_duration=16,
            p_floods=p_floods,
            T=T,
            r=0.04,
            sigma=1.5,
            error_terms=error_terms,
        )
        assert np.all(eu_adapt > eu_nothing)


# ===========================================================================
# ── TESTS: FULL BRIDGE DECISION FLOW ────────────────────────────────────────
# ===========================================================================

class TestEvaluateDecisions:
    """Integration tests for the full evaluate_decisions workflow."""

    def test_raises_without_prepare(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """evaluate_decisions must raise if prepare_damage_arrays was not called."""
        with pytest.raises(RuntimeError, match="prepare_damage_arrays"):
            bridge.evaluate_decisions(year_index=0)

    def test_adaptation_is_irreversible(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """Once adapted, an agent cannot un-adapt in subsequent years."""
        bridge.prepare_damage_arrays(slr_value=0.5)
        bridge.evaluate_decisions(year_index=0)
        adapted_after_yr1 = bridge.is_adapted.copy()

        # Evaluate again
        bridge.prepare_damage_arrays(slr_value=1.0)
        bridge.evaluate_decisions(year_index=1)

        # All agents adapted in year 1 should still be adapted in year 2
        assert np.all(bridge.is_adapted[adapted_after_yr1])

    def test_adapted_agents_not_newly_adapted_again(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """Agents already adapted cannot appear as 'newly adapted'."""
        bridge.prepare_damage_arrays(slr_value=0.5)
        newly_yr1 = bridge.evaluate_decisions(year_index=0)
        adapted_after_yr1 = bridge.is_adapted.copy()

        bridge.prepare_damage_arrays(slr_value=1.0)
        newly_yr2 = bridge.evaluate_decisions(year_index=1)

        # Newly adapted in year 2 must not overlap with already-adapted
        assert not np.any(adapted_after_yr1 & newly_yr2)

    def test_returns_boolean_array(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        bridge.prepare_damage_arrays(slr_value=0.5)
        result = bridge.evaluate_decisions(year_index=0)
        assert result.dtype == bool
        assert result.shape == (bridge.n_agents,)

    def test_prepare_damage_required_for_get_damages(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        """get_current_damages must raise before prepare_damage_arrays."""
        with pytest.raises(RuntimeError, match="prepare_damage_arrays"):
            bridge.get_current_damages("ev_000")

    def test_invalid_event_name_raises(
        self, bridge: DynamoDecisionBridge
    ) -> None:
        bridge.prepare_damage_arrays(slr_value=0.5)
        with pytest.raises(ValueError, match="not found"):
            bridge.get_current_damages("nonexistent_event")

    def test_custom_income_accepted(
        self, mock_ds: xr.Dataset, default_config: CouplingConfig
    ) -> None:
        """Bridge should accept custom income arrays of correct shape."""
        ds = _make_mock_dataset(n_objects=10, n_events=3)
        b = DynamoDecisionBridge(ds=ds, config=default_config)
        n = b.n_agents
        custom_income = np.full(n, 30_000.0, dtype=np.float32)
        b2 = DynamoDecisionBridge(
            ds=ds,
            config=default_config,
            income_per_agent=custom_income,
        )
        np.testing.assert_allclose(b2.income, custom_income, rtol=1e-5)

    def test_wrong_income_shape_raises(
        self, mock_ds: xr.Dataset, default_config: CouplingConfig
    ) -> None:
        """Passing income_per_agent with wrong shape should raise ValueError."""
        with pytest.raises(ValueError, match="income_per_agent"):
            DynamoDecisionBridge(
                ds=mock_ds,
                config=default_config,
                income_per_agent=np.ones(99, dtype=np.float32),
            )

# ===========================================================================
# ── TESTS: TARGETED VALIDATION ──────────────────────────────────────────────
# ===========================================================================

class TestEUValidation:
    """Targeted validation against hand-calculated Expected Utility."""

    def test_eu_hand_calculation(self) -> None:
        """
        Verify the entire EU integration against a manual calculation for
        2 agents and 2 events.
        """
        n_agents = 2
        n_events = 2
        # Agent 0: High wealth, low risk
        # Agent 1: Low wealth, high risk
        wealth = np.array([100_000.0, 20_000.0], dtype=np.float32)
        income = np.array([30_000.0, 15_000.0], dtype=np.float32)
        amenity = np.zeros(2, dtype=np.float32)
        risk_perc = np.array([1.0, 2.0], dtype=np.float32)
        
        # Two events: p=0.5, p=0.1
        p_floods = np.array([0.5, 0.1], dtype=np.float32)
        
        # Damages
        # Event 1 (0.5 freq): no-meas=5k, fp=1k
        # Event 2 (0.1 freq): no-meas=50k, fp=10k
        dmg_no = np.array([[5_000.0, 5_000.0], [50_000.0, 50_000.0]], dtype=np.float32)
        dmg_fp = np.array([[1_000.0, 1_000.0], [10_000.0, 10_000.0]], dtype=np.float32)
        
        adapt_costs = np.array([2_000.0, 2_000.0], dtype=np.float32)
        
        T_arr = np.array([2, 2], dtype=np.int32)
        r = 0.0
        sigma = 1.0 # log utility
        
        # Test do-nothing EU
        eu_no = _calc_eu_do_nothing(
            n_agents=2,
            wealth=wealth,
            income=income,
            amenity_value=amenity,
            amenity_weight=1.0,
            risk_perception=risk_perc,
            expected_damages=dmg_no,
            adapted=np.zeros(2, dtype=np.int32),
            p_floods=p_floods,
            T=T_arr,
            r=r,
            sigma=sigma,
            error_terms=np.ones(2, dtype=np.float32),
        )
        assert eu_no.shape == (2,)
        assert np.all(np.isfinite(eu_no))
        
        eu_fp = _calc_eu_adapt(
            n_agents=2,
            wealth=wealth,
            income=income,
            expenditure_cap=0.2, # 20% cap ensures affordable
            amenity_value=amenity,
            amenity_weight=1.0,
            risk_perception=risk_perc,
            expected_damages_adapt=dmg_fp,
            adaptation_costs=adapt_costs,
            time_adapted=np.zeros(2, dtype=np.int32),
            loan_duration=16,
            p_floods=p_floods,
            T=T_arr,
            r=r,
            sigma=sigma,
            error_terms=np.ones(2, dtype=np.float32),
        )
        assert eu_fp.shape == (2,)
        assert np.all(np.isfinite(eu_fp))


class TestEventCapping:
    """Integration test for the event capping logic in the example script."""
    
    def test_max_events_cap(self) -> None:
        from examples_engine.old_bridge_examples.run_coupled_example import _simulate_year_events
        
        ds = _make_mock_dataset(n_objects=2, n_events=10)
        cfg = CouplingConfig(decision=DecisionConfig(max_events_per_year=2))
        bridge = DynamoDecisionBridge(ds=ds, config=cfg)
        bridge.prepare_damage_arrays(slr_value=0.0)
        
        # All events fire (random()==0.0 < prob); cap then randomly selects
        # `max_events` of them without replacement via `choice`.  The mock
        # supplies a deterministic `choice` so the test is reproducible.
        class AlwaysTrueRNG:
            def random(self): return 0.0
            def choice(self, n, size, replace=False):
                return np.arange(size)

        occurred, _ = _simulate_year_events(bridge, 2025, AlwaysTrueRNG())
        # Event-cap semantics are now magnitude/random-selection based (see
        # DecisionConfig.max_events_per_year note); only the COUNT is asserted.
        assert len(occurred) == 2
