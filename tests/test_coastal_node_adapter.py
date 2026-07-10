"""
test_coastal_node_adapter.py
============================
PRE.4 tests: the lookup-table -> CoastalNode adapter prototype must map every
array **bit-identically** in both directions on a frozen synthetic population,
so the eventual 4b-full ≡ 4b-scaffold gate stays attributable to the
environment swap alone.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import CouplingConfig, SimulationEngine
from floodadapt_abm.coastal_node_adapter import (
    CoastalNodeArrays,
    LookupTableAdapter,
    round_trip_check,
)

from tests.conftest import make_mock_dataset


def _engine(seed: int = 3, n_objects: int = 120) -> SimulationEngine:
    ds = make_mock_dataset(n_objects=n_objects, n_events=6, seed=seed)
    return SimulationEngine(ds=ds, config=CouplingConfig())


class TestForwardMapping:
    def test_round_trip_all_pass(self):
        checks = round_trip_check(_engine(), slr_value=1.0)
        assert checks["all_pass"], f"failed checks: {[k for k, v in checks.items() if not v]}"

    @pytest.mark.parametrize("slr", [0.0, 0.7, 1.0, 2.0])
    def test_damages_bit_identical_across_slr(self, slr):
        eng = _engine()
        node = LookupTableAdapter(eng).populate(slr)
        d_no, d_fp = eng.prepare_damages(slr)
        assert np.array_equal(node.damages_coastal_cells.T, d_no)
        assert np.array_equal(node.damages_coastal_cells_adapt.T, d_fp)

    def test_native_conventions(self):
        eng = _engine()
        node = LookupTableAdapter(eng, geom_id="charleston").populate(1.0)
        assert node.geom_id == "charleston_flood_plain"
        assert node.damages_coastal_cells.shape == (6, eng.n_agents)  # events-first
        assert node.adapt.dtype == np.int8
        assert node.n == eng.n_agents

    def test_populate_reflects_live_state(self):
        eng = _engine()
        eng.state.is_adapted[:5] = True
        eng.state.time_adapted[:5] = 12
        node = LookupTableAdapter(eng).populate(1.0)
        assert node.adapt[:5].all() and not node.adapt[5:].any()
        assert (node.time_adapt[:5] == 12).all()

    def test_populated_arrays_are_copies(self):
        eng = _engine()
        node = LookupTableAdapter(eng).populate(1.0)
        node.property_value[:] = -1.0
        node.adapt[:] = 1
        assert (eng.max_pot_dmg > 0).all()          # source untouched
        assert not eng.state.is_adapted.any()       # state untouched


class TestWriteBack:
    def test_decisions_flow_back_exactly(self):
        eng = _engine()
        adapter = LookupTableAdapter(eng)
        node = adapter.populate(1.0)
        rng = np.random.default_rng(1)
        node.adapt = (rng.random(node.n) < 0.3).astype(np.int8)
        node.time_adapt = rng.integers(0, 75, node.n).astype(np.int32)
        adapter.write_back(node)
        assert np.array_equal(eng.state.is_adapted, node.adapt.astype(bool))
        assert np.array_equal(eng.state.time_adapted, node.time_adapt)

    def test_size_mismatch_raises(self):
        eng = _engine()
        node = LookupTableAdapter(eng).populate(1.0)
        node.n -= 1
        with pytest.raises(ValueError, match="households"):
            LookupTableAdapter(eng).write_back(node)

    def test_object_id_misalignment_raises(self):
        eng = _engine()
        node = LookupTableAdapter(eng).populate(1.0)
        node.object_ids = node.object_ids[::-1].copy()
        with pytest.raises(ValueError, match="object_id"):
            LookupTableAdapter(eng).write_back(node)


class TestEndToEndCouplingLoop:
    def test_engine_step_after_write_back_matches_direct_run(self):
        """Round-tripping the state through the node before a step must not
        change the simulation (the adapter is pure plumbing)."""
        slr = np.linspace(0.0, 1.0, 5)

        eng_a = _engine()
        rng_a = np.random.default_rng(0)
        for t, s in enumerate(slr):
            eng_a.step(t, float(s), rng_a)

        eng_b = _engine()
        adapter = LookupTableAdapter(eng_b)
        rng_b = np.random.default_rng(0)
        for t, s in enumerate(slr):
            node = adapter.populate(float(s))     # forward
            adapter.write_back(node)              # reverse (unchanged)
            eng_b.step(t, float(s), rng_b)

        assert np.array_equal(eng_a.state.is_adapted, eng_b.state.is_adapted)
        assert np.array_equal(eng_a.state.time_adapted, eng_b.state.time_adapted)
        assert np.array_equal(eng_a.state.risk_perception, eng_b.state.risk_perception)
