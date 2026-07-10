"""
test_agent_state.py
===================
Unit tests for the ``AgentState`` container (Phase-3 unified per-agent state).
"""
from __future__ import annotations

import numpy as np

from floodadapt_abm.agent_state import AgentState


def _make(n=6):
    income = np.linspace(30_000, 80_000, n).astype(np.float32)
    wealth = income * 4.14
    return AgentState.initial(n, income, wealth, risk_perc_min=0.01)


def test_initial_shapes_and_dtypes():
    st = _make(6)
    assert st.n_agents == 6
    for arr in (st.wealth, st.income, st.risk_perception):
        assert arr.shape == (6,)
    assert st.is_adapted.dtype == bool
    assert st.time_adapted.dtype == np.int32
    assert st.flood_timer.dtype == np.int32


def test_initial_values():
    st = _make(4)
    assert np.all(st.is_adapted == False)  # noqa: E712
    assert np.all(st.time_adapted == 0)
    assert np.allclose(st.risk_perception, 0.01)
    assert np.all(st.flood_timer == 99)


def test_wealth_income_are_copies_not_views():
    income = np.ones(3, dtype=np.float32)
    wealth = np.ones(3, dtype=np.float32)
    st = AgentState.initial(3, income, wealth, risk_perc_min=0.0)
    st.income[0] = 999.0
    assert income[0] == 1.0  # original untouched


def test_copy_is_deep():
    st = _make(5)
    clone = st.copy()
    clone.is_adapted[0] = True
    clone.time_adapted[1] = 7
    assert st.is_adapted[0] == False  # noqa: E712
    assert st.time_adapted[1] == 0


def test_custom_initial_flood_timer():
    income = np.ones(2, dtype=np.float32)
    st = AgentState.initial(2, income, income, risk_perc_min=0.0, initial_flood_timer=0)
    assert np.all(st.flood_timer == 0)
