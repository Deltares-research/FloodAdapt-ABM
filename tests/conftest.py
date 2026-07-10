"""
conftest.py
===========
Shared pytest fixtures for the FloodAdapt-ABM test-suite.  Provides a mock
xarray lookup table matching the Charleston schema so tests do not need the
real ``.nc`` file.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


def make_mock_dataset(
    n_objects: int = 12,
    n_events: int = 6,
    n_slr: int = 5,
    residential_fraction: float = 0.8,
    residential_type: str = "RES",
    commercial_type: str = "COM",
    max_dmg_value: float = 200_000.0,
    floodproof_factor: float = 0.30,
    seed: int = 7,
) -> xr.Dataset:
    """
    Build a minimal xarray.Dataset mirroring the Charleston lookup table.

    Dimensions: ``object_id x slr x strategy x event``.  Strategies are
    ``["no_measures", "floodproof_all_0"]``; the floodproof damage is
    ``floodproof_factor`` times the no-measures damage.
    """
    rng = np.random.default_rng(seed)

    object_ids = np.array([str(i) for i in range(n_objects)])
    slr_values = np.linspace(0.0, 2.0, n_slr)
    event_names = np.array([f"ev_{i:03d}" for i in range(n_events)])
    strategies = np.array(["no_measures", "floodproof_all_0"])

    n_res = int(round(n_objects * residential_fraction))
    primary_types = [residential_type] * n_res + [commercial_type] * (n_objects - n_res)
    max_pot_dmg = np.full(n_objects, max_dmg_value, dtype=np.float64)

    # Frequencies: decreasing (most frequent first) — a valid RP-like set.
    freqs = np.array([1.0 / (2 ** i) for i in range(n_events)], dtype=np.float64)

    dmg_no_measures = rng.uniform(
        0.0, max_dmg_value * 0.5, size=(n_objects, n_slr, n_events)
    ).astype(np.float32)
    dmg_floodproof = (dmg_no_measures * floodproof_factor).astype(np.float32)

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


@pytest.fixture
def mock_ds():
    """A default mock dataset (12 objects, 80 % residential)."""
    return make_mock_dataset()


@pytest.fixture
def mock_ds_factory():
    """Return the factory so tests can customise the dataset."""
    return make_mock_dataset
