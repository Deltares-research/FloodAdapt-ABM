"""
_shared.py
==========
Small helper shared by the numbered ``examples_engine`` scripts.

Its job is twofold:

1. **Bootstrap imports** — insert the repository root on ``sys.path`` so the
   examples import ``floodadapt_abm`` whether the package is
   ``pip install``-ed.
2. **Provide a dataset** — return the real Charleston probabilistic lookup
   table when it is present, otherwise fall back to a small, deterministic
   *synthetic* "Charleston-like" table so **every example runs anywhere**
   (no large data file or DYNAMO-M checkout required).

None of the examples contain data-plumbing logic themselves; they call
:func:`load_dataset` and focus on the engine / rule API.
"""
from __future__ import annotations

import sys
from pathlib import Path

import os

import numpy as np
import xarray as xr

# --- 1. Make the floodadapt_abm package importable ------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Candidate locations of the real Charleston lookup table (optional).
# FA_ABM_REAL_TABLE_PATH takes precedence (same env var as the PRE.2
# real-table gate); the sibling-checkout convention is the fallback.
_REAL_TABLE_CANDIDATES = [
    *( [Path(os.environ["FA_ABM_REAL_TABLE_PATH"])]
       if os.environ.get("FA_ABM_REAL_TABLE_PATH") else [] ),
    _REPO_ROOT.parent / "DYNAMO-M" / "lookup_table_charleston_beta_release_ABM_probabilistic_set.nc",
]


def _make_synthetic_dataset(
    n_objects: int = 200,
    n_events: int = 6,
    n_slr: int = 5,
    residential_fraction: float = 0.8,
    seed: int = 7,
) -> xr.Dataset:
    """
    Build a minimal, *heterogeneous* xarray.Dataset mirroring the Charleston
    lookup table.

    Dimensions ``object_id x slr x strategy x event``; strategies are
    ``["no_measures", "floodproof_all_0"]`` (floodproof damage = 0.30 x
    no-measures).  Heterogeneity is deliberate so the examples produce a
    realistic *partial* adoption curve with visible Monte-Carlo variance:

    * ``max_pot_dmg`` varies per object (uniform 100k - 300k),
    * each object has its own flood ``exposure`` in [0, 1] (many low-risk,
      some high-risk households),
    * events are relatively rare, so realised flooding differs across
      Monte-Carlo sequences.

    Fully seeded, so example output is reproducible.
    """
    rng = np.random.default_rng(seed)

    object_ids = np.array([str(i) for i in range(n_objects)])
    slr_values = np.linspace(0.0, 2.0, n_slr)
    event_names = np.array([f"ev_{i:03d}" for i in range(n_events)])
    strategies = np.array(["no_measures", "floodproof_all_0"])

    n_res = int(round(n_objects * residential_fraction))
    primary_types = ["RES"] * n_res + ["COM"] * (n_objects - n_res)

    max_pot_dmg = rng.uniform(100_000.0, 300_000.0, n_objects).astype(np.float64)
    exposure = rng.uniform(0.0, 1.0, n_objects)[:, None, None]
    base = rng.uniform(0.0, 1.0, (n_objects, n_slr, n_events))
    dmg_no_measures = (exposure * base * max_pot_dmg[:, None, None]).astype(np.float32)
    dmg_floodproof = (dmg_no_measures * 0.30).astype(np.float32)

    # Rare-ish events (1 / (3 * 2**i)) so per-sequence weather differs.
    freqs = np.array([1.0 / (3 * 2 ** i) for i in range(n_events)], dtype=np.float64)

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


def load_dataset(prefer_real: bool | None = None) -> tuple[xr.Dataset, str]:
    """
    Return ``(dataset, source_label)``.

    By default a small **synthetic** table is used so the examples run fast and
    anywhere.  The real Charleston table (61,858 objects x 207 events) is large
    and slow, so it is *opt-in*:

    * pass ``prefer_real=True``, or
    * set the environment variable ``FA_ABM_REAL_TABLE=1``.

    When opted in but the file is not found, this transparently falls back to
    the synthetic table.  The returned label states which one you got.
    """
    if prefer_real is None:
        prefer_real = os.environ.get("FA_ABM_REAL_TABLE", "0") not in ("0", "", "false", "False")

    if prefer_real:
        for candidate in _REAL_TABLE_CANDIDATES:
            if candidate.exists():
                return xr.open_dataset(candidate), f"real Charleston table ({candidate.name})"
    return _make_synthetic_dataset(), "synthetic Charleston-like table (generated)"


def banner(title: str, width: int = 74) -> None:
    """Print a section banner."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
