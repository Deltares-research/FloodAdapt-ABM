"""
lookup_utils.py
===============
Shared utility for interpolating total damage from the FloodAdapt-ABM
precomputed NetCDF lookup table along the sea-level rise (SLR) axis.

Both ABMSimulator and DynamoDecisionBridge use this module so that
the interpolation logic is defined exactly once.

Public API
----------
``interpolate_damage_at_slr``
    Interpolate damages for a single scalar SLR target.

``interpolate_damage_matrix``
    Batch wrapper: interpolate over a list/array of SLR values, returning a
    3-D damage matrix used by ``ABMSimulator.interpolate_damage_matrix``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def interpolate_damage_at_slr(
    ds: xr.Dataset,
    strategy: str,
    slr_target: float,
    res_mask: np.ndarray | None = None,
    method: str = "linear",
    max_pot_dmg: np.ndarray | None = None,
    dim_object_id: str = "object_id",
    dim_slr: str = "slr",
    dim_event: str = "event",
    dim_strategy: str = "strategy",
    var_total_damage: str = "total_damage",
) -> np.ndarray:
    """
    Interpolate total_damage at a single SLR value for one strategy.

    The result is optionally filtered to a residential subset and clamped to
    [0, max_pot_dmg] to prevent physically implausible values (e.g. cubic
    spline undershoot below zero).

    Parameters
    ----------
    ds : xr.Dataset
        FloodAdapt-ABM lookup table dataset.
    strategy : str
        Strategy name that exists in the strategy dimension.
    slr_target : float
        Sea-level rise value at which to interpolate (same unit as dataset).
    res_mask : np.ndarray or None
        Boolean array of shape (n_all_objects,) for subsetting agents.
        If None, all objects are returned.
    method : str
        One of 'linear', 'nearest', 'cubic', 'floor', 'ceil'.
        'cubic' requires at least 4 SLR grid points.
    max_pot_dmg : np.ndarray or None
        Upper bound for output damages, shape (n_agents,).
        Output is clipped to [0, max_pot_dmg] when provided.
    dim_object_id, dim_slr, dim_event, dim_strategy : str
        Dimension names in ds.
    var_total_damage : str
        Name of the damage data variable in ds.

    Returns
    -------
    np.ndarray
        Shape (n_agents, n_events), dtype float32.

    Raises
    ------
    ValueError
        If method is unknown or cubic is requested with fewer than 4 SLR points.
    """
    from scipy.interpolate import interp1d  # optional heavy dependency

    slr_arr: np.ndarray = ds[dim_slr].values.astype(np.float64)

    # Slice the strategy dimension and transpose to (n_objects, n_slr, n_events)
    da: xr.DataArray = (
        ds[var_total_damage]
        .sel({dim_strategy: strategy})
        .transpose(dim_object_id, dim_slr, dim_event)
    )

    # Optionally filter to a subset of objects (e.g. residential mask)
    if res_mask is not None:
        da = da.isel({dim_object_id: res_mask})

    # Load into memory as float32: shape (n_agents, n_slr, n_events)
    values: np.ndarray = da.values.astype(np.float32)

    return interpolate_cube_at_slr(
        values, slr_arr, slr_target, method=method, max_pot_dmg=max_pot_dmg
    )


def materialize_strategy_cube(
    ds: xr.Dataset,
    strategy: str,
    res_mask: np.ndarray | None = None,
    dim_object_id: str = "object_id",
    dim_slr: str = "slr",
    dim_event: str = "event",
    dim_strategy: str = "strategy",
    var_total_damage: str = "total_damage",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Materialize the (residential) damage cube for one strategy, once.

    Loading and residential-masking the ``(n_objects, n_slr, n_events)`` cube
    from the (lazily backed) dataset is the dominant cost of a single-SLR
    interpolation.  Callers that interpolate at many SLR values should cache the
    returned cube and feed it to :func:`interpolate_cube_at_slr` instead of
    re-reading the dataset each time.

    Returns
    -------
    values : np.ndarray[float32], shape (n_agents, n_slr, n_events)
    slr_arr : np.ndarray[float64], shape (n_slr,)
    """
    slr_arr: np.ndarray = ds[dim_slr].values.astype(np.float64)
    da: xr.DataArray = (
        ds[var_total_damage]
        .sel({dim_strategy: strategy})
        .transpose(dim_object_id, dim_slr, dim_event)
    )
    # Materialize the full cube in NumPy first, then apply the residential mask
    # in-memory.  Boolean ``isel`` on the (lazily backed) DataArray is
    # pathologically slow at scale, whereas a NumPy fancy index is trivial.
    values: np.ndarray = da.values.astype(np.float32)
    if res_mask is not None:
        values = values[res_mask]
    return values, slr_arr


def interpolate_cube_at_slr(
    values: np.ndarray,
    slr_arr: np.ndarray,
    slr_target: float,
    method: str = "linear",
    max_pot_dmg: np.ndarray | None = None,
) -> np.ndarray:
    """
    Interpolate a pre-materialized damage cube along the SLR axis.

    ``values`` has shape ``(n_agents, n_slr, n_events)`` (see
    :func:`materialize_strategy_cube`); the result is ``(n_agents, n_events)``,
    clamped to ``[0, max_pot_dmg]``.  The interpolation math is identical to
    :func:`interpolate_damage_at_slr` (which now delegates here), so results are
    bit-for-bit unchanged.
    """
    from scipy.interpolate import interp1d  # optional heavy dependency

    slr_arr = np.asarray(slr_arr, dtype=np.float64)

    # --- Interpolate along SLR axis (axis=1) ---------------------------------
    if method == "linear":
        f = interp1d(
            slr_arr, values, kind="linear",
            axis=1, bounds_error=False, fill_value="extrapolate",
        )
        interpolated: np.ndarray = f(slr_target).astype(np.float32)

    elif method == "cubic":
        if len(slr_arr) < 4:
            raise ValueError(
                f"Cubic interpolation requires at least 4 SLR grid points; "
                f"lookup table has {len(slr_arr)}."
            )
        f = interp1d(
            slr_arr, values, kind="cubic",
            axis=1, bounds_error=False, fill_value="extrapolate",
        )
        interpolated = f(slr_target).astype(np.float32)

    elif method == "nearest":
        idx: int = int(np.abs(slr_arr - slr_target).argmin())
        interpolated = values[:, idx, :]

    elif method == "floor":
        sort_idx = np.argsort(slr_arr)
        slr_sorted = slr_arr[sort_idx]
        i = np.searchsorted(slr_sorted, slr_target, side="right") - 1
        i = max(i, 0)
        interpolated = values[:, sort_idx[i], :]

    elif method == "ceil":
        sort_idx = np.argsort(slr_arr)
        slr_sorted = slr_arr[sort_idx]
        i = np.searchsorted(slr_sorted, slr_target, side="left")
        i = min(i, len(slr_sorted) - 1)
        interpolated = values[:, sort_idx[i], :]

    else:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            "Choose one of: 'linear', 'nearest', 'cubic', 'floor', 'ceil'."
        )

    # --- Clamp to [0, max_pot_dmg]: prevents spline undershoot/overshoot -----
    if max_pot_dmg is not None:
        interpolated = np.clip(interpolated, 0.0, max_pot_dmg[:, np.newaxis])
    else:
        interpolated = np.clip(interpolated, 0.0, None)  # at least no negatives

    return interpolated  # shape: (n_agents, n_events)


def interpolate_damage_matrix(
    ds: xr.Dataset,
    strategy: str,
    slr_values: np.ndarray,
    event_names_list: list,
    method: str = "linear",
    dim_object_id: str = "object_id",
    dim_slr: str = "slr",
    dim_event: str = "event",
    dim_strategy: str = "strategy",
    var_total_damage: str = "total_damage",
) -> np.ndarray:
    """
    Batch interpolation over a list of SLR values and a subset of events.

    This is the API used by ABMSimulator.interpolate_damage_matrix.

    Parameters
    ----------
    ds : xr.Dataset
        FloodAdapt-ABM lookup table dataset.
    strategy : str
        Strategy name.
    slr_values : array-like
        1-D sequence of SLR targets to interpolate to.
    event_names_list : list of str
        Ordered list of event names to include in the output.
    method : str
        Interpolation method (see interpolate_damage_at_slr).
    dim_object_id, dim_slr, dim_event, dim_strategy, var_total_damage : str
        Dimension / variable names in ds.

    Returns
    -------
    np.ndarray
        Shape (n_objects, n_events, n_slr_values), dtype float32.
    """
    slr_values = np.asarray(slr_values, dtype=np.float64)
    n_objects: int = ds.sizes[dim_object_id]
    n_events: int = len(event_names_list)
    n_slr: int = len(slr_values)
    damage_matrix = np.empty((n_objects, n_events, n_slr), dtype=np.float32)

    # Pre-compute the mapping of requested events to dataset event indices
    all_events: list = list(ds[dim_event].values.astype(str))
    event_indices = [all_events.index(e) for e in event_names_list]

    for i_slr, slr_val in enumerate(slr_values):
        # Full (n_objects, n_events_all) slice at this SLR
        full_slice = interpolate_damage_at_slr(
            ds=ds,
            strategy=strategy,
            slr_target=float(slr_val),
            res_mask=None,
            method=method,
            max_pot_dmg=None,
            dim_object_id=dim_object_id,
            dim_slr=dim_slr,
            dim_event=dim_event,
            dim_strategy=dim_strategy,
            var_total_damage=var_total_damage,
        )
        damage_matrix[:, :, i_slr] = full_slice[:, event_indices]

    return damage_matrix
