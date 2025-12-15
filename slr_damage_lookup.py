import numpy as np
from scipy.interpolate import interp1d

def _extract_damage(ds, slr_sim, strategy, event, building_id):
    damage_vals = []
    for slr in slr_sim:
        sel = ds.sel(slr=slr, strategy=strategy, event=event)
        if building_id is not None:
            try:
                damage = sel.sel(object=building_id).item()
            except Exception:
                damage = sel.isel(object=building_id).item()
        else:
            damage = sel.sum().item()
        damage_vals.append(damage)
    return np.array(damage_vals)

def _interpolate_damage(slr_sim, damage_vals, slr_value, method):
    if method == 'linear':
        f = interp1d(slr_sim, damage_vals, kind='linear', fill_value='extrapolate')
        return float(f(slr_value))
    elif method == 'nearest':
        idx = (np.abs(slr_sim - slr_value)).argmin()
        return float(damage_vals[idx])
    elif method == 'cubic':
        if len(slr_sim) < 4:
            raise ValueError('Cubic interpolation requires at least 4 SLR points.')
        f = interp1d(slr_sim, damage_vals, kind='cubic', fill_value='extrapolate')
        return float(f(slr_value))
    elif method == 'floor':
        slr_sim_sorted = np.sort(slr_sim)
        idxs = np.where(slr_sim_sorted <= slr_value)[0]
        if len(idxs) == 0:
            idx = 0
        else:
            idx = idxs[-1]
        return float(damage_vals[np.argsort(slr_sim)[idx]])
    elif method == 'ceil':
        slr_sim_sorted = np.sort(slr_sim)
        idxs = np.where(slr_sim_sorted >= slr_value)[0]
        if len(idxs) == 0:
            idx = -1
        else:
            idx = idxs[0]
        return float(damage_vals[np.argsort(slr_sim)[idx]])
    else:
        raise ValueError(f'Unknown interpolation method: {method}')

def slr_damage_lookup(ds, slr_value, strategy, event, method='linear', building_id=None):
    """
    Lookup or interpolate damage for a given SLR value, strategy, event, and optionally building/object id.

    Parameters:
        ds: xarray Dataset or DataArray with 'slr' as a coordinate
        slr_value: float, the SLR value to evaluate
        strategy: strategy value to select
        event: event value to select
        method: interpolation method, one of:
            - 'linear': linear interpolation between SLR points (default)
            - 'nearest': use the nearest SLR value
            - 'cubic': cubic spline interpolation (requires at least 4 SLR points)
            - 'floor': returns the value at the largest simulated SLR less than or equal to slr_value (never extrapolates)
            - 'ceil': returns the value at the smallest simulated SLR greater than or equal to slr_value (never extrapolates)
        building_id: optional, id/index of the building/object. If None, sum all objects. Otherwise, select only that object.

    Returns:
        damage: float

    Notes:
        - 'linear', 'cubic', and 'step' use interpolation and may extrapolate outside the simulated SLR range.
        - 'floor' and 'ceil' always return the closest simulated value within the range (no extrapolation).
        - If building_id is provided, returns damage for that object; otherwise, returns the sum over all objects.
    """
    slr_sim = ds['slr'].values
    damage_vals = _extract_damage(ds, slr_sim, strategy, event, building_id)
    return _interpolate_damage(slr_sim, damage_vals, slr_value, method)