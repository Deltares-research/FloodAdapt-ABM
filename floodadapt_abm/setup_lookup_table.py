#In[]: Imports

from pathlib import Path
import numpy as np
import shutil
import xarray as xr
import pandas as pd

from flood_adapt import FloodAdapt
from flood_adapt.misc.exceptions import AlreadyExistsError
from flood_adapt.objects import (
    EventSet,
    Projection, 
    PhysicalProjection, 
    SocioEconomicChange, 
    Scenario, 
    Strategy,
    FloodProof,
    SelectionType,
    )

from flood_adapt.objects.forcing.unit_system import UnitTypesLength, UnitfulLength, UnitfulLengthRefValue, VerticalReference
from flood_adapt.config.config import Settings


def get_events_freq(fa, name_event_set):
    event_set = fa.get_event(name_event_set)
    freqs = []
    for sub in event_set.sub_events:
        freqs.append(sub.frequency)
        
    return freqs

# 1. Create the matrix with all scenario names/combinations
def create_combinations_matrix(fa, name_event_set, slr, unit, fp_height):
    # Allow fp_height to be a list or a single value
    if not isinstance(fp_height, (list, tuple, np.ndarray)):
        fp_heights = [fp_height]
    else:
        fp_heights = list(fp_height)

    event_set = fa.get_event(name_event_set)
    events = []
    for sub in event_set.sub_events:
        events.append(sub.name)
        src_dir = fa.database.events.input_path / name_event_set / sub.name
        dst_dir = fa.database.events.input_path / sub.name
        if src_dir.exists() and src_dir.is_dir():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        else:
            print(f"Source {src_dir} does not exist")

    projections = []
    se_change = SocioEconomicChange(population_growth_existing=0, economic_growth=0)
    for nn, s in enumerate(slr):
        phys_proj = PhysicalProjection(sea_level_rise=UnitfulLength(value=s, units=unit))
        proj = Projection(
            name=f"SLR_{nn}",
            description=f"Sea level rise of {s:.2f} in {unit}",
            physical_projection=phys_proj,
            socio_economic_change=se_change
        )
        projections.append(proj)

    # Create all floodproof measures and strategies for each fp_height
    flood_proofs = []
    strategies = [
        Strategy(
            name="no_measures",
            description="No measures",
            measures=[],
        )
    ]
    for kk, h in enumerate(fp_heights):
        flood_proof = FloodProof(
            name=f"floodproof_all_{kk}",
            description=f"Floodproofing all buildings by {h} {unit}.",
            selection_type=SelectionType.all,
            property_type="ALL",
            elevation=UnitfulLength(value=h, units=unit)
        )
        flood_proofs.append(flood_proof)
        strategies.append(
            Strategy(
                name=f"floodproof_all_{kk}",
                description=f"Floodproofing all buildings by {h} {unit}.",
                measures=[flood_proof.name],
            )
        )

    scenarios = []
    for strat in strategies:
        for proj in projections:
            for event_name in events:
                scen = Scenario(
                    name=f"{proj.name}_{event_name}_{strat.name}",
                    description=f"Scenario with {proj.description}, Event {event_name}, and {strat.description}",
                    event=event_name,
                    projection=proj.name,
                    strategy=strat.name,
                )
                scenarios.append(scen)
    return events, projections, strategies, scenarios, flood_proofs

# 2. Save these in the FloodAdapt database
def save_combinations_to_database(fa, projections, strategies, scenarios, flood_proofs):
    scenarios_existing = fa.get_scenarios()["name"]
    for proj in projections:
        try:
            fa.save_projection(proj)
        except AlreadyExistsError:
            print(f"Projection {proj.name} already exists in database")
    for flood_proof in flood_proofs:
        try:
            fa.save_measure(flood_proof)
        except AlreadyExistsError:
            print(f"Measure {flood_proof.name} already exists in database")
    for strat in strategies:
        try:
            fa.save_strategy(strat)
        except AlreadyExistsError:
            print(f"Strategy {strat.name} already exists in database")
    for scen in scenarios:
        if scen.name not in scenarios_existing:
            fa.save_scenario(scen)

# Helper: delete non-impact files up to a depth of 2 within a scenario folder
def _cleanup_scenario_outputs(
    fa: FloodAdapt,
    scen_name: str,
    keep_substring: str | list[str] | tuple[str, ...] = "Impacts_building_footprints",
    max_depth: int = 2,
) -> None:
    """Delete files inside a scenario output folder except those matching substrings.

    Parameters
    - fa: FloodAdapt instance
    - scen_name: Scenario name (folder under scenarios/output)
    - keep_substring: A single substring or a list/tuple of substrings. Any file whose
      name contains at least one of these substrings will be kept.
    - max_depth: Maximum directory depth to traverse from the scenario folder.
    """

    base: Path = fa.database.scenarios.output_path / scen_name
    if not base.exists() or not base.is_dir():
        return

    # Normalize to a list of substrings to keep
    if isinstance(keep_substring, str):
        keep_list = [keep_substring]
    else:
        keep_list = list(keep_substring)

    def walk_dir(p: Path, depth: int) -> None:
        try:
            for child in p.iterdir():
                if child.is_dir():
                    if depth < max_depth:
                        walk_dir(child, depth + 1)
                else:
                    # Keep file if any of the substrings is present in the filename
                    if not any(sub in child.name for sub in keep_list):
                        try:
                            child.unlink()
                        except Exception as e:
                            print(f"Could not delete {child}: {e}")
        except Exception as e:
            print(f"Could not iterate {p}: {e}")

    walk_dir(base, 0)

# 3. Run all scenarios
def run_scenarios(fa, scenarios, clean=True):
    scenarios_db = pd.DataFrame(fa.get_scenarios())
    for scen in scenarios:
        print(scen.name)
        if not scenarios_db.loc[scenarios_db["name"] == scen.name, "finished"].values[0]:
            fa.run_scenario(scen.name)
        else:
            print(f"Scenario {scen.name} already run.")
        # Cleanup: keep only files with "Impacts_building_footprints" up to depth 2
        if clean:
            _cleanup_scenario_outputs(fa, scen.name, keep_substring=["max_water_level_map.nc", "finished.txt", "Impacts_building_footprints"], max_depth=2)

# 4. Read results and return the dataset
def read_impacts_dataset(fa, projections, strategies, events, slr, events_freq=None):
    scen_name = f"{projections[0].name}_{events[0]}_{strategies[0].name}"
    gdf_temp = fa.get_building_footprint_impacts(scen_name)
    ds_impacts = xr.Dataset(coords={
            "object_id": gdf_temp["Object ID"].values,
            "slr": slr,
            "strategy": [s.name for s in strategies],
            "event": events
        },
        data_vars={
            "inun_depth": (
                ["object_id", "slr", "strategy", "event"],
                np.empty([len(gdf_temp["Object ID"]), len(slr), len(strategies), len(events)])
            ),
            "total_damage": (
                ["object_id", "slr", "strategy", "event"],
                np.empty([len(gdf_temp["Object ID"]), len(slr), len(strategies), len(events)])
            )
        }
    )
    ds_impacts.coords['object_id'].attrs['max_pot_dmg'] = gdf_temp["Max Potential Damage: structure"].values + gdf_temp["Max Potential Damage: content"].values
    ds_impacts.coords['object_id'].attrs['primary_object_type'] = gdf_temp["Primary Object Type"].astype(str).to_list()

    
    object_id_order = gdf_temp["Object ID"].values

    for strat in strategies:
        for proj in projections:
            for event_name in events:
                scen_name = f"{proj.name}_{event_name}_{strat.name}"
                gdf_impacts = fa.get_building_footprint_impacts(scen_name)
                # Align rows by Object ID rather than trusting positional order:
                # different scenarios may return footprints in a different order,
                # which would otherwise silently misalign the damage arrays.
                impacts_by_id = gdf_impacts.set_index("Object ID")
                impacts_by_id = impacts_by_id.reindex(object_id_order)
                missing = impacts_by_id["Total Damage"].isna()
                if missing.any():
                    raise ValueError(
                        f"read_impacts_dataset: scenario '{scen_name}' is missing "
                        f"impacts for {int(missing.sum())} object(s) present in the "
                        "reference footprint set; cannot align damage arrays."
                    )
                ds_impacts["inun_depth"].loc[:, proj.physical_projection.sea_level_rise.value, strat.name, event_name] = impacts_by_id["Inundation Depth"].values
                ds_impacts["total_damage"].loc[:, proj.physical_projection.sea_level_rise.value, strat.name, event_name] = impacts_by_id["Total Damage"].values
    return ds_impacts

# 5. General method
def create_lookup_table(
        fa: FloodAdapt,
        name_event_set: str,
        slr: np.array = np.arange(0, 1.1, 0.25),
        unit: UnitTypesLength = UnitTypesLength.meters,
        fp_height: float | list[float] | tuple | np.ndarray = 0.5,
    ) -> xr.Dataset:

    events, projections, strategies, scenarios, flood_proofs = create_combinations_matrix(fa, name_event_set, slr, unit, fp_height)
    save_combinations_to_database(fa, projections, strategies, scenarios, flood_proofs)
    run_scenarios(fa, scenarios)
    ds_impacts = read_impacts_dataset(fa, projections, strategies, events, slr)
    
    events_freq = get_events_freq(fa, name_event_set)
    
    ds_impacts.coords['event'].attrs['freq'] = events_freq

    return ds_impacts


