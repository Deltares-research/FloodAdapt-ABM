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


def setup_lookup_table(
        DATA_DIR, site, 
        name_event_set, 
        slr=np.arange(0, 1.1, 0.25), 
        unit = "meters", 
        fp_height = 0.5, 
        timestep=1, 
        run_scenarios=False) -> xr.Dataset:


    # Initialize FloodAdapt
    fn_event_set = DATA_DIR / site / "input" / "events" / name_event_set / f"{name_event_set}.toml"
    settings = Settings(
        DATABASE_ROOT=DATA_DIR,
        DATABASE_NAME=site,
        SFINCS_BIN_PATH=Path(r"c:\Users\winter_ga\Offline_data\FloodAdapt-Database\system\win-64\sfincs_v.2.1_alpha\sfincs.exe"),
        FIAT_BIN_PATH=Path(r"c:\Users\winter_ga\Offline_data\FloodAdapt-Database\system\win-64\fiat_v0.2.1\fiat.exe"),
        VALIDATE_BINARIES=True,
    )
    fa = FloodAdapt(database_path=settings.database_path
                    )

    event_set = EventSet.load_file(fn_event_set)
    events = []

    # Copy sub_events directories to events folder
    eventset_folder = DATA_DIR / site / "input" / "events"
    for sub in event_set.sub_events:
        if sub.frequency <= timestep:
            events.append(sub.name)
            src_dir = DATA_DIR / site / "input" / "events" / name_event_set / sub.name
            dst_dir = DATA_DIR / site / "input" / "events" / sub.name
            
            if src_dir.exists() and src_dir.is_dir():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                print(f"Copied {sub.name}")
            else:
                print(f"Source {src_dir} does not exist")


    # create projections for all SLR values
    projections = []
    se_change = SocioEconomicChange(population_growth_existing=0,
                            economic_growth=0)
    for nn,s in enumerate(slr):
        phys_proj = PhysicalProjection(sea_level_rise=UnitfulLength(value=s, units=unit))
        proj = Projection(
            name=f"SLR_{nn}",
            description=f"Sea level rise of {s:.2f} in {unit}",
            physical_projection=phys_proj, 
            socio_economic_change=se_change
        )
        
        try:
            fa.save_projection(proj)
        except AlreadyExistsError:
            print(f"Projection {proj.name} already exists in database")
        projections.append(proj)


    # Create one strategy without measures and one with floodproofing all buildings
    flood_proof = FloodProof(
        name="floodproof_all",
        description="Floodproofing all buildings.",
        selection_type=SelectionType.all,
        property_type="ALL",
        elevation=UnitfulLength(value=fp_height, units=unit)
    )
    try: 
        fa.save_measure(flood_proof)
    except AlreadyExistsError:
        print("Strategy already exists in database")

    strategies = [
        Strategy(
            name="no_measures",
            description="No measures",
            measures=[],
        ),
        Strategy(
            name="floodproof_all",
            description="Floodproof all buildings",
            measures=[flood_proof.name],
        )
    ]
    # save strategies
    for strat in strategies:
        try: 
            fa.save_strategy(strat)
        except AlreadyExistsError:
            print("Strategy already exists in database")

    # Create and save scenarios for all combinations of events, projections, and strategies
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
                try: 
                    fa.save_scenario(scen)
                except AlreadyExistsError:
                    print("Scenario already exists in database")
                scenarios.append(scen)

    # run scenarios
    if run_scenarios:
        for scen in scenarios[1:]:
            print(scen.name)
            fa.run_scenario(scen.name)


    # Arrange impacts in Dataset
    scen_name=f"{projections[0].name}_{events[0]}_{strategies[0].name}"
    gdf_temp = fa.get_building_footprint_impacts(scen_name)

    ds_impacts = xr.Dataset(coords={
            "object_id": pd.to_numeric(gdf_temp["Object ID"], errors="coerce").astype("Int64"),
            "slr": slr, 
            "strategy": [s.name for s in strategies]
            },
            data_vars={
                "inun_depth": (
                    ["object_id", "slr", "strategy"], 
                    np.empty([len(gdf_temp["Object ID"]), len(slr), len(strategies)])
                ),
                "total_damage": (
                    ["object_id", "slr", "strategy"],
                    np.empty([len(gdf_temp["Object ID"]), len(slr), len(strategies)])
                )
            }
    )
    ds_impacts

    for strat in strategies:
        for proj in projections:
            for event_name in events:
                scen_name=f"{proj.name}_{event_name}_{strat.name}"
                gdf_impacts = fa.get_building_footprint_impacts(scen_name)
                ds_impacts["inun_depth"].loc[:, proj.physical_projection.sea_level_rise.value, strat.name] = gdf_impacts["Inundation Depth"].values
                ds_impacts["total_damage"].loc[:, proj.physical_projection.sea_level_rise.value, strat.name] = gdf_impacts["Total Damage"].values

    return ds_impacts


