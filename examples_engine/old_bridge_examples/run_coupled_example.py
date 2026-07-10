"""
run_coupled_example.py
======================
Step-by-step demonstration of the FloodAdapt-ABM × DYNAMO-M coupling.

This script loads the Charleston probabilistic lookup table, constructs a
``DynamoDecisionBridge``, and walks through a multi-year coupled simulation
(Years 1, 2, 3, then a jump to Year 77) printing detailed intermediate
states at each step.

Run directly (no CLI arguments needed) for easy IDE debugging:

    python run_coupled_example.py

Reference
--------------------
Tierolf, L., Haer, T., Botzen, W. J. W., de Bruijn, J. A., Ton, M. J.,
Reimann, L., & Aerts, J. C. J. H. (2023). A coupled agent-based model for
France for simulating adaptation and migration decisions under future coastal
flood risk. Scientific Reports, 13(1), 4176.
https://doi.org/10.1038/s41598-023-31351-y
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import xarray as xr

from floodadapt_abm import CouplingConfig, DecisionConfig, NetCDFMappingConfig
from floodadapt_abm import DynamoDecisionBridge

# ===========================================================================
# ── USER-ADJUSTABLE SETTINGS ────────────────────────────────────────────────
# ===========================================================================

#: Path to the FloodAdapt database
DATA_DIR: Path = Path(r"c:\Users\athanasi\Github\Database\Working_Databases\Charleston\4_FloodAdapt\Database")
SITE_NAME: str = "charleston_beta_release"

#: Path to the precomputed FloodAdapt-ABM lookup table.
LOOKUP_TABLE_PATH: Path = Path(
    r"C:\repos\DYNAMO-M\lookup_table_charleston_beta_release_ABM_probabilistic_set.nc"
)

# Constants
RANDOM_SEED = 42

# TODO: verify these initial values against the full FloodAdapt-ABM configuration
INITIAL_YEAR = 2020
TIME_HORIZON = 30
DEMO_YEARS = list(range(INITIAL_YEAR + 2, INITIAL_YEAR + 7))  # Simulate first 5 years

#: Sub-sample of agents to print in detailed tables (keeps console readable).
N_AGENTS_PRINT: int = 10

# ===========================================================================
# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────
# ===========================================================================


def _header(title: str, width: int = 72) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _subheader(title: str) -> None:
    print(f"  -- {title} --")


def _print_agent_table(
    bridge: DynamoDecisionBridge,
    agent_indices: list[int] | np.ndarray | None = None,
    extra_cols: dict[str, np.ndarray] | None = None,
    cohort_labels: dict[int, str] | None = None,
    n_print: int = N_AGENTS_PRINT,
) -> None:
    """Print a summary table of the specified or first ``n_print`` residential agents."""
    if agent_indices is None:
        agent_indices = list(range(n_print))
    else:
        agent_indices = list(agent_indices)

    cols: dict[str, list] = {
        "object_id": [bridge.object_ids[idx] for idx in agent_indices],
        "max_pot_dmg": [bridge.max_pot_dmg[idx] for idx in agent_indices],
        "income": [bridge.income[idx] for idx in agent_indices],
        "wealth": [bridge.wealth[idx] for idx in agent_indices],
        "risk_perc": [bridge.risk_perception[idx] for idx in agent_indices],
        "flood_timer": [bridge.flood_timer[idx] for idx in agent_indices],
        "adapted": [int(bridge.is_adapted[idx]) for idx in agent_indices],
    }

    if cohort_labels is not None:
        cols["cohort"] = [cohort_labels.get(idx, "Unknown") for idx in agent_indices]

    if extra_cols:
        for k, v in extra_cols.items():
            cols[k] = [v[idx] for idx in agent_indices]

    # Format widths
    col_fmts = {
        "object_id": "<10",
        "max_pot_dmg": ">14",
        "income": ">12",
        "wealth": ">14",
        "risk_perc": ">10",
        "flood_timer": ">12",
        "adapted": ">8",
        "cohort": "<16",
    }
    if extra_cols:
        for k in extra_cols:
            col_fmts[k] = ">14"

    header_line = "  " + "  ".join(
        f"{k:{col_fmts.get(k, '>12')}}" for k in cols
    )
    print(header_line)
    print("  " + "-" * (len(header_line) - 2))

    for i in range(len(agent_indices)):
        row = []
        for k, lst in cols.items():
            val = lst[i]
            fmt = col_fmts.get(k, ">12")
            if isinstance(val, (float, np.floating)):
                row.append(f"{val:{fmt}.2f}")
            elif isinstance(val, (int, np.integer)):
                row.append(f"{val:{fmt}d}")
            else:
                row.append(f"{str(val):{fmt}}")
        print("  " + "  ".join(row))


def _simulate_year_events(
    bridge: DynamoDecisionBridge,
    year: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    """
    Stochastically draw flood events for a single year from the dataset
    using each event's annual frequency.

    Parameters
    ----------
    bridge : DynamoDecisionBridge
        Initialised bridge (provides event names and frequencies).
    year : int
        Current 1-based simulation year (used only for display).
    rng : np.random.Generator
        Seeded RNG for reproducibility.

    Returns
    -------
    occurred_events : list[str]
        Names of events that occurred this year.
    flooded_agents : np.ndarray[bool], shape (n_agents,)
        Per-agent flood flag (True if any event produced positive inundation
        depth for that agent — approximated here by positive damage).
    """
    event_names: np.ndarray = bridge._event_names
    event_freqs: np.ndarray = bridge._event_freqs
    # Done: Refactor this stochastic event drawing and capping logic OUT of the example script
    # and integrate it natively into DynamoDecisionBridge or ABMSimulator.
    # Independent Bernoulli trials for each event (FloodAdapt-ABM approach)
    occurred_list: list[str] = []
    for name, freq in zip(event_names, event_freqs):
        # Bernoulli trial: probability of occurrence = freq * dt (dt=1 year)
        prob = min(freq, 1.0)
        if rng.random() < prob:
            occurred_list.append(str(name))
            
    # Apply the max_events_per_year cap
    max_events = bridge._dec.max_events_per_year
    if len(occurred_list) > max_events:
        # Randomly sample N events from the drawn pool to preserve unbiased distribution
        # We reuse the existing deterministic `rng` to keep it reproducible
        indices = rng.choice(len(occurred_list), size=max_events, replace=False)
        occurred_list = [occurred_list[i] for i in indices]
        
    occurred = occurred_list

    # Aggregate per-agent damages to determine who flooded
    total_dmg = np.zeros(bridge.n_agents, dtype=np.float32)
    for evt in occurred:
        total_dmg += bridge.get_current_damages(evt)

    flooded_agents: np.ndarray = total_dmg > 0
    return occurred, flooded_agents


# ===========================================================================
# ── MAIN SIMULATION LOOP ────────────────────────────────────────────────────
# ===========================================================================


def run() -> None:
    """Execute the step-by-step coupled simulation demo."""

    # -----------------------------------------------------------------------
    # 0.  Configuration
    # -----------------------------------------------------------------------
    _header("STEP 0 — CONFIGURATION")
    cfg = CouplingConfig(
        netcdf=NetCDFMappingConfig(
            # All defaults match the Charleston probabilistic lookup table.
            # Override here if your site uses different names.
            residential_substring="RES",
            strategy_no_measures="no_measures",
            strategy_floodproof="floodproof_all_0",
        ),
        decision=DecisionConfig(
            risk_aversion=1.5,
            discount_rate=0.04,
            decision_horizon=10,
            risk_perc_min=0.01,
            risk_perc_max=2.0,
            risk_perc_coef=-3.6,
            loan_duration=16,
            interest_rate=0.03,
            adaptation_cost_fraction=0.10,
            expenditure_cap=0.06,
            amenity_weight=1.0,
            error_interval=0.0,
            income_to_wealth_ratio=4.14,
            max_events_per_year=4,  # Cap drawn events at 4 per year for the demo
        ),
        random_seed=RANDOM_SEED,
    )

    print(textwrap.dedent(f"""
      NetCDF mapping
      --------------
        object_id dim    : {cfg.netcdf.dimension_object_id}
        event dim        : {cfg.netcdf.dimension_event}
        slr dim          : {cfg.netcdf.dimension_slr}
        strategy dim     : {cfg.netcdf.dimension_strategy}
        total_damage var : {cfg.netcdf.var_total_damage}
        max_pot_dmg attr : {cfg.netcdf.attr_max_pot_dmg}
        event freq attr  : {cfg.netcdf.attr_event_freq}
        building type    : {cfg.netcdf.attr_building_type}
        res. substring   : {cfg.netcdf.residential_substring!r}
        strategy (none)  : {cfg.netcdf.strategy_no_measures!r}
        strategy (adapt) : {cfg.netcdf.strategy_floodproof!r}

      Decision parameters
      -------------------
        risk aversion s  : {cfg.decision.risk_aversion}
        discount rate r  : {cfg.decision.discount_rate}
        decision horizon : {cfg.decision.decision_horizon} years
        rp min / max     : {cfg.decision.risk_perc_min} / {cfg.decision.risk_perc_max}
        rp decay coef    : {cfg.decision.risk_perc_coef}
        loan duration    : {cfg.decision.loan_duration} years  @ {cfg.decision.interest_rate:.0%}
        adapt cost frac  : {cfg.decision.adaptation_cost_fraction:.0%} of max_pot_dmg
        expenditure cap  : {cfg.decision.expenditure_cap:.0%} of income
    """))

    # -----------------------------------------------------------------------
    # 1.  Load the lookup table
    # -----------------------------------------------------------------------
    _header("STEP 1 — LOAD DATASET AND INITIALIZE BRIDGE")

    print(f"\nLoading NetCDF from: {LOOKUP_TABLE_PATH.name}")
    if not LOOKUP_TABLE_PATH.exists():
        print(
            f"  ERROR: file not found at {LOOKUP_TABLE_PATH}.\n"
            "  Please update LOOKUP_TABLE_PATH at the top of this script."
        )
        sys.exit(1)

    ds: xr.Dataset = xr.open_dataset(str(LOOKUP_TABLE_PATH))
    print(
        f"  Dimensions  : {dict(ds.sizes)}\n"
        f"  Strategies  : {list(ds[cfg.netcdf.dimension_strategy].values)}\n"
        f"  SLR levels  : {list(ds[cfg.netcdf.dimension_slr].values)} ft\n"
        f"  Events      : {len(ds[cfg.netcdf.dimension_event])} total\n"
    )

    # Interpolate SLR dynamically based on the lookup table's max SLR
    slr_levels = ds[cfg.netcdf.dimension_slr].values
    max_slr = float(np.max(slr_levels))

    def slr_trajectory(year: int) -> float:
        """Linear SLR projection mapping INITIAL_YEAR to 0 and end year to max_slr."""
        clamped_year = min(max(year, INITIAL_YEAR), INITIAL_YEAR + TIME_HORIZON)
        fraction = (clamped_year - INITIAL_YEAR) / TIME_HORIZON
        return round(fraction * max_slr, 4)

    # -----------------------------------------------------------------------
    # 2.  Initialise the bridge
    # -----------------------------------------------------------------------
    _header("STEP 2 — INITIALISE BRIDGE")
    bridge = DynamoDecisionBridge(ds=ds, config=cfg)

    print(
        f"  Total buildings in dataset : {len(ds[cfg.netcdf.dimension_object_id])}\n"
        f"  Residential agents (RES)   : {bridge.n_agents}\n"
        f"  Annual adapt cost range    : "
        f"${bridge._annual_adapt_cost.min():,.0f} – "
        f"${bridge._annual_adapt_cost.max():,.0f}\n"
    )

    # Pre-calculate initial EAD under SLR=0.0 to identify "never adapt" candidates
    bridge.prepare_damage_arrays(slr_value=0.0, interp_method="linear")
    ead_init = bridge.compute_expected_annual_damages(use_adapted_strategy=False)

    # 1. Cohort: Never adapt (we pick agents with initial expected annual damage = 0)
    zero_ead_indices = np.where(ead_init == 0)[0]
    never_adapt_cohort = list(zero_ead_indices[:5])

    # Track labels for display
    cohort_labels: dict[int, str] = {idx: "Never Adapt" for idx in never_adapt_cohort}
    tracked_indices: list[int] = list(never_adapt_cohort)

    _subheader("Initial agent state (first 10 residential agents)")
    _print_agent_table(bridge)

    # -----------------------------------------------------------------------
    # 3.  Year-by-year simulation
    # -----------------------------------------------------------------------
    # Done: Refactor this time step progression loop. It is currently hardcoded
    # here for demonstration, but time step progression must be natively handled 
    # by the ABM repo (e.g. ABMSimulator or Mesa model integration).
    rng = np.random.default_rng(RANDOM_SEED + 1)

    # Accumulators for summary stats
    total_adapted_history: dict[int, int] = {}
    total_damage_history: dict[int, float] = {}
    cum_damage_history: dict[int, float] = {}
    
    # Cumulative damage array per agent
    cum_dmg = np.zeros(bridge.n_agents, dtype=np.float32)

    for year in DEMO_YEARS:
        _header(f"SIMULATING YEAR {year}", width=50)

        # 2a. Calculate dummy SLR trajectory for this specific year
        slr_ft = slr_trajectory(year)
        print(f"Year {year} SLR: {slr_ft:.3f} ft (Scenario: dummy trajectory)")

        # 2b. Prepare damage arrays via 1D interpolation along the SLR axis
        bridge.prepare_damage_arrays(slr_value=slr_ft, interp_method="cubic")

        # Show EAD statistics
        ead_no_meas = bridge.compute_expected_annual_damages(use_adapted_strategy=False)
        ead_adapted = bridge.compute_expected_annual_damages(use_adapted_strategy=True)
        print(
            f"  EAD (no_measures) — "
            f"mean=${ead_no_meas.mean():,.0f}  "
            f"max=${ead_no_meas.max():,.0f}  "
            f"min=${ead_no_meas.min():,.0f}\n"
            f"  EAD (floodproof)  — "
            f"mean=${ead_adapted.mean():,.0f}  "
            f"max=${ead_adapted.max():,.0f}  "
            f"min=${ead_adapted.min():,.0f}"
        )

        # -- 3c. Draw stochastic flood events --------------------------------
        _subheader("3c. Stochastic flood event draw")
        occurred_events, flooded_agents = _simulate_year_events(bridge, year, rng)

        # Compute total annual damage
        total_dmg_year = np.zeros(bridge.n_agents, dtype=np.float32)
        for evt in occurred_events:
            total_dmg_year += bridge.get_current_damages(evt)

        print(
            f"  Occurred events : {len(occurred_events)} of {len(bridge._event_names)}\n"
            f"  Flooded agents  : {flooded_agents.sum():,} of {bridge.n_agents:,}\n"
            f"  Total damage    : ${total_dmg_year.sum():,.0f}"
        )
        total_damage_history[year] = float(total_dmg_year.sum())
        cum_dmg += total_dmg_year
        cum_damage_history[year] = float(cum_dmg.sum())

        if occurred_events:
            print(
                f"  First 5 events  : {[e for e in occurred_events[:5]]}"
            )

        # -- 3d. Update risk perception --------------------------------------
        _subheader("3d. Update flood experience & risk perception")
        bridge.update_flood_experience(flooded_agents)

        rp_flooded = bridge.risk_perception[flooded_agents]
        rp_not_flooded = bridge.risk_perception[~flooded_agents]
        print(
            f"  Risk perception (flooded agents)     — "
            f"mean={rp_flooded.mean():.4f} (n={flooded_agents.sum()})\n"
            f"  Risk perception (non-flooded agents) — "
            f"mean={rp_not_flooded.mean():.4f} (n={(~flooded_agents).sum()})"
        )

        # -- 3e. SEU decision evaluation ------------------------------------
        _subheader("3e. SEU decision evaluation")
        previously_adapted = bridge.is_adapted.sum()
        newly_adapted = bridge.evaluate_decisions(year_index=year - 1)
        now_adapted = bridge.is_adapted.sum()
        new_adapters = int(newly_adapted.sum())

        print(
            f"  Agents adapted BEFORE this year  : {previously_adapted:,}\n"
            f"  Agents that chose to adapt NOW   : {new_adapters:,}\n"
            f"  Agents adapted AFTER this year   : {now_adapted:,}  "
            f"({now_adapted / bridge.n_agents:.2%} of residential)"
        )
        total_adapted_history[year] = int(now_adapted)

        # Track newly adapted agents from this year (first 5)
        newly_adapted_indices = np.where(newly_adapted)[0]
        new_to_track = [idx for idx in newly_adapted_indices if idx not in cohort_labels]
        for idx in new_to_track[:5]:
            cohort_labels[idx] = f"Adapt Y{year}"
            tracked_indices.append(idx)

        # -- 3f. Detailed table of tracked agents ---------------------------
        _subheader(f"3f. Agent state snapshot (tracing cohorts)")
        _print_agent_table(
            bridge,
            agent_indices=tracked_indices,
            extra_cols={
                "dmg_this_yr": total_dmg_year,
                "cum_dmg": cum_dmg,
                "ead_no_meas": ead_no_meas,
            },
            cohort_labels=cohort_labels,
        )

    # -----------------------------------------------------------------------
    # 4.  Summary
    # -----------------------------------------------------------------------
    _header("SUMMARY")
    print(f"  {'Year':>6}  {'SLR (ft)':>10}  {'Total Damage':>15}  {'Cum Damage':>15}  {'Adapted (cum.)':>16}")
    print("  " + "-" * 70)
    for year in DEMO_YEARS:
        slr_ft = slr_trajectory(year)
        print(
            f"  {year:>6d}  {slr_ft:>10.4f}  "
            f"${total_damage_history[year]:>14,.0f}  "
            f"${cum_damage_history[year]:>14,.0f}  "
            f"{total_adapted_history[year]:>16,d}"
        )


    ds.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
