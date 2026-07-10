"""
run_trace_manual_check.py
=========================
Diagnostic tracing script for manual verification of the
DynamoDecisionBridge pipeline.

Design goals
------------
1. Use the REAL lookup table (no synthetic data).
2. Limit to exactly 10 agents -- 5 with highest Expected Annual Damage
   (the ones that WILL be impacted) and 5 with zero or near-zero EAD
   (controls that should NOT adapt).
3. Print EVERY intermediate value at every computation step so that each
   number can be verified by hand or in a spreadsheet.
4. max_events_per_year = 5 for richer stochastic draws.
5. Deterministic: fixed random seed, fully reproducible.

Run
---
    python example/run_trace_manual_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package is importable even without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import xarray as xr

from floodadapt_abm import CouplingConfig, DecisionConfig, NetCDFMappingConfig
from floodadapt_abm import DynamoDecisionBridge

# -- USER-ADJUSTABLE SETTINGS ----------------------------------------------

#: Path to the precomputed FloodAdapt-ABM lookup table.
LOOKUP_TABLE_PATH: Path = Path(
    r"C:\repos\DYNAMO-M"
    r"\lookup_table_charleston_beta_release_ABM_probabilistic_set.nc"
)

RANDOM_SEED: int = 42
INITIAL_YEAR: int = 2020 # this comes from the SLR initial time step used to make the lookup table
TIME_HORIZON: int = 30 # this comes from the SLR final time step used to make the lookup table
DEMO_YEARS: list[int] = [INITIAL_YEAR, 2040, INITIAL_YEAR + TIME_HORIZON]

# Number of high-damage and low-damage agents to trace
N_HIGH: int = 2
N_LOW: int = 2

# Decimal precision for printing monetary values
PRECISION: int = 2



# -- FORMATTING HELPERS -----------------------------------------------------
def _banner(text: str, char: str = "=", width: int = 80) -> None:
    """Print a wide banner line."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _section(text: str) -> None:
    """Print a subsection header."""
    print(f"\n  -- {text} {'-' * max(1, 60 - len(text))}")


def _kv(label: str, value: object, indent: int = 4) -> None:
    """Print a key-value pair."""
    print(f"{' ' * indent}{label:<40s} : {value}")


def _money(v: float) -> str:
    """Format a float as a dollar string."""
    return f"${v:>14,.{PRECISION}f}"


def _pct(v: float) -> str:
    """Format a float as a percentage string."""
    return f"{v:>10.4%}"


def _print_agent_row(
    idx: int,
    label: str,
    bridge: DynamoDecisionBridge,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    """Print a single agent row with all state variables."""
    fields: list[tuple[str, str]] = [
        ("label", f"{label:<14s}"),
        ("obj_id", f"{bridge.object_ids[idx]:<10s}"),
        ("max_pot_dmg", _money(bridge.max_pot_dmg[idx])),
        ("income", _money(bridge.income[idx])),
        ("wealth", _money(bridge.wealth[idx])),
        ("risk_perc", f"{bridge.risk_perception[idx]:>10.6f}"),
        ("flood_tmr", f"{bridge.flood_timer[idx]:>6d}"),
        ("adapted", f"{int(bridge.is_adapted[idx]):>4d}"),
    ]
    if extras:
        for k, arr in extras.items():
            if np.issubdtype(arr.dtype, np.floating):
                fields.append((k, _money(arr[idx])))
            elif np.issubdtype(arr.dtype, np.bool_):
                fields.append((k, f"{int(arr[idx]):>4d}"))
            else:
                fields.append((k, f"{arr[idx]}"))
    print("    " + "  ".join(v for _, v in fields))


def _print_agent_header(extras: dict[str, np.ndarray] | None = None) -> None:
    """Print the header row matching _print_agent_row."""
    cols = [
        f"{'label':<14s}",
        f"{'obj_id':<10s}",
        f"{'max_pot_dmg':>15s}",
        f"{'income':>15s}",
        f"{'wealth':>15s}",
        f"{'risk_perc':>10s}",
        f"{'flood_tmr':>6s}",
        f"{'adapted':>4s}",
    ]
    if extras:
        for k in extras:
            cols.append(f"{k:>15s}")
    header = "    " + "  ".join(cols)
    print(header)
    print("    " + "-" * (len(header) - 4))


def _print_traced_agents(
    indices: list[int],
    labels: dict[int, str],
    bridge: DynamoDecisionBridge,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    """Print the full table of traced agents."""
    _print_agent_header(extras)
    for idx in indices:
        _print_agent_row(idx, labels[idx], bridge, extras)

# -- EVENT DRAWING (deterministic, traceable) -------------------------------
def _draw_events_traced(
    bridge: DynamoDecisionBridge,
    rng: np.random.Generator,
    max_events: int,
) -> tuple[list[str], np.ndarray]:
    """
    Draw stochastic flood events and print every random draw so the user
    can reproduce each Bernoulli trial by hand.

    Parameters
    ----------
    bridge : DynamoDecisionBridge
        Initialised bridge (provides event names and frequencies).
    rng : np.random.Generator
        Seeded RNG for reproducibility.
    max_events : int
        Maximum number of events to retain per year.

    Returns
    -------
    occurred_events : list[str]
        Names of events that occurred this year.
    total_dmg : np.ndarray[float32], shape (n_agents,)
        Total damage suffered by each agent this year.
    """
    event_names: np.ndarray = bridge._event_names
    event_freqs: np.ndarray = bridge._event_freqs

    print(f"    {'Event':<35s} {'Freq':>10s} {'Prob':>10s}"
          f"  {'Draw':>10s}  {'Hit?':>5s}")
    print(f"    {'-' * 75}")

    print("    (Note: Only showing drawn 'Hit' events to save space)")
    # Done: Refactor this stochastic event drawing and capping logic OUT of the example script
    # and integrate it natively into DynamoDecisionBridge or ABMSimulator.
    occurred_list: list[str] = []
    for name, freq in zip(event_names, event_freqs):
        prob = min(float(freq), 1.0)
        draw = float(rng.random())
        hit = draw < prob
        if hit:
            marker = " *** "
            print(f"    {str(name):<35s} {freq:>10.6f} {prob:>10.6f}"
                  f"  {draw:>10.6f}  {marker}")
            occurred_list.append(str(name))

    # Apply cap
    if len(occurred_list) > max_events:
        print(f"\n    Cap applied: {len(occurred_list)} events drawn "
              f"-> randomly sampling {max_events} events to preserve unbiased distribution")
        indices = rng.choice(len(occurred_list), size=max_events, replace=False)
        occurred_list = [occurred_list[i] for i in indices]

    occurred = occurred_list
    
    # Create a nice string showing event and frequency
    event_with_freq_strs = []
    for evt in occurred:
        evt_idx = list(event_names).index(evt)
        freq = float(event_freqs[evt_idx])
        event_with_freq_strs.append(f"'{evt}' (freq: {freq:.5f})")
        
    print(f"\n    Final events this year ({len(occurred)}): [{', '.join(event_with_freq_strs)}]")

    # Compute per-agent damage
    total_dmg = np.zeros(bridge.n_agents, dtype=np.float32)
    for evt in occurred:
        evt_dmg = bridge.get_current_damages(evt)
        total_dmg += evt_dmg
    return occurred, total_dmg



# -- MAIN -------------------------------------------------------------------
def run() -> None:
    """Execute the manual-check tracing simulation."""

    
    # STEP 0: Configuration

    _banner("STEP 0 -- CONFIGURATION")
    cfg = CouplingConfig(
        netcdf=NetCDFMappingConfig(
            residential_substring="RES",
            strategy_no_measures="no_measures",
            strategy_floodproof="floodproof_all_0",
        ),
        decision=DecisionConfig(
            risk_aversion=1.0,
            discount_rate=0.032,
            decision_horizon=15,
            risk_perc_min=0.01,
            risk_perc_max=2.0,
            risk_perc_coef=-3.6,
            loan_duration=16,
            interest_rate=0.04,
            adaptation_cost_fraction=0.10,
            expenditure_cap=0.06,
            amenity_weight=1.0,
            error_interval=0.0,
            income_to_wealth_ratio=4.14,
            max_events_per_year=5,
        ),
        random_seed=RANDOM_SEED,
    )
    _kv("risk_aversion (sigma)", cfg.decision.risk_aversion)
    _kv("discount_rate (r)", cfg.decision.discount_rate)
    _kv("decision_horizon (T)", cfg.decision.decision_horizon)
    _kv("risk_perc_min", cfg.decision.risk_perc_min)
    _kv("risk_perc_max", cfg.decision.risk_perc_max)
    _kv("risk_perc_coef", cfg.decision.risk_perc_coef)
    _kv("loan_duration", cfg.decision.loan_duration)
    _kv("interest_rate", cfg.decision.interest_rate)
    _kv("adaptation_cost_fraction", cfg.decision.adaptation_cost_fraction)
    _kv("expenditure_cap", cfg.decision.expenditure_cap)
    _kv("income_to_wealth_ratio", cfg.decision.income_to_wealth_ratio)
    _kv("max_events_per_year", cfg.decision.max_events_per_year)
    _kv("random_seed", cfg.random_seed)

    
    # STEP 1: Load dataset
    
    _banner("STEP 1 -- LOAD DATASET")

    if not LOOKUP_TABLE_PATH.exists():
        print(f"  ERROR: file not found at {LOOKUP_TABLE_PATH}")
        sys.exit(1)
    print(f"  File: {LOOKUP_TABLE_PATH.name}")

    ds: xr.Dataset = xr.open_dataset(str(LOOKUP_TABLE_PATH))
    _kv("Dimensions", dict(ds.sizes))
    _kv("Strategies", list(ds.strategy.values))
    _kv("SLR grid points", list(ds.slr.values))
    _kv("Num events", len(ds.event))
    _kv("Num buildings (all)", len(ds.object_id))

    # Show event catalog with frequencies
    _section("Event catalog")
    event_names = ds.event.values
    event_freqs = np.asarray(ds.event.attrs["freq"], dtype=np.float64)
    print(f"    {'Event':<35s} {'Frequency':>12s}  {'Return Period':>14s}")
    print(f"    {'-' * 65}")
    for name, freq in zip(event_names, event_freqs):
        rp_str = f"{1.0 / freq:.1f} yr" if freq > 0 else "∞"
        print(f"    {str(name):<35s} {freq:>12.6f}  {rp_str:>14s}")

    
    # STEP 2: Initialise bridge (all agents)

    _banner("STEP 2 -- INITIALISE BRIDGE (full population)")
    bridge = DynamoDecisionBridge(ds=ds, config=cfg)
    _kv("Total buildings", len(ds.object_id))
    _kv("Residential agents (RES)", bridge.n_agents)

    
    # STEP 3: Select 10 agents to trace

    _banner("STEP 3 -- SELECT 10 AGENTS TO TRACE")

    # Compute initial EAD at SLR=0 to rank agents
    bridge.prepare_damage_arrays(slr_value=0.0, interp_method="linear")
    ead_init = bridge.compute_expected_annual_damages(use_adapted_strategy=False)

    # Top N_HIGH by EAD
    sorted_desc = np.argsort(ead_init)[::-1]
    high_idx = list(sorted_desc[:N_HIGH])

    # Bottom N_LOW by EAD (pick those with EAD == 0)
    zero_mask = ead_init == 0.0
    zero_indices = np.where(zero_mask)[0]
    if len(zero_indices) >= N_LOW:
        low_idx = list(zero_indices[:N_LOW])
    else:
        # Fall back to lowest-EAD agents
        sorted_asc = np.argsort(ead_init)
        low_idx = list(sorted_asc[:N_LOW])

    traced = high_idx + low_idx
    labels: dict[int, str] = {}
    for i, idx in enumerate(high_idx):
        labels[idx] = f"HIGH_{i}"
    for i, idx in enumerate(low_idx):
        labels[idx] = f"LOW_{i}"

    print(f"\n  Selected agents ({len(traced)}):")
    print(f"  High-damage (top {N_HIGH} by EAD at SLR=0): {high_idx}")
    print(f"  Low-damage  (EAD ~= 0 at SLR=0):            {low_idx}")

    
    # STEP 4: Print initial state of traced agents
    
    _banner("STEP 4 -- INITIAL STATE OF TRACED AGENTS")

    _section("Per-agent economic variables (derived from lookup table)")
    for idx in traced:
        label = labels[idx]
        _kv(f"[{label}] object_id", bridge.object_ids[idx], indent=6)
        _kv(f"[{label}] max_pot_dmg", f"${bridge.max_pot_dmg[idx]:,.2f}", indent=6)
        _kv(
            f"[{label}] income = max_pot_dmg / income_to_wealth_ratio",
            f"${bridge.max_pot_dmg[idx]:,.2f} / {cfg.decision.income_to_wealth_ratio}"
            f" = ${bridge.income[idx]:,.2f}",
            indent=6,
        )
        _kv(
            f"[{label}] wealth = income x income_to_wealth_ratio",
            f"${bridge.income[idx]:,.2f} x {cfg.decision.income_to_wealth_ratio}"
            f" = ${bridge.wealth[idx]:,.2f}",
            indent=6,
        )
        adapt_cost_total = cfg.decision.adaptation_cost_fraction * bridge.max_pot_dmg[idx]
        _kv(
            f"[{label}] adapt_cost_total",
            f"{cfg.decision.adaptation_cost_fraction:.0%} x "
            f"${bridge.max_pot_dmg[idx]:,.2f} = ${adapt_cost_total:,.2f}",
            indent=6,
        )
        _kv(
            f"[{label}] annual_adapt_cost",
            f"${bridge._annual_adapt_cost[idx]:,.2f}",
            indent=6,
        )
        expenditure_limit = bridge.income[idx] * cfg.decision.expenditure_cap
        can_afford = expenditure_limit > bridge._annual_adapt_cost[idx]
        _kv(
            f"[{label}] can_afford?",
            f"income * expenditure_cap = ${bridge.income[idx]:,.2f} * "
            f"{cfg.decision.expenditure_cap} = ${expenditure_limit:,.2f} "
            f"{'>' if can_afford else '<='} ${bridge._annual_adapt_cost[idx]:,.2f}"
            f" -> {'YES' if can_afford else 'NO (constrained)'},",
            indent=6,
        )
        _kv(f"[{label}] EAD (SLR=0)", f"${ead_init[idx]:,.2f}", indent=6)
        print()

    _section("Initial state table")
    _print_traced_agents(
        traced, labels, bridge,
        extras={"ead_slr0": ead_init},
    )

    
    # STEP 5: Print damage matrix at SLR=0 for each traced agent
    
    _banner("STEP 5 - DAMAGE LOOKUP (per event, per strategy) at SLR=0.0")

    dmg_no = bridge._damage_no_measures   # shape: (n_agents, n_events)
    dmg_fp = bridge._damage_floodproof

    print("  (Note: Detailed per-event damage is shown below for ONE example agent only)")
    
    for idx in traced[:1]:  # Only show ONE agent as an example
        label = labels[idx]
        print(f"\n  [{label}] object_id={bridge.object_ids[idx]}"
              f"  max_pot_dmg=${bridge.max_pot_dmg[idx]:,.2f}")
        print(f"    {'Event':<35s} {'no_measures':>14s} {'floodproof':>14s}"
              f" {'reduction':>14s}")
        print(f"    {'-' * 80}")
        for j, ename in enumerate(event_names):
            d_no = dmg_no[idx, j]
            d_fp = dmg_fp[idx, j]
            reduction = d_no - d_fp
            print(f"    {str(ename):<35s} {_money(d_no)} {_money(d_fp)}"
                  f" {_money(reduction)}")

        ead_no = float((dmg_no[idx, :] * event_freqs).sum())
        ead_fp = float((dmg_fp[idx, :] * event_freqs).sum())
        
        # EXPLANATION OF EAD DIFFERENCES:
        # The reason the two EAD numbers are different is because they are evaluated at different SLR levels:
        # At SLR = 0.0 ft (Baseline): The EAD for HIGH_0 is $29,041,237 (and fp is $111,323).
        # At SLR = 0.1333 ft (Year 2022): The sea level has risen slightly, so the EAD for HIGH_0 interpolates upward to $36,094,240 (and fp rises to $300,265).
        # But the underlying truth is mathematically sound: the Expected Annual Damage (EAD) increases severely as the sea level rises because the identical storm events cause significantly deeper inundation.
        print(f"    {'EAD (sum dmg*freq)':<35s} {_money(ead_no)} {_money(ead_fp)}"
              f" {_money(ead_no - ead_fp)}")

    
    # STEP 6: SLR trajectory
    
    _banner("STEP 6 -- SLR TRAJECTORY")
    slr_levels = ds.slr.values
    max_slr = float(np.max(slr_levels))
    _kv("Time period (projection)", f"{INITIAL_YEAR} to {INITIAL_YEAR + TIME_HORIZON}")
    _kv("SLR grid from lookup table", list(slr_levels))
    _kv("Max SLR", max_slr)

    def slr_trajectory(year: int) -> float:
        """Linear SLR projection mapping INITIAL_YEAR to 0."""
        clamped = min(max(year, INITIAL_YEAR), INITIAL_YEAR + TIME_HORIZON)
        fraction = (clamped - INITIAL_YEAR) / TIME_HORIZON
        return round(fraction * max_slr, 4)

    print(f"\n    {'Year':>6s}  {'SLR (ft)':>10s}")
    print(f"    {'-' * 20}")
    for yr in DEMO_YEARS:
        print(f"    {yr:>6d}  {slr_trajectory(yr):>10.4f}")

    
    # STEP 7: Year-by-year simulation with full tracing
    
    rng = np.random.default_rng(RANDOM_SEED + 1)
    
    # Track cumulative damage over the entire simulation
    cum_dmg = np.zeros(bridge.n_agents, dtype=np.float32)

    yearly_summary: list[dict] = []

    for year in DEMO_YEARS:

        _banner(f"YEAR {year}", char="#", width=80)

        # -- 7a. SLR interpolation ------------------------------------------
        slr_ft = slr_trajectory(year)
        _section(f"7a. Prepare damage arrays at SLR = {slr_ft:.4f} ft")
        bridge.prepare_damage_arrays(slr_value=slr_ft, interp_method="linear")

        # Print interpolated damage for traced agents
        dmg_no = bridge._damage_no_measures

        for idx in traced:
            label = labels[idx]
            top_events = np.argsort(dmg_no[idx])[::-1][:3]  # top 3 events
            top_str = ", ".join(
                f"{event_names[j]}={_money(dmg_no[idx, j]).strip()}"
                for j in top_events
            )
            print(f"    [{label}] top3 no_measures: {top_str}")

        # EAD at this SLR
        ead_no = bridge.compute_expected_annual_damages(use_adapted_strategy=False)
        ead_fp = bridge.compute_expected_annual_damages(use_adapted_strategy=True)

        _section("7a. EAD comparison at this SLR")
        print(f"    {'Label':<14s} {'EAD(no_meas)':>15s} {'EAD(fp)':>15s}"
              f" {'Avoided':>15s}")
        print(f"    {'-' * 62}")
        for idx in traced:
            avoided = ead_no[idx] - ead_fp[idx]
            print(f"    {labels[idx]:<14s} {_money(ead_no[idx])}"
                  f" {_money(ead_fp[idx])} {_money(avoided)}")

        # -- 7b. Draw stochastic events -------------------------------------
        _section("7b. Stochastic event draw (Bernoulli trials)")
        occurred, total_dmg = _draw_events_traced(
            bridge, rng, cfg.decision.max_events_per_year,
        )
        cum_dmg += total_dmg

        # Print per-agent damage for traced agents
        _section("7b. Realised damage per traced agent")
        print(f"    {'Label':<14s} {'dmg_this_yr':>15s} {'adapted?':>10s}"
              f" {'strategy':>15s}")
        print(f"    {'-' * 57}")
        for idx in traced:
            strat = "floodproof" if bridge.is_adapted[idx] else "no_measures"
            print(f"    {labels[idx]:<14s} {_money(total_dmg[idx])}"
                  f" {int(bridge.is_adapted[idx]):>10d}"
                  f" {strat:>15s}")

        # Show per-event breakdown for high-damage agents
        if occurred:
            _section("7b. Per-event damage breakdown (HIGH agents only)")
            print(f"    {'Label':<14s}", end="")
            for evt in occurred:
                print(f" {evt:>20s}", end="")
            print(f" {'TOTAL':>15s}")
            print(f"    {'-' * (14 + 21 * len(occurred) + 16)}")
            for idx in high_idx:
                print(f"    {labels[idx]:<14s}", end="")
                row_total = 0.0
                for evt in occurred:
                    d = float(bridge.get_current_damages(evt)[idx])
                    row_total += d
                    print(f" {_money(d)}", end="")
                print(f" {_money(row_total)}")

        # -- 7c. Flood experience and risk perception update ----------------
        flooded = total_dmg > 0
        _section("7c. Update flood experience & risk perception")
        _kv("Flooded agents (total)", int(flooded.sum()))
        _kv("Flooded among traced", int(sum(flooded[i] for i in traced)))

        # Print BEFORE update
        print("\n    Risk perception BEFORE update:")
        for idx in traced:
            print(f"      [{labels[idx]}] rp={bridge.risk_perception[idx]:.6f}"
                  f"  flood_timer={bridge.flood_timer[idx]}"
                  f"  flooded_now={int(flooded[idx])}")

        bridge.update_flood_experience(flooded)

        # Print AFTER update with formula
        print("\n    Risk perception AFTER update "
              "(formula: rp = rp_max x 1.6^(coef x timer) + rp_min):")
        for idx in traced:
            timer = bridge.flood_timer[idx]
            expected_rp = (
                cfg.decision.risk_perc_max
                * (1.6 ** (cfg.decision.risk_perc_coef * timer))
                + cfg.decision.risk_perc_min
            )
            print(
                f"      [{labels[idx]}] timer={timer:>3d}"
                f"  -> rp = {cfg.decision.risk_perc_max} x 1.6^"
                f"({cfg.decision.risk_perc_coef} x {timer})"
                f" + {cfg.decision.risk_perc_min}"
                f" = {expected_rp:.6f}"
                f"  (actual: {bridge.risk_perception[idx]:.6f})"
            )

        # -- 7d. SEU decision evaluation ------------------------------------
        _section("7d. SEU decision evaluation")
        prev_adapted = bridge.is_adapted.copy()
        newly_adapted = bridge.evaluate_decisions(year_index=year - 1)
        now_adapted = bridge.is_adapted.copy()

        eu_dn = getattr(bridge, "_eu_do_nothing", None)
        eu_ad = getattr(bridge, "_eu_adapt", None)

        if eu_dn is not None and eu_ad is not None:
            # Find an agent who was flooded and adapted
            flooded_mask = (total_dmg > 0)
            adapted_mask = newly_adapted
            
            flooded_and_adapted = np.where(flooded_mask & adapted_mask)[0]
            # Must also exclude agents who were already adapted previously
            flooded_not_adapted = np.where(flooded_mask & ~adapted_mask & ~prev_adapted)[0]
            
            def _print_agent_seu_trace(a_idx: int, lbl: str, action: str, suffix: str):
                w = float(bridge.wealth[a_idx])
                inc = float(bridge.income[a_idx])
                amen = float(bridge.amenity_value[a_idx])
                rp = float(bridge.risk_perception[a_idx])
                ead_no_a = float((bridge._damage_no_measures[a_idx] * event_freqs).sum())
                ead_fp_a = float((bridge._damage_floodproof[a_idx] * event_freqs).sum())
                cost = float(bridge._annual_adapt_cost[a_idx])
                
                print(f"\n    --- SEU TRACE: Impacted agent who {action} ({lbl}) ---")
                print(f"    State Inputs:")
                print(f"      Wealth          : {_money(w)}")
                print(f"      Income          : {_money(inc)}")
                print(f"      Amenity         : {_money(amen)}")
                print(f"      Risk Perc.      : {rp:.4f}")
                print(f"      EAD (no_meas)   : {_money(ead_no_a)}")
                print(f"      EAD (floodproof): {_money(ead_fp_a)}")
                print(f"      Adapt Cost/yr   : {_money(cost)}")
                print(f"    Results:")
                print(f"      Damage this yr  : {_money(float(total_dmg[a_idx]))}")
                print(f"      EU(do_nothing)  : {eu_dn[a_idx]:,.6f}")
                print(f"      EU(adapt)       : {eu_ad[a_idx]:,.6f}")
                print(f"      Difference      : {eu_ad[a_idx] - eu_dn[a_idx]:,.6f}  ({suffix})")

            if len(flooded_and_adapted) > 0:
                a_idx = flooded_and_adapted[0]
                lbl = labels.get(a_idx, f"agent_{a_idx}")
                _print_agent_seu_trace(a_idx, lbl, "ADAPTED", "> 0, so adapted")
                
            if len(flooded_not_adapted) > 0:
                a_idx = flooded_not_adapted[0]
                lbl = labels.get(a_idx, f"agent_{a_idx}")
                _print_agent_seu_trace(a_idx, lbl, "did NOT adapt", "<= 0, so did not adapt")

        # Print decision details per traced agent
        print(f"\n    {'Label':<14s} {'was_adapted':>12s} {'newly':>8s}"
              f" {'now_adapted':>12s}")
        print(f"    {'-' * 50}")
        for idx in traced:
            was = int(prev_adapted[idx])
            new = int(newly_adapted[idx])
            now = int(now_adapted[idx])
            marker = " <-- ADAPTED!" if new else ""
            print(f"    {labels[idx]:<14s} {was:>12d} {new:>8d}"
                  f" {now:>12d}{marker}")

        total_adapted_now = int(bridge.is_adapted.sum())
        total_new = int(newly_adapted.sum())
        _kv("Total newly adapted this year", total_new)
        _kv("Total adapted cumulative", total_adapted_now)
        _kv("Fraction adapted", f"{total_adapted_now / bridge.n_agents:.4%}")

        # Dynamically add one newly adapted agent to the trace monitor
        if total_new > 0:
            newly_adapted_indices = np.where(newly_adapted)[0]
            for idx in newly_adapted_indices:
                if idx not in traced:
                    traced.append(idx)
                    labels[idx] = f"ADAPT_Y{year}"
                    print(f"\n    [+] Dynamically adding agent {idx} ('ADAPT_Y{year}') to trace monitor")
                    break

        # Update labels for newly adapted agents
        for idx in traced:
            if newly_adapted[idx] and "HIGH" in labels[idx]:
                labels[idx] = labels[idx] + "->Y" + str(year)

        # -- 7e. Full state snapshot ----------------------------------------
        _section("7e. Full state snapshot after year")
        _print_traced_agents(
            traced, labels, bridge,
            extras={
                "dmg_this_yr": total_dmg,
                "cum_dmg": cum_dmg,
                "ead_no_meas": ead_no,
                "ead_fp": ead_fp,
            },
        )

        # Store summary
        yearly_summary.append({
            "year": year,
            "slr": slr_ft,
            "events_drawn": len(occurred),
            "event_names": occurred,
            "total_damage_all": float(total_dmg.sum()),
            "cum_dmg_all": float(cum_dmg.sum()),
            "total_damage_traced": float(sum(total_dmg[i] for i in traced)),
            "newly_adapted": total_new,
            "cum_adapted": total_adapted_now,
        })

    
    # STEP 8: Summary
    
    _banner("SIMULATION SUMMARY")

    print(f"\n  {'Year':>6s}  {'SLR':>8s}  {'Events':>6s}"
          f"  {'Total Dmg (all)':>18s}  {'Cum Dmg (all)':>18s}  {'Total Dmg (10)':>18s}"
          f"  {'New Adapt':>10s}  {'Cum Adapt':>10s}")
    print(f"  {'-' * 102}")
    for s in yearly_summary:
        print(
            f"  {s['year']:>6d}  {s['slr']:>8.4f}  {s['events_drawn']:>6d}"
            f"  ${s['total_damage_all']:>16,.0f}"
            f"  ${s['cum_dmg_all']:>16,.0f}"
            f"  ${s['total_damage_traced']:>16,.0f}"
            f"  {s['newly_adapted']:>10d}"
            f"  {s['cum_adapted']:>10d}"
        )

    print(f"\n  Events per year:")
    for s in yearly_summary:
        print(f"    {s['year']}: {s['event_names']}")

    # Final state
    _section("Final agent state")
    ead_final = bridge.compute_expected_annual_damages(use_adapted_strategy=False)
    _print_traced_agents(
        traced, labels, bridge,
        extras={"ead_final": ead_final},
    )

    ds.close()
    print("\n  [OK] Trace complete. All intermediate values printed above.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
