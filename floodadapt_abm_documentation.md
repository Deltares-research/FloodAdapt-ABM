# FloodAdapt-ABM — Whole-Repo Reference Documentation

**Scope:** the single API-level reference for the `floodadapt_abm` package — every
module, public class, dataclass and function, with signatures, responsibilities,
assumptions and runnable code examples. It also maps the examples/tests/verification
layout and the module dependency graph.

**Companion docs (read alongside this one):**

- [`README.md`](README.md) — repo front door + quickstart.
- [`docs/architecture.md`](docs/architecture.md) — the *why* and the *design*: MVP
  scope, the Strategy-Pattern architecture, the full SEU mathematics, UML/sequence/
  data-flow diagrams, every delivered phase (0 → 4b-full), the roadmap and the
  day-by-day progress log. **This file is the *what* (API); that file is the *why*.**
- [`docs/AGENTS.md`](docs/AGENTS.md) — operational guide (data requirements, NetCDF
  schema, gotchas, engine-performance contract).
- [`docs/adaptation_decisions_complete.md`](docs/adaptation_decisions_complete.md) —
  the DYNAMO-M decision-science reference (source of the ported SEU logic).

**Status:** implementation complete and gated through **Phase 4b-full** (native-class
integration) + the PRE.2 real-table gate; full `pytest` suite **147 tests, all pass**.

---

## Table of contents

1. [Package overview & public API](#1-package-overview--public-api)
2. [Two-stage pipeline & module dependency map](#2-two-stage-pipeline--module-dependency-map)
3. [Configuration — `coupling_config.py`](#3-configuration--coupling_configpy)
4. [Agent state — `agent_state.py`](#4-agent-state--agent_statepy)
5. [Decision rules — `decision_rule.py`](#5-decision-rules--decision_rulepy)
6. [Stochastic events — `event_utils.py`](#6-stochastic-events--event_utilspy)
7. [Simulation engine — `simulation_engine.py`](#7-simulation-engine--simulation_enginepy)
8. [Live parity rule — `dynamo_live_rule.py` (Phase 4a)](#8-live-parity-rule--dynamo_live_rulepy-phase-4a)
9. [Mesa-native scaffold — `mesa_native.py` (Phase 4b)](#9-mesa-native-scaffold--mesa_nativepy-phase-4b)
10. [Native-class integration — `mesa_native_full.py` (Phase 4b-full)](#10-native-class-integration--mesa_native_fullpy-phase-4b-full)
11. [Lookup-table adapter — `coastal_node_adapter.py`](#11-lookup-table-adapter--coastal_node_adapterpy)
12. [Ported kernels — `_core/`](#12-ported-kernels--_core)
13. [Stage-1 pipeline — `setup_lookup_table.py`](#13-stage-1-pipeline--setup_lookup_tablepy)
14. [Legacy simulator — `abm_simulator.py`](#14-legacy-simulator--abm_simulatorpy)
15. [Examples, tests & verification](#15-examples-tests--verification)
16. [Global assumptions & invariants](#16-global-assumptions--invariants)

---

## 1. Package overview & public API

`floodadapt_abm` is an agent-based flood-adaptation simulator whose household
decision logic is the DYNAMO-M **Subjective Expected Utility (SEU)** model. Buildings
("agents") each year decide whether to **dry-floodproof** based on perceived flood
risk, expected damages, income/wealth and the cost of adaptation.

Everything importable from the top-level package (`floodadapt_abm/__init__.py`):

| Symbol | Kind | Summary |
|---|---|---|
| `SimulationEngine` | class | **Recommended entry point.** Owns time, data plumbing, event generation, the lifespan reset, and a pluggable `DecisionRule`. |
| `DecisionRule` | ABC | Strategy interface — subclass to define new adaptation logic. |
| `ThresholdRule` | class | Legacy reactive heuristic (adapt when damage ≥ threshold). |
| `SEURule` | class | The MVP science: ported DYNAMO-M SEU decision rule. |
| `AgentState` | dataclass | Vectorised per-agent state arrays. |
| `CouplingConfig` / `DecisionConfig` / `NetCDFMappingConfig` | dataclasses | Configuration. |
| `draw_year_events` / `generate_event_sequences` | funcs | Unified stochastic event generator. |
| `DynamoLiveRule` / `DynamoMNotAvailable` / `DYNAMO_M_AVAILABLE` | class/exc/flag | Phase 4a live-parity rule (guarded native DYNAMO-M import). |
| `FloodAdaptSLRModel` / `CoastalNodePopulation` / `MesaAgents` / `run_mesa_native` | classes/func | Phase 4b Mesa-native scaffold (framework-free time-ownership inversion). |
| `FloodAdaptSLRModelFull` / `CoastalNodePopulationFull` / `AgentsFull` / `run_mesa_native_full` / `HoneybeesNotAvailable` / `HONEYBEES_AVAILABLE` | classes/func/exc/flag | Phase 4b-full native-class integration (real honeybees `Model`). |
| `ABMSimulator` | class | **Deprecated** legacy stage-2 simulator (kept for the Gate-1 regression). |
| `DynamoDecisionBridge` | class | Internal `_core` plumbing, re-exported for backward compat. |

`setup_lookup_table` is intentionally **not** imported by `__init__.py` (it needs the
full `flood-adapt` library, absent in some envs). Import it explicitly:

```python
from floodadapt_abm.setup_lookup_table import create_lookup_table
```

**Minimal end-to-end run:**

```python
import xarray as xr
from floodadapt_abm import SimulationEngine, SEURule, CouplingConfig

ds = xr.open_dataset("lookup_table_charleston_beta_release_ABM_probabilistic_set.nc")
cfg = CouplingConfig()                      # all defaults (Charleston-calibrated)
rule = SEURule(cfg.decision, rng=None)      # the DYNAMO-M SEU science
engine = SimulationEngine(ds, decision_rule=rule, config=cfg)

slr_values = [0.0, 0.1, 0.2, 0.3]           # SLR (feet) per simulated year
result = engine.run(slr_values, no_seq=100, seed=42, n_jobs=4)
# result -> dict of stacked per-year/per-sequence arrays (damages, adopted, ...)
```

---

## 2. Two-stage pipeline & module dependency map

The coupling is a **two-stage pipeline**:

- **Stage 1 — build the lookup table** (`setup_lookup_table.py`, needs `flood-adapt`):
  runs FloodAdapt hazard/impact scenarios once and bakes a
  `(object_id × event × slr × strategy)` **NetCDF damage cube**. Produced offline.
- **Stage 2 — the ABM** (everything else, needs only NumPy/xarray/SciPy): reads the
  cube and runs the Monte-Carlo adaptation simulation. `flood-adapt` is **not** a
  runtime dependency of stage 2.

```
                 SimulationEngine  ── owns time, data, events, lifespan reset
                    │  holds a
                    ▼
                 DecisionRule (ABC)
        ┌───────────┼───────────────┬──────────────────┐
   ThresholdRule  SEURule       DynamoLiveRule (4a)   (your rule)
                    │                │ delegates to
                    │ uses           ▼
                    │           native DYNAMO-M DecisionModule (guarded)
                    ▼
   _core/dynamo_decision_bridge.py  ── ported SEU math + per-SLR interp cache
                    │ uses
                    ▼
   _core/lookup_utils.py            ── SLR→damage interpolation kernel
                    ▲
                    │ reads
   setup_lookup_table.py (stage 1)  ── writes the NetCDF cube

Time drivers over the SAME engine kernel (bit-for-bit equivalent):
   engine.run(...)                       (engine owns time)
   run_mesa_native(engine, ...)          (4b scaffold owns time)
   run_mesa_native_full(engine, ...)     (4b-full: real honeybees Model owns time)
        │ per tick uses
        ▼
   coastal_node_adapter.LookupTableAdapter  ── lookup-table ↔ CoastalNode arrays

   agent_state.AgentState  ── the vectorised state every rule receives
   event_utils             ── the single stochastic event generator
   coupling_config         ── dataclasses consumed everywhere
```

**Key invariant:** `engine.run`, `run_mesa_native` and `run_mesa_native_full` all
delegate every numeric per-year operation to the same `SimulationEngine.step` kernel
with the same RNG stream, so they are **bit-for-bit identical**. The time drivers only
differ in *who owns the clock*.

---

## 3. Configuration — `coupling_config.py`

Three frozen-by-convention `@dataclass`es. All defaults are calibrated to the
Charleston probabilistic table and DYNAMO-M `settings.yml`; override fields at
construction to retarget.

### `NetCDFMappingConfig`
Maps logical names to the dataset's dimension/variable/attribute names — the *only*
thing to change if the lookup-table schema changes.

| Field | Default | Meaning |
|---|---|---|
| `dimension_object_id` | `"object_id"` | building dimension |
| `dimension_event` | `"event"` | event dimension |
| `dimension_slr` | `"slr"` | sea-level-rise dimension |
| `dimension_strategy` | `"strategy"` | strategy dimension |
| `var_total_damage` | `"total_damage"` | total-damage variable |
| `var_inun_depth` | `"inun_depth"` | inundation-depth variable |
| `attr_max_pot_dmg` | `"max_pot_dmg"` | max potential damage (on `object_id`) |
| `attr_event_freq` | `"freq"` | event frequency (on `event`) |
| `attr_building_type` | `"primary_object_type"` | type list (on `object_id`) |
| `residential_substring` | `"RES"` | residential filter (matches `RES`, `COM_RES`, …) |
| `strategy_no_measures` | `"no_measures"` | baseline strategy label |
| `strategy_floodproof` | `"floodproof_all_0"` | adapted strategy label |

### `DecisionConfig`
SEU behavioural parameters. **Docstring defaults are authoritative** and match
DYNAMO-M `settings.yml`:

| Field | Default | Meaning |
|---|---|---|
| `risk_aversion` (σ) | `1.0` | CRRA coefficient; `1.0` → log-utility |
| `discount_rate` (r) | `0.032` | annual NPV discount rate |
| `decision_horizon` (T) | `15` | planning horizon (years) |
| `risk_perc_min` | `0.01` | risk-perception floor |
| `risk_perc_max` | `2.0` | risk-perception ceiling (post-flood) |
| `risk_perc_coef` | `-3.6` | exponential decay coefficient |
| `loan_duration` | `16` | adaptation loan term (years) |
| `interest_rate` | `0.04` | loan interest rate |
| `adaptation_cost_fraction` | `0.10` | fallback adapt cost as fraction of `max_pot_dmg` |
| `expenditure_cap` | `0.06` | max fraction of income spendable on adaptation |
| `amenity_weight` | `1.0` | weight on amenity value in NPV |
| `error_interval` | `0.0` | half-width of uniform EU error (0 → deterministic) |
| `income_to_wealth_ratio` | `4.14` | income→wealth multiplier when wealth absent |
| `max_events_per_year` | `4` | cap on stochastic events per year (see Sec.6) |
| `lifespan_dryproof` | `75` | dry-floodproofing service life (years); triggers reset |

Risk-perception law:
`risk_perc = risk_perc_max · 1.6^(risk_perc_coef · flood_timer) + risk_perc_min`.

### `CouplingConfig`
Container: `netcdf: NetCDFMappingConfig`, `decision: DecisionConfig`,
`random_seed: int = 42`.

```python
from floodadapt_abm import CouplingConfig
cfg = CouplingConfig()
cfg.netcdf.residential_substring = "COM"   # target commercial buildings
cfg.decision.risk_aversion = 2.0            # more risk-averse households
```

---

## 4. Agent state — `agent_state.py`

### `AgentState` (dataclass)
Vectorised per-agent state; every array has shape `(n_agents,)`. This is the single
container passed to each `DecisionRule.should_adapt`.

| Field | dtype | Meaning |
|---|---|---|
| `wealth` | float32 | household wealth |
| `income` | float32 | annual income |
| `risk_perception` | float32 | subjective risk multiplier |
| `flood_timer` | int32 | years since last flood (decays risk perception) |
| `is_adapted` | bool | current dry-floodproofing status |
| `time_adapted` | int32 | age of the current adaptation (drives the lifespan reset) |

**API:** `n_agents` (property), `AgentState.initial(n_agents, income, wealth,
risk_perc_min, initial_flood_timer=99)` (classmethod; all agents start un-adapted with
`flood_timer=99` so initial risk perception sits at the floor), and `copy()` (deep).

```python
import numpy as np
from floodadapt_abm import AgentState
st = AgentState.initial(3, income=np.array([40e3, 55e3, 70e3]),
                        wealth=np.array([160e3, 220e3, 300e3]), risk_perc_min=0.01)
```

---

## 5. Decision rules — `decision_rule.py`

The Strategy Pattern seam. All rules implement one method:

```python
should_adapt(agent_state, damages_this_year, damages_no_adapt, damages_adapt,
             event_freqs, max_pot_dmg, adaptation_costs) -> np.ndarray[bool]
```

which returns a boolean mask of **currently non-adapted** agents that newly adapt this
year. Adaptation is irreversible within a year and never double-applied.

### `DecisionRule(ABC)`
- `__init__(config)` — stores a `DecisionConfig`.
- `clone(rng_seed=None)` — independent copy for parallel execution (deep-copies RNG
  state); overridden by stochastic rules.
- `@abstractmethod should_adapt(...)`.

### `ThresholdRule(DecisionRule)`
Legacy reactive heuristic (the behaviour the coupling replaces): adapt when this
year's realised damage exceeds `damage_threshold` (default `0.3`) of max potential
damage. Deterministic; used as the bit-for-bit regression oracle.

`ThresholdRule(config, damage_threshold=0.3)`.

### `SEURule(DecisionRule)`
The MVP science. Computes `EU_do_nothing` vs `EU_adapt` (CRRA utility over
time-discounted NPVs, integrated over perceived flood probability) and adapts agents
for whom adapting has higher subjective expected utility. Uses the ported kernels in
`_core`.

`SEURule(config, rng=None, amenity_value=None)` — pass an RNG for stochastic error
terms (`error_interval > 0`); `clone()` forks the RNG for parallel sequences.

```python
from floodadapt_abm import SEURule, ThresholdRule, CouplingConfig
cfg = CouplingConfig()
seu = SEURule(cfg.decision, rng=None)
legacy = ThresholdRule(cfg.decision, damage_threshold=0.3)
```

---

## 6. Stochastic events — `event_utils.py`

The **single** stochastic event generator (consolidated out of the example scripts in
Phase 2).

- `draw_year_events(event_names, event_freqs, rng, max_events_per_year=None, dt=1.0)`
  — draw the events occurring in one year. Each event is an independent Bernoulli
  trial with `p = 1 - exp(-freq·dt)`. When the draw exceeds the cap, events are
  selected **at random without replacement** from the drawn pool (preserves the
  Monte-Carlo distribution — the agreed policy that supersedes the old
  highest-frequency / highest-magnitude heuristics).
- `generate_event_sequences(event_names, event_freqs, n_seq, n_years, rng,
  max_events_per_year=None, dt=1.0)` — `n_seq` independent per-year event sequences.

```python
import numpy as np
from floodadapt_abm import draw_year_events
rng = np.random.default_rng(42)
occurred = draw_year_events(["e0", "e1", "e2"], np.array([0.1, 0.02, 0.2]),
                            rng, max_events_per_year=4)
```

---

## 7. Simulation engine — `simulation_engine.py`

### `SimulationEngine`
The recommended entry point; owns time, data, event generation, the lifespan reset and
a pluggable `DecisionRule`. Wraps a `DynamoDecisionBridge` for the damage plumbing.

`SimulationEngine(ds, decision_rule=None, config=None, income_per_agent=None,
amenity_value_per_agent=None, damage_dtype=np.float32)`.

| Method | Purpose |
|---|---|
| `draw_year_events(rng, dt=1.0)` | unified per-year event draw (delegates to `event_utils`) |
| `prepare_damages(slr_value, interp_method='linear')` | interpolate per-event damage catalogues at an SLR level (memoised per SLR/method) |
| `update_flood_experience(flooded_agents)` | update `flood_timer` + `risk_perception` |
| `step(year_index, slr_value, rng, interp_method='linear')` | advance one year for the live `self.state` (**the authoritative kernel**) |
| `reset_state()` | reset per-agent state for a fresh sequence (bumps `state_epoch`) |
| `run(slr_values, no_seq=1, seed=None, interp_method='linear', track_eu=False, n_jobs=1)` | run `no_seq` Monte-Carlo sequences; `n_jobs>1` parallelises across a thread pool of engine clones sharing a pre-warmed read-only cache |
| `is_residential` (property) | boolean mask (all `True` — engine already operates on residential agents) |

**Performance contract (committed, bit-identical):**
- *Per-SLR interpolation cache* — each strategy cube is materialised once and
  `prepare_damages` is memoised per `(SLR, method)`; cuts per-tick interpolation
  ~5.5 s → ~1 s (first materialize ~24 s → ~3.6 s).
- *Parallel Monte-Carlo sequences* — `run(n_jobs=N)` runs per-worker clones over a
  thread pool sharing a pre-warmed cache; `n_jobs=1` unchanged, parallel ≈ 1.4×,
  bit-for-bit. Internals: `_prewarm_interp_cache`, `_clone_for_worker`,
  `_simulate_one_sequence`.

```python
result = engine.run([0.0, 0.1, 0.2, 0.3], no_seq=200, seed=7, n_jobs=4, track_eu=True)
```

---

## 8. Live parity rule — `dynamo_live_rule.py` (Phase 4a)

A `DecisionRule` that delegates the decision math to the **native** DYNAMO-M
`DecisionModule` — the live parity oracle proving the ported `SEURule` matches upstream
(worst EU abs 1.9e-6, rel 4.8e-7).

- `DYNAMO_M_AVAILABLE: bool` — module-level flag (lightweight probe, no heavy import).
- `DynamoMNotAvailable(ImportError)` — raised on construction when DYNAMO-M is absent.
- `resolve_dynamo_path(dynamo_path=None)` / `load_native_decision_module(dynamo_path=None)`
  — resolve + import the native module (honours the `DYNAMO_M_PATH` env var).
- `DynamoLiveRule(config, dynamo_path=None, amenity_value=None, rng=None,
  geom_id='floodadapt_abm')` — builds the minimal stub object graph the native
  `DecisionModule` needs and forwards `should_adapt` to it.

Guarded: the package imports fine without DYNAMO-M; only *constructing* the rule
requires it. Set `DYNAMO_M_PATH` (e.g. `c:\repos\DYNAMO-M\DYNAMO-M`) to enable.

---

## 9. Mesa-native scaffold — `mesa_native.py` (Phase 4b)

A **framework-free** mirror of DYNAMO-M's `SLRModel.step()` tick loop that inverts
*who owns time*: instead of `engine.run` looping years, a small model advances one tick
at a time — while still delegating all numerics to `engine.step`.

- `CoastalNodePopulation(model)` — vectorised household group; `state`/`n` properties;
  `step()` advances one year.
- `Agents(model)` — steps each agent group per tick (exported as `MesaAgents`).
- `FloodAdaptSLRModel(engine, slr_values, seed, interp_method='linear',
  track_eu=False)` — the framework-free model that owns time. Uses the PRE.3
  `state_epoch` staleness guard (`_check_not_stale`) so a shared engine can't be
  silently invalidated. `step()` / `run_model()` mirror `SLRModel`.
- `run_mesa_native(engine, slr_values, no_seq=1, seed=None, interp_method='linear',
  track_eu=False)` — drop-in analogue of `engine.run`; **bit-for-bit identical**.

---

## 10. Native-class integration — `mesa_native_full.py` (Phase 4b-full)

The **final integration step**: binds the **real honeybees `Model`** as the
time-owning base class (as the upstream `SLRModel` does) and routes decisions through
the native DYNAMO-M `DecisionModule` (via `DynamoLiveRule`), feeding a deterministic
coastal-node population entirely from the FloodAdapt lookup table through the PRE.4
adapter. Every numeric per-year operation is still delegated to `SimulationEngine.step`
with the identical RNG stream, so the whole path stays **bit-for-bit** identical to the
4b scaffold and `engine.run`.

- `HONEYBEES_AVAILABLE: bool` / `HoneybeesNotAvailable(ImportError)` — guarded import;
  the package imports without honeybees, and only *construction* raises.
- `CoastalNodePopulationFull(model)` — native-class analogue of the 4b population;
  per-tick `step()` = `adapter.populate(slr)` (forward) → `engine.step(...)`
  (authoritative) → set `node.adapt/time_adapt` → `adapter.write_back(node)` (reverse).
- `AgentsFull(model)` — mirror of DYNAMO-M's `Agents`.
- `FloodAdaptSLRModelFull(engine, slr_values, seed, interp_method='linear',
  track_eu=False, start_year=2020)` — subclasses the real `honeybees.model.Model`; the
  clock (`current_time`/`current_timestep`/`end_time`) is owned by honeybees.
  `timestep` is a 0-based property alias of `current_timestep`. Reuses the PRE.3
  staleness guard (`self.engine.reset_state()` and `self._state_epoch = engine.state_epoch`).
  **Why it's needed:** A single `SimulationEngine` can be reused to run thousands of models in a loop. Because allocating memory is slow, the engine reuses the exact same memory arrays for agent states (wealth, age, etc.) on every run. If a developer accidentally tried to step two different models at the exact same time using the same engine, their arrays would blindly overwrite each other, ruining the results silently. By calling `reset_state()`, the engine zeroes out its arrays and increments a counter (`state_epoch`). The model grabs that "ticket number". Later, when the model tries to step forward, it checks if its ticket number still matches the engine. If it doesn't, it means another model hijacked the engine, and it throws a loud error rather than corrupting your data.
- `run_mesa_native_full(engine, slr_values, no_seq=1, seed=None,
  interp_method='linear', track_eu=False, start_year=2020)` — drop-in analogue of
  `run_mesa_native` / `engine.run`; same return schema.

**Gate (delivered):** `run_mesa_native_full == run_mesa_native == engine.run`
element-wise across seeds/sequences for `SEURule` and `ThresholdRule`; native-vs-ported
EU parity (EU_adapt max |abs| ≈ 2.9e-6); executed on the real ~58k-household Charleston
table. See `examples_engine/07_mesa_native_full.py` and
`verification/mesa_native_full/`.

**Scope note:** GLOFRIS, gravity CWD, `spin_up_flag`, low-memory `.npz` paging and the
native reporter are out of MVP scope — native `CoastalNode.step()` is too entangled
with that data ecosystem to drive on a dependency-free population, so 4b-full reuses the
validated engine kernel for the per-tick physics and the native `DecisionModule` for
the decision math inside a real honeybees `Model`.

```python
import os, xarray as xr
os.environ["DYNAMO_M_PATH"] = r"c:\repos\DYNAMO-M\DYNAMO-M"
from floodadapt_abm import SimulationEngine, SEURule, CouplingConfig, run_mesa_native_full

ds = xr.open_dataset("lookup_table_charleston_beta_release_ABM_probabilistic_set.nc")
cfg = CouplingConfig()
engine = SimulationEngine(ds, decision_rule=SEURule(cfg.decision), config=cfg)
result = run_mesa_native_full(engine, [0.0, 0.1, 0.2, 0.3], no_seq=10, seed=42)
```

---

## 11. Lookup-table adapter — `coastal_node_adapter.py`

Maps between a `SimulationEngine` (the FloodAdapt lookup-table world) and the native
DYNAMO-M `CoastalNode` array layout — the one genuinely new modelling artefact for
4b-full (prototyped as PRE.4, now driven every tick).

- `CoastalNodeArrays` (dataclass) — dependency-free mirror of the native node array
  set: `property_value`, events-first `damages_coastal_cells`, `p_floods`, `adapt`,
  `time_adapt`, `_flood_plain` geom_id.
- `LookupTableAdapter(engine, geom_id='floodadapt_flood_plain')`
  - `populate(slr_value, interp_method='linear')` — **forward**: build node arrays from
    the lookup table at `slr_value` (read-only, no RNG).
  - `write_back(node)` — **reverse**: route the node's adaptation state back into the
    engine's live `AgentState`, with `object_id` alignment guards (idempotent).
- `round_trip_check(engine, slr_value, interp_method='linear')` — executable bit-parity
  contract (the PRE.4 gate): proves routing state through the node is a simulation
  no-op.

---

## 12. Ported kernels — `_core/`

Import-free numerical layer (only NumPy/SciPy/xarray). DYNAMO-M's Python source is
**not** imported at runtime for the ported MVP path.

### `_core/dynamo_decision_bridge.py`
`DynamoDecisionBridge` couples the xarray lookup table with the ported SEU model.

`DynamoDecisionBridge(ds, config=None, income_per_agent=None,
amenity_value_per_agent=None)`. Key methods:

| Method | Purpose |
|---|---|
| `prepare_damage_arrays(slr_value, interp_method='linear')` | interpolate per-event damage arrays at an SLR level (memoised — the per-SLR cache) |
| `clear_interp_cache()` | drop all memoised interpolation state |
| `compute_expected_annual_damages(use_adapted_strategy=False)` | EAD per agent by integrating damage × frequency |
| `update_flood_experience(flooded_agents)` | advance `flood_timer` + `risk_perception` |
| `evaluate_decisions(year_index)` | apply the SEU model; return newly-adapting agents |
| `get_current_damages(event_name)` | (capped) per-agent damage for one event |

Module-level SEU maths (pure functions): `_iterate_through_flood` (time-discounted
NPV per flood), `_integrate_expected_utility` (CRRA + integrate over perceived
probability), `_calc_eu_do_nothing`, `_calc_eu_adapt`. Private init helpers set up
economic/state arrays, the residential mask and the annualised adaptation cost.

### `_core/lookup_utils.py`
The SLR→damage interpolation kernel:

- `materialize_strategy_cube(ds, strategy, res_mask=None, ...)` — build the
  (residential) damage cube for one strategy **once**.
- `interpolate_cube_at_slr(values, slr_arr, slr_target, method='linear',
  max_pot_dmg=None)` — interpolate a pre-materialized cube along the SLR axis.
- `interpolate_damage_at_slr(ds, strategy, slr_target, ...)` — single-shot convenience.
- `interpolate_damage_matrix(ds, strategy, slr_values, event_names_list, ...)` — batch
  over SLR values and an event subset.

---

## 13. Stage-1 pipeline — `setup_lookup_table.py`

Builds the NetCDF damage cube by running FloodAdapt scenarios. **Requires the
`flood-adapt` library** (hence not imported by `__init__.py`). Functions:

| Function | Purpose |
|---|---|
| `create_lookup_table(fa, name_event_set, slr=np.arange(0,1.1,0.25), unit=UnitTypesLength.meters, fp_height=0.5)` | top-level: build + return the lookup Dataset |
| `get_events_freq(fa, name_event_set)` | read event frequencies from the EventSet |
| `create_combinations_matrix(fa, name_event_set, slr, unit, fp_height)` | enumerate projection/strategy/scenario combinations |
| `save_combinations_to_database(fa, projections, strategies, scenarios, flood_proofs)` | register scenarios in FloodAdapt |
| `run_scenarios(fa, scenarios, clean=True)` | run scenarios (optionally cleaning outputs) |
| `read_impacts_dataset(fa, projections, strategies, events, slr, events_freq=None)` | assemble impacts into the cube (with `object_id`-indexed accessor + alignment assertion) |
| `_cleanup_scenario_outputs(...)` | delete intermediate scenario outputs |

The `EventSet` must be **return-period based** for the frequencies to be meaningful.

---

## 14. Legacy simulator — `abm_simulator.py`

`ABMSimulator` — the **deprecated** stage-2 threshold-rule simulator, retained for
backward compatibility and the Gate-1 bit-for-bit regression against
`SimulationEngine` + `ThresholdRule`. New code should not use it.

`ABMSimulator(ds_impacts, times, slr_values, no_seq, damage_threshold=0.3, seed=42,
dmg_unit='$', slr_unit='feet', damage_dtype=np.int32)`. Notable methods:
`run_simulation`, `generate_event_sequences`, `interpolate_damage_matrix`,
`slr_damage_lookup`, and plotting helpers (`plot_event_damage_timeseries`,
`plot_total_damage_statistics`).

---

## 15. Examples, tests & verification

```
examples_engine/         numbered runnable learning path (01 … 07) + README
  01_...                  engine basics
  ...
  06_mesa_native_driving.py    Phase 4b scaffold demo
  07_mesa_native_full.py       Phase 4b-full native-class demo (+ inline bit-parity)
tests/                   full pytest suite (147 tests; self-contained mock datasets)
  test_mesa_native_full.py     22 tests: triple bit-parity, honeybees clock,
                               object graph/adapter, staleness guard, native path
verification/            vendored, portable batteries emitting md/JSON/figures
  phase1_seu_battery/          V1–V6 SEU validation
  phase4a_parity/              ported vs native EU parity
  phase4b_mesa_native/         4b bit-parity gate
  mesa_native_full/            4b-full G1–G4 battery (gate_pass: True on real table)
  real_table_gate/             PRE.2 full Charleston run
  preflight_4b_full/           import/instantiate checks
```

Run everything (set `DYNAMO_M_PATH` to enable the native-parity tests):

```powershell
$env:DYNAMO_M_PATH = "c:\repos\DYNAMO-M\DYNAMO-M"
pytest -q            # 147 passed
```

Guarded tests skip cleanly when honeybees or DYNAMO-M is unavailable.

---

## 16. Global assumptions & invariants

- **Residential-only MVP:** only buildings whose `primary_object_type` contains `RES`
  are simulated (substring match, case-sensitive).
- **Two strategies:** `no_measures` (baseline) and `floodproof_all_0` (dry-floodproof).
- **RP-based EventSet:** frequencies come from a return-period EventSet; Bernoulli
  per-event occurrence with `p = 1 - exp(-freq·dt)`.
- **Event cap policy:** at most `max_events_per_year`; surplus events chosen **at
  random without replacement** from the drawn pool.
- **Irreversible within-year adaptation:** an agent adapts at most once; adaptations
  age via `time_adapted` and **expire at `lifespan_dryproof`** (default 75 y), after
  which the agent un-adapts and re-decides.
- **Bit-parity is the contract:** all three time drivers share the `engine.step` kernel
  and RNG stream; any divergence between them is a bug.
- **Optional deps are guarded:** DYNAMO-M (`DynamoLiveRule`, native parity) and
  honeybees (`mesa_native_full`) are imported defensively — the package and the FA-ABM
  suite never hard-fail when they are absent.
- **Determinism:** given a seed, results are reproducible; `n_jobs>1` is bit-identical
  to `n_jobs=1` for deterministic rules.

---

*Reference for the `floodadapt_abm` package. For design rationale, the SEU
mathematics, diagrams and the phase history, see
[`docs/architecture.md`](docs/architecture.md). Revised 2026-07-09; reflects the
delivered Phase 4b-full and the 147-test suite.*
