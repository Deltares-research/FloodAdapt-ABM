# FloodAdapt-ABM

FloodAdapt-ABM is a lightweight agent-based simulator that processes a precomputed [FloodAdapt](https://pypi.org/project/flood-adapt/) impact lookup table to generate Monte-Carlo time series of building-level damages and household floodproofing decisions under sea-level rise. Household behaviour is pluggable: a simple damage-threshold rule, the ported **DYNAMO-M** Subjective Expected Utility (SEU) framework (the validated default), or the *live* native DYNAMO-M decision module.
---

## Installation

Requires Python 3.10+. 

> [!NOTE]
> The `[pipeline]` extra (which installs `flood-adapt` to build new lookup tables from scratch) currently requires Python < 3.13. If you are on Python 3.13+ and already have a precomputed lookup table, you can skip installing `[pipeline]` and simply install the core/dev/dynamo packages to run the simulation.

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell (or: source venv/bin/activate)

pip install -e .                 # core
pip install -e .[dev]            # + pytest (run the test suite)
pip install -e .[pipeline]       # + flood-adapt (stage-1 lookup-table builds)
pip install -e .[dynamo]         # + mesa/honeybees (native DYNAMO-M / Phase-4b-full; optional)
```
---

## Repository structure

```
FloodAdapt-ABM/
├── floodadapt_abm/                     # the package
│   ├── __init__.py                     # public API (SimulationEngine recommended)
│   ├── simulation_engine.py            # ⭐ unified engine: time, data, events, state
│   ├── decision_rule.py                # DecisionRule ABC + ThresholdRule + SEURule
│   ├── dynamo_live_rule.py             # Phase 4a: native-DYNAMO-M rule (guarded import)
│   ├── mesa_native.py                  # Phase 4b: Mesa-native tick driver (mirror)
│   ├── coastal_node_adapter.py         # PRE.4: lookup-table → CoastalNode adapter prototype
│   ├── agent_state.py                  # vectorised per-agent state container
│   ├── event_utils.py                  # unified event drawing (Bernoulli + random-pool cap)
│   ├── coupling_config.py              # configuration dataclasses
│   ├── abm_simulator.py                # DEPRECATED legacy simulator (backward compat)
│   ├── setup_lookup_table.py           # stage-1 FloodAdapt orchestration
│   └── _core/                          # internal plumbing (not public API)
│       ├── dynamo_decision_bridge.py   #   ported SEU kernels + data layer
│       └── lookup_utils.py             #   NetCDF & SLR interpolation utilities
├── examples_engine/                    # ⭐ numbered learning path (01–06)
│   ├── 01_quickstart.py … 06_mesa_native_driving.py
│   ├── _shared.py                      # helper: dataset bootstrap (synthetic by default)
│   ├── README.md                       # usage guide & architecture
│   └── old_bridge_examples/            # DEPRECATED pre-refactor demos (reference only)
├── tests/                              # pytest suite (incl. all parity gates)
├── docs/                               # design record (architecture, phase docs)
├── notebooks/                          # top-level pipelines
│   ├── 1_create_lookup_table.ipynb     # stage 1: build the lookup table (SFINCS+FIAT)
│   └── 2_run_coupled_abm.ipynb         # stage 2: ⭐ Coupled ABM guide (SimulationEngine + SEURule)
└── pyproject.toml                      # package metadata & dependencies (primary)
```
---

## Quick start

```python
from floodadapt_abm import SimulationEngine, CouplingConfig
import xarray as xr, numpy as np

ds = xr.open_dataset("lookup_table.nc")                     # stage-1 output
engine = SimulationEngine(ds=ds, config=CouplingConfig())   # SEURule by default
results = engine.run(np.linspace(0, 1.5, 30), no_seq=10, seed=42)

results["damage_history"]      # (no_seq, n_agents, n_years)
results["adapted_history"]     # (no_seq, n_agents, n_years) bool
results["adoption_fraction"]   # (no_seq, n_years)
```

No lookup table yet? The numbered examples run out-of-the-box on a synthetic one:

```bash
cd examples_engine
python 01_quickstart.py
```

---

## The two-stage pipeline

1. **Build the lookup table** — [notebooks/1_create_lookup_table.ipynb](notebooks/1_create_lookup_table.ipynb) runs FloodAdapt (SFINCS + FIAT) over every `event × SLR × strategy` combination and saves `lookup_table_<site>_<event_set>.nc` (dims `object_id × slr × strategy × event`).
2. **Simulate adaptation** — [notebooks/2_run_coupled_abm.ipynb](notebooks/2_run_coupled_abm.ipynb) (or the API above) draws Monte-Carlo event sequences, interpolates damages along the SLR axis, and applies the pluggable household decision rule each year.

The `.nc` lookup table is the **only** interface between the stages — keep it stable.

## Performance & parallelisation

Two bit-identical engine speedups (landed while closing the real-table gate, commit `6f45d6f`):

- **Per-SLR interpolation cache** — the residential strategy cube is materialized once and `prepare_damage_arrays` is memoised per `(SLR, method)`. Because the SLR trajectory repeats across Monte-Carlo sequences this removes the `no_seq×` redundant interpolation (per-tick ~5.5 s → ~1 s on the real Charleston table; first cube materialize ~24 s → ~3.6 s). `bridge.clear_interp_cache()` frees the memory.
- **Parallel Monte-Carlo sequences** — `engine.run(..., n_jobs=N)` runs the independent sequences across a thread pool of per-worker clones sharing a pre-warmed, read-only cache. `n_jobs=1` (default) is the unchanged sequential path; `n_jobs>1` / `-1` is **bit-identical** for deterministic rules (~1.4× on the real table). Use `DecisionRule.clone()` for isolated per-worker rule instances.

```python
results = engine.run(np.linspace(0, 1.5, 30), no_seq=100, seed=42, n_jobs=-1)  # all cores, bit-identical
```

## Decision rules (Strategy Pattern)

| Rule | Behaviour | Use |
|---|---|---|
| `SEURule` *(default)* | DYNAMO-M SEU, ported: ex-ante expected-utility maximisation with CRRA utility, risk-perception decay, affordability cap, loan amortisation, 75-y lifespan reset | The validated MVP science |
| `ThresholdRule` | Legacy ex-post rule: adapt when `damage/max_pot_dmg > 0.3` | Backward compat; reproduces `ABMSimulator` bit-for-bit |
| `DynamoLiveRule` | Calls the **native** DYNAMO-M `DecisionModule` (optional, guarded import via `DYNAMO_M_PATH`) | Parity oracle — proves the port hasn't drifted |
| your own | Subclass `DecisionRule`, implement `should_adapt(...)` | See `examples_engine/03_custom_rule.py` |

## Examples

The learning path is [examples_engine/](examples_engine/) — six numbered scripts from a minimal run (01) through custom rules (03), Monte-Carlo uncertainty (04), the Phase-4a live-DYNAMO-M parity oracle (05), and Phase-4b Mesa-native driving (06). All run on a synthetic dataset by default; set `FA_ABM_REAL_TABLE=1` to opt into the real Charleston table.

## Tests & verification

```bash
pytest tests/ -q                 # full suite incl. bit-parity gates
```

Phase-gate evidence (reports, metrics, re-runnable harnesses) lives in [verification/](verification/); CI runs the suite and the Phase-4b gate on every push (see `.github/workflows/ci.yml`).

## Documentation

- [examples_engine/README.md](examples_engine/README.md) — architecture & usage walkthrough
- [docs/architecture.md](docs/architecture.md) — the design record detailing the coupling architecture, the SEU mathematical formulation, and the phase-gate history
- [floodadapt_abm_documentation.md](floodadapt_abm_documentation.md) — the comprehensive API reference covering every module, class, and configuration dataclass

