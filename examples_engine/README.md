# Examples — `SimulationEngine` (learning path)

A guided, runnable tour of the FloodAdapt-ABM × DYNAMO-M coupling. The examples
are **numbered in the order you should read them** and each ends by pointing at
the next one.

> **They run anywhere.** By default the examples use a small, deterministic
> *synthetic* "Charleston-like" lookup table (built in `_shared.py`), so you
> need **no large data file and no DYNAMO-M checkout** to try them. See
> [Using the real Charleston table](#using-the-real-charleston-table) to opt in.

## Quick start

```bash
cd examples_engine
python 01_quickstart.py
```

(Use the Python environment that has `floodadapt_abm` and its dependencies —
`numpy`, `xarray`, `numba`, `scipy`, `pyyaml`.)

## The examples

| # | File | What it teaches |
|---|------|-----------------|
| 01 | `01_quickstart.py` | Smallest end-to-end run; result array shapes & meaning. Engine uses `SEURule` by default. |
| 02 | `02_rules_comparison.py` | Strategy Pattern: swap `SEURule` ↔ `ThresholdRule` without touching the engine; EU diagnostics via `track_eu=True`. |
| 03 | `03_custom_rule.py` | Open/Closed: write your own `DecisionRule` (`should_adapt`) and plug it in. |
| 04 | `04_monte_carlo_uncertainty.py` | Why `no_seq` exists: average across sequences for expected outcomes + uncertainty (std). |
| 05 | `05_dynamo_live_parity.py` | **Phase 4a**: `DynamoLiveRule` drives *native* DYNAMO-M as a parity oracle proving the ported `SEURule` has not drifted. Guarded/optional dependency. |
| 06 | `06_mesa_native_driving.py` | **Phase 4b**: invert time ownership — `FloodAdaptSLRModel.step()` ticks drive the shared kernels (mirrors DYNAMO-M `SLRModel.run_model()`); reproduces `engine.run` bit-for-bit. |
| 07 | `07_mesa_native_full.py` | **Phase 4b-full**: native-class integration — `FloodAdaptSLRModelFull` subclasses the **real honeybees `Model`** (owns time) and routes decisions through the native DYNAMO-M `DecisionModule`; triple-parity `full == scaffold == engine.run`. |

`_shared.py` is a helper (not an example): it bootstraps `sys.path` and provides
the dataset. You never need to run it directly.

## Key concepts (as the examples demonstrate them)

```
SimulationEngine        owns TIME + DATA (NetCDF load, interpolation,
  ├── _data             stochastic event draw, per-agent state, the year loop)
  ├── decision_rule ◄── owns BEHAVIOUR — pluggable (Strategy Pattern):
  │        ThresholdRule   legacy: adapt when realised damage > 0.3·max_pot_dmg
  │        SEURule         DYNAMO-M SEU (ported); adapt when EU_adapt > EU_stay
  │        DynamoLiveRule  Phase 4a: calls NATIVE DYNAMO-M (parity oracle)
  │        <your rule>     inherit DecisionRule, implement should_adapt(...)
  └── state: AgentState  wealth, income, risk_perception, flood_timer,
                          is_adapted, time_adapted
```

* **Time** is driven by the length of the SLR trajectory you pass to
  `engine.run(slr_values, ...)` — one value per year. In **example 06** time is
  instead driven by `model.step()` ticks (Phase 4b), producing identical results.
* **`no_seq`** independent Monte-Carlo sequences each get a fresh `AgentState`
  and their own random weather; aggregate across them for expected behaviour.
* **`n_jobs`** parallelizes those sequences: `engine.run(..., n_jobs=N)` spreads
  them across a thread pool of per-worker clones that share a pre-warmed,
  read-only interpolation cache. `n_jobs=1` (default) is the unchanged
  sequential path; `n_jobs>1` / `-1` (all cores) is **bit-identical** for
  deterministic rules. The engine also memoises the SLR→damage interpolation
  per `(SLR, method)`, so repeated ticks/sequences reuse the cube instead of
  re-interpolating — this is what makes the real table (below) tractable.
* **`SEURule`** is *ex-ante* (forward-looking utility), **`ThresholdRule`** is
  *ex-post* (reacts to realised damage) — hence example 04 uses the latter to
  show adoption variance.

## Minimal code (from `01_quickstart.py`)

```python
from floodadapt_abm import SimulationEngine, CouplingConfig
import xarray as xr, numpy as np

ds = xr.open_dataset("lookup_table.nc")
engine = SimulationEngine(ds=ds, config=CouplingConfig())   # SEURule by default
results = engine.run(np.linspace(0, 1.5, 30), no_seq=10, seed=42)

results["damage_history"]      # (no_seq, n_agents, n_years)
results["adapted_history"]     # (no_seq, n_agents, n_years) bool
results["adoption_fraction"]   # (no_seq, n_years)
```

## Using the real Charleston table

The real probabilistic lookup table (~61,858 objects × 207 events) is large and
slow, so it is **opt-in**:

```bash
set FA_ABM_REAL_TABLE=1            # Windows (PowerShell: $env:FA_ABM_REAL_TABLE=1)
python 01_quickstart.py
```

`_shared.load_dataset()` looks for the file next to the DYNAMO-M checkout and
falls back to the synthetic table if it is not found. Expect multi-minute
runtimes on the full table — pass `n_jobs=-1` to `engine.run(...)` to parallelize
the Monte-Carlo sequences (bit-identical), and reuse a single `engine` across
runs so its per-SLR interpolation cache stays warm (the cache is released when the
engine is discarded, or via `engine._data.clear_interp_cache()`). On the real
table these cut per-tick interpolation from ~5.5 s to ~1 s (first cube
materialize ~24 s → ~3.6 s) and give ~1.4× from parallel sequences.

## Phase 4a — the live DYNAMO-M rule (example 05)

`DynamoLiveRule` drives the upstream `DecisionModule.calcEU_*` directly. The
DYNAMO-M dependency is **optional and guarded**:

* `DYNAMO_M_AVAILABLE` reports whether it is importable; the package works
  without it (example 05 prints a notice and exits cleanly).
* Point it at a checkout with `DynamoLiveRule(cfg.decision, dynamo_path=...)` or
  the `DYNAMO_M_PATH` environment variable.

The parity gate (identical decisions, EU differences at float32 level) is the
executable Phase-1 cross-check. See the verification bundle in
[`verification/phase4a_parity/`](../verification/phase4a_parity/).

## Phase 4b — Mesa-native driving (example 06)

Phase 4b **inverts time ownership**: instead of `engine.run()` owning the year
loop, a small `FloodAdaptSLRModel` advances one tick at a time via
`model.step()`, mirroring the native DYNAMO-M `SLRModel.run_model()`
(`while True: self.step()`). The object graph mirrors DYNAMO-M
(`FloodAdaptSLRModel → Agents → CoastalNodePopulation → DecisionRule`) and
`run_mesa_native(...)` reproduces `engine.run(...)` **bit-for-bit** — proving the
migration is non-breaking because the `DecisionRule.should_adapt` seam is
unchanged.

```python
from floodadapt_abm import SimulationEngine, CouplingConfig, run_mesa_native
engine = SimulationEngine(ds=ds, config=CouplingConfig())
res = run_mesa_native(engine, slr_values, no_seq=5, seed=42)   # time owned by model.step()
```

Binding the *real* honeybees `SLRModel` in full ("4b-full") needs the full
DYNAMO-M data ecosystem; the **native-class integration** below delivers the
documented 4b-full gate without it. See the verification bundle in
[`verification/phase4b_mesa_native/`](../verification/phase4b_mesa_native/).

## Phase 4b-full — native-class integration (example 07)

Phase 4b-full replaces the framework-free mirror with a **genuine honeybees
`Model` subclass**: `FloodAdaptSLRModelFull` inherits the real framework clock
(`current_time` / `current_timestep` / `end_time`) exactly as the upstream
`SLRModel` does, and each year's decision runs through the **native DYNAMO-M
`DecisionModule`** (via `DynamoLiveRule`). The coastal-node population is fed
**entirely from the FloodAdapt lookup table** through the PRE.4
`LookupTableAdapter` — no GLOFRIS, gravity or geodata. Because every numeric
per-year operation is still delegated to the same `SimulationEngine.step` kernel
with the same RNG stream, `run_mesa_native_full(...)` reproduces both
`run_mesa_native(...)` and `engine.run(...)` **bit-for-bit** — the triple-parity
4b-full gate.

```python
from floodadapt_abm import SimulationEngine, CouplingConfig, run_mesa_native_full
engine = SimulationEngine(ds=ds, config=CouplingConfig())
res = run_mesa_native_full(engine, slr_values, no_seq=5, seed=42)  # honeybees Model owns time
```

The full verification battery (multi-seed parity, native-vs-ported EU parity,
adapter round-trip, real-table run) lives in
[`verification/mesa_native_full/`](../verification/mesa_native_full/).

## Folder structure

```
examples_engine/                     ← YOU ARE HERE (canonical learning path)
  ├── _shared.py                     (helper: sys.path + dataset)
  ├── 01_quickstart.py
  ├── 02_rules_comparison.py
  ├── 03_custom_rule.py
  ├── 04_monte_carlo_uncertainty.py
  ├── 05_dynamo_live_parity.py       (Phase 4a)
  ├── 06_mesa_native_driving.py      (Phase 4b)
  ├── 07_mesa_native_full.py         (Phase 4b-full)
  ├── README.md                      (this file)
  └── old_bridge_examples/           ⚠️ DEPRECATED (pre-refactor bridge demos)
       ├── run_coupled_example.py
       ├── run_trace_manual_check.py
       └── README.md
```

## Legacy examples

`old_bridge_examples/` holds the original `DynamoDecisionBridge`-based scripts
from before the Phase 2+3 refactor. They still work but are **reference only** —
`DynamoDecisionBridge` is now internal (`floodadapt_abm._core`). Prefer the
numbered examples above for all new work.

## Tests

The library behaviour these examples exercise is covered by the suite:

```bash
pytest tests/ -q
```
