# FloodAdapt-ABM

FloodAdapt-ABM is a lightweight agent-based simulator that processes a precomputed [FloodAdapt](https://pypi.org/project/flood-adapt/) impact lookup table to generate Monte-Carlo time series of building-level damages and household floodproofing (and optionally flood-insurance) decisions under sea-level rise.

Household behaviour is pluggable. The **preferred path is the fully native coupling**: a real honeybees `Model` owns the clock and the native **DYNAMO-M** decision module decides both floodproofing and insurance. A parity-gated pure-NumPy port stands in when DYNAMO-M is not installed, and a simple damage-threshold rule is kept as a comparison baseline.

---

## Installation

Requires **Python 3.11+** (the main engine depends on `honeybees`, whose dependency chain sets the floor).

> [!NOTE]
> The `[pipeline]` extra (which installs `flood-adapt` to build new lookup tables from scratch) currently requires Python < 3.13. If you already have a precomputed lookup table you never need it.

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell (or: source venv/bin/activate)

pip install -e .                 # core (includes honeybees — the main engine)
pip install -e .[dev]            # + pytest (run the test suite)
pip install -e .[pipeline]       # + flood-adapt (stage-1 lookup-table builds)
```

---

## One kernel, clearly-labelled roles

There is exactly **one** compute kernel, `SimulationEngine.step`. Everything else is either a driver that owns the clock around it, or a rule that supplies behaviour. Each carries a `STATUS` tag.

**Drivers** (which object owns the clock):

| Entry point | Status | Use it for |
|---|---|---|
| `run_mesa_native_full` | **preferred** | Application runs. A real honeybees `Model` owns the clock |
| `engine.run(n_jobs=...)` | kernel | Experiments, sweeps, the parallel Monte-Carlo backend |
| `run_mesa_native` | verification | The bit-parity gate |
| `ABMSimulator` | deprecated | Backward compatibility only |

**Decision rules** (which behaviour applies):

| Rule | Status | Use it for |
|---|---|---|
| `DynamoLiveRule` | **preferred** | Native DYNAMO-M decides floodproofing and insurance |
| `SEURule` | reference | Parity-gated port: used when DYNAMO-M is absent, and required for per-household (risk-based) premiums |
| `ThresholdRule` | experiment | The pre-coupling damage-threshold baseline |

`preferred_decision_rule(config)` picks the right rule for the configuration and environment. `DynamoLiveRule` and `SEURule` are parity-gated (relative expected-utility error below 1e-4, identical actions), so their results are interchangeable. The port is selected only when DYNAMO-M is absent or when per-household premiums are configured, which the native kernel cannot express.

All drivers delegate every numeric operation to the same kernel with the same RNG stream, so their results are **bit-for-bit identical**.

## Quick start

```python
from floodadapt_abm import (
    SimulationEngine, CouplingConfig, preferred_decision_rule, run_mesa_native_full,
)
import xarray as xr, numpy as np

ds = xr.open_dataset("lookup_table.nc")                  # stage-1 output
config = CouplingConfig()
engine = SimulationEngine(ds=ds, config=config)

# The preferred rule: native DYNAMO-M when installed, else the parity-gated port.
engine.decision_rule = preferred_decision_rule(config.decision)

results = run_mesa_native_full(engine, np.linspace(0, 1.5, 30), no_seq=10, seed=42)

results["damage_history"]      # (no_seq, n_agents, n_years) — gross damage
results["adapted_history"]     # (no_seq, n_agents, n_years) bool
results["adoption_fraction"]   # (no_seq, n_years)
```

No lookup table yet? The numbered examples run out-of-the-box on a synthetic one:

```bash
cd examples_engine
python 01_quickstart.py
```

## The two-stage pipeline

1. **Build the lookup table** — [notebooks/1_create_lookup_table.ipynb](notebooks/1_create_lookup_table.ipynb) runs FloodAdapt (SFINCS + FIAT) over every `event × SLR × strategy` combination and saves `lookup_table_<site>_<event_set>.nc` (dims `object_id × slr × strategy × event`).
2. **Simulate adaptation** — [notebooks/2_run_coupled_abm.ipynb](notebooks/2_run_coupled_abm.ipynb) (or the API above) draws Monte-Carlo event sequences, interpolates damages along the SLR axis, and applies the pluggable household decision rule each year.

The `.nc` lookup table is the **only** interface between the stages — keep it stable.

## Behaviour configuration (post-2026-07-review defaults)

The `DecisionConfig` defaults implement the fixes from the 2026-07 review (rationale: [docs/architecture.md](docs/architecture.md), Appendix B):

| Switch | Default | Meaning |
|---|---|---|
| `event_draw_mode` | `"poisson"` | Statistically exact hazard draw — every event's realised rate equals its nominal frequency; extremes are never crowded out |
| `max_events_per_year` | `None` | No discard cap (damage is already bounded per event by `max_pot_dmg`) |
| `nuisance_freq_threshold` | `None` | Set `1.0` for event sets with sub-annual (`freq > 1`) events — recommended for the real Charleston table |
| `seu_prob_mode` | `"exceedance"` | SEU integral uses `p = 1 − e^(−freq)` exceedance probabilities |
| `perception_mode` | `"severity"` | Post-flood risk-perception spike scales with damage severity (concave, γ = 0.5) |
| `income_mode` | `"synthetic_lognormal"` | Native DYNAMO-M income port — income independent of building value, so affordability genuinely binds |
| `include_insurance` | `False` | Optional third decision option (ported native `calcEU_insure`, 10 % deductible) |
| `insurance_pricing` | `"community"` | Premium rating: flat mean-EAD rate (native) or `"risk_based"` (each household's own expected payout). Levers: `insurance_loading`, `insurance_subsidy` |

**Reproducing historical results:** `CouplingConfig.legacy()` pins every switch to the pre-review behaviour **bit-exactly** (guarded by the golden regression `tests/test_legacy_mode.py`); the `verification/` harnesses are pinned to it.

Per-agent economic inputs can be supplied directly (`income_per_agent`, `wealth_per_agent`) or via **income percentiles** (`income_percentile_per_agent`; derive from ACS block-group data with `floodadapt_abm.income_utils.derive_income_percentiles`).

Income has **two independent axes**, and only one of them needs building locations. The *marginal distribution* (`median_income`, `mean_median_inc_ratio`) is a property of the study area and is measurable from ACS county totals with no geometry at all (`income_utils.fetch_acs_county_income`; needs a free key in `CENSUS_API_KEY`). The *rank assignment* (which building sits at which percentile) is the axis that needs footprints; without them, `percentiles_from_value_proxy` is the documented fallback. Validate the fitted marginal against the held-out ACS B19001 histogram with `lognormal_bracket_shares` and `bracket_fit_distance`. See [docs/calibration_validation_guide.md](docs/calibration_validation_guide.md).

## Decision rules (Strategy Pattern)

| Rule | `STATUS` | Behaviour | Use |
|---|---|---|---|
| `DynamoLiveRule` | **preferred** | Calls the **native** DYNAMO-M `DecisionModule` (optional dependency, guarded import via `DYNAMO_M_PATH`), including native `calcEU_insure` | Application runs, and the seam for any future DYNAMO-M coupling |
| `SEURule` | reference | The same SEU science ported to pure NumPy: ex-ante expected-utility maximisation with CRRA utility, risk-perception decay, affordability cap, loan amortisation, 75-y lifespan reset; 3-way `decide()` when insurance is on | Fallback when DYNAMO-M is absent; **required for `insurance_pricing="risk_based"`** (native prices one flat rate only). Faster in parallel (releases the GIL). Doubles as the parity oracle proving the port has not drifted |
| `ThresholdRule` | experiment | Legacy ex-post rule: adapt when `damage/max_pot_dmg > 0.3` | Baseline comparison; reproduces `ABMSimulator` bit-for-bit |
| your own | experiment | Subclass `DecisionRule`, implement `should_adapt(...)` (two-way rules need no changes for the 3-way contract) | See `examples_engine/03_custom_rule.py` |

Call `preferred_decision_rule(config.decision)` rather than branching on `DYNAMO_M_AVAILABLE` yourself.

## Performance & parallelisation

- **Per-SLR interpolation cache** — interpolation is memoised per `(SLR, method)`; `bridge.clear_interp_cache()` frees the memory.
- **Parallel Monte-Carlo sequences** — `engine.run(..., n_jobs=N)` runs independent sequences across a thread pool of per-worker clones sharing a pre-warmed read-only cache; `n_jobs>1` / `-1` is **bit-identical** to sequential for deterministic rules.

## Tests & verification

```bash
pytest tests/ -q                 # full suite incl. all bit-parity gates + golden legacy regression
```

Phase-gate evidence (reports, metrics, re-runnable harnesses) lives in [verification/](verification/); the harnesses are pinned to `CouplingConfig.legacy()` (set `FA_ABM_HARNESS_CONFIG=new` to re-run them under the current defaults; outputs get a `_newdefaults` suffix). CI runs the suite and the examples on every push (see `.github/workflows/ci.yml`).

## Documentation

Two core documents with distinct roles, plus one method guide:

| Document | Role | Read it when you ask |
|---|---|---|
| [docs/architecture.md](docs/architecture.md) | **Design record** | *Why is it built this way?* Coupling architecture, the SEU decision science (adaptation and insurance), deviations from native DYNAMO-M, phase history. |
| [floodadapt_abm_documentation.md](floodadapt_abm_documentation.md) | **Reference manual** | *How do I use it?* Every module, class, function and configuration field, with examples. |
| [docs/calibration_validation_guide.md](docs/calibration_validation_guide.md) | **Method guide** | *How do I calibrate and validate it?* Parameter inventory, tiered approach, open-data sources. |

Reading order for newcomers:

1. **This README**: overview, installation, quick start.
2. [notebooks/2_run_coupled_abm.ipynb](notebooks/2_run_coupled_abm.ipynb): the R1 DYNAMO ABM run walkthrough, including the household-income workflow and the insurance scenarios.
3. [examples_engine/README.md](examples_engine/README.md): the numbered learning path (01 to 08; 08 tours the income, perception and insurance features).
4. The three documents above, in the order of your question.

(The 2026-07 review response and the government-extension feasibility assessment are project deliverables kept outside the repository.)
