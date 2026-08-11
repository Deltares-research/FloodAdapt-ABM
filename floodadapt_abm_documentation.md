# FloodAdapt-ABM — Whole-Repo Reference Documentation

**Scope:** the single API-level reference for the `floodadapt_abm` package — every
module, public class, dataclass and function, with signatures, responsibilities,
assumptions and runnable code examples. It also maps the examples/tests/verification
layout and the module dependency graph.

**Documentation roles.** Two core documents with distinct jobs, plus one method guide:

- **This file is the reference manual.** It answers *how to use the package*: every
  module, public class, dataclass, function and configuration field, with examples.
- [`docs/architecture.md`](docs/architecture.md) is **the design record**. It answers
  *why and how the system is built*: MVP scope, the Strategy-Pattern architecture, the
  full SEU mathematics (including insurance), diagrams and every deviation
  from native DYNAMO-M.
- [`docs/calibration_validation_guide.md`](docs/calibration_validation_guide.md) is
  **the method guide**: how to calibrate and validate the behavioural parameters.

[`README.md`](README.md) is the front door and quickstart.

**Status:** implementation complete. Every parity gate, the real-table gate and
the full `pytest` suite pass; run `pytest tests/ -q` for the current count.

---

## Table of contents

1. [Package overview & public API](#1-package-overview--public-api)
2. [Two-stage pipeline & module dependency map](#2-two-stage-pipeline--module-dependency-map)
3. [Configuration — `coupling_config.py`](#3-configuration--coupling_configpy)
4. [Agent state — `agent_state.py`](#4-agent-state--agent_statepy)
5. [Decision rules — `decision_rule.py`](#5-decision-rules--decision_rulepy)
6. [Stochastic events — `event_utils.py`](#6-stochastic-events--event_utilspy)
7. [Simulation engine — `simulation_engine.py`](#7-simulation-engine--simulation_enginepy)
8. [Preferred rule — `dynamo_live_rule.py`](#8-preferred-rule--dynamo_live_rulepy)
9. [Verification mirror — `mesa_native.py`](#9-verification-mirror--mesa_nativepy)
10. [Preferred driver — `mesa_native_full.py`](#10-preferred-driver--mesa_native_fullpy)
11. [Lookup-table adapter — `coastal_node_adapter.py`](#11-lookup-table-adapter--coastal_node_adapterpy)
12. [Ported kernels — `_core/`](#12-ported-kernels--_core)
13. [Stage-1 pipeline — `setup_lookup_table.py`](#13-stage-1-pipeline--setup_lookup_tablepy)
14. [Standalone simulator — `abm_simulator.py`](#14-standalone-simulator--abm_simulatorpy)
15. [Examples, tests & verification](#15-examples-tests--verification)
16. [Global assumptions & invariants](#16-global-assumptions--invariants)

---

## 1. Package overview & public API

`floodadapt_abm` is an agent-based flood-adaptation simulator whose household
decision logic is the DYNAMO-M **Subjective Expected Utility (SEU)** model
([Tierolf et al., 2023](https://doi.org/10.1038/s41598-023-31351-y); source: [VU-IVM/DYNAMO-M](https://github.com/VU-IVM/DYNAMO-M/tree/v0.1.4),
v0.1.4, the version the parity gates run against). Buildings
("agents") each year decide whether to **dry-floodproof** based on perceived flood
risk, expected damages, income/wealth and the cost of adaptation.

Everything importable from the top-level package (`floodadapt_abm/__init__.py`):

| Symbol | Kind | Summary |
|---|---|---|
| `SimulationEngine` | class | The compute kernel: owns time, data plumbing, event generation, the lifespan reset, and a pluggable `DecisionRule`. Also the parallel Monte-Carlo backend. |
| `DecisionRule` | ABC | Strategy interface. Subclass to define new adaptation logic. Carries a `STATUS` tag. |
| `ThresholdRule` | class | *(status: experiment)* Reactive heuristic (adapt when damage exceeds a threshold). |
| `SEURule` | class | *(status: reference)* Parity-gated NumPy port of the DYNAMO-M SEU rule. |
| `preferred_decision_rule` | func | **Returns the preferred rule for this environment**, with a parity-verified fallback. |
| `AgentState` | dataclass | Vectorised per-agent state arrays. |
| `CouplingConfig` / `DecisionConfig` / `NetCDFMappingConfig` | dataclasses | Configuration. |
| `draw_year_events` / `generate_event_sequences` | funcs | Unified stochastic event generator. |
| `DynamoLiveRule` / `DynamoMNotAvailable` / `DYNAMO_M_AVAILABLE` | class/exc/flag | *(status: preferred)* Live coupling to the native DYNAMO-M `DecisionModule`, for floodproofing and insurance. Guarded import. |
| `STATUS_PREFERRED` / `STATUS_REFERENCE` / `STATUS_EXPERIMENT` / `STATUS_VERIFICATION` / `STATUS_DEPRECATED` | str | The status vocabulary; compare against `rule.STATUS`. |
| `FloodAdaptSLRModel` / `CoastalNodePopulation` / `MesaAgents` / `run_mesa_native` | classes/func | *(status: verification)* Framework-free mirror of the tick loop; the bit-parity gate. |
| `FloodAdaptSLRModelFull` / `CoastalNodePopulationFull` / `AgentsFull` / `run_mesa_native_full` / `HoneybeesNotAvailable` / `HONEYBEES_AVAILABLE` | classes/func/exc/flag | *(status: preferred)* **Recommended entry point.** A real honeybees `Model` owns the clock. |
| `ABMSimulator` | class | **Deprecated** standalone simulator (kept for a regression test). |
| `DynamoDecisionBridge` | class | Internal `_core` plumbing, re-exported for backward compat. |

`setup_lookup_table` is intentionally **not** imported by `__init__.py` (it needs the
full `flood-adapt` library, absent in some envs). Import it explicitly:

```python
from floodadapt_abm.setup_lookup_table import create_lookup_table
```

**Minimal end-to-end run:**

```python
import xarray as xr
from floodadapt_abm import SimulationEngine, preferred_decision_rule, CouplingConfig

ds = xr.open_dataset("lookup_table_charleston_beta_release_ABM_probabilistic_set.nc")
cfg = CouplingConfig()                      # all defaults (Charleston-calibrated)
rule = preferred_decision_rule(cfg.decision)   # native DYNAMO-M when available
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
   ThresholdRule  SEURule       DynamoLiveRule        (your rule)
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
   run_mesa_native(engine, ...)          (framework-free mirror owns time)
   run_mesa_native_full(engine, ...)     (real honeybees Model owns time)
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
| `income_to_wealth_ratio` | `4.14` | income→wealth multiplier, explicit-income runs only (the default mode interpolates the full percentile table) |
| `max_events_per_year` | `None` | occurrence cap per year; `None` disables it (see Sec.6) |
| `lifespan_dryproof` | `75` | dry-floodproofing service life (years); triggers reset |
| `event_draw_mode` | `"poisson"` | hazard draw: `"poisson"` (exact rates) / `"bernoulli_clip"` (clips rates above 1) |
| `nuisance_freq_threshold` | `None` | drop events with `freq >` threshold from the whole catalogue (set `1.0` for the Charleston set) |
| `cap_policy` | `"largest_damage"` | surplus-occurrence discard: keep most damaging (deterministic) / `"random"` (uniform) |
| `seu_prob_mode` | `"exceedance"` | SEU probabilities: `p = 1 − e^(−freq)` / `"raw_freq"` (rates used directly) |
| `perception_mode` | `"severity"` | post-flood spike scales with damage severity / `"binary"` (native behaviour) |
| `flood_significance_threshold` | `0.01` | min damage severity to register as flood experience (`0.0` = any positive damage) |
| `perception_severity_form` | `"power"` | the only supported value (`"saturating_exp"` / `"threshold_linear"` were removed, see below) |
| `perception_severity_exponent` | `0.5` | severity exponent γ > 0: γ→0 approaches binary, γ<1 concave, γ=1 linear, γ>1 near-miss. `γ = 0` raises |
| `income_mode` | `"synthetic_lognormal"` | native income port; the only supported value (`"mpd_ratio"` was removed, see below) |
| `median_income` | `70000.0` | regional median income for the synthetic distribution (site-specific) |
| `mean_median_inc_ratio` | `1.15` | lognormal spread (native UN-WIID fallback) |
| `adaptation_total_cost` | `None` | fixed per-household cost (native style); `None` → `fraction·max_pot_dmg` |
| `include_insurance` | `False` | offer insurance as a third decision option (matches native default) |
| `insurance_deductible` | `0.1` | damage share still borne when insured (native hard-coded value) |
| `insurance_pricing` | `"community"` | premium rating: `"community"` = flat mean-EAD rate (native `InsurerAgent`) / `"risk_based"` = each household's own expected payout `(1-deductible)*EAD_i` |
| `insurance_loading` | `1.0` | multiplier on the actuarial premium (insurer margin); `1.0` = fair |
| `insurance_subsidy` | `0.0` | fraction of the premium paid by a public scheme (premium analogue of native's adaptation subsidy) |

**No preset bundles the alternatives.**  Each switch above accepts its
alternative individually, so name the ones you want explicitly; no test pins a
stored run of any bundle.  One value is unsupported and raises:
`income_mode="mpd_ratio"` made income and adaptation cost both proportional to
`max_pot_dmg`, so the affordability gate reduced to one population-wide constant
and never bound for any household.

Risk-perception law:

$$\mathrm{risk\_perc} = P \cdot 1.6^{\,\text{risk\_perc\_coef} \cdot \text{flood\_timer}} + \mathrm{risk\_perc\_min},$$

where the peak $P = \mathrm{risk\_perc\_max}$ in binary mode and, in severity
mode,

$$P = \mathrm{risk\_perc\_max} \cdot s^{\gamma}, \qquad
s = \min\!\left(\frac{\text{realised damage}}{\text{max\_pot\_dmg}},\, 1\right),$$

with $\gamma$ = `perception_severity_exponent` > 0. The severity $s$ is the
share of the building's maximum potential damage realised in the flood, so
$s = 0.25$ means the flood destroyed a quarter of what the home could lose.

A total loss always reproduces the full binary-mode spike, whatever γ is.
γ is the single shape parameter and covers the whole hypothesis range:

| γ | Shape | Reading | s = 0.10 | s = 0.25 |
|---|---|---|---|---|
| → 0⁺ | step-like | approaches binary/native | 63 % (γ=0.2) | 76 % (γ=0.2) |
| 0.5 (default) | concave | availability heuristic | 32 % | 50 % |
| 1.0 | linear | damage-proportional | 10 % | 25 % |
| 2.0 | convex | near-miss, small floods discounted | 1 % | 6 % |

`γ = 0` raises a `ValueError` rather than being accepted: `0.0 ** 0.0 == 1.0`
would spike every agent, including agents that never flooded. Use
`perception_mode="binary"` for the exact native response.

Calibration status: the default γ = 0.5 is argued from the availability
heuristic, not fitted to data, and is not calibrated for Charleston. Sweep it
or fit it from a small survey; both recipes are in
`docs/calibration_validation_guide.md` (Tier 1 and Tier 3).

**Two further severity forms are not supported.** `"saturating_exp"`
($P = \mathrm{rp}_{\max}\,(1 - e^{-k s})/(1 - e^{-k})$) and
`"threshold_linear"`
($P = \mathrm{rp}_{\max}\operatorname{clip}((s - s_0)/(1 - s_0), 0, 1)$) were
removed after
their model outcomes were measured on both the synthetic and the real Charleston
table and found to lie *on* the γ curve: `saturating_exp` at k = 3 reproduces
γ ≈ 0.5, and `threshold_linear` at s0 = 0.1 reproduces γ ≈ 1.3. They were
reparameterisations of γ, not distinct hypotheses. Passing either now raises a
directed `ValueError` naming the equivalent γ, so an old config migrates in one
edit. Rationale, the measured equivalence table and the figures are in
`docs/architecture.md` ("Severity response"); the survey-fitting recipe is in
`docs/calibration_validation_guide.md`.

**Insurance pricing modes, in plain language.** `"community"`: every household
pays the same flat premium, equal to the pool's mean expected annual damage
(native's rule). Low-risk households cross-subsidise high-risk ones.
`"risk_based"`: each household pays its own expected payout,
$\pi_i = (1 - d)\,\mathrm{EAD}_i$ with $d$ = `insurance_deductible`. This is
the actuarially fair premium (the price that exactly covers the insurer's
expected payments): no cross-subsidy, cheap for low-risk households, expensive
for high-risk ones. The offer the household actually faces is

$$\pi_i^{\text{offer}} = \lambda\,(1 - \sigma_s)\,\pi_i,$$

with $\lambda$ = `insurance_loading` and $\sigma_s$ = `insurance_subsidy`.

How to set the three knobs:

- `insurance_deductible` ($d$, default 0.1, native's hard-coded value): from
  the policy terms of the scheme being modelled.
- `insurance_loading` ($\lambda$, default 1.0 = "at cost"): the loading is
  what a real insurer adds on top of the expected-loss price to pay staff,
  reinsurance and capital held for bad years. Set it from the scheme's **loss
  ratio** (claims paid / premiums collected):
  $\lambda \approx 1 / \text{loss ratio}$,
  so paying out 75 cents per premium dollar means $\lambda \approx 1.3$.
- `insurance_subsidy` ($\sigma_s$, default 0.0): the share of the bill paid
  publicly. The default is 0 because the unsubsidised market is the baseline
  (and the native-parity setting); any positive value is a policy scenario.
  For scale, the pre-reform NFIP's implicit subsidy was about 0.6 (subsidised
  policies paid 35-40 % of the full-risk rate; GAO-13-607), and the notebook's
  0.9 is deliberately deeper. Sweep it rather than calibrate it.

Provenance: community rating is native DYNAMO-M's `InsurerAgent` and mirrors
real community-rated flood schemes (e.g. the US NFIP before its 2021 reform;
Michel-Kerjan 2010); risk-based pricing is the standard actuarial benchmark
from insurance economics (and the direction of NFIP "Risk Rating 2.0"; FEMA
2021). The trade-off between them is real and measured: risk-based premiums
reward risk reduction but become unaffordable exactly in the high-risk tail
(Hudson et al. 2016; after the NFIP's reform, Gourevitch, Snyder & Kousky
2025), which is why the literature pairs them with means-tested,
outside-the-pool subsidies (Kousky & Kunreuther 2014) — the design
`insurance_subsidy` implements, since it discounts the household's bill while
the risk-based price itself stays intact. The beyond-native modes exist (1) as
a diagnostic — to test whether the flat community rate is what suppresses
uptake (it is not; the expenditure cap binds under either rule, see notebook
§5) — and (2) so that rating rule × loading × subsidy spans the premium
designs a public insurer or regulator could realistically set. Full argument
and citations: `docs/architecture.md` §9.7 and its References.

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

The Strategy Pattern seam. The primary method is the three-way `decide`:

```python
decide(agent_state, damages_this_year, damages_no_adapt, damages_adapt,
       event_freqs, max_pot_dmg, adaptation_costs,
       insurance_premium=None) -> np.ndarray[int8]   # 0 nothing / 1 adapt / 2 insure
```

The base class implements it on top of the two-way `should_adapt`, which returns a
boolean mask of **currently non-adapted** agents that newly adapt this year, so
two-way rules (including third-party ones) work unchanged. Adaptation is never
double-applied within a year.

### Rule status

Every rule carries a `STATUS` class attribute saying how it is meant to be used.

| Rule | `STATUS` | Use it for |
|---|---|---|
| `DynamoLiveRule` | `preferred` | Application runs: native DYNAMO-M decides floodproofing and insurance |
| `SEURule` | `reference` | When DYNAMO-M is absent, and for per-household (risk-based) premiums |
| `ThresholdRule` | `experiment` | Simple baseline, for comparison |

`DynamoLiveRule` and `SEURule` are parity-gated (relative EU error < 1e-4, identical
actions), so their results are interchangeable. Third-party subclasses inherit
`STATUS = "experiment"` unless they override it.

### `preferred_decision_rule(config, dynamo_path=None, ...)`

Returns the preferred rule available in this environment, so callers do not
hand-write availability checks:

```python
from floodadapt_abm import CouplingConfig, preferred_decision_rule

cfg = CouplingConfig()
rule = preferred_decision_rule(cfg.decision)
rule.STATUS      # "preferred" if DYNAMO-M is installed, else "reference"
```

It returns `DynamoLiveRule` whenever DYNAMO-M is importable **and** the
configuration is expressible natively; `SEURule` otherwise. The port is selected in
exactly two cases: DYNAMO-M is absent, or per-agent premiums are configured
(`include_insurance=True` with a pricing mode other than `"community"`). Native
`calcEU_insure` discounts `premium.mean()`, so risk-based pricing is inexpressible
natively, and `DynamoLiveRule.decide` raises on a varying premium rather than
silently averaging it.

Parallel runs are safe with either rule (`DynamoLiveRule.clone()` builds a fresh
native module per worker), but the native kernels hold the GIL, so a parallel
native run is correct yet effectively serial. Pass `SEURule` explicitly when
parallel throughput matters.

### `DecisionRule(ABC)`
- `__init__(config)` — stores a `DecisionConfig`.
- `STATUS` — status tag; defaults to `"experiment"` for subclasses.
- `clone(rng_seed=None)` — independent copy for parallel execution (forks the RNG);
  overridden by stochastic rules.
- `decide(...)` — the three-way contract; defaults to delegating to `should_adapt`.
- `@abstractmethod should_adapt(...)`.

### `ThresholdRule(DecisionRule)`  *(status: experiment)*
Reactive damage-threshold heuristic: adapt when this
year's realised damage exceeds `damage_threshold` (default `0.3`) of max potential
damage. Ignores income, affordability, risk perception and insurance. Deterministic;
used as the bit-for-bit regression oracle and as a comparison baseline.

`ThresholdRule(config, damage_threshold=0.3)`.

### `SEURule(DecisionRule)`  *(status: reference)*
The pure-NumPy port of the DYNAMO-M SEU science. Computes `EU_do_nothing`,
`EU_adapt` and (when insurance is enabled) `EU_insure` (CRRA utility over
time-discounted NPVs, integrated over perceived flood probability) and picks the
highest. Uses the ported kernels in `_core`.

`SEURule(config, rng=None, amenity_value=None)` — pass an RNG for stochastic error
terms (`error_interval > 0`); `clone()` forks the RNG for parallel sequences.

```python
from floodadapt_abm import SEURule, ThresholdRule, CouplingConfig
cfg = CouplingConfig()
seu = SEURule(cfg.decision, rng=None)          # reference port
baseline = ThresholdRule(cfg.decision, damage_threshold=0.3)
```

---

## 6. Stochastic events — `event_utils.py`

The **single** stochastic event generator.

- `draw_year_events(event_names, event_freqs, rng, max_events_per_year=None, dt=1.0,
  mode="poisson", cap_policy="largest_damage", event_severity=None)` — draw the event
  occurrences of one year.
  - `mode="poisson"` (default, **use this**): each event occurs `n ~ Poisson(freq·dt)`
    times. The returned list may contain the same event more than once, and realised
    damages sum per occurrence.
  - `mode="bernoulli_clip"`: one Bernoulli trial per event with `p = min(freq·dt, 1)`.
    Rates above `1/dt` are clipped to certainty. Kept as an option for
    sensitivity work; its RNG call order is frozen so matched-hazard comparisons
    stay controlled.
  - **Choosing between them.** The two distributions answer different questions.
    Bernoulli asks *did it happen* and takes a probability in `[0, 1]`, so it
    returns 0 or 1 and must clip any rate above 1. Poisson asks *how many times
    did it happen* and takes a rate of any size, so it returns 0, 1, 2, … and
    represents the rate exactly. `freq` is a rate, so Poisson is the matching
    model. Even where nothing clips the two still differ: `P(at least one)` is
    `freq` under Bernoulli and `1 − exp(−freq)` under Poisson, which changes how
    often agents experience a flood year and therefore their risk perception.
    See `docs/architecture.md` §9.3.
  - When a cap binds, `cap_policy="largest_damage"` keeps the most damaging
    occurrences deterministically (requires `event_severity`); `"random"`
    discards uniformly, at the same rate for extremes and nuisance events.
- `generate_event_sequences(...)` — `n_seq` independent per-year event sequences
  (same parameters).

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

## 8. Preferred rule — `dynamo_live_rule.py`  *(status: preferred)*

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

## 9. Verification mirror — `mesa_native.py`  *(status: verification)*

A **framework-free** mirror of DYNAMO-M's `SLRModel.step()` tick loop that inverts
*who owns time*: instead of `engine.run` looping years, a small model advances one tick
at a time — while still delegating all numerics to `engine.step`.

- `CoastalNodePopulation(model)` — vectorised household group; `state`/`n` properties;
  `step()` advances one year.
- `Agents(model)` — steps each agent group per tick (exported as `MesaAgents`).
- `FloodAdaptSLRModel(engine, slr_values, seed, interp_method='linear',
  track_eu=False)` — the framework-free model that owns time. Uses the
  `state_epoch` staleness guard (`_check_not_stale`) so a shared engine can't be
  silently invalidated. `step()` / `run_model()` mirror `SLRModel`.
- `run_mesa_native(engine, slr_values, no_seq=1, seed=None, interp_method='linear',
  track_eu=False)` — drop-in analogue of `engine.run`; **bit-for-bit identical**.

---

## 10. Preferred driver — `mesa_native_full.py`  *(status: preferred)*

The **final integration step**: binds the **real honeybees `Model`** as the
time-owning base class (as the upstream `SLRModel` does) and routes decisions through
the native DYNAMO-M `DecisionModule` (via `DynamoLiveRule`), feeding a deterministic
coastal-node population entirely from the FloodAdapt lookup table through the
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
  `timestep` is a 0-based property alias of `current_timestep`. Reuses the
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
with that data ecosystem to drive on a dependency-free population, so it reuses the
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
the native driver, exercised every tick.

- `CoastalNodeArrays` (dataclass) — dependency-free mirror of the native node array
  set: `property_value`, events-first `damages_coastal_cells`, `p_floods`, `adapt`,
  `time_adapt`, `_flood_plain` geom_id.
- `LookupTableAdapter(engine, geom_id='floodadapt_flood_plain')`
  - `populate(slr_value, interp_method='linear')` — **forward**: build node arrays from
    the lookup table at `slr_value` (read-only, no RNG).
  - `write_back(node)` — **reverse**: route the node's adaptation state back into the
    engine's live `AgentState`, with `object_id` alignment guards (idempotent).
- `round_trip_check(engine, slr_value, interp_method='linear')` — executable bit-parity
  contract: proves routing state through the node is a simulation
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
- `_linear_at_slr(values, slr_arr, slr_target)`: internal, the `linear` branch.
  A dtype-pinned linear interpolation used instead of SciPy's `interp1d`.
  The cube is `float32` and the SLR grid `float64`, and SciPy's internal
  promotion of that mix is not stable across SciPy builds or NumPy promotion
  regimes, which flipped stored damages by one ulp and broke the bit-parity
  gates on some platforms. This helper takes the y-difference at `float32` (the
  historical semantics) and runs the division and
  affine step at `float64`, so results are identical on every platform. Do not
  route the `linear` branch back through SciPy; `tests/test_lookup_interpolation.py`
  asserts it imports none. `cubic` still uses SciPy and carries no parity gate.
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

## 14. Standalone simulator — `abm_simulator.py`

`ABMSimulator` — the **deprecated** stage-2 threshold-rule simulator, retained for
backward compatibility and the bit-for-bit regression against
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
  06_mesa_native_driving.py    verification-mirror demo
  07_mesa_native_full.py       preferred-path demo (+ inline bit-parity)
tests/                   full pytest suite (219 tests; self-contained mock datasets)
  test_mesa_native_full.py     22 tests: triple bit-parity, honeybees clock,
                               object graph/adapter, staleness guard, native path
verification/            vendored, portable batteries emitting md/JSON/figures
  phase1_seu_battery/          V1–V6 SEU validation
  phase4a_parity/              ported vs native EU parity
  phase4b_mesa_native/         tick-driver bit-parity gate
  mesa_native_full/            native-driver G1–G4 battery (gate_pass: True on real table)
  real_table_gate/             full Charleston run
  preflight_4b_full/           import/instantiate checks
```

Run everything (set `DYNAMO_M_PATH` to enable the native-parity tests):

```powershell
$env:DYNAMO_M_PATH = "c:\repos\DYNAMO-M\DYNAMO-M"
pytest -q            # 219 passed
```

Guarded tests skip cleanly when honeybees or DYNAMO-M is unavailable.

---

## 16. Global assumptions & invariants

- **Residential-only MVP:** only buildings whose `primary_object_type` contains `RES`
  are simulated (substring match, case-sensitive).
- **Two strategies:** `no_measures` (baseline) and `floodproof_all_0` (dry-floodproof).
- **RP-based EventSet:** frequencies are annual occurrence rates from a return-period
  EventSet. The hazard draw is Poisson by default (`event_draw_mode`); the SEU
  integral consumes exceedance probabilities `p = 1 − e^(−freq)` (`seu_prob_mode`).
  Sub-annual (`freq > 1`) events should be dropped via `nuisance_freq_threshold=1.0`.
- **Event cap policy:** disabled by default (`max_events_per_year=None`); when set,
  surplus occurrences are discarded by `cap_policy` (`"largest_damage"` default,
  `"random"`).
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
mathematics and the diagrams, see
[`docs/architecture.md`](docs/architecture.md).*


---

## Appendix: three-way decisions, insurance and income utilities

Design rationale for everything below: `docs/architecture.md` (the hazard-draw
section and Appendix B).

### Three-way decision contract

`DecisionRule.decide(agent_state, damages_this_year, damages_no_adapt,
damages_adapt, event_freqs, max_pot_dmg, adaptation_costs,
insurance_premium=None) -> np.ndarray[int8]` returns one action code per agent:
`ACTION_DO_NOTHING` (0), `ACTION_ADAPT` (1), `ACTION_INSURE` (2).  The base-class
default delegates to `should_adapt` and never insures, so existing two-way rules
(including third-party subclasses) work unchanged.  `SEURule` and `DynamoLiveRule`
implement the native three-way comparison (`adapt = EU_a > EU_dn and EU_a ≥ EU_i`;
`insure = EU_i > EU_dn and EU_i > EU_a`; both restricted to non-adapted agents).
The `event_freqs` parameter receives the engine's `p_floods_seu` (converted per
`seu_prob_mode`); its name is retained for backward compatibility.

### Insurance results (only present when `include_insurance=True`)

| Key | Shape | Meaning |
|---|---|---|
| `insured_history` | `(no_seq, n_agents, n_years)` bool | insured status per year |
| `out_of_pocket_history` | same, damage dtype | damages after the deductible for insured agents |
| `premium_history` | `(no_seq, n_years)` | **mean** premium offered that year (equals the flat rate under community pricing) |
| `premium_paid_history` | `(no_seq, n_agents, n_years)` | premium actually paid per household (0 when uninsured) |
| `insured_fraction` | `(no_seq, n_years)` | share of households insured |

`damage_history` always stays **gross**.  Coverage timing: a policy decided in
year *t* covers year *t+1*; year 0 starts uninsured.  Insurance is annual and
never coexists with physical floodproofing.

Premiums are re-derived every year from the expected annual damage at that
year's SLR (`SimulationEngine._compute_premium_offer`).  Note the native
community rate charges the full mean EAD while covering only
`1 - deductible` of each loss (an implicit ~11 % loading); the `risk_based`
mode prices the cover actually sold, `(1 - deductible) * EAD_i`.

### Per-agent state additions

`AgentState.last_flood_severity` (float32; severity of the most recent significant
flood, drives the severity-scaled perception peak) and `AgentState.is_insured`
(bool).  The `CoastalNodeArrays.adapt` field uses the native encoding
`0/1/2 = nothing/floodproofed/insured`.

### Income utilities

The synthetic income model takes **two independent inputs**, and they need
different data.  Calibrate them separately:

| Axis | What it fixes | Set by | Needs building locations? |
|---|---|---|---|
| **Marginal distribution** | how much a household at percentile *p* earns | `DecisionConfig.median_income`, `mean_median_inc_ratio` | **No** |
| **Rank assignment** | which building sits at percentile *p* | `SimulationEngine(..., income_percentile_per_agent=...)` | **Yes**, unless a proxy is assumed |

The percentile helpers below address the second axis only.  The first is a
property of the study area as a whole and is measurable from published county
totals even when the lookup table carries no geometry, so it is worth
calibrating at every tier.

#### Rank assignment (per-agent percentiles)

`derive_income_percentiles(buildings_gdf, regions_gdf, income_column)`
spatially joins building footprints to (e.g. ACS block-group) income regions and
converts each building's regional income to its empirical percentile (±5-percentile
jitter within a region).  Save as `income_percentiles_<site>.npy` and pass via
`SimulationEngine(..., income_percentile_per_agent=...)`.

**The building footprints are a hard requirement.**  The lookup `.nc` carries
only `max_pot_dmg` and `primary_object_type` on its `object_id` axis, so
nothing in the table says where a building is.  The geometry must come from the
FloodAdapt database that produced the table
(`fa.get_building_footprint_impacts(scenario)`, needs the `[pipeline]` extra).
Without it, ACS income data alone cannot be linked to agents and the value
proxy below is the only option.

When the lookup table carries no building locations (the shipped `.nc` schema
does not), `percentiles_from_value_proxy(values, rank_correlation=0.5, seed=0)`
derives percentiles from the building value (`max_pot_dmg`) through a *noisy*
rank link (Gaussian copula): the percentile tends to follow the building-value
rank with target Spearman correlation `rank_correlation` (income and housing
value correlate at roughly 0.4–0.6 in household-finance data), without
recreating a degenerate `income ≡ value` identity.  `rho = 0` equals
the uniform fallback; `rho = 1` is strictly monotone.  Numpy-only, dedicated
RNG, an explicit assumption to be replaced by the spatial join when footprints
are available.

#### Marginal distribution (measure it, then check the shape)

Four helpers calibrate and validate the regional income distribution.  They
need no geometry, no geopandas and no new dependency (stdlib `urllib` plus
scipy).  A free Census Data API key is read from `CENSUS_API_KEY`.

**Why a lognormal, and what the two numbers do.**  Regional income is modelled
as lognormal (`log(income) ~ Normal(mu, sigma)`), the standard choice because
income is right-skewed: most households sit below the average and a thin tail
of high earners pulls the average up.  ACS publishes statistics, not `mu` and
`sigma`, so the code converts.  The **median** is the middle household; the
**mean** is total income over households, which a few rich households lift
above the median.  Both conversions are exact for a lognormal:
`mu = ln(median)` and `sigma = sqrt(2 ln(mean/median))`.  The pair is
therefore *median plus mean/median ratio*, not *mean plus standard
deviation*: the median fixes the centre and the ratio fixes the spread, and
the ratio is the one that moves results.  `_synthesize_income_wealth`
(`_core/dynamo_decision_bridge.py`) draws 5,000 sorted samples from it once
per run on a dedicated seed-derived generator; each household reads its income
off that curve at its own percentile, wealth follows via the native
wealth-to-income table, and both feed the `expenditure_cap` affordability
check.  Charleston: ratio 1.4879 gives `sigma = 0.892`, against 0.529 for the
default 1.15.

**Neither fetcher caches.**  Each call issues a fresh HTTP request; there is no
disk cache and no memoisation, so an offline run cannot fetch and callers should
pin the values they get (notebook 2 pins them as constants and re-fetches only
to report drift).

Calibration fits exactly two parameters (`median_income`,
`mean_median_inc_ratio`, which map to the lognormal's `mu`/`sd`) from two
independent published numbers (B19013 median; B19025/B11001 mean), so the
fit is exact by construction, not an optimisation.  Validation does not
split rows: ACS never publishes individual responses, only aggregate
tabulations.  It checks the fit against a *different* published table of
the same population, B19001's 16-bracket histogram, which the fit never
touches.  See `docs/calibration_validation_guide.md` for the full
mechanics.

| Function | Purpose |
|---|---|
| `fetch_acs_county_income(state_fips, county_fips, api_key=None, year=2024)` | Returns `median_income`, `mean_income`, `mean_median_ratio`, `n_households`. ACS publishes no mean, so it is computed as aggregate household income (B19025) over households (B11001); the median is B19013 |
| `fetch_acs_income_brackets(state_fips, county_fips, ...)` | Observed household counts and shares across the 16 income brackets of ACS table B19001 |
| `lognormal_bracket_shares(median_income, mean_median_ratio, edges=ACS_B19001_EDGES)` | Pure function: the bracket shares implied by the engine's own fit (`mu = ln(median)`, `sd = sqrt(2 ln(mean/median))`) |
| `bracket_fit_distance(observed, predicted)` | Total-variation distance, readable as *the fraction of households placed in the wrong bracket* (0 is perfect) |

`ACS_B19001_EDGES` and `ACS_B19001_LABELS` are the bracket edges and labels.

Two numbers fit a lognormal, but they do not show that its **shape** is right;
that is what the bracket comparison is for.  Worked example, Charleston County
SC (`state:45 county:019`, ACS 2020-2024 5-year), verified against the live API:

```python
from dataclasses import replace
from floodadapt_abm.income_utils import (
    bracket_fit_distance, fetch_acs_county_income,
    fetch_acs_income_brackets, lognormal_bracket_shares,
)

stats = fetch_acs_county_income("45", "019")          # ACS 2020-2024 5-year
# median_income 88,494 | mean_income 131,674 | mean_median_ratio 1.4879

observed = fetch_acs_income_brackets("45", "019")["shares"]
bracket_fit_distance(observed, lognormal_bracket_shares(70_000, 1.15))    # 0.326
bracket_fit_distance(observed, lognormal_bracket_shares(88_494, 1.4879))  # 0.072

cfg = replace(config.decision,
              median_income=stats["median_income"],
              mean_median_inc_ratio=stats["mean_median_ratio"])
```

The package defaults (70,000, ratio 1.15) misplace about 33 % of Charleston
households, chiefly by giving the county a thin upper tail where the real one
is fat (2.4 % predicted above $200k against 17.2 % observed).  The measured
pair misplaces about 7 %.  Because `sd = sqrt(2 ln(mean/median))`, the ratio
and not the median controls dispersion: 1.15 to 1.4879 widens `sd` from 0.529
to 0.892, which is what makes the affordability constraint bite realistically.

The figures above are the ACS 2020-2024 5-year release, the latest published.
`fetch_acs_county_income` and `fetch_acs_income_brackets` default to
`year=2024`; pass `year=` explicitly to pin a vintage, and use the same
vintage for the fit and for the B19001 validation.

The residual misfit is concentrated in the lowest brackets, where a lognormal
cannot reproduce the spike of near-zero-income households.  That is a limit of
the functional form (inherited from native DYNAMO-M), recorded rather than
tuned away.

Pin the measured values as constants in the run script or notebook so results
reproduce offline, and re-fetch to check for drift rather than silently
substituting new numbers.
