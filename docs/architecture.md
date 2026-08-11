# FloodAdapt-ABM x DYNAMO-M: architecture and model description

**What this document is.** The design record for the coupled model as it is
implemented today. It answers *why and how the system is built*, and it
specifies the agent-based model precisely enough to reproduce or review it.

**What it is not.** It is not a user manual and not a project history. For the
API, see [`../floodadapt_abm_documentation.md`](../floodadapt_abm_documentation.md)
(the reference manual). For calibration and validation method, see
[`calibration_validation_guide.md`](calibration_validation_guide.md) (the method
guide). Development history lives in the git log and in the local
`docs/progress/` folder, not here.

**Audience.** Developers extending the model, and reviewers checking the
science.

**How it is organised.** Part I is the software architecture: components,
contracts, execution paths. Part II describes the model itself, following the
**ODD protocol** (Overview, Design concepts, Details), the standard structure
for describing agent-based models in the scientific literature. Part III covers
verification and validation. The appendices hold the glossary, the deviations
from native DYNAMO-M, and worked-example pointers.

---

# Part I. Software architecture

## 1. Purpose and system context

**Research question.** How do policy incentives and lived flood experience
shape household-level adaptation, and therefore the flood risk a coastal city
carries under sea-level rise?

Two repositories collaborate to answer it:

| Repository | Role | Key sources |
|---|---|---|
| **FloodAdapt-ABM** (this repo) | Owns time, data and bookkeeping. Turns a precomputed impact table into Monte-Carlo time series of damages and household decisions. | `simulation_engine.py`, `decision_rule.py`, `mesa_native_full.py` |
| **DYNAMO-M** (upstream, optional at runtime) | Supplies the Subjective Expected Utility (SEU) decision science. | `decision_module.py`, `agents/coastal_nodes.py`, `hazards/flooding/flood_risk.py`, `settings.yml` |

### 1.1 The two-stage pipeline

The system is deliberately split into two stages joined by **one file format**.
Stage 1 is expensive and runs rarely; stage 2 is cheap and runs many times.

```
Stage 1 (offline, hours)                    Stage 2 (online, seconds to minutes)
+-------------------------------+           +-----------------------------------+
| FloodAdapt: SFINCS + FIAT     |           | floodadapt_abm                    |
| over event x SLR x strategy   |  --.nc--> | Monte-Carlo ABM over years        |
| setup_lookup_table.py         |           | SimulationEngine + DecisionRule   |
+-------------------------------+           +-----------------------------------+
     needs flood-adapt                          pure NumPy / SciPy / xarray
```

**Why the split.** Hydrodynamic and damage modelling is far too slow to run
inside an agent decision loop. Precomputing the full grid of outcomes turns
per-year damage evaluation into an array lookup plus a one-dimensional
interpolation, which is what makes tens of thousands of Monte-Carlo sequences
feasible. The cost is that the ABM can only ask questions the grid already
answers.

### 1.2 The coupling contract

The `.nc` lookup table is the **only** interface between the stages. Keep it
stable.

| Element | Content |
|---|---|
| Dimensions | `object_id x slr x strategy x event` |
| Data variables | `total_damage`, `inun_depth` |
| `object_id` attributes | `max_pot_dmg`, `primary_object_type` |
| `event` attribute | `freq` (annual occurrence **rate**, not a probability) |
| `strategy` values | `"no_measures"`, `"floodproof_all_0"` |

Schema strings live in `NetCDFMappingConfig`; `setup_lookup_table.py` is the
other side of the contract. Change both together. `object_id` and its
`max_pot_dmg` attribute must stay aligned: never reorder one independently.

**Hard requirement on the event set.** The SEU integral treats event
frequencies as points on an exceedance curve, so the event set must be a
genuine probabilistic set with meaningful rates. Sub-annual events
(`freq > 1`) cannot sit on an exceedance curve and must be filtered (§9.3).

**Reference schema** (the validated Charleston table):

| Property | Value |
|---|---|
| `object_id` | 61,858 buildings, of which 57,976 (about 94 %) are residential |
| `slr` | `[0.0, 0.5, 1.0, 1.5, 2.0]` feet |
| `strategy` | `['no_measures', 'floodproof_all_0']` |
| `event` | 207 events (`event_0000` to `event_0206`) |

The residential filter is substring-based (`np.char.find(types, 'RES') >= 0`),
so it captures both `'RES'` and mixed types such as `'COM_RES'`.

Units (`dmg_unit`, `slr_unit`) are display-only. Numeric values must already
match what was fed to FloodAdapt; Charleston uses feet end to end.

## 2. Component architecture

**The organising principle: `SimulationEngine` owns *time and data*; a
`DecisionRule` owns *behaviour*.** Swapping the rule is the only change needed
to move between the threshold heuristic and the DYNAMO-M science. This is the
Strategy pattern, and it is what keeps the numeric kernel single-sourced.

```mermaid
classDiagram
    direction LR

    class SimulationEngine {
        <<kernel: owns time and data>>
        +AgentState state
        +DecisionRule decision_rule
        +prepare_damages(slr)
        +step(year, slr, rng)
        +run(slr_values, no_seq, n_jobs) dict
        -_clone_for_worker() SimulationEngine
    }

    class DecisionRule {
        <<abstract: owns behaviour>>
        +STATUS str
        +DecisionConfig config
        +should_adapt(...) ndarray
        +decide(...) ndarray
        +clone(seed) DecisionRule
    }

    class DynamoLiveRule {
        +STATUS = preferred
        +decide(...) ndarray
    }
    class SEURule {
        +STATUS = reference
        +decide(...) ndarray
    }
    class ThresholdRule {
        +STATUS = experiment
        +damage_threshold float
        +should_adapt(...) ndarray
    }

    class AgentState {
        <<vectorised state>>
        +is_adapted ndarray
        +is_insured ndarray
        +flood_timer ndarray
        +risk_perception ndarray
        +time_adapted ndarray
        +last_flood_severity ndarray
    }

    class DynamoDecisionBridge {
        <<internal data layer>>
        +income ndarray
        +wealth ndarray
        +p_floods_seu ndarray
        +prepare_damages(slr, method)
        +compute_expected_annual_damages()
        +clear_interp_cache()
    }

    class LookupTable {
        <<xarray Dataset>>
        +object_id_x_slr_x_strategy_x_event
    }

    class FloodAdaptSLRModelFull {
        <<honeybees Model: owns the clock>>
        +step()
        +run_model()
    }

    class NativeDecisionModule {
        <<DYNAMO-M, optional dependency>>
        +calcEU_do_nothing()
        +calcEU_adapt()
        +calcEU_insure()
    }

    DecisionRule <|-- DynamoLiveRule
    DecisionRule <|-- SEURule
    DecisionRule <|-- ThresholdRule

    SimulationEngine o-- DecisionRule : delegates behaviour
    SimulationEngine *-- AgentState
    SimulationEngine *-- DynamoDecisionBridge
    DynamoDecisionBridge ..> LookupTable : reads
    FloodAdaptSLRModelFull ..> SimulationEngine : delegates every numeric op
    DynamoLiveRule ..> NativeDecisionModule : calls
```

### 2.1 Module responsibilities

| Module | Responsibility |
|---|---|
| `simulation_engine.py` | The single per-year kernel `step()`, the year loop, Monte-Carlo sequences, the parallel backend |
| `decision_rule.py` | The `DecisionRule` interface, the status vocabulary, `ThresholdRule`, `SEURule` |
| `dynamo_live_rule.py` | `DynamoLiveRule` (native decisions) and `preferred_decision_rule()` |
| `_core/dynamo_decision_bridge.py` | Internal data layer, the ported SEU kernels, the interpolation cache |
| `agent_state.py` | Vectorised per-agent state |
| `event_utils.py` | The unified stochastic event draw |
| `coupling_config.py` | Configuration dataclasses |
| `income_utils.py` | Case-study income-percentile helpers |
| `mesa_native_full.py` | The preferred driver: a real honeybees `Model` owns the clock |
| `mesa_native.py` | Framework-free mirror of the tick loop (verification) |
| `coastal_node_adapter.py` | Lookup table to `CoastalNode` array adapter |
| `abm_simulator.py` | Deprecated standalone simulator; do not add call sites |

### 2.2 The decision-rule interface

```python
def decide(
    agent_state,          # AgentState
    damages_this_year,    # (n_agents,)            realised damage this year
    damages_no_adapt,     # (n_agents, n_events)   catalogue at SLR_t, no measures
    damages_adapt,        # (n_agents, n_events)   catalogue at SLR_t, floodproofed
    event_freqs,          # (n_events,)            exceedance probabilities
    max_pot_dmg,          # (n_agents,)
    adaptation_costs,     # (n_agents,)            annualised loan repayment
    insurance_premium=None,   # (n_agents,) or None
) -> np.ndarray:          # (n_agents,) int8: 0 nothing / 1 adapt / 2 insure
```

The signature is deliberately wide enough to serve both an *ex-post* heuristic
(which needs only `damages_this_year`) and the *ex-ante* SEU science (which
integrates the full catalogues). A rule ignores what it does not need. The base
class implements `decide` on top of `should_adapt`, so two-way rules, including
third-party ones, work unchanged.

**Design constraints, enforced.** No `flood_adapt` or DYNAMO-M imports inside
engine or rule kernels. Fully vectorised: no per-household Python loops in the
hot path, which are about 100 times slower at real-table scale. Rules take
plain NumPy arrays and provide `clone(seed)` for parallel workers.

## 3. Execution paths and their status

Every rule and driver carries a machine-readable `STATUS` tag. The preferred
path is the **fully native coupling**: a real honeybees `Model` owning the
clock, with the native DYNAMO-M `DecisionModule` making the decisions.

### 3.1 Decision rules

| Rule | `STATUS` | What it runs | Use it for |
|---|---|---|---|
| `DynamoLiveRule` | `preferred` | Native DYNAMO-M `calcEU_do_nothing`, `calcEU_adapt`, `calcEU_insure` | Application runs. Floodproofing and insurance both decided by native code |
| `SEURule` | `reference` | The pure-NumPy port of those kernels | When DYNAMO-M is absent, and for per-household (risk-based) premiums |
| `ThresholdRule` | `experiment` | `damage/max_pot_dmg > threshold` | Simple baseline, for comparison |

`DynamoLiveRule` and `SEURule` are parity-gated: identical agent state gives a
relative EU error below 1e-4 and **identical actions**, so their results are
interchangeable and comparisons across runs stay valid.

**The native rule is used everywhere it can be.** It falls back to the port in
exactly two cases:

1. **DYNAMO-M is not installed** (it is an optional dependency).
2. **Per-household premiums are configured.** Native `calcEU_insure` discounts
   `premium.mean()` (`decision_module.py:337`): its insurer only ever issues
   one flat community rate, so it cannot express `insurance_pricing =
   "risk_based"` at all. `DynamoLiveRule.decide` raises on a varying premium
   rather than silently charging everyone the pool average. This is a
   capability limit of the native kernel, not a preference.

Parallel runs are safe with either rule: `DynamoLiveRule.clone()` builds a
fresh native module per worker, so nothing is shared across threads. The
native kernels are `@njit` without `nogil=True`, though, so they hold the GIL
and a parallel native run is correct but effectively serial. Pass `SEURule`
explicitly when parallel throughput matters; the results are parity-verified
identical.

`preferred_decision_rule(config)` encodes exactly this policy and should be
preferred over hand-written availability checks:

```python
from floodadapt_abm import preferred_decision_rule

rule = preferred_decision_rule(config.decision)
rule.STATUS   # "preferred" when DYNAMO-M is installed, else "reference"
```

Any **future DYNAMO-M coupling** (migration, the government or dike agent)
should arrive through the same seam as a new rule, inherit
`STATUS = "preferred"`, and gain a ported reference implementation plus a parity
gate. That is the extension contract.

### 3.2 Drivers (which object owns the clock)

| Driver | Status | Clock owner | Use it for |
|---|---|---|---|
| `run_mesa_native_full` | `preferred` | A real honeybees `Model` (`FloodAdaptSLRModelFull`) | Application runs |
| `SimulationEngine.run` | kernel | The engine's own year loop | Experiments, sweeps, the parallel backend `n_jobs` |
| `run_mesa_native` | `verification` | A framework-free mirror of the native tick loop | The bit-parity gate only |
| `ABMSimulator` | `deprecated` | Threshold | Backward compatibility only |

**The drivers are not alternative implementations.** They are wrappers whose
contract is bit-for-bit equality; each delegates all numeric work to
`SimulationEngine.step` with the same RNG stream (§10).

### 3.3 Why not run native DYNAMO-M end to end?

DYNAMO-M's own geography stack is heavy: multi-gigabyte amenity rasters,
gravity models for inter-municipal migration, and internal flood routing. The
goal here is a fast Monte-Carlo evaluator of the SEU decision science against a
precomputed FloodAdapt lookup table. Stripping the heavy geodata lets the
simulator run many sequences in parallel in seconds. Native
`CoastalNode.step()` is too entangled with that data ecosystem to reuse
directly, so the preferred path reuses the **validated engine kernel** for
per-tick physics and the **native `DecisionModule`** for the decision maths,
inside a real honeybees `Model`. Those subsystems stay available for a future
full-geodata run.

## 4. Runtime behaviour: one simulated year

```mermaid
sequenceDiagram
    autonumber
    participant M as FloodAdaptSLRModelFull
    participant E as SimulationEngine.step
    participant B as DynamoDecisionBridge
    participant S as AgentState
    participant R as DecisionRule

    M->>E: step(year, slr_t, rng)
    E->>B: prepare_damages(slr_t)
    B-->>E: damages_no_adapt, damages_adapt (memoised per SLR)

    E->>E: draw_year_events(freq, rng) - Poisson counts
    E->>E: realised damage, severity s = realised / max_pot_dmg

    E->>S: update_flood_experience(severity)
    Note over S: flood_timer resets on flood, else +1<br/>risk_perception = peak(s) x 1.6^(coef x timer) + min

    E->>S: lifespan reset where time_adapted >= lifespan_dryproof

    opt insurance enabled
        E->>B: premium from expected annual damages
        E->>S: settle payouts, out-of-pocket, premiums paid
    end

    E->>R: decide(state, damages, freqs, costs, premium)
    Note over R: EU_do_nothing / EU_adapt / EU_insure<br/>affordability encoded as EU = -inf
    R-->>E: actions (0 nothing / 1 adapt / 2 insure)

    E->>S: apply actions, advance time_adapted
    E-->>M: per-year records (damage, adopted, insured, premiums)
```

**Ordering matters and is fixed.** Flood experience updates *before* the
decision, so a household that floods this year decides with the elevated
perception. The lifespan reset also happens before the decision, so an expired
measure can be renewed in the same year.

## 5. Configuration

Configuration is Python dataclasses, not YAML: type-checked, IDE-completable
and diffable.

```python
from floodadapt_abm import CouplingConfig

config = CouplingConfig()                       # current defaults
config.decision.nuisance_freq_threshold = 1.0   # recommended for Charleston

# Alternative behaviours, named explicitly (see section 11)
config.decision.event_draw_mode = "bernoulli_clip"
config.decision.perception_mode = "binary"
```

`CouplingConfig` holds `DecisionConfig` (behaviour) and `NetCDFMappingConfig`
(schema strings). There is no preset bundling the alternative behaviours:
each behaviour switch accepts its alternative individually, so name the ones
you want explicitly.

## 6. Performance and scaling

At real-table scale (61,858 objects x 207 events x 5 SLR points) two
optimisations carry the runtime:

1. **Interpolation memoisation.** Damage matrices are interpolated along the
   SLR axis only, and memoised per `(SLR, method)` in the bridge, so repeated
   years at the same sea level are free. The cache costs roughly 90 MB per SLR
   value, so long trajectories with many engine instances must call
   `clear_interp_cache()` between runs or they will exhaust memory.
2. **Parallel Monte-Carlo sequences.** `run(n_jobs=N)` distributes sequences
   over a thread pool of `_clone_for_worker` engines sharing a pre-warmed,
   read-only interpolation cache. Output is bit-identical to `n_jobs=1` for
   deterministic rules.

**Interpolate only along the SLR axis** (`linear`, `nearest`, `cubic`, `floor`,
`ceil`; `cubic` needs at least four SLR points). It extrapolates outside the
grid: extend the grid in stage 1 rather than relying on that.

**The linear kernel is dtype-pinned, and deliberately does not use SciPy.**
The damage cube is `float32` while the SLR grid is `float64`, so a mixed-dtype
interpolation depends on *where* the promotion to `float64` happens. SciPy's
`interp1d` decides that internally, and that decision is an
implementation detail of a deprecated API rather than a guaranteed contract:
it can differ between SciPy builds and between NumPy promotion regimes
(NEP 50). Interpolated damages routinely land within one `float32` unit in the
last place of a rounding boundary, so such a difference flips stored damages by
a single ulp. That is invisible scientifically but fatal to the bit-parity
contract, and it produced bit-parity failures that reproduced on CI
runners while passing on developer machines. `_linear_at_slr` therefore fixes
every intermediate dtype explicitly: the y-difference is taken at the cube's
own `float32` precision, and the
division and affine step run in `float64`. The kernel then depends only on
IEEE-754 add, subtract, multiply and divide, which are exactly rounded and so
identical on every platform. `cubic` still delegates to SciPy, and is not
covered by any bit-parity gate for that reason.
`tests/test_lookup_interpolation.py` pins the contract, including a guard that
the linear path imports no SciPy.

---

# Part II. Model description (ODD protocol)

## 7. Overview

### 7.1 Purpose

The model simulates how individual households in a coastal city decide, year by
year, whether to protect themselves against flooding under rising sea levels,
and what that means for the damage the city suffers. It is built to compare
**policy settings** (for example insurance pricing rules) and **behavioural
assumptions** (for example how strongly flood experience shapes risk
perception). It is not a forecasting tool.

Patterns it is expected to reproduce: adaptation uptake rising after flood
events and decaying between them; uptake constrained by affordability; and
voluntary insurance uptake staying low under realistic premiums.

### 7.2 Entities, state variables and scales

There is **one entity type: the household**, identified with one residential
building (one `object_id`). Non-residential buildings are excluded from the
agent population but remain in the damage table.

| State variable | Type | Meaning |
|---|---|---|
| `is_adapted` | bool | Dry-floodproofing in place |
| `is_insured` | bool | Insured this year (annual contract) |
| `flood_timer` | int | Years since the last flood; drives perception decay |
| `risk_perception` | float | Multiplier on the actual flood probability |
| `time_adapted` | int | Years since floodproofing; drives the lifespan reset |
| `last_flood_severity` | float | Damage fraction of the most recent flood |

Static per-agent attributes: `max_pot_dmg` (replacement value, from FIAT),
`income`, `wealth`, `income_percentile`, `adaptation_costs`.

**Scales.** One timestep is one year; the default horizon is 30 years. Space is
implicit: households are not placed on a grid and have no neighbours. Each
agent's exposure comes from its own row of the lookup table.

### 7.3 Process overview and scheduling

Each year, in this fixed order (`SimulationEngine.step`):

1. **Hazard.** Draw event occurrences for the year (Poisson by default, §9.3).
2. **Impact.** Interpolate damage at the current sea level for each drawn
   occurrence, using the strategy matching each agent's adaptation state.
   Compute severity `s = realised damage / max_pot_dmg`.
3. **Experience.** Update `flood_timer` and `risk_perception` (§9.4).
4. **Lifespan.** Reset floodproofing that has reached `lifespan_dryproof`.
5. **Insurance bookkeeping** (when enabled): price premiums, settle payouts.
6. **Decision.** Each non-adapted household compares expected utilities and
   chooses to do nothing, floodproof, or insure (§9.6).
7. **Bookkeeping.** Apply actions and record the year.

The schedule is vectorised across agents: within a year all households observe
the same state and decide simultaneously. There is no within-year ordering
among agents, and therefore no order effect.

## 8. Design concepts

**Basic principles.** Household behaviour follows *Subjective Expected Utility*
theory as implemented in DYNAMO-M (Tierolf et al., 2023): agents choose the
option with the highest expected utility, computed over the whole flood
probability distribution, using a CRRA utility function and a discounted
planning horizon. Risk perception follows the *availability heuristic*: recent
salient events dominate risk judgement.

**Emergence.** City-level adaptation uptake and the resulting damage trajectory
emerge from independent household decisions; they are not imposed.

**Adaptation.** Agents adapt by floodproofing or insuring. Both are explicit
utility comparisons rather than rules of thumb, except in `ThresholdRule`, the
baseline.

**Objectives.** Agents maximise the expected utility of net present value over
`decision_horizon` years.

**Prediction.** Agents are myopic in one specific respect: they evaluate
today's damage distribution over the horizon and do not anticipate future
sea-level rise (Appendix B, deviation 8).

**Sensing.** Agents sense their own damage history, income, wealth, expected
damages under both strategies, and the offered premium. They do not sense other
agents.

**Interaction.** Households do not interact directly. The only coupling between
them is the insurance pool, where the community premium is the mean expected
annual damage across all households.

**Stochasticity.** Two sources: the yearly event draw, and optional decision
noise (`error_interval`, zero by default). Sequence `s` uses
`default_rng(base_seed + s)`. Decision rules never consume the hazard stream,
so runs sharing hazard settings and seed see **identical flood histories**
whichever rule is used. This is what makes controlled rule comparisons
possible.

**Collectives.** None. The insurance pool is an accounting construct, not an
agent.

**Observation.** Per year and per sequence the model records damages, adoption,
insured fraction, out-of-pocket costs, premiums offered and paid, and
optionally the expected utilities behind each decision.

## 9. Details

### Notation

The submodels below use a consistent set of symbols, defined once here. Each maps
to a config field or a derived quantity.

| Symbol | Meaning | In the model |
|---|---|---|
| $f_e$ | Annual occurrence rate of event $e$ (events/year) | `freq` attribute of the lookup table |
| $D_{i,e}$ | Damage to household $i$ in event $e$ at the current sea level | interpolated from the lookup table |
| $p$ | Annual exceedance probability of an event | $p = 1 - e^{-f}$ under `seu_prob_mode="exceedance"` |
| $s_i$ | Damage severity: the share of household $i$'s maximum potential damage realised in a flood, clipped to $[0, 1]$ | $s_i = \min(D_i / \text{max\_pot\_dmg}_i,\, 1)$ |
| $\gamma$ | Severity exponent of the perception response | `perception_severity_exponent`, default 0.5 |
| $\tau_i$ | Years since household $i$'s last significant flood | `flood_timer` |
| $c$ | Perception decay coefficient | `risk_perc_coef`, default $-3.6$ |
| $\sigma$ | CRRA risk-aversion coefficient | `risk_aversion`, default 1 (log utility) |
| $r$ | Annual discount rate | `discount_rate`, default 0.032 |
| $T$ | Decision horizon in years | `decision_horizon`, default 15 |
| $W, Y, A$ | Wealth, income, amenity value of a household | derived; see §9.5 |
| $D_i$ | Expected damages of household $i$ under the strategy being evaluated | derived |
| $\mathrm{EAD}_i$ | Expected annual damage, $\sum_e f_e D_{i,e}$ | derived (premium base) |
| $d$ | Insurance deductible | `insurance_deductible`, default 0.1 |
| $\lambda$ | Premium loading | `insurance_loading`, default 1.0 |
| $\sigma_s$ | Premium subsidy share | `insurance_subsidy`, default 0.0 |
| $r_L, L$ | Loan interest rate and duration for adaptation | `interest_rate` (0.04), `loan_duration` (16) |

### 9.1 Initialisation

All households start non-adapted and uninsured with `flood_timer = 99`, which
puts risk perception at its baseline minimum. Incomes are drawn from a
lognormal distribution parameterised by `median_income` and
`mean_median_inc_ratio`, read at each agent's income percentile. Percentiles
come from real data when available and otherwise from a uniform fallback; see
`income_utils.py` and the method guide.

### 9.2 Input data

The lookup table (§1.2) plus a sea-level trajectory. In production the
trajectory comes from FloodAdapt's projection database (`fa.interp_slr(...)`);
the `slr` coordinate of the table holds interpolation grid points, not the
trajectory itself.

### 9.3 Submodel: the hazard draw

Event frequencies are annual occurrence **rates**. The default draw is
therefore **Poisson**: each event $e$ occurs $n_e \sim \mathrm{Poisson}(f_e)$
times per year (`event_draw_mode="poisson"`), with no discard cap
(`max_events_per_year=None`). Realised damage is bounded per occurrence by
`max_pot_dmg`.

**Why Poisson and not Bernoulli.** The two distributions answer different
questions, and only one of them matches what `freq` means.

| | Bernoulli | Poisson |
|---|---|---|
| Question it answers | did it happen? | how many times did it happen? |
| Parameter | probability $p \in [0, 1]$ | rate $\lambda \ge 0$, of any size |
| Possible outcomes | 0 or 1 | 0, 1, 2, … |
| Mean | $p$ | $\lambda$ |
| A rate above 1 | inexpressible, must be clipped | represented exactly |
| P(at least one) | $p$ | $1 - e^{-\lambda}$ |

A frequency of 0.01 means "about once every 100 years"; a frequency of 3 means
"about three times a year". A Bernoulli trial can only carry a probability, so
it must round any rate above 1 down to certainty. Poisson carries the rate
itself. Four consequences follow:

1. every event keeps its true long-run rate, rare extremes included;
2. an event with rate 3 can genuinely happen 0, 2 or 5 times in one year;
3. nothing is clipped or discarded, so simulated damage statistics match the
   hazard input;
4. the simple sum $\mathrm{EAD}_i = \sum_e f_e\, D_{i,e}$ becomes the exact
   expected annual damage, which the insurance premium builds on.

Poisson also keeps the decision side consistent: the exceedance conversion
below is $1 - e^{-f}$, which is $P(n \ge 1)$ for exactly this process. A
clipped Bernoulli draw cannot be reconciled with any exceedance curve above
rate 1, because it has already discarded the rate.

This differs deliberately from DYNAMO-M's `stochastic_flood`
(`flood_risk.py:583`), which performs a single draw and allows at most one
flood per year against a fixed return-period set.

**Both draws are available; Poisson is the one to use.** `"bernoulli_clip"`
runs one Bernoulli trial per event with $p_e = \min(f_e, 1)$. Paired with a
random cap it clips sub-annual events to certainty and discards rare extremes
at the same rate as nuisance floods. It is kept as an ordinary config option so
the two hazards can be compared inside one script, which is what notebook 2's
matched-hazard runs do, and its RNG call order is frozen so those runs stay
controlled. It is not a setting for a real study.

**Where the two differ once the nuisance filter is on.** Every retained rate is
then below 1, nothing clips, and the two draws have the *same* expected
occurrence count. They still differ in the count *distribution*, and that is
what an agent-based model feels. Bernoulli cannot place two occurrences in one
year, so it converts multiplicity into extra **flood years**. Since
`flood_timer` resets on any flood in a year, the decision-relevant quantity is
$P(n \ge 1)$, where the two disagree: $f$ against $1 - e^{-f}$. At a rate of
0.85 that is 85 % of years against 57 %. Under Bernoulli risk perception stays
permanently elevated, which suppresses the alarm-then-complacency cycle the
perception submodel exists to represent.

**Nuisance filter.** For event sets containing sub-annual events (the
Charleston set has about 11), set `nuisance_freq_threshold = 1.0`. Those events
are dropped once at data load, from the hazard draw **and** the SEU integral
consistently: a near-certain event cannot sit on an exceedance curve and would
hit the 0.998 perceived-probability cap regardless.

**Exceedance conversion.** On the decision side, `seu_prob_mode="exceedance"`
converts rates to annual exceedance probabilities

$$p_e = 1 - e^{-f_e},$$

the exact probability of at least one Poisson arrival in a year, before the
trapezoidal integration. Rates and probabilities are distinct quantities, and
this keeps the decision side on the probability one: raw rates above 1 would
clamp onto the 0.998 perceived-probability cap and collapse the no-flood band
of the integral.

### 9.4 Submodel: risk perception

Risk perception is a deterministic decay function of the flood timer
$\tau_i$, not a random variable. Household $i$'s perception multiplier is

$$\mathrm{rp}_i = P_i \cdot 1.6^{\,c\,\tau_i} + \mathrm{rp}_{\min},$$

where $P_i$ is the post-flood peak (below), $c$ the decay coefficient and
$\mathrm{rp}_{\min}$ the baseline floor. Defaults from Tierolf et al. (2023,
§2.2): $\mathrm{rp}_{\max} = 2.0$, $\mathrm{rp}_{\min} = 0.01$, $c = -3.6$,
base $1.6$. With $\tau_i = 99$ this evaluates to essentially the minimum; a
flood resets the timer to 0 and perception spikes to $P_i$.

The base 1.6 is hard-coded in native DYNAMO-M (`flood_risk.py:650-651`); the
`base` key in its `settings.yml` is never read. The port mirrors this as
`_RISK_PERC_BASE = 1.6`.

**Severity response.** Under `perception_mode="severity"` the peak $P_i$ is not
the fixed $\mathrm{rp}_{\max}$: it scales with the **damage severity** $s_i$,
the share of the household's maximum potential damage realised in the flood,
through a **single one-parameter form**, the power law
(`simulation_engine.py::_severity_peak`). The decay law is unchanged; only the
starting height varies:

$$P_i = \mathrm{rp}_{\max}\, s_i^{\,\gamma}, \qquad
s_i = \min\!\left(\frac{\text{realised damage}_i}{\text{max. potential damage}_i},\; 1\right),
\qquad \gamma > 0.$$

The response is monotone in $s$ and pinned at both ends for every $\gamma$:
$s = 0$ gives no spike, $s = 1$ gives the full $\mathrm{rp}_{\max}$ spike, so a
total loss gives the full peak. **γ is the only shape parameter
in the perception block, and it spans the whole hypothesis range.**

![Severity response: the γ family](images/perception_severity_gamma.png)

### What each region of γ means

| Region | Shape | Hypothesis | Behaviour at s = 0.10 | Behaviour at s = 0.25 |
|---|---|---|---|---|
| γ → 0⁺ | step-like | Approaches binary/native: any flood is a full spike | 63 % of peak (γ=0.2) | 76 % (γ=0.2) |
| γ = 0.5 **(default)** | concave | Availability heuristic: small floods already register strongly | 32 % | 50 % |
| γ = 1 | linear | Damage-proportional response | 10 % | 25 % |
| γ > 1 | convex | Near-miss: small floods are largely ignored | 1 % (γ=2) | 6 % (γ=2) |

`γ = 0` exactly is **rejected** with a `ValueError`, not silently accepted:
`0.0 ** 0.0 == 1.0` in NumPy, so it would spike *every* agent to the full peak
including agents that have never flooded. Use `perception_mode="binary"` for the
exact native response; small positive γ approaches it continuously.

**Why the concave default.** The perception model is built on the availability
heuristic, which is exactly why native DYNAMO-M is binary. A concave response
preserves that insight while restoring magnitude information: at γ = 0.5 a flood
damaging 25 % of the home already triggers 50 % of the maximum spike and one
damaging 50 % triggers about 71 %, whereas linear scaling would underweight
moderate floods at both points.

### Why there is only one form

Two further one-parameter forms are **not supported**, because measurement
showed they add no hypothesis γ cannot already express. `saturating_exp`,
concave with finite slope at zero,

$$P_i = \mathrm{rp}_{\max}\,\frac{1 - e^{-k s_i}}{1 - e^{-k}},$$

and `threshold_linear`, a hard deadband below $s_0$,

$$P_i = \mathrm{rp}_{\max}\,
\operatorname{clip}\!\left(\frac{s_i - s_0}{1 - s_0},\, 0,\, 1\right).$$

They were introduced to bracket the power law on the small-flood response, on
the assumption that they represented distinct hypotheses.

They do not. Sweeping γ traces a curve in outcome space, and both alternatives
land **on** that curve rather than outside it, at every parameter setting tested:

| Retired configuration | Equivalent γ, synthetic table | Equivalent γ, real Charleston table |
|---|---|---|
| `saturating_exp` k = 1 | ≈ 0.79 | not run |
| `saturating_exp` k = 3 (its default) | ≈ 0.53 | **0.48** |
| `saturating_exp` k = 6 | ≈ 0.34 | not run |
| `threshold_linear` s0 = 0.05 | ≈ 1.15 | not run |
| `threshold_linear` s0 = 0.1 (its default) | ≈ 1.35 | **1.25** |
| `threshold_linear` s0 = 0.2 | ≈ 1.80 | not run |

The synthetic column is interpolated from a 5-point γ grid and the Charleston
column from a 10-point grid, so read the synthetic figures as approximate; the
two agree to about 0.1 in γ.

![Measured outcomes against γ on the real Charleston table, with the retired forms at their equivalent γ](images/perception_gamma_outcomes.png)

The equivalences reproduce across two tables that differ by three orders of
magnitude in adoption level (70–93 % synthetic against 1.9–5.5 % Charleston),
which is why this reads as structural rather than as a coincidence of one
fixture.

**The quantitative test, and why it is not circular.** Each retired form is
placed at the γ its *adoption* implies, so the adoption panel is fitted by
construction and proves nothing on its own. Cumulative damage is then the
independent check, because it took no part in placing the marker. Predicting
damage from the power curve at that γ leaves:

| Retired configuration | Fitted γ | Damage predicted | Damage measured | Residual |
|---|---|---|---|---|
| `saturating_exp` k = 3 | 0.485 | 2,842.0 M\$ | 2,862.1 M\$ | 20.1 M\$ = **1.5 %** of the sweep spread |
| `threshold_linear` s0 = 0.1 | 1.249 | 3,345.4 M\$ | 3,346.2 M\$ | 0.8 M\$ = **0.1 %** of the sweep spread |

against a 1,326 M$ spread across the γ range. **At most 1.5 % of a retired
form's effect is form-specific; the rest is reproducible by moving γ**, and
that residual is far below the uncertainty in γ itself, which is not calibrated
for Charleston at all.

The reason is structural, and is why the result should generalise: all three are
monotone maps of `[0, 1]` onto `[0, 1]` pinned at both endpoints, and γ already
sweeps that space from step-like through linear to strongly convex. There is no
room left for a second form to occupy. Note that the *curves* differ pointwise
(panel b of the first figure shows exactly how); it is the *model outcomes* that
coincide.

Both retired names raise a directed `ValueError` naming their equivalent γ rather
than an "unknown mode" error, so an old config migrates in one edit.

**What this bought.** The perception block went from four parameters
(`perception_severity_form`, `_exponent`, `_rate`, `_threshold`) to two
(`perception_severity_form`, pinned to `"power"`, and `_exponent`). Sensitivity
analysis is a one-dimensional sweep over γ that can be plotted on one axis;
comparing non-nested forms would instead need model-selection machinery (AIC/BIC
or Bayes factors) that a survey of a few hundred households cannot support.

**The deadband hypothesis keeps its own home.** `flood_significance_threshold`
decides whether a flood registers *at all* (timer reset and spike), which is the
mechanistically honest deadband, and it is pinned by
`tests/test_perception.py::test_significance_threshold_creates_deadband`. It is
*not* a substitute for γ, and this was measured too: raising it from 0.01 to 0.20
at γ = 1 moves final adoption only from 79.25 % to 78.75 % on the synthetic
table, because it changes *whether* a flood is remembered rather than *how
strongly*. Keep the threshold for the registration question and let γ carry the
response shape.

**A logistic form was considered and rejected** on the original one-parameter
identifiability argument: a defensible logistic needs two free parameters.

**Recommended analysis order** (survey design in the method guide):

1. Sweep γ over `0.25 / 0.5 / 1.0 / 2.0`. If outcomes are insensitive, the
   response shape does not matter for the question at hand and no data
   collection is needed.
2. If sensitive, the sweep already tells you *which region* matters, so a survey
   can be designed to discriminate within it rather than between functional
   forms. γ is what the survey estimates.
4. When survey data arrives, fit all three by least squares (stated risk
   perception against experienced damage fraction) and keep the best fit.

This front-loads the free analysis and isolates one behavioural property per
step, so even a small survey can discriminate between the forms.

### 9.5 Submodel: income and wealth

Native DYNAMO-M derives wealth from income with a percentile-based multiplier
(`decision_module.py:27-30`):

```python
perc  = np.array([0, 20, 40, 60, 80, 100])        # income percentile
ratio = np.array([0, 1.06, 4.14, 4.19, 5.24, 6])  # wealth / income multiplier
```

The bridge uses that table in full. Under the default
`income_mode="synthetic_lognormal"` it interpolates all six anchor points at
each household's income percentile
(`DynamoDecisionBridge._WEALTH_RATIO_PERCENTILES` / `_WEALTH_RATIO_VALUES`),
exactly as native does, so richer households hold proportionally more wealth.

The scalar `DecisionConfig.income_to_wealth_ratio` (default `4.14`, the 40th
percentile) is a *separate* path. It applies only when a caller supplies an
explicit `income_per_agent` array without a matching `wealth_per_agent`. It is
not used by the default mode.

**These six numbers are not calibrated.** They are inherited from native
DYNAMO-M, whose source documents them only with the inline comment
"wealth in relation to income" and gives no citation. Nothing about them is
specific to Charleston or to any other study area. Changing them breaks
bit-parity with native, so treat them as a sensitivity target rather than a
free parameter, and see `docs/calibration_validation_guide.md` for how they sit
alongside the parameters that *are* measured.

### 9.6 Submodel: the SEU decision

**Ex-ante versus ex-post.** The event catalogue is used in two different ways,
and conflating them is a modelling error:

| Concept | Purpose | Which events | When |
|---|---|---|---|
| **Ex-post realised damage** | Physical: what actually hits the agent | Only the occurrences drawn this year | During the year loop |
| **Ex-ante expected utility** | Cognitive: how the agent perceives future risk | **All** events in the distribution | At the decision moment |

**Utility (CRRA).** Consumption-equivalent value $c$ is mapped to utility with
constant relative risk aversion $\sigma$:

$$U(c) =
\begin{cases}
\dfrac{c^{1-\sigma}}{1-\sigma} & \sigma \neq 1,\\[2ex]
\ln c & \sigma = 1.
\end{cases}$$

**Net present value under flood event $e$.** With wealth $W$, income $Y$,
amenity value $A$ (`amenity_premium` $\cdot\, W\, \cdot$ `amenity_weight`),
event damage $D_e$, horizon $T$ and discount rate $r$:

$$\mathrm{NPV}_e = \left(W + Y + A - D_e\right)
\left(1 + \sum_{t=1}^{T-1} \frac{1}{(1+r)^t}\right).$$

**Perceived probability.** Each event's exceedance probability is scaled by the
household's risk perception and capped:

$$\tilde{p}_e = \min\!\left(p_e \cdot \mathrm{rp}_i,\; 0.998\right).$$

**Expected utility.** The expectation integrates utility over the whole
perceived exceedance curve:

$$\mathrm{EU} = \int_0^1 U\!\big(\mathrm{NPV}(p)\big)\, dp,$$

evaluated with the trapezoidal rule over the discrete curve
(`_integrate_expected_utility`). The grid is built as follows: the events are
sorted by ascending perceived probability $\tilde{p}_1 < \dots < \tilde{p}_n$
(so the rarest, most damaging scenario comes first); a lower-bound row at
$p = 0$ repeats the rarest scenario's NPV; and two no-flood rows are appended
above $\tilde{p}_n$ (a transition band of width 0.001, then $p = 1$) so the
remaining probability mass carries the undamaged NPV. Over that grid,

$$\mathrm{EU} \approx \sum_{j}
\frac{U\!\big(\mathrm{NPV}_{j+1}\big) + U\!\big(\mathrm{NPV}_{j}\big)}{2}\,
\big(p_{j+1} - p_{j}\big).$$

**Do nothing** (`calcEU_do_nothing`). $D_e$ comes from the no-measures damage
matrix, and the integral above gives $\mathrm{EU}_{\text{do nothing}}$. For an
already-adapted household $\mathrm{EU}_{\text{do nothing}} = -\infty$, which
prevents un-adapting.

**Dry floodproofing** (`calcEU_adapt`). $D_e$ comes from the floodproofed
damage matrix (`floodproof_all_0`). The one-off cost $C$ is financed as a loan
of duration $L$ at rate $r_L$ and annuitised:

$$\text{annual cost} = C \cdot
\frac{r_L\,(1+r_L)^{L}}{(1+r_L)^{L} - 1},$$

and the household counts the discounted remaining payments inside its horizon,
with $\ell = L - \text{time adapted}$ years left on the loan:

$$\text{cost} = \sum_{t=0}^{\min(\ell,\,T)}
\frac{\text{annual cost}}{(1+r)^t},
\qquad \mathrm{NPV}^{\text{adapt}}_e = \mathrm{NPV}_e - \text{cost}.$$

Affordability is enforced inside the function: if
$Y \cdot \text{expenditure cap} \leq \text{annual cost}$, then
$\mathrm{EU}_{\text{adapt}} = -\infty$.

**The NPV floor.** Both native and port apply
$\mathrm{NPV} \leftarrow \max(1, \mathrm{NPV})$ before $U$ is applied, because
the utility function is undefined at zero or negative values. Native prints a
per-call diagnostic counting the floored entries; see Appendix B.

**Decision.**

```python
adapt_decision = (EU_adapt > EU_do_nothing) & (~is_floodproofed) & is_residential
# Affordability is NOT re-checked here: it is already encoded as EU_adapt = -inf
# inside calcEU_adapt, which avoids logic drift.
```

With insurance enabled the comparison becomes three-way over `EU_do_nothing`,
`EU_adapt` and `EU_insure`, restricted to non-adapted agents (Appendix B,
deviation 4).

### 9.7 Submodel: insurance pricing — two rating rules for insurance policy

`insurance_pricing` selects the insurer's rating rule:

| Mode | Premium | Meaning |
|---|---|---|
| `"community"` (default) | mean expected annual damage of the pool, one flat rate for everybody | Native DYNAMO-M's rule. Low-risk households cross-subsidise high-risk ones. Mirrors community-rated schemes such as the pre-2021 US NFIP (Michel-Kerjan 2010) |
| `"risk_based"` | $\pi_i = (1-d)\,\mathrm{EAD}_i$ per household | The actuarially fair price: it exactly covers the insurer's expected payments for that household. No cross-subsidy. Mirrors the direction of NFIP Risk Rating 2.0 (FEMA 2021; GAO 2023) |

Insurance is annual, re-decided every year, and never overrides physical
floodproofing.

**The symbols, mapped to the model.** Every quantity in this submodel is either a
config field or derived from the lookup table:

| Symbol | Meaning | In this model | Default | How to set |
|---|---|---|---|---|
| $\mathrm{EAD}_i$ | Expected annual damage of household $i$: its average yearly flood loss at the current sea level | $\sum_e f_e\, D_{i,e}$ over the no-measures slice of the lookup table, recomputed each year at the current SLR (`_compute_premium_offer`, `simulation_engine.py`) | derived | Nothing to set; it is the hazard model's output |
| $d$ | Deductible: the share of each loss the insured household still pays itself | `insurance_deductible` | 0.1 (native DYNAMO-M's hard-coded value) | From the policy terms of the scheme being modelled (e.g. NFIP deductible options) |
| $\pi_i$ | The actuarial base premium | risk-based: $(1-d)\,\mathrm{EAD}_i$ per household; community: one flat value, the pool mean of that quantity | mode via `insurance_pricing`, default `"community"` | A scenario choice, not a calibration target: pick the rating rule of the scheme under study |
| $\lambda$ | Loading: what the insurer adds on top of the expected-loss price | `insurance_loading`, multiplies the base premium | 1.0 (at cost) | From loss ratios; see below |
| $\sigma_s$ | Subsidy: the share of the household's bill paid by a public scheme | `insurance_subsidy` | 0.0 (no intervention; the native-parity baseline) | A policy lever to sweep, not calibrate; see below |

The offer the household actually faces is

$$\pi_i^{\text{offer}} = \lambda\,(1-\sigma_s)\,\pi_i,$$

and the affordability gate then tests $\pi_i^{\text{offer}}$ against six percent of
income (`expenditure_cap`), setting $\mathrm{EU}_{\text{insure}} = -\infty$ when it
fails.

**Why both rules ship.** The two modes are the two poles of the real policy debate,
and the trade-off between them follows from the premium formula itself:

1. *Risk-based rating creates a price signal.* Because $\pi_i$ is household $i$'s
   own expected payout, anything that lowers its expected damage lowers its premium,
   so risk reduction becomes privately profitable — the incentive argument for
   risk-based pricing (Hudson et al. 2016; Kousky 2019). Community rating severs
   this link: everyone pays the pool mean, so floodproofing does not change your
   bill. The signal is muted and the low-risk majority is overcharged relative to
   its own risk, a selection problem measured by Wagner (2022): under 60 % of
   high-risk-zone US homeowners buy flood insurance even at premiums around
   two-thirds of actuarial cost, and selection runs on observable adaptation. (One
   honest model caveat: the ported insurer prices on the no-measures damage matrix,
   so *within* a run a household's premium does not fall when it floodproofs — the
   signal argument applies across scenarios. This is the deliberate
   no-feedback-loop simplification of Appendix B row 6.)
2. *The same formula creates the affordability failure.* $\pi_i$ scales
   one-for-one with $\mathrm{EAD}_i$; flood risk is concentrated in specific places
   and correlated with lower incomes; so the premium-to-income ratio is largest
   exactly where cover matters most. In this model that arrives through one
   concrete gate: $\pi_i^{\text{offer}} > 0.06 \cdot \text{income}_i$ excludes the
   household. The notebook's risk-based run shows the result: the median premium
   collapses to about \$122/yr and 83 % of households can afford cover, but only
   4.5 % of the highest-risk decile can. Hudson et al. (2016) measure the same
   failure empirically (about 20 % of at-risk households in France and Germany
   cannot afford risk-based premiums), and Gourevitch, Snyder and Kousky (2025)
   measure it after the NFIP's move to risk-based rates: 11–39 % fewer new
   policies and 5–13 % fewer renewals, with declines up to 60 % in lower-income
   zip codes. This is a problem a policy design has to address, and the next point
   is the model's lever for it.
3. *The remedy the literature recommends is a subsidy that sits outside the premium
   formula — which is exactly what `insurance_subsidy` is.* The recommended design
   is to keep charging the risk-based price, so the signal survives, and pay part
   of the household's *bill* from public money rather than from other
   policyholders' premiums: means-tested vouchers coupled to mitigation loans
   (Kousky & Kunreuther 2014), vouchers costing less than the risk reduction they
   enable (Hudson et al. 2016), and the same design recommended alongside Risk
   Rating 2.0 (Zhang, Lin & Kunreuther 2023; Gourevitch, Snyder & Kousky 2025). In
   the implementation this is literally the order of operations in
   `_compute_premium_offer`: the actuarial $\pi_i$ is computed first from
   $\mathrm{EAD}_i$, and $\sigma_s$ only discounts what the household is asked to
   pay. High-risk households still face premiums proportional to their risk, but
   the affordability gate now tests $\lambda(1-\sigma_s)\pi_i$ against income, so
   raising $\sigma_s$ relaxes the constraint that point 2 imposes. Sweeping
   $\sigma_s$ is how a modeller studies voucher-style policy with this package.

   **How much it relaxes depends on how skewed the pool is, and on the real
   Charleston table a uniform subsidy is much weaker than it sounds.** On the
   package's synthetic demonstration table
   (`examples_engine/08_income_perception_insurance.py`) a 90 % subsidy lifts
   uptake from 0 % to a 48.8 % peak. On the real table it does far less, because
   the subsidy scales the premium while the affordability gate is absolute. A
   top-decile household with $\mathrm{EAD}_i \approx 52{,}000$ pays about
   46,000 under fair pricing; a 60 % subsidy still leaves 18,600 against a 6 %
   budget of roughly 4,200, so it stays excluded. Measured on the real table, a
   60 % subsidy moves risk-based uptake not at all (0.000 % either way), while
   the same subsidy on the flat community rate lifts peak uptake from 0.98 % to
   3.06 % and *reduces* floodproofing from 4.65 % to 4.06 %. A uniform subsidy
   therefore helps the households that least need cover, and can crowd out
   physical adaptation. This is precisely why the literature specifies
   **means-tested** support sized to each household's affordability gap (Kousky
   & Kunreuther 2014; Hudson et al. 2016) rather than a uniform discount.
   `insurance_subsidy` implements the uniform version; a per-household
   $\sigma_{s,i}$ is the natural extension at the same seam.

**Setting the loading.** $\lambda = 1$ prices "at cost": over many years the insurer
collects exactly what it pays out, with nothing left for running the company. A real
insurer also pays staff and administration, buys reinsurance, and holds capital for
the bad years, and prices all of that into the premium; the loading is that
addition. It connects to a published statistic, the **loss ratio**: claims paid
divided by premiums collected. An insurer paying out about 75 cents of claims per
premium dollar (loss ratio 0.75) charges about $1/0.75 \approx 1.3$ times expected
claims, so $\lambda \approx 1/\text{loss ratio}$ of the scheme being modelled.
Disaster lines sit above ordinary property lines here, because correlated
catastrophe losses force insurers to hold expensive capital (Kousky 2019).

**Setting the subsidy.** The default is 0 because the unsubsidised market is the
honest baseline: it is native DYNAMO-M's behaviour (parity), and any positive value
is a policy intervention that should be a deliberate scenario choice. For a
defensible illustration value, the NFIP itself ran an implicit subsidy for decades:
GAO found pre-FIRM subsidised policies paid on average only 35–40 % of the
full-risk rate, an implicit $\sigma_s \approx 0.60$–$0.65$ (GAO 2013; the menu of
explicit affordability designs is in Horn 2023). A US-anchored sweep is therefore
$\sigma_s \in \{0, 0.3, 0.6, 0.9\}$: 0.6 mirrors the NFIP's historical implicit
subsidy, and the notebook's 0.9 is deliberately deeper, chosen to reveal when the
high-risk tail finally comes under cover. One stated simplification:
`insurance_subsidy` is a flat share of every bill, whereas the voucher literature
sizes support to each household's affordability gap (Hudson et al. 2016; Kousky &
Kunreuther 2014) — a means-tested $\sigma_{s,i}$ would be a straightforward
extension at the same seam.

### 9.8 Submodel: adaptation lifespan

When `time_adapted` reaches `lifespan_dryproof` (75 years) the measure expires:
`is_adapted` becomes `False`, `time_adapted` resets to 0, and the household
re-decides the following year. Adaptation is therefore **not permanent**, and
agents adapt **sequentially**, never cumulatively: elevations do not stack.

---

# Part III. Verification and validation

## 10. The bit-parity contract

The drivers are wrappers, not alternative implementations, and these equalities
are enforced:

```
engine.run(...) = run_mesa_native(...) = run_mesa_native_full(...)    bit-for-bit
SEURule EU values = native DYNAMO-M DecisionModule EU values          rel. err < 1e-4
```

Any change to `SimulationEngine.step`, the event draw or the SEU kernels must
preserve these. `engine.run(n_jobs=-1)` is likewise bit-identical to `n_jobs=1`
for deterministic rules.

## 11. Verification suite

Verification means checking that the code does what the equations say.

- `pytest tests/ -q` runs the whole suite, including every parity gate. Tests
  auto-skip when an optional dependency is missing, and that guard is itself
  under test.
- `verification/` holds re-runnable harnesses with full reports and metrics
  (`phase1_seu_battery`, `phase4a_parity`, `phase4b_mesa_native`,
  `real_table_gate`, `mesa_native_full`). They run under the current package
  defaults.

**There is no bit-exact reproduction contract for older behaviour.** The
alternative algorithms are available as ordinary config options, but no test
pins a stored run of them. One mode is unsupported outright:
`income_mode="mpd_ratio"`, which made income and adaptation cost both
proportional to `max_pot_dmg`, so the affordability gate reduced to a single
population-wide constant (measured at 1.6887 for every household) and never
bound for anybody.

## 12. Validation status

Verification is complete; **empirical validation is not**, and the model should
be read as a comparative instrument until it is. The parameters carrying the
most weight (risk-perception dynamics, the severity form, adaptation cost, the
income distribution) are listed with their provenance and calibration data
needs in [`calibration_validation_guide.md`](calibration_validation_guide.md),
which also sets out a tiered, open-data-first plan.

## 13. Reference parameters

| Parameter | Provenance | Default | Unit | Description |
|---|---|---|---|---|
| `income` | new input | site data | USD/yr | Household annual income |
| `income_percentile` | new input | site data | 0–100 | Position in the income distribution |
| `wealth` | derived | computed | USD | `income × income_to_wealth_ratio` |
| `property_value` | lookup table | from table | USD | FIAT `max_pot_dmg`, capped at wealth |
| `adaptation_costs` | new input | site data | USD/yr | Annual loan repayment for dry-proofing |
| `risk_aversion` (σ) | DYNAMO-M settings | 1 | – | CRRA coefficient (σ=1 gives log utility) |
| `decision_horizon` (T) | DYNAMO-M settings | 15 | yr | NPV planning horizon |
| `discount_rate` (r) | DYNAMO-M settings | 0.032 | /yr | Time discounting |
| `expenditure_cap` | DYNAMO-M settings | 0.06 | frac | Max income share for adaptation or premium |
| `loan_duration` (L) | DYNAMO-M settings | 16 | yr | Loan repayment period |
| `interest_rate` (r_loan) | DYNAMO-M settings | 0.04 | /yr | Loan interest rate |
| `lifespan_dryproof` | DYNAMO-M settings | 75 | yr | Measure lifetime; triggers re-decision |
| `error_interval` | DYNAMO-M settings | 0.0 | frac | Uniform noise on EU |
| `amenity_weight` | DYNAMO-M settings | 1 | – | Amenity weight in utility |
| `amenity_value` (A) | derived | 0 | USD | Location value; 0 in the validated configuration |
| `risk_perception` | dynamic | 0.01–2.01 | – | Subjective probability multiplier |
| `flood_timer` | dynamic | 99 at init | yr | Years since the last flood |

Behavioural parameters added by this project (the severity exponent γ, insurance
pricing knobs, income synthesis) are inventoried in the method guide.

---

# Appendix A. Glossary

| Term | Meaning |
|---|---|
| **ABM** | Agent-based model: a model of individual decision-makers whose interactions produce system-level outcomes |
| **SEU** | Subjective Expected Utility: choosing the option with the highest expected utility, using the agent's own perceived probabilities |
| **CRRA** | Constant Relative Risk Aversion, the utility function family used here |
| **NPV** | Net Present Value: future money expressed in today's value |
| **EAD** | Expected Annual Damage: the damage expected in an average year |
| **FPS** | Flood Protection Standard, expressed as a return period in years |
| **ODD** | Overview, Design concepts, Details: the standard protocol for describing an agent-based model |
| **Dry floodproofing** | Measures that keep water out of a building (DYNAMO-M models 1 m elevation; here it is the table's `floodproof_all_0` strategy) |
| **Flood timer** | Years since the last flood; drives risk-perception decay |
| **Risk perception** | Multiplier applied to the actual flood probability, capturing the availability heuristic |
| **Exceedance probability** | The probability that an event of at least a given size occurs in a year |
| **Occurrence rate** | The expected number of times an event happens per year; may exceed 1 |
| **Lookup table (LUT)** | The precomputed `.nc` impact table joining stage 1 to stage 2 |
| **Household** | One residential building, one `object_id`, one agent |
| **Actuarially fair** | Priced exactly at expected cost, with no margin |
| **Cross-subsidy** | One group's premium covering another's risk: under community rating, low-risk households pay above their own risk so high-risk households can pay below theirs |
| **Cross-subsidised tail** | The high-risk households that benefit from that transfer. Under community rating everyone pays the pool's *mean* EAD, so a household whose own EAD exceeds the mean buys cover below its own risk. On the real Charleston table the top EAD decile pays about 0.24 times its own risk. In the model runs this is the only group for which insuring beats the alternatives, and therefore the only source of non-zero uptake (§9.7) |
| **Expenditure cap** | The hard budget rule `expenditure_cap = 0.06`: if an option costs more than 6 % of annual income, its expected utility is set to `-inf` and it leaves the choice set entirely. Applies to both floodproofing and insurance |
| **Loading** | What an insurer adds on top of the expected-loss price for expenses, reinsurance and capital; `insurance_loading`, estimated as 1 / loss ratio |
| **Loss ratio** | Claims paid divided by premiums collected; the published statistic a loading is derived from |
| **Bit-parity gate** | A test asserting two code paths produce *byte-identical* arrays, not merely close ones; used to prove the drivers are wrappers around one kernel |

# Appendix B. Deviations from native DYNAMO-M

The coupling is a faithful port of the DYNAMO-M SEU science (parity gates:
ported versus native EU relative error < 1e-6, identical decisions). The
following are the **deliberate** deviations, each with its rationale.

| # | Deviation | Native behaviour | Rationale |
|---|---|---|---|
| 1 | **Multi-event Poisson hazard draw** | One draw per node per year against descending return periods; at most 1 event/yr (`flood_risk.py:599-622`) | The lookup table carries a probabilistic event *set* with occurrence rates, not a fixed return-period ladder; Poisson is the exact model for rates |
| 2 | **Severity-scaled risk perception** (`perception_mode="severity"`, one power-law form, default γ=0.5) | Binary: any positive depth triggers the full spike (`flood_risk.py:619`) | A nuisance flood and a catastrophe should not look identical to the agent; γ→0 approaches the native response (§9.4) |
| 3 | **Exceedance conversion** `p = 1 − e^(−freq)` for the SEU integral | `p = 1/rt` are already exceedance probabilities ≤ 0.5 | The table stores rates that may exceed 1; the conversion restores valid exceedance semantics |
| 4 | **Insurance never overrides floodproofing** (`decide()` masks adapted agents) | An adapted agent may flip to insured, silently discarding its floodproofing (`coastal_nodes.py:1938-1952`) | Physical measures should persist until the lifespan reset; avoids un-modelled capital destruction |
| 5 | **Realised-damage bookkeeping** (`damage_history`, `out_of_pocket_history`) | No realised-damage accounting; wealth is never decremented by floods | Damage time series are a core output here |
| 6 | **Premium at current-SLR EAD, residential-only, always no-measures** | Premium = node `ead_total / n`, where `ead_total` **includes commercial and industrial** damage, is FPS-truncated, and is adaptation-aware (`flood_risk.py:528-554`, `insurer_agent.py:20-26`) | The lookup table is the hazard source; `Σ freq·dmg` is the exact EAD under Poisson semantics; households are the agents, so only residential exposure is pooled; the no-measures matrix keeps the premium independent of uptake, avoiding a feedback loop |
| 6b | **Pricing knobs beyond native**: `insurance_pricing="risk_based"`, `insurance_loading`, `insurance_subsidy` | Native has exactly one rule: the flat community premium | Community rating is native's rule, mirroring real community-rated schemes such as the pre-2021 US NFIP (Michel-Kerjan 2010); risk-based is the standard actuarial benchmark (premium = expected loss, cf. NFIP Risk Rating 2.0; FEMA 2021). Added (1) as a diagnostic, to test whether the flat rate is what suppresses uptake (it is not; the expenditure cap is — the trade-off measured by Hudson et al. 2016 and, post-reform, by Gourevitch, Snyder & Kousky 2025), and (2) to span the premium designs a public insurer could realistically set (Kousky & Kunreuther 2014). Full argument: §9.7 |
| 7 | **Parameterised income synthesis** (`median_income`, `mean_median_inc_ratio`) | GDL raster + World Bank table + UN WIID per region | Those datasets are not shipped here; the pipeline (lognormal to percentile to wealth ratio) is ported unchanged, and real percentiles can be supplied per agent |
| 8 | **Damages evolve with SLR inside the horizon integration inputs** | Native NPV holds damages constant over the horizon (myopic) | Inherited coupling behaviour; revisit with calibration |

**Faithful-port details worth noting.** The hard-coded risk-perception base 1.6
(native's `settings.yml` `base` key is dead code); the `max(1, NPV)` floor;
error terms applied even when `error_interval = 0`; the 0.998
perceived-probability cap; the annuity loan formula; the 75-year lifespan reset;
and the native `0/1/2 = nothing/floodproofed/insured` encoding on the adapter
seam.

**About the `[calcEU_*] Warning, N negative NPVs` console lines.** These are
diagnostics printed by the *native* `DecisionModule`, so they appear only when
`DynamoLiveRule` drives; the ported kernels apply the identical maths silently.
They report how many entries of the households x probability-grid NPV matrix
went negative before the `max(1, NPV)` floor, that is, cases where discounted
expected damages exceed the discounted value of staying put. At FIAT damage
scales on the real Charleston table the floor engages for roughly 0.5 % to 2 %
of entries and grows with sea level. This is expected, harmless for parity
(both rules floor identically, bit-for-bit), and a known joint income and
damage calibration point.

**How aligned is `_calc_eu_insure` with native `calcEU_insure`?** The kernel
maths (`_core/dynamo_decision_bridge.py:1176-1321` versus native
`decision_module.py:243-367`) is a faithful port: identical deductible
application, perceived-probability construction (0.998 cap, +0.001 band,
`[0, 1]` trapezoid domain), NPV floor, CRRA utility, affordability gate and
error-term handling. The **one substantive generalisation** is premium
discounting: native discounts the scalar `premium.mean()` because its premium
*is* a scalar, whereas the port discounts each agent's own premium. These are
identical under the flat community premium and diverge only under the port-only
`risk_based` pricing. Two further notes for anyone comparing decisions against
upstream: (1) native's three-way selection
`np.logical_and(EU_a > EU_dn, EU_a >= EU_i, EU_migr_cond)`
(`coastal_nodes.py:1938-1952`) passes the third argument into NumPy's `out=`
parameter, so the migration comparison is silently discarded; the port
reproduces what native actually *computes*, not what its source appears to say;
(2) the port adds the sticky-floodproofing guard of deviation 4. The verified
equivalence envelope of the parity gate
(`tests/test_dynamo_live_rule.py::TestInsuranceParity`) is a flat premium,
relative EU error < 1e-4, and bit-identical decisions.

# Appendix C. Worked examples and hand checks

Worked traces are kept as **runnable code** rather than prose, so they cannot
drift from the implementation:

- `examples_engine/02_seu_rule.py` shows the SEU comparison for a small
  population, printing `EU_do_nothing`, `EU_adapt` and the resulting decision.
- `examples_engine/08_income_perception_insurance.py` walks the income
  percentiles, the severity-exponent sweep, Poisson rate recovery, and the
  insurance pricing modes side by side.
- `notebooks/2_run_coupled_abm.ipynb` runs the full scenario set on the real
  Charleston table, including the matched-hazard comparison and the
  insurance experiment.

For the single-event case the trapezoidal integral collapses to a closed form
useful for hand checks:

$$\mathrm{EU} = (p + 0.0005)\, U\!\big(\mathrm{NPV}_{\text{flood}}\big)
+ (1 - p - 0.0005)\, U\!\big(\mathrm{NPV}_{\text{no flood}}\big),$$

where 0.0005 is half the 0.001 transition band inserted between the flood and
no-flood rows of the exceedance curve.

**Capping constraints** applied before the SEU evaluation, and why:

1. **Wealth capping**: `property_value = min(max_pot_dmg, W)`. A household
   cannot own a home worth more than its total wealth, and this keeps `NPV`
   positive so the log utility stays defined.
2. **Damage capping**: `D_i = min(D_i, property_value)`. A flood cannot destroy
   more than the building is worth. This also filters interpolation artefacts
   at extreme sea levels.

# References

Model lineage and protocol:

Tierolf, L., Haer, T., Botzen, W. J. W., de Bruijn, J. A., Ton, M. J.,
Reimann, L., & Aerts, J. C. J. H. (2023). *A coupled agent-based model for
France for simulating adaptation and migration decisions under future coastal
flood risk.* Scientific Reports 13, 4176. <https://doi.org/10.1038/s41598-023-31351-y>
Source code: [VU-IVM/DYNAMO-M](https://github.com/VU-IVM/DYNAMO-M/tree/v0.1.4), v0.1.4, the version every parity
gate in this repository is run against.

Grimm, V., Railsback, S. F., Vincenot, C. E., et al. (2020). *The ODD protocol
for describing agent-based and other simulation models: a second update to
improve clarity, replication, and structural realism.* Journal of Artificial
Societies and Social Simulation 23(2), 7.

Insurance pricing, affordability and subsidies (§9.7, Appendix B row 6b):

Gourevitch, J. D., Snyder, M., & Kousky, C. (2025). *Effects of risk-based
pricing reform on flood insurance uptake.* Journal of Catastrophe Risk and
Resilience 3, article 7.

Hudson, P., Botzen, W. J. W., Feyen, L., & Aerts, J. C. J. H. (2016).
*Incentivising flood risk adaptation through risk based insurance premiums:
trade-offs between affordability and risk reduction.* Ecological Economics
125, 1–13.

Hudson, P., Botzen, W. J. W., & Aerts, J. C. J. H. (2019). *Flood insurance
arrangements in the European Union for future flood risk under climate and
socioeconomic change.* Global Environmental Change 58, 101966.

Kousky, C. (2019). *The role of natural disaster insurance in recovery and
risk reduction.* Annual Review of Resource Economics 11.

Kousky, C., & Kunreuther, H. (2014). *Addressing affordability in the National
Flood Insurance Program.* Journal of Extreme Events 1(1), 1450001.

Michel-Kerjan, E. O. (2010). *Catastrophe economics: the National Flood
Insurance Program.* Journal of Economic Perspectives 24(4), 165–186.

Wagner, K. R. H. (2022). *Adaptation and adverse selection in markets for
natural disaster insurance.* American Economic Journal: Economic Policy 14(3),
380–421.

Zhang, F., Lin, N., & Kunreuther, H. (2023). *Benefits of and strategies to
update premium rates in the US National Flood Insurance Program under climate
change.* Risk Analysis 43, 1627–1640.

Government and programme reports:

FEMA (2021). *Risk Rating 2.0: Equity in Action* (methodology). Federal
Emergency Management Agency.

GAO (2013). *Flood Insurance: More Information Needed on Subsidized
Properties.* GAO-13-607. US Government Accountability Office.

GAO (2023). *Flood Insurance: FEMA's New Rate-Setting Methodology Improves
Actuarial Soundness but Highlights Need for Broader Program Reform.*
GAO-23-105977. US Government Accountability Office.

Horn, D. P. (2023). *Options for Making the National Flood Insurance Program
More Affordable.* CRS Report R47000. Congressional Research Service.
