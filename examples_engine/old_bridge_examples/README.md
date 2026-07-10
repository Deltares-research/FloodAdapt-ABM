# Old Bridge Examples (Deprecated)

**Status**: ⚠️ DEPRECATED — kept for reference and backward compatibility only.

This folder contains the original `DynamoDecisionBridge`-based examples from before the Phase 2+3 refactor. They still work but are **not recommended for new code**.

## Files

- `run_coupled_example.py` — Original manual step-by-step demo using `DynamoDecisionBridge`
- `run_trace_manual_check.py` — Debug trace example

## Migration Path

**Old code** (deprecated):
```python
from floodadapt_abm import DynamoDecisionBridge

bridge = DynamoDecisionBridge(ds=ds, config=cfg)
bridge.prepare_damage_arrays(1.0)
newly_adapted = bridge.evaluate_decisions(year_index=0)
```

**New code** (recommended):
```python
from floodadapt_abm import SimulationEngine

engine = SimulationEngine(ds=ds, config=cfg)
results = engine.run(slr_values, no_seq=10)
```

See `examples_engine/` for the canonical new examples.

## Why the Change?

The unified `SimulationEngine` (Phase 2+3 refactor) consolidates:
- Time ownership (was external loop)
- Event drawing (unified Bernoulli + random-pool cap)
- Per-agent state (standardised `AgentState`)
- Decision logic (pluggable `DecisionRule`)

This eliminates code duplication and makes the API clearer and more maintainable.

---

**Use `examples_engine/run_coupled_example_engine.py` for new projects.**
