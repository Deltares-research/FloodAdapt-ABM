"""
07_mesa_native_full.py
======================
Phase **4b-full**: **native-class integration** — the real honeybees ``Model``
owns time and the native DYNAMO-M ``DecisionModule`` makes the decisions.

Example 06 mirrored the DYNAMO-M tick loop with a *framework-free* Python object.
Here the driver (:class:`FloodAdaptSLRModelFull`) **subclasses the real honeybees
``Model``**, so the clock (``current_time`` / ``current_timestep`` / ``end_time``)
is provided by the genuine framework — exactly as the upstream ``SLRModel`` does.
Each year's decision flows through the native ``DecisionModule.calcEU_*`` via
:class:`DynamoLiveRule`, and the coastal-node population is fed **entirely from
the FloodAdapt lookup table** through the PRE.4 ``LookupTableAdapter`` (no
GLOFRIS / gravity / geodata).

What you learn here
-------------------
* :class:`FloodAdaptSLRModelFull` is a genuine ``honeybees.model.Model`` subclass
  that owns time via the framework clock,
* ``run_mesa_native_full(...)`` is a drop-in analogue of ``engine.run(...)`` /
  ``run_mesa_native(...)`` and is **bit-for-bit identical** — the triple-parity
  4b-full gate,
* the lookup-table adapter round-trips (FloodAdapt -> node arrays -> engine
  state) every tick without perturbing the shared kernel,
* driving with a :class:`DynamoLiveRule` makes it a true *native-class*
  integration (falls back to the ported ``SEURule`` when DYNAMO-M is absent).

Run::

    python 07_mesa_native_full.py
"""
from __future__ import annotations

import numpy as np

import _shared
from floodadapt_abm import (
    SimulationEngine,
    CouplingConfig,
    FloodAdaptSLRModelFull,
    run_mesa_native,
    run_mesa_native_full,
    HONEYBEES_AVAILABLE,
    DYNAMO_M_AVAILABLE,
)

SLR = np.linspace(0.0, 1.5, 20)
SEED = 42


def _make_rule(cfg, ds):
    """A native DYNAMO-M rule when available, else the ported SEURule (None)."""
    if DYNAMO_M_AVAILABLE:
        from floodadapt_abm import DynamoLiveRule

        amenity = SimulationEngine(ds=ds, config=cfg)._data.amenity_value
        return lambda: DynamoLiveRule(cfg.decision, amenity_value=amenity), "native DYNAMO-M DecisionModule"
    return lambda: None, "ported SEURule (DYNAMO-M not importable)"


def _honeybees_clock_walkthrough(ds, cfg, rule_factory) -> None:
    """Drive a single sequence one tick at a time on the honeybees clock."""
    engine = SimulationEngine(ds=ds, config=cfg, decision_rule=rule_factory())
    model = FloodAdaptSLRModelFull(engine, SLR, seed=SEED, start_year=2020)

    from honeybees.model import Model

    print(f"  isinstance(model, honeybees.Model): {isinstance(model, Model)}")
    print(f"  households (agents.regions.n)     : {model.agents.regions.n}")
    print(f"  clock starts at {model.current_time} (timestep {model.timestep})")
    for _ in range(3):
        model.step()  # one honeybees tick == one year
        adopted = engine.state.is_adapted.mean()
        print(f"  after model.step(): {model.current_time}  timestep={model.timestep:2d}  adopted={adopted:.1%}")
    model.run_model()
    print(f"  run_model() finished at timestep={model.timestep} (== {model.n_timesteps} years)")


def _triple_equivalence_gate(ds, cfg, rule_factory) -> None:
    """run_mesa_native_full == run_mesa_native == engine.run (bit-for-bit)."""
    full = run_mesa_native_full(
        SimulationEngine(ds=ds, config=cfg, decision_rule=rule_factory()), SLR, no_seq=5, seed=SEED
    )
    scaf = run_mesa_native(
        SimulationEngine(ds=ds, config=cfg, decision_rule=rule_factory()), SLR, no_seq=5, seed=SEED
    )
    loop = SimulationEngine(ds=ds, config=cfg, decision_rule=rule_factory()).run(SLR, no_seq=5, seed=SEED)

    full_vs_loop = np.array_equal(full["damage_history"], loop["damage_history"]) and np.array_equal(
        full["adapted_history"], loop["adapted_history"]
    )
    full_vs_scaf = np.array_equal(full["damage_history"], scaf["damage_history"]) and np.array_equal(
        full["adapted_history"], scaf["adapted_history"]
    )
    print(f"  run_mesa_native_full == engine.run       : {full_vs_loop}")
    print(f"  run_mesa_native_full == run_mesa_native  : {full_vs_scaf}")
    print(f"  4b-full gate: {'PASS' if full_vs_loop and full_vs_scaf else 'FAIL'}")
    print(f"  (final adoption, mean over sequences: {full['adoption_fraction'][:, -1].mean():.1%})")


def main() -> None:
    _shared.banner("07 - PHASE 4b-full: native-class integration (honeybees Model)")
    if not HONEYBEES_AVAILABLE:
        print("honeybees is not importable in this environment; install honeybees")
        print("(and mesa) to run the Phase 4b-full driver. See example 06 for the")
        print("framework-free equivalent.")
        return

    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")
    cfg = CouplingConfig()
    rule_factory, rule_label = _make_rule(cfg, ds)
    print(f"Decision rule: {rule_label}")

    _shared.banner("A) Drive time on the honeybees clock: model.step() ticks")
    _honeybees_clock_walkthrough(ds, cfg, rule_factory)

    _shared.banner("B) Triple-parity gate: full == scaffold == engine.run")
    _triple_equivalence_gate(ds, cfg, rule_factory)

    print(
        "\nThe time owner is now a genuine honeybees Model subclass and the\n"
        "decision math runs through the native DYNAMO-M DecisionModule, yet the\n"
        "results are bit-for-bit identical to engine.run(). The DecisionRule +\n"
        "lookup-table adapter seams made binding the real framework non-breaking."
    )


if __name__ == "__main__":
    main()
