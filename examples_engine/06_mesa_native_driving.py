"""
06_mesa_native_driving.py
=========================
Phase 4b: **Mesa-native driving** — inverting time ownership.

In examples 01-05 the year loop lives inside ``engine.run()``. Here the loop is
owned by a small model that advances one tick at a time via ``model.step()``,
mirroring the native DYNAMO-M ``SLRModel.run_model()``. The decision science is
unchanged; only *who drives time* changes.

What you learn here
-------------------
* the object graph mirrors DYNAMO-M: ``FloodAdaptSLRModel -> Agents ->
  CoastalNodePopulation`` (households) -> shared ``DecisionRule``,
* ``model.step()`` advances exactly one year; ``model.run_model()`` is the tick
  loop,
* ``run_mesa_native(...)`` is a drop-in analogue of ``engine.run(...)`` and
  produces **bit-for-bit identical** results (the Phase-4b gate),
* the ``DecisionRule`` seam is what makes this time-ownership migration
  non-breaking (and paves the way to the native honeybees ``model.step()``).

Run::

    python 06_mesa_native_driving.py
"""
from __future__ import annotations

import numpy as np

import _shared
from floodadapt_abm import (
    SimulationEngine,
    CouplingConfig,
    FloodAdaptSLRModel,
    run_mesa_native,
)

SLR = np.linspace(0.0, 1.5, 20)
SEED = 42


def _manual_tick_walkthrough(ds, cfg) -> None:
    """Drive a single sequence one tick at a time to show time ownership."""
    engine = SimulationEngine(ds=ds, config=cfg)
    model = FloodAdaptSLRModel(engine, SLR, seed=SEED)

    print(f"  households (agents.regions.n): {model.agents.regions.n}")
    print(f"  clock starts at timestep      : {model.timestep}")
    for _ in range(3):
        model.step()  # one native tick == one year
        adopted = engine.state.is_adapted.mean()
        print(f"  after model.step(): timestep={model.timestep:2d}  adopted={adopted:.1%}")
    model.run_model()  # finish the remaining years
    print(f"  run_model() finished at timestep={model.timestep} (== {model.n_timesteps} years)")


def _equivalence_check(ds, cfg) -> None:
    """Show run_mesa_native == engine.run bit-for-bit (the Phase-4b gate)."""
    native = run_mesa_native(SimulationEngine(ds=ds, config=cfg), SLR, no_seq=5, seed=SEED)
    loop = SimulationEngine(ds=ds, config=cfg).run(SLR, no_seq=5, seed=SEED)

    dmg_ok = np.array_equal(native["damage_history"], loop["damage_history"])
    adapt_ok = np.array_equal(native["adapted_history"], loop["adapted_history"])
    print(f"  damage_history  identical: {dmg_ok}")
    print(f"  adapted_history identical: {adapt_ok}")
    print(f"  gate: {'PASS' if dmg_ok and adapt_ok else 'FAIL'}")
    print(f"  (final adoption, mean over sequences: {native['adoption_fraction'][:, -1].mean():.1%})")


def main() -> None:
    _shared.banner("06 - PHASE 4b: Mesa-native driving (time-ownership inversion)")
    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")
    cfg = CouplingConfig()

    _shared.banner("A) Drive time by hand: model.step() ticks")
    _manual_tick_walkthrough(ds, cfg)

    _shared.banner("B) Equivalence gate: run_mesa_native == engine.run")
    _equivalence_check(ds, cfg)

    print(
        "\nThe loop owner changed (engine.run -> model.step ticks) but the\n"
        "results did not. The DecisionRule.should_adapt seam is unchanged, so\n"
        "migrating time toward the native honeybees model.step() is non-breaking.\n"
        "Example 07 (Phase 4b-full) binds the real honeybees Model and the native\n"
        "DYNAMO-M DecisionModule on top of this same seam."
    )
    print("\nDone. Continue with 07_mesa_native_full.py (Phase 4b-full).")


if __name__ == "__main__":
    main()
