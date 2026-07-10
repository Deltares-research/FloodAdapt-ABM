"""
04_monte_carlo_uncertainty.py
=============================
Why ``no_seq`` exists: flood events are drawn randomly each year, so a single
run is one noisy "weather history".  Running many independent sequences and
aggregating across them yields the *expected* behaviour plus its uncertainty.

What you learn here
-------------------
* each sequence is an independent random realisation of the SAME scenario,
* averaging over the sequence axis (axis 0) gives expected outcomes,
* the spread (std) over sequences quantifies uncertainty,
* how to reduce ``damage_history`` / ``adoption_fraction`` correctly.

Run::

    python 04_monte_carlo_uncertainty.py
"""
from __future__ import annotations

import numpy as np

import _shared
from floodadapt_abm import SimulationEngine, CouplingConfig, ThresholdRule


def main() -> None:
    _shared.banner("04 - MONTE-CARLO: averaging & uncertainty across sequences")
    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")

    cfg = CouplingConfig()
    # The reactive ThresholdRule is used here on purpose: its adaptation depends
    # on *realised* floods, so it shows genuine spread across sequences. (The
    # ex-ante SEURule is nearly deterministic, so its variance lives in the
    # damage arrays rather than in the adoption curve.)
    engine = SimulationEngine(
        ds=ds, config=cfg,
        decision_rule=ThresholdRule(cfg.decision, damage_threshold=0.30),
    )

    slr = np.linspace(0.0, 1.5, 30)
    no_seq = 20
    print(f"Agents: {engine.n_agents}  |  sequences: {no_seq}  |  years: {len(slr)}")

    res = engine.run(slr, no_seq=no_seq, seed=42)

    dmg = res["damage_history"]          # (no_seq, n_agents, n_years)
    adoption = res["adoption_fraction"]  # (no_seq, n_years)

    # Expected damage per agent per year (average over the sequence axis).
    exp_damage = dmg.mean(axis=0)        # (n_agents, n_years)

    # Expected adoption curve and its uncertainty band across sequences.
    mean_adopt = adoption.mean(axis=0)   # (n_years,)
    std_adopt = adoption.std(axis=0)     # (n_years,)

    print("\nExpected damage array shape (n_agents, n_years):", exp_damage.shape)

    print("\nAdoption curve (mean +/- 1 std across sequences), every 5th year:")
    print(f"  {'year':>4} {'mean':>8} {'std':>8}")
    for t in range(0, len(slr), 5):
        print(f"  {t:>4} {mean_adopt[t]:>8.1%} {std_adopt[t]:>8.1%}")
    t = len(slr) - 1
    print(f"  {t:>4} {mean_adopt[t]:>8.1%} {std_adopt[t]:>8.1%}   <- final year")

    print(
        "\nEach sequence shares identical inputs (SLR trajectory, config); only\n"
        "the random flood draws differ. Averaging across them is the Monte-Carlo\n"
        "method: the mean is the 'typical' outcome, the std is the uncertainty."
    )
    print("\nDone. Next: 05_dynamo_live_parity.py")


if __name__ == "__main__":
    main()
