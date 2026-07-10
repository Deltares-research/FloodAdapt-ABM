"""
03_custom_rule.py
=================
Open/Closed principle: add new behaviour by writing a new ``DecisionRule``,
*not* by editing the engine.  Here we implement a simple "insurance-like"
rule that adapts when the annualised adaptation cost is a small enough share
of income AND the household has recently been flooded.

What you learn here
-------------------
* the ``DecisionRule`` contract (one method: ``should_adapt``),
* what arrays the engine hands the rule each year and their shapes,
* how ``agent_state`` exposes wealth / income / risk_perception / flood_timer,
* that a custom rule plugs in with zero engine changes.

Run::

    python 03_custom_rule.py
"""
from __future__ import annotations

import numpy as np

import _shared
from floodadapt_abm import SimulationEngine, CouplingConfig, DecisionRule


class AffordableAndScaredRule(DecisionRule):
    """
    Toy rule: a household adapts this year when

    * it is *not* already adapted, and
    * the annualised adaptation cost is <= ``cost_income_ratio`` of income, and
    * it was flooded within the last ``recent_years`` years
      (``flood_timer`` counts years since the last flood).

    Demonstrates the interface only — it is not calibrated science.
    """

    def __init__(self, config, cost_income_ratio: float = 0.05, recent_years: int = 3):
        super().__init__(config)
        self.cost_income_ratio = cost_income_ratio
        self.recent_years = recent_years

    def should_adapt(
        self,
        agent_state,
        damages_this_year,   # (n_agents,)          [unused here]
        damages_no_adapt,    # (n_agents, n_events) [unused here]
        damages_adapt,       # (n_agents, n_events) [unused here]
        event_freqs,         # (n_events,)          [unused here]
        max_pot_dmg,         # (n_agents,)          [unused here]
        adaptation_costs,    # (n_agents,)  annualised loan repayment
    ) -> np.ndarray:
        affordable = adaptation_costs <= self.cost_income_ratio * agent_state.income
        recently_flooded = agent_state.flood_timer <= self.recent_years
        return affordable & recently_flooded & (~agent_state.is_adapted)


def main() -> None:
    _shared.banner("03 - CUSTOM RULE: write your own DecisionRule")
    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")
    cfg = CouplingConfig()

    rule = AffordableAndScaredRule(cfg.decision, cost_income_ratio=0.05, recent_years=3)
    engine = SimulationEngine(ds=ds, config=cfg, decision_rule=rule)
    print(f"Agents: {engine.n_agents}  |  Rule: {type(rule).__name__}")

    results = engine.run(np.linspace(0.0, 1.5, 30), no_seq=5, seed=42)

    adoption = results["adoption_fraction"]
    print(f"\nFinal adoption fraction: {adoption[:, -1].mean():.1%}")
    print(f"Total damage           : ${results['damage_history'].sum():,.0f}")
    print(
        "\nThe engine never changed - only the injected rule did. That is the\n"
        "Open/Closed principle: new science arrives as a new DecisionRule."
    )
    print("\nDone. Next: 04_monte_carlo_uncertainty.py")


if __name__ == "__main__":
    main()
