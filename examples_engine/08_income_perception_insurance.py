"""
08_income_perception_insurance.py
=================================
Tour of the post-review features: household **income percentiles**, the
**severity-response forms** of flood risk perception, the **Poisson** event
draw, and **insurance pricing** modes.

What you learn here
-------------------
* two numpy-only ways to build ``income_percentile_per_agent``:
  from regional income data (``percentiles_from_income_values``) and from
  the building-value proxy when no locations are known
  (``percentiles_from_value_proxy``),
* how the three ``perception_severity_form`` options shape the
  risk-perception spike after a flood,
* why the Poisson draw recovers every event's nominal occurrence rate
  (the legacy clip+cap draw does not),
* what ``insurance_pricing="community"`` vs ``"risk_based"`` means for
  premiums and uptake.

The geopandas-based spatial join (``derive_income_percentiles``) is not run
here because it needs building footprints and geopandas; see
``floodadapt_abm/income_utils.py`` and notebook 2 for that workflow.

Run::

    python 08_income_perception_insurance.py
"""
from __future__ import annotations

import numpy as np

import _shared
from floodadapt_abm import CouplingConfig, SimulationEngine
from floodadapt_abm.event_utils import draw_year_events
from floodadapt_abm.income_utils import (
    percentiles_from_income_values,
    percentiles_from_value_proxy,
)

SLR = np.linspace(0.0, 1.5, 20)
SEED = 42


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two 1-D arrays."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _residential_max_pot_dmg(ds) -> np.ndarray:
    """Building values of the residential agents, in engine order."""
    types = np.asarray(ds["object_id"].attrs["primary_object_type"], dtype=str)
    mask = np.char.find(types, "RES") >= 0
    return np.asarray(ds["object_id"].attrs["max_pot_dmg"], dtype=np.float64)[mask]


def _income_percentiles(ds) -> np.ndarray:
    """Derive percentiles two ways and print the correlation contrast."""
    mpd = _residential_max_pot_dmg(ds)
    n = mpd.shape[0]

    # (a) From regional income data: pretend the agents sit in 4 districts
    # with different median household incomes (in a real study these come
    # from a census join; here they are mocked).
    district = np.arange(n) % 4
    district_income = np.array([45_000.0, 62_000.0, 80_000.0, 115_000.0])
    pct_regional = percentiles_from_income_values(
        district_income[district], jitter=5, seed=SEED
    )

    # (b) From the building-value proxy (no locations needed): a NOISY rank
    # link at Spearman rho = 0.5 — richer households tend to hold more
    # valuable homes, but income is not a function of the home value.
    pct_proxy = percentiles_from_value_proxy(mpd, rank_correlation=0.5, seed=SEED)

    print(f"  residential agents                 : {n}")
    print(f"  regional percentiles vs value rank : rho = {_spearman(pct_regional, mpd):+.2f}")
    print(f"  value-proxy percentiles vs value   : rho = {_spearman(pct_proxy, mpd):+.2f}  (target 0.50)")
    print("  (uniform fallback would give rho ~= 0; legacy income mode gave rho = 1)")
    return pct_proxy


def _income_into_engine(ds, pct: np.ndarray) -> None:
    """Feed percentiles to the engine and show what the agents earn."""
    cfg = CouplingConfig(random_seed=SEED)
    engine = SimulationEngine(ds=ds, config=cfg, income_percentile_per_agent=pct)
    income = engine._data.income
    cost = engine._annual_adapt_cost
    constrained = income * cfg.decision.expenditure_cap <= cost
    print(f"  income  p10 / median / p90         : "
          f"${np.percentile(income, 10):,.0f} / ${np.median(income):,.0f} / ${np.percentile(income, 90):,.0f}")
    print(f"  affordability gate binds for       : {constrained.mean():.1%} of agents")


def _severity_forms(ds) -> None:
    """Perception spike right after a flood, per severity form."""
    severities = np.array([0.05, 0.25, 0.50, 1.00])
    forms = ["power", "saturating_exp", "threshold_linear"]
    print(f"  {'damage severity':>16} | " + " | ".join(f"{f:>16}" for f in forms))
    rows = {}
    for form in forms:
        cfg = CouplingConfig(random_seed=SEED)
        cfg.decision.perception_severity_form = form
        engine = SimulationEngine(ds=ds, config=cfg)
        n = engine.n_agents
        flooded = np.zeros(n, dtype=bool)
        flooded[: severities.size] = True
        sev = np.zeros(n)
        sev[: severities.size] = severities
        engine.update_flood_experience(flooded, sev)
        rows[form] = engine.state.risk_perception[: severities.size]
    for i, s in enumerate(severities):
        cells = " | ".join(f"{rows[f][i]:16.3f}" for f in forms)
        print(f"  {s:16.0%} | {cells}")
    print("  power: strong response even to small floods; saturating_exp: finite")
    print("  slope at zero; threshold_linear: no response below 10 % damage.")


def _poisson_rate_recovery() -> None:
    """A rare extreme keeps its nominal rate under the Poisson draw."""
    names = np.array([f"ev_{i}" for i in range(12)])
    freqs = np.array([0.01] + [1.5] * 11)      # 1 extreme + 11 nuisance events
    n_years = 3_000

    def realised_rate(mode: str, cap: int | None) -> float:
        rng = np.random.default_rng(SEED)
        hits = 0
        for _ in range(n_years):
            occ = draw_year_events(
                names, freqs, rng, max_events_per_year=cap, mode=mode,
                cap_policy="random" if mode == "bernoulli_clip" else "largest_damage",
                event_severity=np.arange(12, dtype=np.float64)[::-1],
            )
            hits += sum(1 for name in occ if name == "ev_0")
        return hits / n_years

    poisson = realised_rate("poisson", None)
    legacy = realised_rate("bernoulli_clip", 4)
    print(f"  nominal rate of the extreme        : 0.0100 events/year")
    print(f"  realised, poisson (no cap)         : {poisson:.4f}  (matches)")
    print(f"  realised, legacy clip + cap of 4   : {legacy:.4f}  (extreme mostly discarded)")


def _insurance_pricing(ds) -> None:
    """Community vs risk-based (and subsidised) premiums, same simulation."""
    scenarios = [
        ("community", 0.0),
        ("risk_based", 0.0),
        ("risk_based", 0.9),
    ]
    for pricing, subsidy in scenarios:
        cfg = CouplingConfig(random_seed=SEED)
        cfg.decision.include_insurance = True
        cfg.decision.insurance_pricing = pricing
        cfg.decision.insurance_subsidy = subsidy
        engine = SimulationEngine(ds=ds, config=cfg)
        res = engine.run(SLR, no_seq=3, seed=SEED)
        uptake = res["insured_fraction"].mean(axis=0)
        premium = res["premium_history"].mean(axis=0)
        label = pricing if subsidy == 0.0 else f"{pricing} + {subsidy:.0%} subsidy"
        print(f"  {label:<26}: premium y1 ${premium[0]:>7,.0f} -> yN ${premium[-1]:>7,.0f}"
              f" | uptake peak {uptake.max():.1%} final {uptake[-1]:.1%}")
    print("  community: one flat premium = mean expected annual damage (native);")
    print("  risk_based: each household pays its own expected payout; the subsidy")
    print("  is the share of the premium paid by a public scheme (the household")
    print("  pays the rest) - the lever that moves uptake when the 6 % income cap binds.")


def main() -> None:
    _shared.banner("08 - New features: incomes, perception forms, Poisson, insurance")
    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")

    _shared.banner("A) Income percentiles: regional data vs building-value proxy")
    pct = _income_percentiles(ds)
    _income_into_engine(ds, pct)

    _shared.banner("B) Severity-response forms of the risk-perception spike")
    _severity_forms(ds)

    _shared.banner("C) Poisson event draw: rate recovery for a rare extreme")
    _poisson_rate_recovery()

    _shared.banner("D) Insurance pricing: community vs risk-based")
    _insurance_pricing(ds)

    print("\nDone. This is the end of the numbered path; notebook 2 runs the")
    print("full coupled model (and the income workflow) on the real table.")


if __name__ == "__main__":
    main()
