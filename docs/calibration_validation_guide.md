# Calibration and validation guide

This guide explains how to calibrate and validate the FloodAdapt-ABM
parameters in a pragmatic, low-cost way. It is written for modellers.
Jargon is defined where it first appears.

Role in the documentation set: this is the **method guide**. The design
record is `docs/architecture.md` (why the model is built this way). The
reference manual is `floodadapt_abm_documentation.md` (how to use the
package). This guide covers the behavioural parameters of both the
adaptation and the insurance decisions.

Two terms used throughout:

- **Calibration**: choosing parameter values so the model reproduces
  observed data or patterns.
- **Validation**: checking that the calibrated model also reproduces data
  or patterns it was *not* tuned on.

The guiding principle is cost ordering. Do the free checks first. Collect
new data only when the free checks show it matters.

## 1. Parameter inventory

The table lists every behavioural parameter, where its default comes from,
and what data would calibrate it. Priority marks how much the headline
results depend on it (H = high, M = medium, L = low).

| Parameter | Role | Default | Provenance | Priority | Calibration data |
|---|---|---|---|---|---|
| `risk_perc_max` | Perception spike right after a flood | 2.0 | Tierolf et al. (2023), France | H | Post-flood household survey |
| `risk_perc_coef` | Speed of perception decay (memory length) | -3.6 | Tierolf et al. (2023), France | H | Repeated surveys, or insurance drop-off rates after floods |
| `risk_perc_min` | Baseline perception with no flood memory | 0.01 | Tierolf et al. (2023) | M | Survey of never-flooded households |
| `perception_severity_form` | Shape of the severity response | `"power"` | This project (beyond native) | H | Survey: stated risk perception vs experienced damage |
| `perception_severity_exponent` (γ) | Concavity of the power form | 0.5 | This project, availability-heuristic argument | H | Same survey |
| `perception_severity_rate` (k) | Rate of the saturating form | 3.0 | This project | M | Same survey |
| `perception_severity_threshold` (s0) | Deadband of the threshold form | 0.1 | This project | M | Same survey |
| `flood_significance_threshold` | Minimum damage that counts as a flood experience | 0.01 | This project | L | Survey ("did you consider this a flood?") |
| `risk_aversion` | Curvature of the utility function | 1.0 | DYNAMO-M settings | M | Economics literature priors (0.5 to 2) |
| `discount_rate` | Time preference in the NPV | 0.032 | DYNAMO-M settings | M | Standard public rates; sensitivity only |
| `decision_horizon` | Years the household looks ahead | 15 | DYNAMO-M settings | M | Survey; or expected remaining residence time |
| `median_income` | Median of the synthetic income distribution | 70,000 | Charleston order of magnitude | H | **Directly measurable**: ACS B19013 for the study county (`fetch_acs_county_income`) |
| `mean_median_inc_ratio` | Spread of the income distribution | 1.15 | UN WIID fallback (native) | H | **Directly measurable**: ACS B19025 / B11001 over B19013. Controls `sd`, so it matters more than the median |
| `rank_correlation` (value proxy) | Income-to-building-value sorting strength | 0.5 | Household-finance literature (0.4 to 0.6) | M | Bounds [0.3, 0.7]; replaced entirely by a spatial income join |
| wealth-to-income table (`_WEALTH_RATIO_*`, not a config field) | Wealth multiplier by income percentile, used by the default income mode | 6 anchor points, `[0, 1.06, 4.14, 4.19, 5.24, 6]` at percentiles `[0, 20, 40, 60, 80, 100]` | Native `decision_module.py:27-29`, **no citation given** | M | **Not calibrated here.** Would need a household wealth survey (in the US, the Survey of Consumer Finances) |
| `income_to_wealth_ratio` | Scalar wealth multiplier, explicit-income runs only | 4.14 | Native table, 40th percentile | L | Same source; unused by the default income mode |
| `adaptation_total_cost` | One-off dry-floodproofing cost | None (legacy fraction) | Native: 10,800 EUR France, GDP-scaled | H | Local contractor quotes, FEMA mitigation cost tables |
| `expenditure_cap` | Max share of income spent on adaptation or premium | 0.06 | DYNAMO-M settings | H | Household budget surveys |
| `insurance_pricing` | Rating rule: `"community"` (one flat premium) or `"risk_based"` (own expected loss) | `"community"` | Native rule / this project | n/a | Scenario input, not calibrated |
| `insurance_deductible` | Damage share still paid when insured | 0.1 | Native hard-coded value | L | Actual policy terms (NFIP) |
| `insurance_loading` | Premium markup for insurer costs | 1.0 | This project | L | Industry loss ratios |
| `insurance_subsidy` | Premium share paid publicly | 0.0 | This project (policy lever) | n/a | Scenario input, not calibrated |
| `nuisance_freq_threshold` | Drop events more frequent than this | None (1.0 for Charleston) | Site decision | M | Event-set inspection |
| `lifespan_dryproof` | Service life of floodproofing | 75 | DYNAMO-M settings | L | Engineering literature |

## 2. The tiered approach

### Tier 0: verification (free, already in place)

Verification means checking the code does what the equations say. This is
not calibration, but it must come first. The repository ships it: the
parity gates prove the ported SEU kernels match native DYNAMO-M to a
relative error below 1e-4, and the golden regression pins the legacy
behaviour bit-exactly (`pytest tests/ -q`, `verification/`).

### Tier 1: sensitivity screening (free)

Before collecting any data, find out which parameters actually move the
results. A parameter that does not move the results does not need
calibration.

- Start with one-at-a-time sweeps on the high-priority rows above. For the
  perception form: run γ at 0.25 / 0.5 / 1.0, then the other two forms at
  their defaults. Compare adoption and damage trajectories.
- If interactions matter, use Morris screening. Morris screening is a
  standard method that varies parameters along random one-step paths and
  ranks them by the mean and spread of the output changes. It needs only
  tens of model runs, not thousands.
- Report the ranking. Parameters that rank low get literature values and a
  note. Parameters that rank high move to Tier 2 or 3.

### Tier 2: pattern-oriented calibration (cheap, open data)

Pattern-oriented calibration means tuning parameters so the model
reproduces a few robust observed patterns, rather than fitting every data
point.

**Start by separating what is measurable from what must be assumed.** The
income model is the clearest case, and the pattern generalises. It takes
two independent inputs:

1. the **marginal distribution**, or how much a household at percentile
   *p* earns (`median_income`, `mean_median_inc_ratio`);
2. the **rank assignment**, or which building sits at percentile *p*
   (`income_percentile_per_agent`).

Only the second needs building locations. The first is a property of the
study area and is published for every US county. Losing the footprints
therefore costs you axis 2 and nothing on axis 1, so the regional
distribution should never be left at a generic default, even when the
percentiles come from the value proxy. Ask this question of every
parameter before tuning it: is it measurable from aggregate data, or does
it genuinely require micro-data?

**Why a lognormal, and what the two parameters do.** The model assumes
regional income follows a lognormal distribution: `log(income)` is normally
distributed with parameters `mu` and `sigma`. A lognormal is the standard
choice for income because it is right-skewed. Most households cluster below
the average, and a thin tail of high earners pulls the average up. That is
what real income data looks like.

Nobody publishes `mu` and `sigma`, so the code converts from statistics
that are published. The **median** is the middle household, half above and
half below. The **mean** is total income over households, the plain
average. Because a few very rich households lift the average without moving
the middle household, the mean sits at or above the median, and the gap
between them measures the spread. For a lognormal both conversions are
exact:

- `median = exp(mu)`, so `mu = ln(median)`;
- `mean / median = exp(sigma^2 / 2)`, so `sigma = sqrt(2 ln(mean/median))`.

So the pair is not "mean and standard deviation". It is the median, which
fixes the centre, and `mean_median_inc_ratio`, which fixes the spread. Both
are needed: two counties can share a median and be nothing alike, one equal
and one very unequal, and only the ratio separates them. This is why the
ratio matters more than the median for the results.

Downstream, `_synthesize_income_wealth` in
`_core/dynamo_decision_bridge.py` draws 5,000 samples from that lognormal
once per run, sorts them, and lets each household read its income off the
curve at its own income percentile (axis 2). Wealth follows from income
through the native wealth-to-income ratio table. Income and wealth then
drive the affordability check (`expenditure_cap`) in the adaptation and
insurance decision. The draw uses a dedicated seed-derived generator, so
building the distribution never perturbs the event-draw or decision RNG
streams and the bit-parity gates stay valid.

For Charleston the measured pair gives `sigma = sqrt(2 ln(1.4879)) = 0.892`.
The default ratio 1.15 implies `sigma = 0.529`, a much narrower and, for
this county, plainly wrong distribution: it predicts 2.4 % of households
above $200k against 17.2 % observed.

**What "calibrating the marginal" actually means.** There are exactly two
free parameters on this axis, `median_income` and `mean_median_inc_ratio`,
which map to the lognormal's own parameters as above. Fitting them is not an
optimisation: there is no loss function and nothing is searched. Two
parameters are matched to two independent published numbers.
`median_income` is read straight off ACS table B19013, which already is a
median. `mean_median_inc_ratio` is the mean, aggregate household income
(B19025) divided by households (B11001), divided again by that same
median. The fit is exact by construction, so there is no under- or
over-fitting question at this step; `fetch_acs_county_income` performs the
whole computation in one call. What the fit cannot tell you is whether a
two-parameter lognormal is even the right *shape* for the county's real
income distribution. That question needs data the fit never touched, which
is what Tier 4's bracket check, below, answers.

This is genuine calibration only on axis 1. The Tier A/B rank assignment
(the spatial join) is calibrated too, to real per-region income. The Tier C
fallback (`rank_correlation = 0.5`) is not: it is a literature value from
household-finance studies on how income ranks correlate with housing-value
ranks, used because no per-building income ground truth exists for
Charleston to fit it to. Call Tier C "assumed", not "calibrated".

Useful free sources for a US coastal study:

- **OpenFEMA NFIP policies** (openfema data sets): flood-insurance
  take-up rates by county and year. Target for the insurance module:
  simulated uptake in the observed range.
- **OpenFEMA NFIP claims**: claim counts and paid amounts after events.
  Target: simulated out-of-pocket and insured shares.
- **ACS tables B19013, B19025, B11001** (Census Data API): median
  household income, aggregate household income, and household count for
  the study county. These *measure* `median_income` and
  `mean_median_inc_ratio` rather than approximating them, so neither
  should be left at its default for a real case study. ACS is the US
  Census Bureau's American Community Survey. Use
  `income_utils.fetch_acs_county_income`; a free API key is required
  (`CENSUS_API_KEY`).
- **ACS table B19001** (same API): the household-income histogram in 16
  brackets. This is a *validation* source, not a fitting one. See the
  income-marginal check below.
- **Elevation certificates and mitigation records** (local floodplain
  manager, USACE): counts of floodproofed or elevated homes. Target:
  simulated adoption levels and their timing after major floods.
- **Building permits** after flood events: a proxy for the adaptation
  response speed, which constrains the perception decay.

Fit by hand or by grid search over the few high-ranked parameters. Keep
the number of tuned parameters below the number of independent patterns.

### A worked calibration, end to end

The income marginal, done for real, with Charleston County numbers. Steps 1
to 5 are the whole calibration. Step 6 is the check, and it uses data the fit
never touched.

**Step 1. Fetch the three published numbers.** One call does it:

```python
from floodadapt_abm.income_utils import fetch_acs_county_income
stats = fetch_acs_county_income("45", "019")     # Charleston County, SC
```

| Source | Meaning | Value |
|---|---|---|
| B19013 | median household income | 88,494 |
| B19025 | aggregate household income | 23,566,353,400 |
| B11001 | households | 178,975 |

**Step 2. Derive the mean.** ACS publishes no mean, so compute it:

```
mean = 23,566,353,400 / 178,975 = 131,674
```

**Step 3. Form the ratio.** This is the parameter the model wants:

```
mean_median_inc_ratio = 131,674 / 88,494 = 1.4879
```

**Step 4. Convert to the lognormal's own parameters.** Both conversions are
exact, no fitting involved:

```
mu    = ln(88,494)              = 11.3907
sigma = sqrt(2 ln 1.4879)       = 0.8915
```

**Step 5. Assign them.** Two parameters, two published numbers:

```python
config.decision.median_income = stats["median_income"]            # 88,494
config.decision.mean_median_inc_ratio = stats["mean_median_ratio"]  # 1.4879
```

That is the entire calibration. There is no loss function, nothing is
searched, and the fit is exact by construction, so there is no risk of
over-fitting at this step. What it cannot tell you is whether a
two-parameter lognormal is the right *shape* for this county.

**Step 6. Check the shape against data the fit never saw.**

```python
from floodadapt_abm.income_utils import (
    bracket_fit_distance, fetch_acs_income_brackets, lognormal_bracket_shares,
)
observed = fetch_acs_income_brackets("45", "019")["shares"]        # B19001
bracket_fit_distance(observed, lognormal_bracket_shares(88_494, 1.4879))  # 0.072
bracket_fit_distance(observed, lognormal_bracket_shares(70_000, 1.15))    # 0.326
```

The measured pair misplaces about 7 % of households across the 16 brackets;
the package defaults misplace about 33 %. See Tier 4 for why B19001 counts as
held-out data and for how to read the residual.

### Where the ACS data lives

Which published table supplies which parameter:

| ACS table | Variables | Fetched by | Ends up in |
|---|---|---|---|
| B19013 | `B19013_001E` | `fetch_acs_county_income` | `median_income` |
| B19025 | `B19025_001E` | same call | numerator of the derived mean |
| B11001 | `B11001_001E` | same call | denominator of the derived mean |
| B19001 | `B19001_002E` to `B19001_017E` (16 brackets) | `fetch_acs_income_brackets` | validation only, never stored in the config |

If a browser view of B19001 appears to show only a handful of rows, it is a
collapsed view. The definitive list is the API variable index,
`api.census.gov/data/2024/acs/acs5/groups/B19001.html`, which shows
`B19001_001E` (the total) plus the 16 bracket variables.

Where the numbers are actually kept, which is easy to assume wrongly:

- **The package caches nothing.** Both fetchers issue a fresh HTTP request on
  every call. There is no disk cache and no memoisation, so a run without
  network access cannot fetch anything.
- The fitted values live **in memory only**, on `config.decision.median_income`
  and `config.decision.mean_median_inc_ratio`.
- Notebook 2 therefore **pins them as literal constants** so it reproduces
  offline, and re-fetches only to report drift against those constants.
- `tests/test_income_marginal.py` pins its own copy so the suite stays offline.
- The only things written to disk are the **TIGER/Line shapefiles**, cached by
  the notebook helper to `%TEMP%\floodadapt_abm_census` (override with
  `FA_ABM_CENSUS_CACHE`), and the Tier A/B percentile file
  `income_percentiles_charleston.npy` if you ever produce one.
- The API key is read from `CENSUS_API_KEY` and is never written to any file.

Both fetchers take a `year` argument and default to `2024`, meaning the ACS
2020-2024 5-year release. Use the same vintage for the fit and for the
validation.

### Inherited constants that are not calibrated

Not every number in the model is measured, and the honest move is to say
which. Two stand out.

**The wealth-to-income table.** Wealth is not sampled independently; it is
`ratio(percentile) * income`, interpolated from six anchor points that rise
from 0 at the bottom to 6.0 at the top. Those six numbers come straight from
native DYNAMO-M (`decision_module.py:27-29`), where the only documentation is
the inline comment "wealth in relation to income". No source is cited, and
nothing about them is specific to any study area, Charleston included.
Changing them breaks bit-parity with native, so treat them as a sensitivity
target rather than a free parameter: vary them, see whether your conclusions
move, and report that.

**The value-proxy rank correlation.** `rank_correlation = 0.5` is a literature
value for how income ranks track housing-value ranks, not a Charleston
measurement. There is no per-building income ground truth to fit it against.
Report it as an assumption with bounds [0.3, 0.7].

The contrast worth keeping in view: the income *marginal* is measured, in the
sense that two published numbers determine it exactly. The wealth multiplier
and the rank correlation are assumed. A calibration report should not present
them in the same tone.

### Tier 3: a small survey (the first real cost)

Only needed if Tier 1 shows the perception parameters matter and Tier 2
patterns cannot pin them down. A short household survey (n in the low
hundreds) can discriminate the severity forms:

1. Ask flooded households for the approximate damage as a share of their
   home value, and their current flood-risk concern on a fixed scale.
2. Ask when the flood happened. The concern-versus-years-since curve
   fits the decay pair (`risk_perc_max`, `risk_perc_coef`).
3. Plot concern against damage share. Fit all three severity forms by
   least squares. Keep the best-fitting form. Each form has one free
   parameter, so a small sample is enough to compare them.
4. Include never-flooded households. Their mean concern estimates
   `risk_perc_min`.

Questions map one-to-one to parameters, which keeps the survey short and
the analysis simple.

### Tier 4: validation

Validate on material not used for calibration:

- **Check fitted shapes, not just fitted moments.** Two numbers pin a
  lognormal, but matching a median and a mean does not prove the
  distribution has the right shape. Compare the fitted distribution
  against an observed histogram that was *not* used in the fit.

  **This is not a row-level train/test split, because there is no
  row-level data to split.** The Census Bureau never publishes individual
  ACS responses, only aggregate tabulations, for privacy. What plays the
  role of held-out data here is a *different published tabulation of the
  same population*: ACS table B19001 reports household counts across 16
  income brackets, far richer than the single median and mean the fit
  consumed. The fitting numbers (B19013, B19025, B11001) collapse to two
  parameters; B19001 carries 16. Because the fit never sees the extra
  information B19001 holds, comparing the fitted lognormal's implied
  bracket shares against the observed ones is a genuine out-of-sample
  check. "Held out" means a different published table here, not
  held-out rows.

  `income_utils.lognormal_bracket_shares` produces the model-implied
  shares from the two fitted parameters; `bracket_fit_distance` scores
  them against the observed B19001 shares as a total-variation distance,
  the fraction of households the model places in the wrong bracket (0 is
  perfect). Worked result for Charleston County (ACS 2020-2024 5-year):
  the package defaults (70,000, ratio 1.15) misplace 33 % of households,
  while the measured pair (88,494, ratio 1.4879) misplaces 7 %. The
  residual sits in the lowest brackets, where a lognormal cannot
  reproduce the spike of near-zero-income households. Record that limit
  rather than tuning around it, since the functional form is inherited
  from native DYNAMO-M and changing it would break parity.
- **Hold-out patterns**: calibrate on some patterns (say, income and
  uptake), then check the untouched ones (say, post-flood permit spikes).
- **Cross-site transfer**: calibrate on one county, run another with only
  its own income and hazard inputs changed. Structural parameters should
  transfer; if they do not, they were overfitted.
- **Backcasting**: if a historical flood with documented responses exists,
  run the model over that period and compare.

Report validation as ranges and directions, not single numbers. An ABM
with stochastic weather should reproduce the pattern, not the exact path.

## 3. If you have little or no data

This is the expected starting point, and it is workable:

- Use the literature priors in the inventory table. Every default has a
  stated source.
- Replace point claims with bounds. Run the model at the low and high end
  of each uncertain parameter and report both. If a conclusion holds
  across the bounds, it does not depend on the calibration.
- Say what is an assumption. The value proxy (`rank_correlation = 0.5`)
  is the clearest example: it is a documented assumption with bounds, not
  data, and the docs and notebook say so.
- Prefer relative statements ("risk-based pricing shifts affordability
  from the high-risk tail to the majority") over absolute ones ("uptake
  will be 3.2 %"). Relative statements are robust to most calibration
  error.

## 4. The event draw needs no calibration

The Poisson event draw is not a calibrated behaviour; it is the exact
statistical model for the event set. Each event in the lookup table has an
occurrence rate (events per year). The Poisson distribution gives the
number of occurrences per year for a known rate, so every event keeps its
true long-run frequency, rare extremes included, and sub-annual events can
occur several times a year. The only site decision is
`nuisance_freq_threshold`, which drops very frequent nuisance events from
the catalogue (recommended: 1.0 for the Charleston set). See
`docs/architecture.md`, "Event Drawing Approach", for the full rationale.

## 5. Suggested first pass for Charleston

1. Tier 1 sweep: γ (0.25 / 0.5 / 1.0), `risk_perc_coef` (-2 / -3.6 / -6),
   `adaptation_total_cost` (5k / 10.8k / 25k USD), `rank_correlation`
   (0.3 / 0.5 / 0.7). About 12 runs.
2. Set `median_income` and `mean_median_inc_ratio` from ACS Charleston
   County. Zero modelling effort, removes two free parameters. **Done**:
   `fetch_acs_county_income("45", "019")` gives 88,494 and 1.4879 on the ACS
   2020-2024 5-year release, and the B19001 bracket check scores the fit at
   0.072 against 0.326 for the defaults. Notebook 2 (§2.6) pins and
   re-verifies these. Refresh both when a new ACS vintage is published, and
   keep the fit and the B19001 validation on the same vintage.
3. Pull NFIP take-up for Charleston County from OpenFEMA and check the
   insurance scenarios against it.
4. Only then decide whether a survey (Tier 3) is worth the cost.
