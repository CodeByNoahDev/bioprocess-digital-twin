# Walkthrough: the biology, the statistics and the design decisions

---

## 0. The manufacturing context

Most modern drugs from companies like BMS are **biologics**: large proteins such as **monoclonal antibodies (mAbs)** that are too complex to make by chemistry. Instead, living cells, usually **Chinese Hamster Ovary (CHO)** cells engineered with the antibody gene, make them inside a stirred **bioreactor**.

A typical **fed-batch** run:
1. **Inoculate**: put a small number of cells into fresh medium.
2. **Growth phase** (days 0–5): cells divide roughly once a day at 37 °C.
3. **Feeding** (daily from day 3): add concentrated nutrients because the cells would otherwise starve.
4. **Temperature shift** (around day 5): drop to ~33 °C. Growth slows, but each cell makes more antibody and lives longer.
5. **Harvest** (day ~14): remove the cells and send the antibody-rich liquid to purification ("downstream processing").

Process engineers watch **critical process parameters (CPPs)** such as temperature, pH, DO and osmolality, because they affect **critical quality attributes (CQAs)** of the product. Regulators expect the process to be understood and controlled. That is the idea behind **Quality by Design (ICH Q8)** and **Process Analytical Technology (PAT)**.

---

## 1. The mechanistic model (`model.py`)

The model is a set of **ordinary differential equations**: equations for how fast each quantity changes. `scipy.integrate.solve_ivp` (LSODA solver) steps them forward in time.

### Growth: the Monod equation
Cells grow fastest when food is plentiful, and growth slows as a nutrient runs out:

    mu = mu_max · GLC/(K_glc + GLC) · GLN/(K_gln + GLN) · inhibition terms · environment terms

`C/(K + C)` is close to 1 when C ≫ K and close to 0 when C ≪ K. Waste products **inhibit** growth with `K_I/(K_I + C)`, which is close to 1 when C is small and falls as waste accumulates.

**Environment terms:** temperature, `exp(0.12·(T − 37))`, so 33 °C gives about 60 % of the growth rate; pH, a bell curve around 7.0; DO, a Monod term with a small K, so CHO cells only suffer at very low oxygen; and osmolality (below).

### Death
A small basal death rate, plus ammonia toxicity (a sigmoidal Hill function that stays harmless at low levels and then rises sharply), glucose starvation, osmotic stress and pH stress. Dead cells slowly **lyse** (break apart).

### Metabolism
- **Glucose and glutamine** are used for growth (via *yield coefficients*: cells made per mM consumed) and for **maintenance** (energy to stay alive even without growing).
- **Lactate** is made by fast glycolysis while cells grow fast, and **consumed** once growth slows. This "lactate shift" is seen in most good CHO processes, and failing to shift is a known warning sign.
- **Ammonia** comes from glutamine metabolism and from glutamine breaking down in the medium.

### Product: Luedeking–Piret
Antibody production per cell = α·mu + β: a growth-linked part plus a constant part. A lower temperature raises β, which is why the temperature shift increases titer.

### Osmolality: why "just feed more" doesn't work
Feed is concentrated. Every bolus adds salts and sugars, and the base added to neutralise lactate adds sodium. Osmolality (≈ sum of dissolved particles) rises from ~300 to ~400 mOsm/kg. Above ~360 the model slows growth and speeds up death, so there is an **optimum feed rate** rather than "more is better".

### Feeding (`simulate_batch`)
The ODEs are integrated one day at a time. Each morning the reactor is **sampled first**, then fed: the feed volume is mixed in and everything is diluted. If glucose is still below 15 mM, a glucose stock is added to bring it to 25 mM. Tests check that the volume balance closes exactly and that antibody *mass* (titer × volume) never decreases, even though *concentration* drops at each feed.

---

## 2. A fleet of batches (`batches.py`)

No two real batches are identical, so each simulated batch gets small random (lognormal, a few %) changes to growth rate, productivity, death rate, seeding density, media and feed amounts, and shift timing.

Five **fault types** are injected, each starting at a random time. The most interesting is the **pH probe drift**: the probe reads high, so the controller adds acid until the *probe* reads 7.0, while the *real* pH sinks. Everything looks perfect on the pH trend. This is why sensor faults are dangerous.

**Measurement noise** mimics real analysers: ~5 % on cell counts, ~3 % on glucose, ±1 % on viability, and so on. Online probe values are averaged over the 24 h before each sample.

---

## 3. Multivariate Statistical Process Control (`mspc.py`)

### Why multivariate?
With 11 variables you could draw 11 control charts, but problems often show up as a *combination* (VCD a bit low AND lactate a bit high AND glutamine a bit high) where no single chart alarms. And 11 charts each with a 1 % false-alarm rate give about 10 % false alarms overall.

### PCA in one paragraph
Principal Component Analysis finds a few directions ("components") that capture most of the variation in correlated data. Normal batches mostly vary along these directions (e.g. "a bit more growth overall" moves VCD, titer and lactate together). Each batch then gets **scores** (where it sits along the components) and a **residual** (what the components can't explain).

### The two statistics
- **Hotelling's T²**: distance from the centre *within* the model, measured in units of normal spread. Large means an extreme but "normal-shaped" batch.
- **SPE (Q, squared prediction error)**: size of the residual. Large means the batch has a pattern normal batches never have, so the correlation structure is broken. Most faults show up here first.

### Batch-wise unfolding
Batch data is 3-D (batch × variable × day). Following Nomikos & MacGregor, each batch becomes one long row (vcd_d0, viability_d0, …, vcd_d1, …), so ordinary PCA can be used. To monitor a batch **while it is still running**, there is one model per day using only data up to that day (`EvolvingMonitor`).

### Control limits, and the most important lesson in this project
Limits are set with Box's approximation: the statistic is treated as a scaled chi-squared distribution matched to its mean and variance. The question is *which* values to match.
- **Naive:** use the training batches' own T²/SPE. But PCA was fitted to those batches, so their residuals are artificially small. Result: **100 % false alarms** on new normal batches.
- **Leave-one-out:** for each training batch, fit PCA without it and score it, which mimics a future batch. Result: 10 % of new normal batches alarm (2/20) and 3.6 % of batch-days are out of limit. With only 40 training batches the limits are themselves uncertain, so this is a reasonable outcome.

This is a general machine-learning lesson: **never judge a model on the data it was trained on.**

### Alarm rule and diagnosis
An alarm requires **2 consecutive days** outside the 99 % limits, which filters single noisy samples. When an alarm fires, **contributions** show which variables are responsible:
- SPE contribution = squared residual of each variable;
- T² contribution = how much each variable pushes the batch along the model directions.

The heatmap shows the method working: temperature and DO faults are blamed on temp and DO, the glutamine feed fault on glutamine, lactate and osmolality, and the poor inoculum on viability and VCD.

---

## 4. Raman soft sensor (`softsensor.py`)

### What Raman spectroscopy measures
Laser light scattered by molecules shifts in energy by amounts that depend on their chemical bonds. A spectrum is a molecular fingerprint, and a probe inside the bioreactor gives one every few minutes without taking a sample.

### Simulating spectra
Each molecule contributes Gaussian bands at (approximate) positions: glucose ~1125 cm⁻¹, lactate ~856 cm⁻¹, protein/cells ~1003 (phenylalanine) and ~1660 (amide I), scaled by concentration. On top come the realistic problems:
- a large, smooth **fluorescence background** that grows with biomass and differs between batches;
- **laser power drift** (±3 % multiplicative);
- **detector noise**.

### Preprocessing
- **Savitzky–Golay first derivative**: fits a small polynomial in a moving window and takes its slope. Flat or sloped baselines become ~0, while sharp bands survive.
- **SNV**: subtract each spectrum's mean and divide by its standard deviation, to remove multiplicative effects.

Both were compared by cross-validation, as shown in the README table. SNV made things worse here because the fluorescence background dominates each spectrum's standard deviation.

### PLS regression
Spectra have 701 correlated wavenumbers, many more than the number of samples, so ordinary regression would overfit badly. **Partial Least Squares** compresses the spectra into a few latent variables chosen to predict glucose, then regresses on those.

### Validation done right
- **Grouped cross-validation by batch** (`GroupKFold`). Spectra from the same batch are similar; if one batch were in both training and validation folds, the score would be over-optimistic (**data leakage**).
- **Number of components** = the smallest within 5 % of the best RMSECV. Simpler models are more robust.
- **Spiked calibration samples**: in real calibration runs, samples are split and spiked with known glucose. That widens the range and breaks the correlation between glucose and culture age. Without it, PLS could learn "time" as a proxy for glucose. The benefit shows up on faulty batches where glucose leaves the normal range.
- **The reference method sets the floor**: the offline analyser has ±3 % error (~1 mM), so RMSEP against it can't go much below that.

---

## 5. Process optimisation (`optimize.py`)

The twin runs a 14-day batch in ~0.05 s. Scanning 8 temperature-shift days × 5 feed rates (40 virtual batches) shows the trade-offs:
- shift too early: too few cells; shift too late or never: low productivity and viability crashes;
- too little feed: starvation; too much: osmotic stress.

The **constraint** of harvest viability ≥ 70 % reflects downstream processing, since dead cells release host-cell proteins and DNA that are harder to remove. The feasible region is a simple **design space** in the ICH Q8 sense. The nominal recipe sits on a broad plateau, which means it is robust.

---

## 6. Tests (`tests/`)

- **Physics/biology**: no negative concentrations, the volume balance closes, product mass never decreases, textbook peak VCD/viability/titer ranges, the lactate shift happens, the temperature shift raises titer, more feed raises osmolality, and every fault changes the batch.
- **Statistics**: the unfolding layout is correct; leave-one-out limits give fewer false alarms than naive limits (averaged over several training sets); a broken correlation shows up in SPE and the contribution points to the right variable; the alarm rule needs consecutive days.
- **Chemometrics**: SNV gives zero-mean, unit-variance rows; the derivative removes constant offsets; pure spectra peak at the right bands; the soft sensor predicts held-out batches with R² > 0.8.
- **End-to-end** pipeline, plus a check that the optimum is interior (not at the edge of the grid).

---

## 7. What I would do next

1. **Parameter estimation**: fit the kinetic parameters to real (published) CHO data with least squares, and quantify parameter uncertainty.
2. **Model predictive control**: use the soft-sensor glucose and the twin to feed continuously to a glucose setpoint instead of daily boluses.
3. **Product quality**: add a glycosylation (a CQA) model, so the optimiser trades titer against quality.
4. **Batch-wise PLS**: predict final titer from the first few days of data, to decide early whether a batch is worth continuing.
