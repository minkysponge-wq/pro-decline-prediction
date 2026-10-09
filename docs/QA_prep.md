# Viva / Q&A preparation

Short, honest answers to the questions the panel is most likely to ask. All numbers come from
`results/metrics.csv` and `results/ablation.csv`.

## About the problem and data

**Why didn't you use the real dataset?**
It's Stanford's private EHR (STARR) and requires institutional access. We built a generator that
produces *raw event tables* (not a ready feature matrix), so every step of the paper's pipeline
runs for real. We say this openly everywhere. The absolute AUROCs describe our simulator; what we
reproduce and test is the *method*.

**How does the simulator create signal?**
- A hidden "frailty" variable per patient (worse with age) shifts labs (low albumin and
  haemoglobin, high CRP), vitals, the probability of diagnoses and medications, and how often the
  patient visits hospital.
- The change in GPH depends on:
  - frailty
  - cancer type (pancreatic is worse, breast milder)
  - regression to the mean of the baseline score
  - an age > 70 × low-albumin interaction
  - noise (SD 5.5)

**Why two labels?**
- *Drop* is relative to the patient's own baseline: did they get worse?
- *Threshold* is absolute: are they in poor health during treatment?

They answer different clinical questions, and the paper uses both.

## Method

**Why AUROC and not accuracy?**
Only ~19–23% of patients are positive, so a model that always says "no" gets ~80% accuracy.
AUROC measures how well the model *ranks* patients by risk, independent of any threshold, which is
what a clinician needs to prioritise. We also report AUPRC, which is more sensitive to the rare
class.

**What is the exponential weighting?**
A lab from yesterday matters more than one from 5 months ago. Each event gets weight
`exp(−t/365)` (t = days before chemo), and we take the weighted mean. A value from 1 year ago
counts e⁻¹ ≈ 37% as much as one from today.

**Why KNN imputation vs binning?**
- *KNN*: fills a missing lab with the average of the 5 most similar patients. It keeps continuous
  information.
- *Binned*: tertiles plus a "missing" bin. It treats "not measured" as informative (sicker
  patients get tested more).

In our data neither clearly wins: KNN was better in 10 of 18 comparisons, with overlapping CIs.

**How did you prevent data leakage?** *(our strongest point)*
1. *Temporal*: features use only events on or before the first chemo date. The simulator
   deliberately plants post-chemo labs and diagnoses that correlate with the outcome. Our test
   deletes all of them and checks the feature matrix is **bit-identical**. We confirmed the test
   fails if the date filter is broken.
2. *Train/test*: the 2.5% filter, imputer, binning cut-points, one-hot categories and scaler are
   inside the sklearn `Pipeline`, so they're re-fitted on each training fold and never see
   validation or test rows.
3. The test set is used **once**, after all tuning.

**Isn't using the baseline PROMIS score cheating?**
No. It's collected *before* the first dose, so it's genuinely available at prediction time; the
paper's clinicians confirmed the same. Leakage would be using the *on-treatment* survey, and that
is only ever used for the label.

**What did you do differently from the paper?**
- The paper early-stopped LightGBM and the MLP **on the test set**, which is a leak.
  - We tune LightGBM's number of rounds in CV.
  - The MLP early-stops on 15% of each *training* fold.
- The paper applied the 2.5% filter to the whole cohort; we fit it per training fold.
- Grids are slightly smaller for compute (500 trees, not 1,000), all documented in `registry.py`.

**How does the ensemble work and why does it help?**
- It averages the predicted probabilities of the tuned LASSO, Ridge, Elastic Net, Random Forest,
  LightGBM and SVM.
- The SVM gets probabilities via Platt scaling (`CalibratedClassifierCV`).
- Different models make different errors, and averaging cancels some of them, which reduces
  variance.
- It had the **highest CV AUROC in all 4 settings** (0.753 / 0.749 / 0.821 / 0.828).

## Results

**On Drop, LASSO (0.730) beat the ensemble (0.692–0.716) on test. Why pick the ensemble?**
- We select by CV, not by test, because choosing by test would be overfitting to the test set.
- With only 47 positive test patients, the 95% CI is about ±0.09, so all these differences are
  statistical noise.
- Honest summary: the ensemble and the penalised linear models are equivalent here, and the
  ensemble is the more robust choice by CV.

**Why are the MLP and KNN worst?**
Both are poor fits for this data: about 1,000 training patients, several hundred mostly sparse
binary features, and a weak signal.
- The MLP has far more parameters than patients, and its early stopping uses even less data.
- KNN suffers from the curse of dimensionality: distances lose meaning in high dimensions. Its best
  k was 100, the grid maximum, meaning it wanted to average as many neighbours as possible.

The paper found the same.

**Ridge picked C = 10⁻⁸. Isn't that weird?**
With extreme shrinkage, every coefficient becomes proportional to that feature's covariance with
the label, like a naive "sum of risk factors" score. AUROC only depends on the ranking, so this
simple, very stable scorer does well. It's a sign the data is noisy relative to the number of
features.

**What does the ablation show?**
- Removing baseline PROMIS lowers Drop AUROC from 0.692 to 0.559.
- Removing demographics and PROMIS together lowers it to 0.551. The paper saw 0.770 → 0.599.
- Removing any other single group changes AUROC by less than the CI, because frailty shows up
  redundantly across labs, diagnoses, meds and so on.

**Why does a *high* baseline predict a Drop?**
Regression to the mean, plus room to fall: someone at 65 can drop 10 points, while someone at 30
is already near the floor. That's why the sign flips between the Drop and Threshold labels.

**Some features in the LASSO top-15 (like `dx_197`) have no real meaning. Problem?**
They're filler codes with no true signal. LASSO picked them up by chance, which is overfitting
noise in a small, high-dimensional dataset. This is honest evidence for using the ensemble and
stronger regularisation, and we point it out ourselves.

**Is the model useful?**
Ranked by predicted risk, the top fifth of test patients had a **42%** observed Drop rate, versus
about **10%** in the bottom two fifths. That's roughly 4× enrichment, which is enough to target
supportive care.

## Engineering

**How is the code organised?** `data/` (simulate, cohort) → `features/` (build, preprocess) →
`models/` (registry, training) → `evaluation/` (metrics, plots). There's a CLI with independent
stages, and every constant from the paper lives once in `config.py`.

**How do you know it's correct?** 19 pytest tests, covering:
- leakage
- label definitions
- inclusion windows
- determinism
- that LASSO really is L1-sparse across sklearn versions (sklearn 1.8 changed that API)
- that every model and the ensemble fit end-to-end, including with a feature group removed

**Limitations?**
- The data is simulated.
- There is one train/test split.
- The ensemble's CV score is mildly optimistic, because its members were tuned on the same folds;
  nested CV would fix this.
- We didn't check calibration.
