# Project tracker: ML Mini-Project #8 (Team E-17)

**Deadline:** Sat 10 Oct 2026, 11:59 PM (GitHub repo + PDF write-up). Review and demo window ends Fri 9 Oct.
**Paused:** Fri 9 Oct 2026, 01:07 IST

## Done
- [x] Read the guidelines, the problem statement and the reference paper (Ostberg & Peterson, CS229 2020)
- [x] Synthetic EHR generator, because the paper's Stanford STARR data is private (`data/simulate.py`)
- [x] Cohort selection and drop/threshold labels: 1,254 patients (the paper had 1,252)
- [x] Featurisation: 180-day look-back, exp(-t/365) weighting, impossible-vital removal
- [x] Leakage-safe preprocessing: 2.5% filter, KNN-imputed or binned labs, all fitted inside CV
- [x] 8 model families with paper grids, soft-voting ensemble, ablation, figures, CLI
- [x] 19 pytest tests passing; the leakage test was confirmed to catch a broken date filter
- [x] Fast run (`python -m prodecline all --fast`, ~5 min) verified end-to-end
- [x] README drafted (Results section still a placeholder)
- [x] Demo notebook generator written (`notebooks/build_demo_notebook.py`)

## In progress
- [ ] Full run (`python -m prodecline all`, 10-fold CV, full grids) was started in the background at 18:57 container time
  - Finished so far: drop/knn (ensemble test AUROC 0.692) and drop/binned (0.716)
  - Remaining: threshold/knn, threshold/binned, then ablation and figures
  - Log: scratchpad `full_run.log`
  - If the session restarted and `results/metrics.csv` is missing, re-run `python -m prodecline all`

## To do (in order)
1. [ ] Check that the full run completed: `python -m prodecline summary`
2. [ ] Look at the figures in `results/figures/`
3. [ ] Fill in the README Results section with real numbers
4. [ ] Build and execute `notebooks/demo.ipynb`
5. [ ] Write the 1–2 page PDF write-up (`docs/`): problem, dataset, approach, implementation, conclusions
6. [ ] Slide deck, ~8 slides: problem → data → features → models → results → ablation → limitations → demo
7. [ ] Create a **private** GitHub repo, share it with faculty and TA, and have both members commit (individual contribution is graded)
8. [ ] Prepare for Q&A: AUROC vs accuracy, KNN imputation, baseline-score leakage, why the ensemble helps, overfitting

## Decisions made
- **Synthetic data**, stated openly in the README, write-up and slides
- **Ensemble members:** best LASSO, Ridge, Elastic Net, Random Forest, LightGBM and SVM, soft vote
- **Fixed a flaw in the paper:** no early stopping on the test set
- **Reduced grids** for Random Forest, Elastic Net and the MLP for compute reasons (documented in `registry.py`)
