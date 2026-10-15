"""Command-line interface.

    python -m prodecline data        # simulate EHR, build cohort + features + split
    python -m prodecline train       # tune 8 model families x 2 outcomes x 2 lab strategies
    python -m prodecline ablate      # feature-group ablation of the ensemble
    python -m prodecline figures     # plots into results/figures
    python -m prodecline all         # everything above, in order
    python -m prodecline summary     # print the results table
    python -m prodecline improve     # extension: burden features + tuned cut-off (accuracy)

Add ``--fast`` for a ~5-minute run (5-fold CV, small grids) - useful for demos.
Training is checkpointed per model: an interrupted ``train`` resumes where it
stopped (use ``--fresh`` to retrain everything).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import time
import warnings

from sklearn.exceptions import ConvergenceWarning

from .config import CONFIG, Config


def _fast(cfg: Config) -> Config:
    return dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, cv_folds=5, n_bootstrap=300))


def summary(cfg: Config) -> None:
    import pandas as pd
    m = pd.read_csv(cfg.paths.results_dir / "metrics.csv")
    m["test AUROC (95% CI)"] = m.apply(
        lambda r: f"{r.test_auroc:.3f} ({r.test_auroc_ci_low:.3f}-{r.test_auroc_ci_high:.3f})",
        axis=1)
    m["CV AUROC"] = m.apply(lambda r: f"{r.cv_auroc_mean:.3f} ± {r.cv_auroc_std:.3f}", axis=1)
    for (o, lab), d in m.groupby(["outcome", "lab_strategy"], sort=False):
        print(f"\n### outcome={o}  labs={lab}")
        print(d[["model", "CV AUROC", "test AUROC (95% CI)", "test_auprc"]]
              .round(3).to_string(index=False))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="prodecline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["data", "train", "ablate", "figures", "all",
                                          "summary", "improve", "improve-summary"])
    parser.add_argument("--fast", action="store_true", help="small grids, 5-fold CV")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore training checkpoints and retrain everything")
    parser.add_argument("--n-jobs", type=int, default=None, help="parallel workers")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    # Non-converging configs at extreme regularisation are expected in a grid
    # search; they simply score poorly. Silence the noise, keep real errors.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    cfg = _fast(CONFIG) if args.fast else CONFIG
    if args.n_jobs is not None:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train,
                                                                 n_jobs=args.n_jobs))

    # imported lazily so `--help` is instant
    from . import pipeline
    from .evaluation import plots

    t0 = time.perf_counter()
    if args.stage in ("data", "all"):
        pipeline.stage_data(cfg)
    if args.stage in ("train", "all"):
        pipeline.stage_train(cfg, fast=args.fast, fresh=args.fresh)
    if args.stage in ("ablate", "all"):
        pipeline.stage_ablation(cfg)
    if args.stage in ("figures", "all"):
        plots.make_all(cfg)
    if args.stage in ("summary", "all"):
        summary(cfg)
    if args.stage == "improve":                 # extension; needs `train` to have run
        from . import improve
        improve.stage_improve(cfg, fast=args.fast)
        improve.summary(cfg)
    if args.stage == "improve-summary":
        from . import improve
        improve.summary(cfg)
    logging.getLogger("prodecline").info("Done in %.1f min", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
