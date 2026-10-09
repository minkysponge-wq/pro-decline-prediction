"""End-to-end stages. Each stage reads the previous stage's outputs from disk,
so stages can be re-run independently (e.g. re-plot without re-training)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn.base import clone

from .config import FEATURE_GROUPS, LAB_STRATEGIES, OUTCOMES, Config
from .data.cohort import build_cohort
from .data.simulate import EHRSimulator, RawEHR
from .evaluation.metrics import evaluate, predict_score
from .features.build import DEMO_CATEGORICAL, build_features
from .models.registry import ENSEMBLE_MEMBERS, get_model_specs
from .models.training import cv_auroc, ensemble_from, split_patients, tune

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- I/O
def _processed(cfg: Config, name: str):
    return cfg.paths.processed_dir / name


def load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load (features X, cohort with labels, split) produced by ``stage_data``."""
    path = _processed(cfg, "features.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m prodecline data` first.")
    dtypes = {f"demo__{c}": str for c in DEMO_CATEGORICAL}
    X = pd.read_csv(path, index_col="patient_id", dtype=dtypes)
    cohort = pd.read_csv(_processed(cfg, "cohort.csv"), index_col="patient_id")
    split = pd.read_csv(_processed(cfg, "split.csv"), index_col="patient_id")["split"]
    if not (X.index.equals(cohort.index) and X.index.equals(split.index)):
        raise ValueError("features / cohort / split are not aligned - regenerate data.")
    return X, cohort, split


def _model_path(cfg: Config, outcome: str, lab: str, model: str):
    return cfg.paths.models_dir / f"{outcome}__{lab}__{model}.joblib"


# ------------------------------------------------------------------ stages
def stage_data(cfg: Config) -> None:
    """Simulate raw EHR -> cohort + labels -> features -> train/test split."""
    paths = cfg.paths.ensure()
    ehr = EHRSimulator(cfg.cohort).generate()
    ehr.save(paths.raw_dir)
    ehr = RawEHR.load(paths.raw_dir)            # build features from what is on disk

    cohort, flow = build_cohort(ehr, cfg.cohort, cfg.labels)
    X = build_features(ehr, cohort, cfg.features)
    split = split_patients(cohort, cfg)

    cohort.to_csv(_processed(cfg, "cohort.csv"))
    flow.to_csv(_processed(cfg, "cohort_flow.csv"), index=False)
    X.to_csv(_processed(cfg, "features.csv"))
    split.to_csv(_processed(cfg, "split.csv"))
    log.info("Cohort flow:\n%s", flow.to_string(index=False))
    log.info("Split: %s", split.value_counts().to_dict())


def _fingerprint(cfg: Config, fast: bool) -> str:
    """Identifies the exact data + settings a checkpoint was trained on."""
    h = hashlib.sha256()
    for name in ("features.csv", "cohort.csv", "split.csv"):
        h.update(_processed(cfg, name).read_bytes())
    train = dataclasses.replace(cfg.train, n_jobs=0)    # parallelism doesn't change results
    h.update(repr((fast, train, cfg.features,
                   [(s.name, s.grid) for s in get_model_specs(cfg.train.seed, fast)])).encode())
    return h.hexdigest()[:16]


class _Checkpoints:
    """One small JSON + CSV per (outcome, lab strategy, model) so an interrupted
    run resumes where it stopped. Checkpoints from different data/settings are
    ignored automatically (fingerprint mismatch)."""

    def __init__(self, cfg: Config, fingerprint: str):
        self.dir = cfg.paths.results_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fp = fingerprint
        self.cfg = cfg

    def _paths(self, key: str):
        return self.dir / f"{key}.json", self.dir / f"{key}__pred.csv"

    def load(self, key: str, outcome: str, lab: str, model: str):
        meta_p, pred_p = self._paths(key)
        model_p = _model_path(self.cfg, outcome, lab, model)
        if not (meta_p.exists() and pred_p.exists() and model_p.exists()):
            return None
        meta = json.loads(meta_p.read_text())
        if meta.get("fingerprint") != self.fp:
            return None
        return meta, pd.read_csv(pred_p), joblib.load(model_p)

    def save(self, key: str, row: dict, params: dict, pred: pd.DataFrame) -> None:
        meta_p, pred_p = self._paths(key)
        pred.to_csv(pred_p, index=False)
        meta_p.write_text(json.dumps({"fingerprint": self.fp, "row": row,
                                      "best_params": params}, indent=2))


def stage_train(cfg: Config, fast: bool = False, fresh: bool = False) -> pd.DataFrame:
    paths = cfg.paths.ensure()
    X, cohort, split = load_dataset(cfg)
    tr, te = split.eq("train").to_numpy(), split.eq("test").to_numpy()
    memory = Memory(paths.cache_dir / "pipelines", verbose=0)
    cv_dir = paths.results_dir / "cv_results"
    cv_dir.mkdir(exist_ok=True)
    ckpt = _Checkpoints(cfg, _fingerprint(cfg, fast))   # fresh=True ignores, then overwrites

    rows, preds, best_params = [], [], {}
    for outcome in OUTCOMES:
        y = cohort[outcome]
        y_tr, y_te = y[tr], y[te]
        for lab in LAB_STRATEGIES:
            log.info("== outcome=%s  labs=%s  (train n=%d, test n=%d, prevalence %.1f%%)",
                     outcome, lab, tr.sum(), te.sum(), 100 * y_tr.mean())
            best = {}
            for spec in get_model_specs(cfg.train.seed, fast=fast):
                key = f"{outcome}__{lab}__{spec.name}"
                hit = None if fresh else ckpt.load(key, outcome, lab, spec.name)
                if hit is not None:
                    meta, pred, est = hit
                    log.info("  %-14s resumed from checkpoint (CV AUROC %.3f)",
                             spec.name, meta["row"]["cv_auroc_mean"])
                    row, params = meta["row"], meta["best_params"]
                else:
                    search, secs = tune(spec, X[tr], y_tr, lab, cfg, memory)
                    est = search.best_estimator_
                    est.set_params(memory=None)
                    score, is_prob = predict_score(est, X[te])
                    row = {"outcome": outcome, "lab_strategy": lab, "model": spec.name,
                           "cv_auroc_mean": float(search.best_score_),
                           "cv_auroc_std": float(search.cv_results_["std_test_score"]
                                                 [search.best_index_]),
                           "fit_seconds": secs,
                           **evaluate(y_te, score, cfg.train.n_bootstrap,
                                      cfg.train.seed, is_prob)}
                    params = {k.replace("clf__", ""): _jsonable(v)
                              for k, v in search.best_params_.items()}
                    pred = _pred_frame(outcome, lab, spec.name, X.index[te], y_te, score)
                    pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(
                        cv_dir / f"{key}.csv", index=False)
                    joblib.dump(est, _model_path(cfg, outcome, lab, spec.name))
                    ckpt.save(key, row, params, pred)
                best[spec.name] = est
                rows.append(row)
                preds.append(pred)
                best_params[f"{outcome}/{lab}/{spec.name}"] = params

            # ---- soft-voting ensemble of tuned members
            key = f"{outcome}__{lab}__ensemble"
            hit = None if fresh else ckpt.load(key, outcome, lab, "ensemble")
            if hit is not None:
                meta, pred, _ = hit
                row = meta["row"]
                log.info("  %-14s resumed from checkpoint", "ensemble")
            else:
                t0 = time.perf_counter()
                ens = ensemble_from(best)
                cv_mean, cv_std = cv_auroc(ens, X[tr], y_tr, cfg)
                ens.fit(X[tr], y_tr)
                score, is_prob = predict_score(ens, X[te])
                row = {"outcome": outcome, "lab_strategy": lab, "model": "ensemble",
                       "cv_auroc_mean": cv_mean, "cv_auroc_std": cv_std,
                       "fit_seconds": time.perf_counter() - t0,
                       **evaluate(y_te, score, cfg.train.n_bootstrap, cfg.train.seed,
                                  is_prob)}
                pred = _pred_frame(outcome, lab, "ensemble", X.index[te], y_te, score)
                joblib.dump(ens, _model_path(cfg, outcome, lab, "ensemble"))
                ckpt.save(key, row, {m: best_params[f"{outcome}/{lab}/{m}"]
                                     for m in ENSEMBLE_MEMBERS}, pred)
            rows.append(row)
            preds.append(pred)
            log.info("  %-14s CV AUROC %.3f | TEST AUROC %.3f", "ensemble",
                     row["cv_auroc_mean"], row["test_auroc"])

    metrics = pd.DataFrame(rows)
    metrics.to_csv(paths.results_dir / "metrics.csv", index=False)
    pd.concat(preds).to_csv(paths.results_dir / "test_predictions.csv", index=False)
    (paths.results_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    log.info("Saved metrics to %s", paths.results_dir / "metrics.csv")
    return metrics


def stage_ablation(cfg: Config, lab: str = "knn") -> pd.DataFrame:
    """Re-fit the tuned ensemble with feature groups removed (paper Table 3)."""
    paths = cfg.paths.ensure()
    X, cohort, split = load_dataset(cfg)
    tr, te = split.eq("train").to_numpy(), split.eq("test").to_numpy()

    conditions = {"all features": []}
    for g in FEATURE_GROUPS:
        conditions[f"- {g}"] = [g]
    conditions["- demo & promis"] = ["demo", "promis"]

    def run(outcome: str, name: str, groups: list[str]) -> dict:
        warnings.filterwarnings("ignore")
        ens = clone(joblib.load(_model_path(cfg, outcome, lab, "ensemble")))
        cols = [c for c in X.columns if c.split("__", 1)[0] not in groups]
        y = cohort[outcome]
        ens.fit(X.loc[tr, cols], y[tr])
        score, _ = predict_score(ens, X.loc[te, cols])
        m = evaluate(y[te], score, cfg.train.n_bootstrap, cfg.train.seed, True)
        return {"outcome": outcome, "lab_strategy": lab, "condition": name,
                "n_raw_features": len(cols), **m}

    jobs = [(o, n, g) for o in OUTCOMES for n, g in conditions.items()]
    rows = Parallel(n_jobs=cfg.train.n_jobs)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(paths.results_dir / "ablation.csv", index=False)
    log.info("Ablation:\n%s", df[["outcome", "condition", "test_auroc",
                                  "test_auroc_ci_low", "test_auroc_ci_high"]]
             .round(3).to_string(index=False))
    return df


# ----------------------------------------------------------------- helpers
def _pred_frame(outcome, lab, model, ids, y, score) -> pd.DataFrame:
    return pd.DataFrame({"outcome": outcome, "lab_strategy": lab, "model": model,
                         "patient_id": ids, "y_true": np.asarray(y), "score": score})


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v
