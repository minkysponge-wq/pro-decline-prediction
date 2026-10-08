"""Synthetic EHR generator that mirrors the structure of Stanford STARR data.

The reference paper uses de-identified Stanford EHR data that is not publicly
available. To replicate the *method* faithfully we simulate raw, event-level
EHR tables (not a ready-made feature matrix), so that every step of the paper's
pipeline - cohort selection, look-back windows, exponential time weighting,
prevalence filtering, lab imputation/binning - is genuinely exercised.

Generative story
----------------
* Each patient has a latent frailty ``f ~ N(0, 1)`` (worse with age).
* Observable EHR signals (labs, vitals, diagnoses, medications, procedures,
  healthcare utilisation, baseline PROMIS scores) are noisy functions of ``f``.
* The change in PROMIS Global Physical Health (GPH) during chemotherapy depends
  on ``f``, cancer type, regression-to-the-mean of the baseline score, a
  non-linear age x albumin interaction and substantial unexplained noise.
* Events are also generated *after* the chemotherapy start date, and some of
  them are deliberately correlated with the outcome. A correct pipeline must
  ignore them; ``tests/test_leakage.py`` verifies this.
* Some patients violate inclusion criteria (diagnosed before 2013, missing
  baseline or on-treatment survey) so that cohort construction is non-trivial.

All randomness flows from a single ``numpy.random.Generator`` -> fully
reproducible given the seed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CohortConfig
from . import vocab

log = logging.getLogger(__name__)

WINDOW_START, WINDOW_END = -400, 180      # event days relative to chemo start
DX_HISTORY_DAYS = 3650                    # diagnoses can be up to 10 y old
GPH_MIN, GPH_MAX = 16.2, 67.7             # PROMIS GPH T-score range


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


@dataclass
class RawEHR:
    """Container for the raw event-level tables (one DataFrame per table)."""
    patients: pd.DataFrame
    encounters: pd.DataFrame
    vitals: pd.DataFrame
    labs: pd.DataFrame
    diagnoses: pd.DataFrame
    procedures: pd.DataFrame
    medications: pd.DataFrame
    promis: pd.DataFrame

    TABLES = ("patients", "encounters", "vitals", "labs", "diagnoses",
              "procedures", "medications", "promis")

    def save(self, directory) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name in self.TABLES:
            getattr(self, name).to_csv(directory / f"{name}.csv", index=False)

    @classmethod
    def load(cls, directory) -> "RawEHR":
        tables = {}
        for name in cls.TABLES:
            path = directory / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found - run `python -m prodecline generate-data` first.")
            date_cols = [c for c in ("date", "birth_date", "cancer_dx_date")
                         if c in pd.read_csv(path, nrows=0).columns]
            tables[name] = pd.read_csv(path, parse_dates=date_cols)
        return cls(**tables)


class EHRSimulator:
    def __init__(self, cfg: CohortConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------ helpers
    def _event_days(self, n: int, lo: int = WINDOW_START, hi: int = WINDOW_END) -> np.ndarray:
        return self.rng.integers(lo, hi + 1, size=n)

    @staticmethod
    def _to_dates(index_dates: pd.Series, pids: np.ndarray, offsets: np.ndarray) -> pd.Series:
        base = index_dates.loc[pids].to_numpy()
        return pd.Series(base + offsets.astype("timedelta64[D]"))

    # --------------------------------------------------------------- patients
    def _patients(self, n: int) -> pd.DataFrame:
        r = self.rng
        age = np.clip(r.normal(61, 12.5, n), 18, 95).round(1)
        frailty = 0.025 * (age - 61) + r.normal(0, 0.95, n)
        cancer = r.choice(len(vocab.CANCER_TYPES), size=n, p=vocab.CANCER_TYPE_PROBS)
        cancer_effect = np.array([vocab.CANCER_TYPES[i][1] for i in cancer])

        start, end = np.datetime64("2012-01-01"), np.datetime64("2019-10-01")
        dx_offset = r.integers(0, int((end - start).astype(int)), n)
        cancer_dx_date = start + dx_offset.astype("timedelta64[D]")
        chemo_start = cancer_dx_date + r.integers(10, 91, n).astype("timedelta64[D]")
        birth_date = chemo_start - np.round(age * 365.25).astype("timedelta64[D]")

        insurance = np.where(
            age >= 65,
            r.choice(vocab.INSURANCE, n, p=[0.25, 0.68, 0.03, 0.01, 0.03]),
            r.choice(vocab.INSURANCE, n, p=[0.74, 0.04, 0.15, 0.03, 0.04]))

        return pd.DataFrame({
            "patient_id": [f"P{i:05d}" for i in range(1, n + 1)],
            "birth_date": birth_date,
            "sex": r.choice(["female", "male"], n, p=[0.55, 0.45]),
            "race": r.choice(vocab.RACES, n, p=vocab.RACE_PROBS),
            "ethnicity": r.choice(vocab.ETHNICITIES, n, p=vocab.ETHNICITY_PROBS),
            "insurance": insurance,
            "language": r.choice(vocab.LANGUAGES, n, p=vocab.LANGUAGE_PROBS),
            "marital_status": r.choice(vocab.MARITAL, n, p=vocab.MARITAL_PROBS),
            "cancer_type": [vocab.CANCER_TYPES[i][0] for i in cancer],
            "cancer_dx_date": cancer_dx_date,
            # latent quantities kept in a private column prefix; never featurised
            "_age": age, "_frailty": frailty, "_cancer_effect": cancer_effect,
            "_chemo_start": chemo_start,
        })

    # ---------------------------------------------------------------- outcome
    def _outcome(self, p: pd.DataFrame, albumin_level: np.ndarray) -> pd.DataFrame:
        r, n = self.rng, len(p)
        f = p["_frailty"].to_numpy()
        baseline = np.clip(50 - 4.2 * f + r.normal(0, 7.0, n), GPH_MIN, GPH_MAX)
        low_alb_old = ((albumin_level < 3.5) & (p["_age"].to_numpy() > 70)).astype(float)
        delta = (-3.0
                 - 3.5 * f
                 - 2.0 * p["_cancer_effect"].to_numpy()
                 - 0.60 * (baseline - 50)          # regression to the mean
                 - 3.0 * low_alb_old               # non-linear interaction
                 + r.normal(0, 5.5, n))            # unexplained variation
        on_treatment = np.clip(baseline + delta, GPH_MIN, GPH_MAX)
        gmh = np.clip(50 - 2.5 * f + 0.35 * (baseline - 50) + r.normal(0, 7, n),
                      21.2, 67.6)
        return pd.DataFrame({"_gph_baseline": baseline.round(1),
                             "_gph_on": on_treatment.round(1),
                             "_gmh_baseline": gmh.round(1)})

    # ------------------------------------------------------------------- labs
    def _lab_levels(self, p: pd.DataFrame) -> list[tuple]:
        """Per-lab spec plus each patient's true (latent) level for that lab."""
        r, n = self.rng, len(p)
        f = p["_frailty"].to_numpy()
        specs = [(name, mu, sd, load, r.uniform(0.75, 0.97))
                 for name, mu, sd, load in vocab.INFORMATIVE_LABS]
        n_filler = self.cfg.n_labs - len(specs)
        if n_filler > len(vocab.FILLER_LABS):
            raise ValueError(f"n_labs too large; max {len(specs) + len(vocab.FILLER_LABS)}")
        for name in vocab.FILLER_LABS[:n_filler]:
            mu = r.uniform(1, 100)
            # include some rarely-ordered labs so the 2.5% filter has work to do
            specs.append((name, mu, mu * r.uniform(0.05, 0.3), 0.0,
                          float(np.exp(r.uniform(np.log(0.005), np.log(0.9))))))
        out = []
        for name, mu, sd, load, p_ordered in specs:
            level = mu + sd * (load * f + np.sqrt(1 - load ** 2) * r.normal(0, 1, n))
            out.append((name, sd, load, p_ordered, level))
        return out

    def _labs(self, p: pd.DataFrame, lab_levels: list[tuple],
              delta: np.ndarray) -> pd.DataFrame:
        r, n = self.rng, len(p)
        f = p["_frailty"].to_numpy()
        rows = []
        for name, sd, load, p_ordered, level in lab_levels:
            # sicker patients are tested more often -> informative missingness
            ordered = r.random(n) < np.clip(p_ordered + 0.05 * f, 0, 1)
            counts = np.where(ordered, 1 + r.poisson(2.0 + 0.6 * np.clip(f, 0, None)), 0)
            pid_idx = np.repeat(np.arange(n), counts)
            days = self._event_days(len(pid_idx))
            values = level[pid_idx] + r.normal(0, 0.25 * sd, len(pid_idx))
            # LEAKAGE TRAP: post-index values move with the (future) outcome
            post = days > 0
            values[post] += np.sign(load) * sd * (-delta[pid_idx][post] / 8.0)
            rows.append(pd.DataFrame({"pid_idx": pid_idx, "day": days,
                                      "lab": name, "value": values.round(3)}))
        return pd.concat(rows, ignore_index=True)

    # ----------------------------------------------------------------- vitals
    def _vitals(self, p: pd.DataFrame) -> pd.DataFrame:
        r, n = self.rng, len(p)
        f = p["_frailty"].to_numpy()
        counts = 3 + r.poisson(4 + np.clip(f, 0, None), n)
        pid_idx = np.repeat(np.arange(n), counts)
        days = self._event_days(len(pid_idx))
        rows = []
        for name, mu, sd, load, (lo, hi) in vocab.VITALS:
            level = mu + sd * (load * f + np.sqrt(1 - load ** 2) * r.normal(0, 1, n))
            values = np.clip(level[pid_idx] + r.normal(0, 0.3 * sd, len(pid_idx)), lo, hi)
            # ~0.7% charting errors: zeros, unit slips (x10) -> must be cleaned
            err = r.random(len(values)) < 0.007
            values[err] = np.where(r.random(err.sum()) < 0.5, 0.0, values[err] * 10)
            rows.append(pd.DataFrame({"pid_idx": pid_idx, "day": days,
                                      "vital": name, "value": values.round(2)}))
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------ binary code tables (dx/med/proc)
    def _code_events(self, n_codes: int, informative, prefix: str, f: np.ndarray,
                     delta: np.ndarray, lo: int, repeat_rate: float):
        r, n = self.rng, len(f)
        specs = list(informative)
        for i in range(n_codes - len(specs)):
            prev = float(np.exp(r.uniform(np.log(0.003), np.log(0.30))))
            specs.append((f"{prefix}_{i + 1:03d}", prev, 0.0))
        rows = []
        for name, prevalence, load in specs:
            prob = _sigmoid(_logit(prevalence) + load * f)
            has = r.random(n) < prob
            counts = np.where(has, 1 + r.poisson(repeat_rate, n), 0)
            pid_idx = np.repeat(np.arange(n), counts)
            days = r.integers(lo, 1, size=len(pid_idx))          # pre-index history
            rows.append(pd.DataFrame({"pid_idx": pid_idx, "day": days, "code": name}))
            if load > 0:
                # LEAKAGE TRAP: new post-index codes for patients who decline
                p_post = _sigmoid(_logit(prevalence / 3) - delta / 6.0)
                post = np.flatnonzero(r.random(n) < p_post)
                rows.append(pd.DataFrame({"pid_idx": post,
                                          "day": r.integers(1, WINDOW_END + 1, len(post)),
                                          "code": name}))
        return pd.concat(rows, ignore_index=True)

    def _encounters(self, f: np.ndarray) -> pd.DataFrame:
        r, n = self.rng, len(f)
        base_rate = {"emergency": (-1.2, 0.6), "inpatient": (-1.6, 0.7),
                     "outpatient": (1.8, 0.2), "telehealth": (0.3, 0.1),
                     "urgent_care": (-1.8, 0.3)}
        rows = []
        for enc_type, (b0, b1) in base_rate.items():
            counts = r.poisson(np.exp(b0 + b1 * f) * (WINDOW_END - WINDOW_START) / 180)
            pid_idx = np.repeat(np.arange(n), counts)
            rows.append(pd.DataFrame({"pid_idx": pid_idx,
                                      "day": self._event_days(len(pid_idx)),
                                      "encounter_type": enc_type}))
        return pd.concat(rows, ignore_index=True)

    def _chemo_admins(self, n: int) -> pd.DataFrame:
        cycles = 1 + self.rng.integers(3, 8, n)
        pid_idx = np.repeat(np.arange(n), cycles)
        days = np.concatenate([np.arange(c) * 21 for c in cycles])
        return pd.DataFrame({"pid_idx": pid_idx, "day": days,
                             "code": vocab.CHEMO_MED_CLASS})

    # ----------------------------------------------------------------- PROMIS
    def _promis(self, out: pd.DataFrame) -> pd.DataFrame:
        r, n = self.rng, len(out)
        rows = []
        has_baseline = r.random(n) > 0.07
        has_on = r.random(n) > 0.10
        for i in range(n):
            gph0, gph1, gmh0 = out.loc[i, ["_gph_baseline", "_gph_on", "_gmh_baseline"]]
            if r.random() < 0.40:   # older pre-treatment survey
                rows.append((i, -int(r.integers(91, 366)),
                             np.clip(gph0 + r.normal(0, 4), GPH_MIN, GPH_MAX),
                             np.clip(gmh0 + r.normal(0, 4), 21.2, 67.6)))
            if has_baseline[i]:
                rows.append((i, -int(r.integers(0, 91)), gph0, gmh0))
            if r.random() < 0.15:   # early on-treatment survey (< 30 d, not a label)
                rows.append((i, int(r.integers(1, 30)),
                             np.clip(gph0 + 0.4 * (gph1 - gph0) + r.normal(0, 3),
                                     GPH_MIN, GPH_MAX), gmh0))
            if has_on[i]:
                d = int(r.integers(30, 181))
                rows.append((i, d, gph1, np.clip(gmh0 + r.normal(0, 5), 21.2, 67.6)))
                if r.random() < 0.30:
                    rows.append((i, int(r.integers(d + 1, d + 120)),
                                 np.clip(gph1 + r.normal(0, 4), GPH_MIN, GPH_MAX),
                                 gmh0))
        df = pd.DataFrame(rows, columns=["pid_idx", "day", "gph_tscore", "gmh_tscore"])
        df[["gph_tscore", "gmh_tscore"]] = df[["gph_tscore", "gmh_tscore"]].round(1)
        return df

    # ------------------------------------------------------------------- main
    def generate(self, n_raw: int | None = None) -> RawEHR:
        cfg = self.cfg
        # simulate extra raw patients: ~27% fail inclusion criteria
        n = n_raw or int(round(cfg.n_patients * 1.37))
        p = self._patients(n)
        f = p["_frailty"].to_numpy()

        # latent lab levels -> outcome (uses true albumin) -> observed lab events
        lab_levels = self._lab_levels(p)
        albumin = next(lv[-1] for lv in lab_levels if lv[0] == "albumin")
        out = self._outcome(p, albumin)
        delta = (out["_gph_on"] - out["_gph_baseline"]).to_numpy()
        labs = self._labs(p, lab_levels, delta)

        vitals = self._vitals(p)
        dx = self._code_events(cfg.n_diagnoses, vocab.INFORMATIVE_DX, "dx", f, delta,
                               lo=-DX_HISTORY_DAYS, repeat_rate=1.0)
        procs = self._code_events(cfg.n_procedures, vocab.INFORMATIVE_PROCS, "proc", f,
                                  delta, lo=WINDOW_START, repeat_rate=0.6)
        meds = self._code_events(cfg.n_med_classes, vocab.INFORMATIVE_MEDS, "med_class",
                                 f, delta, lo=WINDOW_START, repeat_rate=1.5)
        meds = pd.concat([meds, self._chemo_admins(n)], ignore_index=True)
        enc = self._encounters(f)
        promis = self._promis(out)

        index_dates = pd.Series(p["_chemo_start"].to_numpy())
        pids = p["patient_id"].to_numpy()

        def finalize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.insert(0, "patient_id", pids[df["pid_idx"].to_numpy()])
            df.insert(1, "date", self._to_dates(index_dates, df["pid_idx"].to_numpy(),
                                                df["day"].to_numpy()).to_numpy())
            return (df.drop(columns=["pid_idx", "day"])
                      .sort_values(["patient_id", "date"]).reset_index(drop=True))

        patients = p.drop(columns=[c for c in p.columns if c.startswith("_")])
        log.info("Simulated %d raw patients", n)
        return RawEHR(
            patients=patients,
            encounters=finalize(enc),
            vitals=finalize(vitals),
            labs=finalize(labs),
            diagnoses=finalize(dx).rename(columns={"code": "dx_code"}),
            procedures=finalize(procs).rename(columns={"code": "proc_code"}),
            medications=finalize(meds).rename(columns={"code": "med_class"}),
            promis=finalize(promis),
        )
