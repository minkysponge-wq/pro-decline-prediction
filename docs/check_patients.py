"""Manually check the classifier patient by patient on the held-out test set.

    python docs/check_patients.py                  # Drop outcome, first 15 test patients
    python docs/check_patients.py --outcome threshold --n 25
"""
import argparse
import warnings

import joblib
import pandas as pd

from prodecline.config import CONFIG as cfg
from prodecline.pipeline import load_dataset

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--outcome", default="drop", choices=["drop", "threshold"])
ap.add_argument("--n", type=int, default=15)
args = ap.parse_args()

X, cohort, split = load_dataset(cfg)
test = split.eq("test").to_numpy()
model = joblib.load(cfg.paths.models_dir / f"{args.outcome}__knn__ensemble.joblib")

# cut-off chosen on TRAINING data by `python -m prodecline improve`
m = pd.read_csv(cfg.paths.results_dir / "improved" / "metrics.csv")
cutoff = m.query("outcome == @args.outcome and variant == 'paper ensemble' "
                 "and cutoff_rule == 'tuned on train OOF'")["threshold"].item()

prob = model.predict_proba(X[test])[:, 1]
df = pd.DataFrame({
    "baseline_GPH": X.loc[test, "promis__gph_baseline"].round(1),
    "age": X.loc[test, "demo__age"].round(0),
    "cancer": X.loc[test, "demo__cancer_type"],
    "P(yes)": prob.round(3),
    "predicted": (prob >= cutoff).astype(int),
    "actual": cohort.loc[test, args.outcome],
})
df["correct?"] = (df["predicted"] == df["actual"]).map({True: "yes", False: "NO"})

print(f"Outcome: {args.outcome} | cut-off {cutoff} (chosen on training data) | "
      "1 = will decline, 0 = won't\n")
print(df.head(args.n).to_string())
print(f"\nAll {len(df)} test patients: {(df['correct?'] == 'yes').sum()} correct "
      f"-> accuracy {(df['correct?'] == 'yes').mean():.1%}")
print(pd.crosstab(df["actual"], df["predicted"], rownames=["actual"], colnames=["predicted"]))
