"""Writes the Results section of README.md from results/*.csv (no hand-copied numbers).

    python docs/update_readme.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LABELS = {"lasso": "LASSO", "ridge": "Ridge", "elasticnet": "Elastic Net",
          "random_forest": "Random Forest", "lightgbm": "LightGBM", "mlp": "MLP",
          "svm": "SVM", "knn": "KNN", "ensemble": "**Ensemble**"}
COLS = [("drop", "knn"), ("drop", "binned"), ("threshold", "knn"), ("threshold", "binned")]


def results_markdown() -> str:
    m = pd.read_csv(ROOT / "results" / "metrics.csv")
    abl = pd.read_csv(ROOT / "results" / "ablation.csv")
    n_test = int(pd.read_csv(ROOT / "data" / "processed" / "split.csv")["split"]
                 .eq("test").sum())

    def cell(o, lab, model):
        r = m[(m.outcome == o) & (m.lab_strategy == lab) & (m.model == model)].iloc[0]
        best = m[(m.outcome == o) & (m.lab_strategy == lab)]["test_auroc"].max()
        s = f"{r.test_auroc:.3f} <sub>({r.test_auroc_ci_low:.2f}–{r.test_auroc_ci_high:.2f})</sub>"
        return f"**{s}**" if r.test_auroc == best else s

    lines = [
        f"Test-set AUROC (n = {n_test} held-out patients) with 95% bootstrap CI; best per "
        "column in bold. Full metrics (CV mean ± SD, AUPRC, Brier) are in "
        "`results/metrics.csv`; chosen hyper-parameters are in `results/best_params.json`.",
        "",
        "| Model | Drop, KNN labs | Drop, binned labs | Threshold, KNN labs | Threshold, binned labs |",
        "|---|---|---|---|---|",
    ]
    for model in LABELS:
        lines.append(f"| {LABELS[model]} | " + " | ".join(cell(o, lab, model) for o, lab in COLS)
                     + " |")
    lines += ["", "Paper (ensemble, KNN labs): Drop 0.771, Threshold 0.752.", "",
              "**Ablation** (ensemble, KNN labs, test AUROC):", "",
              "| Condition | Drop | Threshold |", "|---|---|---|"]
    piv = abl.pivot(index="condition", columns="outcome", values="test_auroc")
    for cond in abl[abl.outcome == "drop"]["condition"]:
        lines.append(f"| {cond} | {piv.loc[cond, 'drop']:.3f} | {piv.loc[cond, 'threshold']:.3f} |")
    lines += ["",
              "![Model comparison](results/figures/model_comparison.png)",
              "![ROC curves](results/figures/roc_curves.png)",
              "![Ablation](results/figures/ablation.png)",
              "![Top LASSO features](results/figures/top_features.png)"]
    return "\n".join(lines)


def main() -> None:
    text = README.read_text()
    block = f"{START}\n{results_markdown()}\n{END}"
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text,
                      flags=re.S)
    elif "RESULTS_PLACEHOLDER" in text:
        text = text.replace("RESULTS_PLACEHOLDER", block)
    else:
        raise SystemExit("README has no results markers")
    README.write_text(text)
    print("README results updated")


if __name__ == "__main__":
    main()
