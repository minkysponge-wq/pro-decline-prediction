"""Publication figures for the write-up and slides (static PNG, 200 dpi)."""
from __future__ import annotations

import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from ..config import OUTCOMES, Config  # noqa: E402

log = logging.getLogger(__name__)

# Validated categorical palette (fixed slot order) + recessive ink
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

MODEL_LABELS = {"lasso": "LASSO", "ridge": "Ridge", "elasticnet": "Elastic Net",
                "random_forest": "Random Forest", "lightgbm": "LightGBM", "mlp": "MLP",
                "svm": "SVM", "knn": "KNN", "ensemble": "Ensemble"}
OUTCOME_LABELS = {"drop": "Drop (decline >= 10 pts)", "threshold": "Threshold (GPH <= 40)"}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def model_comparison(metrics: pd.DataFrame, out) -> None:
    """Test AUROC (95% bootstrap CI) per model, KNN- vs binned-lab features."""
    order = [m for m in MODEL_LABELS if m in metrics["model"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, outcome in zip(axes, OUTCOMES):
        d = metrics[metrics["outcome"] == outcome]
        x = np.arange(len(order))
        for k, (lab, color) in enumerate([("knn", BLUE), ("binned", ORANGE)]):
            r = d[d["lab_strategy"] == lab].set_index("model").loc[order]
            err = [r["test_auroc"] - r["test_auroc_ci_low"],
                   r["test_auroc_ci_high"] - r["test_auroc"]]
            ax.bar(x + (k - 0.5) * 0.38, r["test_auroc"], width=0.36, color=color,
                   label="KNN-imputed labs" if lab == "knn" else "Binned labs",
                   yerr=err, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        ax.axhline(0.5, color=INK_2, lw=1, ls="--")
        ax.text(-0.55, 0.505, "chance", color=INK_2, ha="left", va="bottom", fontsize=8)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in order], rotation=35, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test AUROC (95% CI)")
    axes[1].legend(loc="upper right")
    fig.savefig(out)
    plt.close(fig)


def roc_curves(preds: pd.DataFrame, metrics: pd.DataFrame, lab: str, out) -> None:
    """ROC curves of the ensemble vs its strongest single competitors."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, outcome in zip(axes, OUTCOMES):
        m = metrics[(metrics["outcome"] == outcome) & (metrics["lab_strategy"] == lab)]
        singles = (m[m["model"] != "ensemble"].sort_values("test_auroc", ascending=False)
                   ["model"].head(3).tolist())
        for model, color, lw in zip(["ensemble"] + singles, [BLUE, ORANGE, AQUA, VIOLET],
                                    [2.6, 1.6, 1.6, 1.6]):
            p = preds[(preds["outcome"] == outcome) & (preds["lab_strategy"] == lab)
                      & (preds["model"] == model)]
            fpr, tpr, _ = roc_curve(p["y_true"], p["score"])
            auc = m.set_index("model").loc[model, "test_auroc"]
            ax.plot(fpr, tpr, color=color, lw=lw, label=f"{MODEL_LABELS[model]}  {auc:.3f}")
        ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               xlim=(0, 1), ylim=(0, 1.01), title=OUTCOME_LABELS[outcome])
        ax.set_aspect("equal")
        ax.legend(loc="lower right", title="Test AUROC", title_fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def ablation(abl: pd.DataFrame, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, outcome in zip(axes, OUTCOMES):
        d = abl[abl["outcome"] == outcome].reset_index(drop=True)
        full = d.loc[d["condition"] == "all features", "test_auroc"].item()
        y = np.arange(len(d))[::-1]
        colors = [INK_2 if c == "all features" else (RED if c == "- demo & promis" else BLUE)
                  for c in d["condition"]]
        ax.barh(y, d["test_auroc"], color=colors, height=0.6,
                xerr=[d["test_auroc"] - d["test_auroc_ci_low"],
                      d["test_auroc_ci_high"] - d["test_auroc"]],
                error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        ax.axvline(full, color=INK_2, lw=1, ls=":")
        ax.set_yticks(y, d["condition"])
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("Ensemble test AUROC (95% CI)")
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="y", visible=False)
    fig.savefig(out)
    plt.close(fig)


def top_features(cfg: Config, lab: str, out, k: int = 15) -> pd.DataFrame:
    """Largest-magnitude LASSO coefficients (standardised features)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    tables = []
    for ax, outcome in zip(axes, OUTCOMES):
        pipe = joblib.load(cfg.paths.models_dir / f"{outcome}__{lab}__lasso.joblib")
        names = pipe.named_steps["prep"].get_feature_names_out()
        coef = pipe.named_steps["clf"].coef_.ravel()
        s = pd.Series(coef, index=names)
        s = s[s != 0].sort_values(key=np.abs, ascending=False).head(k)[::-1]
        tables.append(s.rename(outcome))
        ax.barh(range(len(s)), s.values, color=[RED if v > 0 else BLUE for v in s.values],
                height=0.65)
        ax.set_yticks(range(len(s)), [n.replace("__", ": ") for n in s.index], fontsize=8.5)
        ax.axvline(0, color=INK_2, lw=1)
        ax.set_xlabel("LASSO coefficient (red = higher risk)")
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return pd.concat(tables, axis=1)


def make_all(cfg: Config, lab: str = "knn") -> None:
    _style()
    fig_dir = cfg.paths.ensure().figures_dir
    res = cfg.paths.results_dir
    metrics = pd.read_csv(res / "metrics.csv")
    preds = pd.read_csv(res / "test_predictions.csv")
    model_comparison(metrics, fig_dir / "model_comparison.png")
    roc_curves(preds, metrics, lab, fig_dir / "roc_curves.png")
    top_features(cfg, lab, fig_dir / "top_features.png")
    if (res / "ablation.csv").exists():
        ablation(pd.read_csv(res / "ablation.csv"), fig_dir / "ablation.png")
    log.info("Figures written to %s", fig_dir)
