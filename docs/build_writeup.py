"""Builds docs/writeup.pdf (2 pages) directly from the results files, so every
number in the write-up is exactly what the pipeline produced.

    python docs/build_writeup.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
OUT = ROOT / "docs" / "writeup.pdf"

LABELS = {"lasso": "LASSO", "ridge": "Ridge", "elasticnet": "Elastic Net",
          "random_forest": "Random Forest", "lightgbm": "LightGBM", "mlp": "MLP",
          "svm": "SVM", "knn": "KNN", "ensemble": "Ensemble (soft vote)"}

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=8.8,
                      leading=11.2, spaceAfter=3, alignment=4)
H = ParagraphStyle("h", parent=BODY, fontName="Helvetica-Bold", fontSize=10, leading=12,
                   spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1d3557"),
                   alignment=0)
TITLE = ParagraphStyle("t", parent=BODY, fontName="Helvetica-Bold", fontSize=13.5,
                       leading=16, alignment=TA_CENTER, spaceAfter=2)
SUB = ParagraphStyle("s", parent=BODY, fontSize=8.8, alignment=TA_CENTER,
                     textColor=colors.HexColor("#52514e"), spaceAfter=6)
CAP = ParagraphStyle("c", parent=BODY, fontSize=7.8, leading=9.5,
                     textColor=colors.HexColor("#52514e"), alignment=0)


def f3(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    m = pd.read_csv(RES / "metrics.csv")
    abl = pd.read_csv(RES / "ablation.csv")
    flow = pd.read_csv(PROC / "cohort_flow.csv")
    cohort = pd.read_csv(PROC / "cohort.csv")
    X_cols = pd.read_csv(PROC / "features.csv", nrows=0).columns.drop("patient_id")
    n_test = int(pd.read_csv(PROC / "split.csv")["split"].eq("test").sum())

    n = len(cohort)
    n_raw = int(flow["n_patients"].iloc[0])
    prev_drop, prev_thr = cohort["drop"].mean() * 100, cohort["threshold"].mean() * 100
    groups = pd.Series([c.split("__")[0] for c in X_cols]).value_counts()

    def get(o, lab, model, col="test_auroc"):
        return m[(m.outcome == o) & (m.lab_strategy == lab) & (m.model == model)][col].item()

    def ci(o, lab, model):
        r = m[(m.outcome == o) & (m.lab_strategy == lab) & (m.model == model)].iloc[0]
        return f"{r.test_auroc:.3f} (95% CI {r.test_auroc_ci_low:.3f}-{r.test_auroc_ci_high:.3f})"

    def abl_auc(o, cond):
        return abl[(abl.outcome == o) & (abl.condition == cond)]["test_auroc"].item()

    # data-driven statements
    singles = m[m.model != "ensemble"]
    best_single = {o: singles[singles.outcome == o].sort_values("test_auroc").iloc[-1]
                   for o in ("drop", "threshold")}
    knn_vs_bin = (m[m.lab_strategy == "knn"].set_index(["outcome", "model"])["test_auroc"]
                  - m[m.lab_strategy == "binned"].set_index(["outcome", "model"])["test_auroc"])
    knn_wins = int((knn_vs_bin > 0).sum())
    worst = (singles.groupby("model")["test_auroc"].mean().sort_values().index[:2])
    top_cv = m.loc[m.groupby(["outcome", "lab_strategy"])["cv_auroc_mean"].idxmax()]
    ens_cv_wins = int((top_cv["model"] == "ensemble").sum())
    cv_summary = ", ".join(
        f"{o} / {lab} labs {r:.3f}" for o, lab, r in
        m[m.model == "ensemble"][["outcome", "lab_strategy", "cv_auroc_mean"]].itertuples(
            index=False))
    full_drop = abl_auc("drop", "all features")
    preds = pd.read_csv(RES / "test_predictions.csv")
    n_pos_drop = int(preds.query("model == 'ensemble' and lab_strategy == 'knn' "
                                 "and outcome == 'drop'")["y_true"].sum())
    # guard the prose claim that single-group removals (other than PROMIS) stay inside the CI
    for o in ("drop", "threshold"):
        a = abl[abl.outcome == o].set_index("condition")
        half_ci = (a.loc["all features", "test_auroc_ci_high"]
                   - a.loc["all features", "test_auroc_ci_low"]) / 2
        others = a.drop(["all features", "- promis", "- demo & promis"])["test_auroc"]
        assert ((others - a.loc["all features", "test_auroc"]).abs() < half_ci).all(), o

    story = [
        Paragraph("Predicting a Decline in Patient-Reported Outcomes for Cancer Patients "
                  "on Chemotherapy", TITLE),
        Paragraph("UE24CS352A Machine Learning Mini-Project, Problem 8 &nbsp;|&nbsp; Team E-17: "
                  "Sampreeth Jaladi (PES1UG24AM246), Sharvesh K (PES1UG24AM260) &nbsp;|&nbsp; "
                  "Replication of Ostberg &amp; Peterson, CS229 Spring 2020", SUB),

        Paragraph("1. Problem statement", H),
        Paragraph(
            "Chemotherapy often worsens patients' physical well-being, captured by the PROMIS "
            "Global Physical Health (GPH) T-score (population mean 50, SD 10). The goal is to "
            "predict, <b>at the first chemotherapy dose and using only data available then</b>, "
            "which patients will report poor physical health during treatment, so clinicians "
            "can target supportive care. Two binary outcomes from the paper are modelled: "
            "<b>Drop</b>, on-treatment GPH falls by at least 10 points (1 SD) from baseline; and "
            "<b>Threshold</b>, on-treatment GPH is 40 or lower (1 SD below the mean). Models "
            "are compared by AUROC.", BODY),

        Paragraph("2. Dataset", H),
        Paragraph(
            "The paper uses de-identified Stanford EHR data (STARR), which is not publicly "
            "available. We therefore wrote a <b>synthetic EHR generator</b> that produces "
            "<i>raw event-level</i> tables (patients, encounters, vitals, labs, diagnoses, "
            "procedures, medication classes, PROMIS surveys) with the same structure, so the "
            "paper's entire pipeline runs unchanged. A latent frailty variable drives both the "
            "observable EHR signals and the change in GPH, together with cancer type, "
            "regression to the mean of the baseline score, a non-linear age by albumin "
            "interaction and substantial unexplained noise. The generator also deliberately "
            "plants <i>post-chemotherapy</i> events that correlate with the outcome, to test "
            "for leakage, and charting errors in the vitals. "
            f"Applying the paper's inclusion criteria to {n_raw:,} simulated patients "
            f"(cancer diagnosis from 2013, received chemotherapy, a baseline survey within 90 days "
            f"before and an on-treatment survey 30-180 days after the first dose) gives "
            f"<b>{n:,} patients</b> (paper: 1,252). Outcome prevalence is Drop {prev_drop:.1f}% "
            f"and Threshold {prev_thr:.1f}%. Absolute AUROCs therefore describe the simulator; "
            "the contribution is a faithful, tested re-implementation of the method.", BODY),

        Paragraph("3. Approach", H),
        Paragraph(
            "<b>Features</b> (only events on or before the index date): vitals with "
            "physiologically impossible values removed; labs; procedures; diagnoses (full history, "
            "binary); medication classes; healthcare utilisation; demographics; and 6 baseline "
            "PROMIS features. Time-varying data from the last 180 days is summarised with the "
            f"paper's exponential weight exp(-t/365). This gives {len(X_cols)} raw features "
            f"({groups.get('dx', 0)} diagnosis, {groups.get('proc', 0)} procedure, "
            f"{groups.get('med', 0)} medication, {groups.get('lab', 0)} lab, "
            f"{groups.get('vital', 0)} vital, {groups.get('util', 0)} utilisation, "
            f"{groups.get('demo', 0)} demographic, {groups.get('promis', 0)} PROMIS). "
            "<b>Preprocessing</b>: a 2.5% prevalence filter on labs, diagnoses and procedures, "
            "then labs handled by one of two strategies: (a) KNN imputation, or (b) tertile bins "
            "plus a missing indicator. "
            "<b>Models</b>: logistic regression (LASSO, Ridge, Elastic Net), random forest, "
            "LightGBM, a 3-layer MLP, an SVM (4 kernels) and KNN, plus a <b>soft-voting "
            "ensemble</b> that averages the probabilities of the tuned LASSO, Ridge, Elastic Net, "
            "random forest, LightGBM and Platt-scaled SVM. "
            "<b>Protocol</b>: a stratified 80/20 patient split; grid search with stratified "
            "10-fold CV on the training set, scored by AUROC; each best model refitted on the "
            "training set and evaluated <i>once</i> on the test set, with 1,000-sample "
            "bootstrap 95% CIs.", BODY),

        Paragraph("4. Implementation", H),
        Paragraph(
            "We wrote a Python package (<font face='Courier'>prodecline</font>) using "
            "scikit-learn and LightGBM with a staged CLI (data, train, ablate, figures). "
            "Training is checkpointed so an interrupted run resumes. <b>Every fitted "
            "preprocessing step</b> (prevalence filter, imputation, binning cut-points, one-hot "
            "categories, scaling) <b>sits inside the model pipeline</b>, so it is re-fitted on "
            "each training fold and never sees validation or test data. We also corrected two "
            "issues in the original study: it early-stopped LightGBM and the MLP on the "
            "<i>test</i> set, and it applied the prevalence filter to the whole cohort. "
            "Nineteen unit tests cover the pipeline; the key one deletes every post-index event "
            "and checks that the feature matrix is bit-identical, which proves that no future "
            "information is used.", BODY),
    ]

    # ---------------- results table
    order = list(LABELS)
    header = ["Model", "Drop, KNN labs", "Drop, binned labs", "Threshold, KNN labs",
              "Threshold, binned labs"]
    cols = [("drop", "knn"), ("drop", "binned"), ("threshold", "knn"), ("threshold", "binned")]
    data = [header]
    best_in_col = {c: m[(m.outcome == c[0]) & (m.lab_strategy == c[1])]["test_auroc"].max()
                   for c in cols}
    for model in order:
        row = [LABELS[model]]
        for c in cols:
            v = get(*c, model)
            row.append(f"<b>{v:.3f}</b>" if abs(v - best_in_col[c]) < 1e-12 else f"{v:.3f}")
        data.append([Paragraph(x, ParagraphStyle("tc", parent=BODY, fontSize=8, leading=9.5,
                                                 alignment=0 if i == 0 else 1))
                     for i, x in enumerate(row)])
    data[0] = [Paragraph(f"<b>{h}</b>", ParagraphStyle("th", parent=BODY, fontSize=8,
                                                        leading=9.5, alignment=0 if i == 0 else 1))
               for i, h in enumerate(header)]
    table = Table(data, colWidths=[3.6 * cm] + [3.3 * cm] * 4, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#eef3fb")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))

    story += [
        Paragraph("5. Results", H),
        KeepTogether([table, Spacer(1, 2), Paragraph(
            f"Table 1. Test-set AUROC (n = {n_test} held-out patients); best per column in bold. Mean CV AUROCs and "
            "bootstrap CIs are in <font face='Courier'>results/metrics.csv</font>.", CAP)]),
        Spacer(1, 4),
        Paragraph(
            f"<b>Model selection.</b> Ranked by cross-validated AUROC (the selection criterion), "
            f"the ensemble was the top model in {ens_cv_wins} of 4 settings "
            f"({cv_summary}). On the held-out test set it reached {ci('drop', 'knn', 'ensemble')} "
            f"for Drop and {ci('threshold', 'knn', 'ensemble')} for Threshold with KNN labs "
            "(paper: 0.771 and 0.752). The best single test scores were "
            f"{LABELS[best_single['drop'].model]} for Drop ({best_single['drop'].test_auroc:.3f}, "
            f"{best_single['drop'].lab_strategy} labs) and {LABELS[best_single['threshold'].model]} "
            f"for Threshold ({best_single['threshold'].test_auroc:.3f}). Every gap between the "
            "ensemble and the best single model lies well inside the 95% CIs (about +/-0.09 for "
            f"Drop with only {n_pos_drop} positive test patients), so"
            "  the honest reading is that the "
            "ensemble and the penalised linear models perform equivalently, and the ensemble is "
            "the more robust choice by CV. As in the paper, "
            f"{LABELS[worst[0]]} and {LABELS[worst[1]]} performed worst; both struggle with "
            "several hundred mostly sparse features and ~1,000 training patients. KNN-imputed labs beat "
            f"binned labs in {knn_wins} of {len(knn_vs_bin)} model and outcome pairs with "
            "overlapping CIs, so unlike the paper we find no clear winner between the two lab "
            "strategies.", BODY),
        Paragraph(
            f"<b>Ablation</b> (ensemble, KNN labs): with all features, Drop AUROC is "
            f"{f3(full_drop)}. Removing the baseline PROMIS features lowers it to "
            f"{f3(abl_auc('drop', '- promis'))}, and removing demographics and PROMIS together "
            f"lowers it to {f3(abl_auc('drop', '- demo & promis'))} (paper: 0.770 to 0.599). "
            "Removing any other single group changes AUROC by less than the width of the CI. For "
            f"Threshold the equivalent figures are {f3(abl_auc('threshold', 'all features'))} and "
            f"{f3(abl_auc('threshold', '- demo & promis'))}. Baseline patient-reported "
            "status is the dominant predictor, which is clinically legitimate because it is "
            "routinely collected before treatment.", BODY),
        Spacer(1, 3),
        KeepTogether([Image(str(RES / "figures" / "ablation.png"), width=17.2 * cm,
                            height=17.2 * cm * 4.4 / 11 * 0.92),
                      Paragraph("Figure 1. Ensemble test AUROC with each feature group removed "
                                "(error bars: 95% bootstrap CI).", CAP)]),

        KeepTogether([Image(str(RES / "figures" / "roc_curves.png"), width=13.5 * cm,
                            height=13.5 * cm * 4.4 / 10 * 0.93),
                      Paragraph("Figure 2. Test ROC curves: the ensemble and the three best "
                                "single models (KNN-imputed labs).", CAP)]),
        Paragraph("6. Conclusions", H),
        Paragraph(
            "We re-implemented the full Ostberg &amp; Peterson pipeline, from cohort selection "
            "through time-weighted EHR featurisation, prevalence filtering, two lab strategies, "
            "eight tuned model families, a soft-voting ensemble and ablation, on a synthetic "
            "EHR built to mirror STARR. The paper's qualitative findings were reproduced: the "
            "ensemble is the most robust choice by cross-validation, with penalised linear models "
            "a close equal; KNN and the MLP underperform; and baseline "
            "PROMIS scores carry most of the signal, especially for Drop. Our protocol is "
            "stricter than the original (no test-set early stopping, preprocessing fitted "
            "inside CV, a regression test for leakage). <b>Limitations</b>: the data is "
            "simulated, so the AUROCs do not indicate clinical performance; there is a single "
            "train/test split; and the ensemble's CV score is mildly optimistic because its "
            "members were tuned on the same folds. <b>Future work</b>: validate on real EHR "
            "data (for example MIMIC-IV with a PRO proxy), use nested CV, check calibration, "
            "and weight the ensemble members by their CV performance.", BODY),
    ]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.3 * cm,
                            title="Predicting PRO decline during chemotherapy",
                            author="Sampreeth Jaladi, Sharvesh K")
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
