"""Generates notebooks/demo.ipynb (kept as code so the notebook stays reviewable).

    python notebooks/build_demo_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/demo.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Live demo - predicting PROMIS physical-health decline during chemotherapy\n"
       "Replication of *Ostberg & Peterson, CS229 Spring 2020*.\n\n"
       "Run `python -m prodecline all` (or `all --fast`) once before this notebook; it loads "
       "the saved cohort, features and trained models."),
    code("import warnings; warnings.filterwarnings('ignore')\n"
         "import joblib, numpy as np, pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "from IPython.display import Image, display\n"
         "from sklearn.metrics import roc_auc_score\n"
         "from prodecline.config import CONFIG as cfg\n"
         "from prodecline.pipeline import load_dataset\n"
         "pd.set_option('display.width', 140)\n"
         "X, cohort, split = load_dataset(cfg)\n"
         "tr, te = split.eq('train').to_numpy(), split.eq('test').to_numpy()\n"
         "print(f'{len(cohort)} patients x {X.shape[1]} raw features | train {tr.sum()} / test {te.sum()}')"),
    md("## 1. Cohort construction (inclusion criteria)"),
    code("pd.read_csv(cfg.paths.processed_dir / 'cohort_flow.csv')"),
    md("## 2. Outcomes\n*Drop*: GPH falls by >= 10 points (1 SD). "
       "*Threshold*: on-treatment GPH <= 40."),
    code("change = cohort.gph_on_treatment - cohort.gph_baseline\n"
         "fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))\n"
         "ax[0].hist(change, bins=40, color='#2a78d6'); ax[0].axvline(-10, color='#e34948', ls='--')\n"
         "ax[0].set(title='Change in GPH (on - baseline)', xlabel='T-score points')\n"
         "ax[1].hist(cohort.gph_on_treatment, bins=40, color='#2a78d6'); ax[1].axvline(40, color='#e34948', ls='--')\n"
         "ax[1].set(title='On-treatment GPH', xlabel='T-score')\n"
         "plt.tight_layout(); plt.show()\n"
         "cohort[['drop', 'threshold']].mean().rename('prevalence').to_frame()"),
    md("## 3. Feature groups (raw, before train-fold preprocessing)"),
    code("X.columns.str.split('__').str[0].value_counts().rename('n_features').to_frame()"),
    md("## 4. Results (test set, 95% bootstrap CI)"),
    code("m = pd.read_csv(cfg.paths.results_dir / 'metrics.csv')\n"
         "m.pivot_table(index='model', columns=['outcome', 'lab_strategy'], values='test_auroc')"
         ".round(3).sort_values(('drop', 'knn'), ascending=False)"),
    code("for f in ['model_comparison', 'roc_curves', 'ablation', 'top_features']:\n"
         "    display(Image(filename=str(cfg.paths.figures_dir / f'{f}.png'), width=900))"),
    md("## 5. Live scoring with the trained ensemble\n"
       "Rank held-out patients by predicted risk of a *drop*; compare with what actually happened."),
    code("ens = joblib.load(cfg.paths.models_dir / 'drop__knn__ensemble.joblib')\n"
         "risk = pd.Series(ens.predict_proba(X[te])[:, 1], index=X.index[te], name='predicted_risk')\n"
         "print('Test AUROC:', round(roc_auc_score(cohort.loc[risk.index, 'drop'], risk), 3))\n"
         "view = pd.concat([risk, X.loc[risk.index, ['demo__age', 'demo__cancer_type', "
         "'promis__gph_baseline', 'lab__albumin']], cohort.loc[risk.index, ['gph_on_treatment', 'drop']]], axis=1)\n"
         "view.sort_values('predicted_risk', ascending=False).head(10).round(2)"),
    code("# Risk stratification: observed drop rate by predicted-risk quintile\n"
         "q = pd.qcut(risk, 5, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)'])\n"
         "cohort.loc[risk.index, 'drop'].groupby(q, observed=True).agg(['mean', 'size'])"
         ".rename(columns={'mean': 'observed drop rate', 'size': 'n'}).round(3)"),
    md("## 6. What-if: how does one patient's risk change?"),
    code("pid = view.sort_values('predicted_risk').index[len(view) // 2]   # a median-risk patient\n"
         "base = X.loc[[pid]]\n"
         "rows = {'as recorded': base}\n"
         "for label, col, val in [('albumin 2.8 g/dL', 'lab__albumin', 2.8),\n"
         "                        ('baseline GPH 62', 'promis__gph_baseline', 62.0),\n"
         "                        ('baseline GPH 35', 'promis__gph_baseline', 35.0)]:\n"
         "    r = base.copy(); r[col] = val; rows[label] = r\n"
         "pd.Series({k: ens.predict_proba(v)[0, 1] for k, v in rows.items()}, name='P(drop)').round(3)"),
    md("## 7. Retrain a model live (seconds)\nFull pipeline (prevalence filter -> KNN imputation "
       "-> scaling -> LASSO) fitted on the training split only."),
    code("import json\n"
         "from prodecline.models.registry import logistic\n"
         "from prodecline.models.training import make_pipeline\n"
         "best = json.loads((cfg.paths.results_dir / 'best_params.json').read_text())['drop/knn/lasso']\n"
         "pipe = make_pipeline(logistic('lasso', cfg.train.seed), 'knn', cfg).set_params(clf__C=best['C'])\n"
         "%time pipe.fit(X[tr], cohort['drop'][tr])\n"
         "coef = pipe.named_steps['clf'].coef_.ravel()\n"
         "print(f\"best C={best['C']}: {(coef != 0).sum()} of {coef.size} features kept | test AUROC \"\n"
         "      f\"{roc_auc_score(cohort['drop'][te], pipe.predict_proba(X[te])[:, 1]):.3f}\")"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).with_name("demo.ipynb")
nbf.write(nb, out)
print("wrote", out)
