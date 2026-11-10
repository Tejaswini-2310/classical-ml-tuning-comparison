"""
Default vs tuned classical ML models across small tabular datasets.
 
Run:  pip install scikit-learn xgboost pandas numpy matplotlib scipy
      python tuning_vs_default_experiment.py
 
Design (paired): for each dataset, model and repeat r, the SAME outer 5-fold split is used for
  (a) the model with default settings
  (b) the model tuned by RandomizedSearchCV in an inner 3-fold loop (nested CV, no leakage)
Outputs go to ./results/ (CSV tables and PNG figures). Report YOUR numbers from these files.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint, uniform
from sklearn.datasets import load_breast_cancer, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
 
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
 
warnings.filterwarnings("ignore")
 
# ---------------- settings you can change ----------------
N_REPEATS = 5        # outer repeats (increase for tighter estimates if time allows)
OUTER_FOLDS = 5
INNER_FOLDS = 3
N_ITER = 15          # random-search candidates per fit
SEED = 42
OUT = "results"
os.makedirs(OUT, exist_ok=True)
 
 
# ---------------- data ----------------
def from_openml(name, version=1, zero_as_nan=None):
    X, y = fetch_openml(name, version=version, as_frame=True, return_X_y=True)
    if zero_as_nan:
        for c in zero_as_nan:
            X[c] = X[c].replace(0, np.nan)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X.values, LabelEncoder().fit_transform(y)
 
 
def load_datasets():
    data = {}
    d = load_breast_cancer(); data["BreastCancer"] = (d.data, d.target)
    d = load_wine(); data["Wine"] = (d.data, d.target)
    openml_sets = {
        "PimaDiabetes": dict(name="diabetes", version=1,
                             zero_as_nan=["plas", "pres", "skin", "insu", "mass"]),
        "HeartStatlog": dict(name="heart-statlog", version=1),
        "CreditG": dict(name="credit-g", version=1),
    }
    for key, kw in openml_sets.items():
        try:
            data[key] = from_openml(**kw)
        except Exception as e:  # no internet or dataset unavailable
            print(f"Skipping {key}: {e}")
    return data
 
 
# ---------------- models ----------------
def make_models(seed):
    pre = lambda: [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    models = {
        "LogReg": (Pipeline(pre() + [("clf", LogisticRegression(max_iter=2000))]),
                   {"clf__C": loguniform(1e-3, 1e2)}),
        "SVM": (Pipeline(pre() + [("clf", SVC(probability=True, random_state=seed))]),
                {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e0)}),
        "KNN": (Pipeline(pre() + [("clf", KNeighborsClassifier())]),
                {"clf__n_neighbors": randint(1, 31), "clf__weights": ["uniform", "distance"]}),
        "RandomForest": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier(random_state=seed, n_jobs=1))]),
                         {"clf__n_estimators": randint(100, 500), "clf__max_depth": [None, 3, 5, 8, 12],
                          "clf__min_samples_leaf": randint(1, 10), "clf__max_features": ["sqrt", "log2", 0.5]}),
    }
    if HAS_XGB:
        models["XGBoost"] = (Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("clf", XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1))]),
                             {"clf__n_estimators": randint(100, 500), "clf__max_depth": randint(2, 8),
                              "clf__learning_rate": loguniform(1e-2, 3e-1), "clf__subsample": uniform(0.6, 0.4),
                              "clf__colsample_bytree": uniform(0.6, 0.4)})
    return models
 
 
# ---------------- experiment ----------------
def run():
    datasets = load_datasets()
    rows = []
    for dname, (X, y) in datasets.items():
        multiclass = len(np.unique(y)) > 2
        scoring = {"accuracy": "accuracy", "f1": "f1_macro",
                   "auc": "roc_auc_ovr" if multiclass else "roc_auc"}
        print(f"\n=== {dname}: {X.shape[0]} rows, {X.shape[1]} features, {len(np.unique(y))} classes")
        for r in range(N_REPEATS):
            outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + r)
            for mname, (pipe, space) in make_models(SEED + r).items():
                # (a) default settings
                res = cross_validate(pipe, X, y, cv=outer, scoring=scoring, n_jobs=-1)
                # (b) tuned with nested CV
                inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED + 100 + r)
                search = RandomizedSearchCV(pipe, space, n_iter=N_ITER, cv=inner, scoring="f1_macro",
                                            random_state=SEED + r, n_jobs=1)
                res_t = cross_validate(search, X, y, cv=outer, scoring=scoring, n_jobs=-1)
                for setting, rr in (("default", res), ("tuned", res_t)):
                    for k in range(OUTER_FOLDS):
                        rows.append(dict(dataset=dname, model=mname, setting=setting, repeat=r, fold=k,
                                         accuracy=rr["test_accuracy"][k], f1=rr["test_f1"][k],
                                         auc=rr["test_auc"][k]))
            print(f"  repeat {r + 1}/{N_REPEATS} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(f"{OUT}/raw_scores.csv", index=False)
    return raw
 
 
# ---------------- analysis ----------------
def analyse(raw):
    # summary table: mean +- std over all outer folds and repeats
    g = raw.groupby(["dataset", "model", "setting"])
    mean = g[["accuracy", "f1", "auc"]].mean()
    std = g[["accuracy", "f1", "auc"]].std()
    summary = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    summary.to_csv(f"{OUT}/summary_mean_std.csv")
 
    # paired gain per fold (tuned - default) for RQ1 and RQ3
    piv = raw.pivot_table(index=["dataset", "model", "repeat", "fold"], columns="setting", values="f1").reset_index()
    piv["gain"] = piv["tuned"] - piv["default"]
    gain = piv.groupby(["dataset", "model"])["gain"].agg(mean_gain="mean", std_gain="std",
                                                          frac_tuned_better=lambda s: (s > 0).mean()).round(4)
    gain.to_csv(f"{OUT}/f1_gain_tuned_minus_default.csv")
 
    # ranks by mean F1 within each dataset (1 = best)
    m = raw.groupby(["dataset", "model", "setting"])["f1"].mean().unstack("setting")
    rank_def = m["default"].groupby("dataset").rank(ascending=False)
    rank_tun = m["tuned"].groupby("dataset").rank(ascending=False)
    ranks = pd.DataFrame({"rank_default": rank_def, "rank_tuned": rank_tun})
    ranks["rank_change"] = ranks["rank_default"] - ranks["rank_tuned"]  # positive = moved up
    ranks.to_csv(f"{OUT}/ranks.csv")
 
    datasets = sorted(raw["dataset"].unique())
    models = list(raw["model"].unique())
 
    # Figure 1: default vs tuned mean F1 per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(models))
    for ax, d in zip(axes, datasets):
        sub = m.loc[d].loc[models]
        ax.bar(x - 0.2, sub["default"], 0.4, label="default")
        ax.bar(x + 0.2, sub["tuned"], 0.4, label="tuned")
        ax.set_title(d); ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(max(0, m.values.min() - 0.05), 1.0)
    axes[0].set_ylabel("Mean macro-F1"); axes[0].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_default_vs_tuned_f1.png", dpi=200); plt.close()
 
    # Figure 2: rank heatmap (default vs tuned), datasets x models
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in zip(axes, ["rank_default", "rank_tuned"], ["Rank, default", "Rank, tuned"]):
        mat = ranks[col].unstack("model").loc[datasets, models]
        im = ax.imshow(mat.values, cmap="viridis_r")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets); ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, int(mat.values[i, j]), ha="center", va="center", color="white")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_rank_heatmap.png", dpi=200); plt.close()
 
    # Figure 3: box plots of per-fold F1 gain (tuned - default)
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4))
    data = [piv[piv["model"] == mo]["gain"].values for mo in models]
    ax.boxplot(data, labels=models); ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("F1 gain (tuned - default)"); ax.set_title("Per-fold gain from tuning, all datasets")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3_gain_boxplot.png", dpi=200); plt.close()
 
    print("\nSaved tables and figures to ./results/")
    print(ranks.round(2))
 
 
if __name__ == "__main__":
    analyse(run())