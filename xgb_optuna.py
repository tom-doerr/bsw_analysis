#!/usr/bin/env python3
"""XGBoost HPO with Optuna for BSW prediction."""

import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor

from xgb_enhanced import load_all, SEED

DATA = Path("data")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_est", 100, 1500),
        "max_depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("lr", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("sub", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("col", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("mcw", 1, 50),
        "reg_alpha": trial.suggest_float("alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("lam", 1e-8, 10, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5, log=True),
    }


def make_objective(X, y, cv):
    def objective(trial):
        p = suggest_params(trial)
        xgb = XGBRegressor(**p, random_state=SEED,
                            n_jobs=-1, tree_method="hist",
                            device="cuda")
        scores = cross_val_score(xgb, X, y, cv=cv, scoring="r2")
        return scores.mean()
    return objective


def run_study(name, X, y, cv, n_trials=100):
    study = optuna.create_study(
        study_name=name, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(make_objective(X, y, cv),
                    n_trials=n_trials, show_progress_bar=True)
    return study


# Parties to tune (BSW + major parties for comparison)
PARTIES = ["BSW", "FDP", "SPD", "AfD", "CDU", "CSU",
           "GRÜNE", "Die Linke", "FREIE WÄHLER"]


def eval_best(X, y, cv, best_params):
    """Evaluate best params with cross_val_predict."""
    xgb = XGBRegressor(**best_params, random_state=SEED,
                        n_jobs=-1, tree_method="hist",
                        device="cuda")
    yp = cross_val_predict(xgb, X, y, cv=cv)
    rho, _ = spearmanr(y, yp)
    return {
        "R2": r2_score(y, yp),
        "Spearman": rho,
        "MAE_pp": mean_absolute_error(y, yp),
        "Mean_share": y.mean(),
    }


PARAM_MAP = {
    "n_est": "n_estimators", "depth": "max_depth",
    "lr": "learning_rate", "sub": "subsample",
    "col": "colsample_bytree", "mcw": "min_child_weight",
    "alpha": "reg_alpha", "lam": "reg_lambda",
    "gamma": "gamma",
}


def tune_party(party, X, y, cv, n_trials, baseline):
    print(f"\n{'='*60}")
    print(f"Optimizing {party} ({n_trials} trials)...")
    print(f"{'='*60}")
    study = run_study(party, X, y, cv, n_trials)
    best = study.best_params
    xgb_p = {PARAM_MAP[k]: v for k, v in best.items()}
    print(f"  Best CV R²: {study.best_value:.4f}")
    print(f"  Params: {xgb_p}")
    metrics = eval_best(X, y, cv, xgb_p)
    bl_r2 = baseline.loc[party, "R2"] if party in baseline.index else None
    print(f"  Tuned R²={metrics['R2']:.4f}  MAE={metrics['MAE_pp']:.4f}pp")
    if bl_r2 is not None:
        print(f"  Default R²={bl_r2:.4f}  Δ={metrics['R2']-bl_r2:+.4f}")
    return {**metrics, **xgb_p}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-trials", type=int, default=100)
    ap.add_argument("-p", "--parties", nargs="*", default=None)
    args = ap.parse_args()

    X_df, z_map, meta = load_all()
    X = X_df.values.astype(np.float64)
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    parties = args.parties or PARTIES
    parties = [p for p in parties if p in z_map]
    baseline = pd.read_csv(DATA / "xgb_enhanced_metrics.csv",
                            index_col=0)
    results = {}
    for party in parties:
        y = z_map[party]
        results[party] = tune_party(
            party, X, y, cv, args.n_trials, baseline)
    print_summary(results, baseline)


def print_summary(results, baseline):
    print(f"\n{'='*60}")
    print("Summary: Default XGB vs Optuna-tuned XGB")
    print(f"{'='*60}")
    hdr = f"{'Party':>15s}  {'Def':>7s}  {'Tuned':>7s}  {'Δ':>7s}"
    print(hdr)
    for p in sorted(results, key=lambda x: -results[x]["R2"]):
        r = results[p]
        bl = baseline.loc[p, "R2"] if p in baseline.index else float("nan")
        d = r["R2"] - bl
        print(f"{p:>15s}  {bl:7.4f}  {r['R2']:7.4f}  {d:+7.4f}")
    out = DATA / "xgb_optuna_results.csv"
    pd.DataFrame(results).T.to_csv(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
