#!/usr/bin/env python3
import argparse
from collections import Counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ridge regression on Bundestagswahl precinct data with plain 10-fold CV "
            "and nested CV (double CV)."
        )
    )
    parser.add_argument(
        "--input-zip",
        default="btw25_wbz.zip",
        help="Path to zip containing precinct csv.",
    )
    parser.add_argument(
        "--csv-name",
        default="btw25_wbz_ergebnisse.csv",
        help="CSV file name inside zip archive.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=10,
        help="Number of outer CV folds.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=5,
        help="Number of inner CV folds for nested CV.",
    )
    parser.add_argument(
        "--min-valid-zweit",
        type=int,
        default=1,
        help="Keep rows with Gültige - Zweitstimmen >= this threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold shuffling.",
    )
    parser.add_argument(
        "--alpha-min-exp",
        type=float,
        default=-3.0,
        help="Minimum exponent for alpha grid (10^exp).",
    )
    parser.add_argument(
        "--alpha-max-exp",
        type=float,
        default=5.0,
        help="Maximum exponent for alpha grid (10^exp).",
    )
    parser.add_argument(
        "--alpha-steps",
        type=int,
        default=25,
        help="Number of alpha values in log-space grid.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional output prefix. If set, writes "
            "<prefix>_plain.csv and <prefix>_nested.csv."
        ),
    )
    return parser.parse_args()


def load_data(zip_path: str, csv_name: str) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh, sep=";", skiprows=4, low_memory=False)
    return df


def extract_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    erst_party_cols = [
        c
        for c in df.columns
        if c.endswith(" - Erststimmen")
        and not c.startswith("Ungültige -")
        and not c.startswith("Gültige -")
    ]
    zweit_party_cols = [
        c
        for c in df.columns
        if c.endswith(" - Zweitstimmen")
        and not c.startswith("Ungültige -")
        and not c.startswith("Gültige -")
    ]
    base_feature_cols = [
        "Wahlberechtigte (A)",
        "Wählende (B)",
        "Ungültige - Erststimmen",
        "Gültige - Erststimmen",
    ]
    feature_cols = base_feature_cols + erst_party_cols
    return feature_cols, zweit_party_cols


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    min_valid_zweit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = df.copy()
    numeric_cols = feature_cols + target_cols + ["Gültige - Zweitstimmen"]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    work = work[work["Gültige - Zweitstimmen"] >= min_valid_zweit].copy()
    X = work[feature_cols].to_numpy(dtype=np.float64)
    y = work[target_cols].to_numpy(dtype=np.float64)
    return X, y, work.index.to_numpy()


def party_names(target_cols: list[str]) -> list[str]:
    return [c.replace(" - Zweitstimmen", "") for c in target_cols]


def per_party_error_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    parties: list[str],
) -> pd.DataFrame:
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2
    total_true = y_true.sum(axis=0)
    n = y_true.shape[0]
    out = pd.DataFrame(
        {
            "party": parties,
            "sum_abs_error": abs_err.sum(axis=0),
            "mean_abs_error": abs_err.mean(axis=0),
            "sum_sq_error": sq_err.sum(axis=0),
            "rmse": np.sqrt(sq_err.mean(axis=0)),
            "sum_true_votes": total_true,
            "abs_error_pct_of_votes": np.where(
                total_true > 0, abs_err.sum(axis=0) / total_true, np.nan
            ),
            "mean_abs_error_per_row": abs_err.sum(axis=0) / n,
        }
    )
    return out.sort_values("sum_abs_error", ascending=False).reset_index(drop=True)


def build_pipe(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=0)),
        ]
    )


def run_plain_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    outer_folds: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        estimator=build_pipe(),
        param_grid={"ridge__alpha": alphas.tolist()},
        scoring="neg_mean_absolute_error",
        cv=outer,
        n_jobs=1,
        refit=True,
    )
    grid.fit(X, y)
    best_alpha = float(grid.best_params_["ridge__alpha"])

    pred = cross_val_predict(
        build_pipe(alpha=best_alpha),
        X,
        y,
        cv=outer,
        n_jobs=1,
    )
    return best_alpha, pred


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[list[float], np.ndarray]:
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=np.float64)
    best_alphas: list[float] = []

    # Inner CV tuning is repeated independently in each outer fold.
    for train_idx, test_idx in outer.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        inner = KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed,
        )
        grid = GridSearchCV(
            estimator=build_pipe(),
            param_grid={"ridge__alpha": alphas.tolist()},
            scoring="neg_mean_absolute_error",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best_alpha = float(grid.best_params_["ridge__alpha"])
        best_alphas.append(best_alpha)
        pred[test_idx] = grid.best_estimator_.predict(X_test)

    return best_alphas, pred


def print_summary(
    title: str,
    alpha_text: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    parties: list[str],
) -> pd.DataFrame:
    mae_all = mean_absolute_error(y_true, y_pred)
    total_abs = np.abs(y_true - y_pred).sum()
    print(f"\n=== {title} ===")
    print(alpha_text)
    print(f"Global MAE: {mae_all:.4f}")
    print(f"Total absolute error (all parties, all rows): {total_abs:.2f}")
    table = per_party_error_table(y_true, y_pred, parties)
    print("\nTop 15 parties by sum_abs_error:")
    print(
        table.head(15).to_string(
            index=False,
            formatters={
                "sum_abs_error": "{:.2f}".format,
                "mean_abs_error": "{:.4f}".format,
                "sum_sq_error": "{:.2f}".format,
                "rmse": "{:.4f}".format,
                "sum_true_votes": "{:.0f}".format,
                "abs_error_pct_of_votes": "{:.6f}".format,
                "mean_abs_error_per_row": "{:.4f}".format,
            },
        )
    )
    return table


def main() -> None:
    args = parse_args()
    alphas = np.logspace(args.alpha_min_exp, args.alpha_max_exp, args.alpha_steps)

    df = load_data(args.input_zip, args.csv_name)
    feature_cols, target_cols = extract_columns(df)
    X, y, _ = prepare_xy(df, feature_cols, target_cols, args.min_valid_zweit)
    parties = party_names(target_cols)

    print("Rows used:", X.shape[0])
    print("Features:", X.shape[1])
    print("Targets (parties):", y.shape[1])
    print("Alpha grid:", ", ".join(f"{a:.5g}" for a in alphas))

    plain_alpha, plain_pred = run_plain_cv(
        X, y, alphas, args.outer_folds, args.seed
    )
    plain_table = print_summary(
        title="Plain 10-fold CV (single tuning on full data)",
        alpha_text=f"Best alpha: {plain_alpha:.8g}",
        y_true=y,
        y_pred=plain_pred,
        parties=parties,
    )

    nested_alphas, nested_pred = run_nested_cv(
        X,
        y,
        alphas,
        args.outer_folds,
        args.inner_folds,
        args.seed,
    )
    alpha_counter = Counter(nested_alphas)
    most_common_alpha, count = alpha_counter.most_common(1)[0]
    nested_table = print_summary(
        title="Nested CV (double CV)",
        alpha_text=(
            f"Outer-fold best alphas: {nested_alphas}\n"
            f"Most common alpha: {most_common_alpha:.8g} ({count}/{len(nested_alphas)} folds)\n"
            f"Median alpha: {np.median(nested_alphas):.8g}"
        ),
        y_true=y,
        y_pred=nested_pred,
        parties=parties,
    )

    if args.output_prefix:
        plain_path = f"{args.output_prefix}_plain.csv"
        nested_path = f"{args.output_prefix}_nested.csv"
        plain_table.to_csv(plain_path, index=False)
        nested_table.to_csv(nested_path, index=False)
        print(f"\nWrote: {plain_path}")
        print(f"Wrote: {nested_path}")


if __name__ == "__main__":
    main()
