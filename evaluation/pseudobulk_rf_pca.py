#!/usr/bin/env python3
"""Pseudobulk Random Forest-PCA benchmark for the GB radiotherapy data.

The model aggregates expression by biological sample and malignant cell type,
predicts PCA-compressed expression deltas, and reconstructs gene-level
pseudobulk predictions. Two evaluations are reported:

1. saved_cell_split: train/test cells from the notebook's saved split. Samples
   occur in both partitions, so this is within-sample interpolation.
2. leave_one_sample_out: each treated biological sample is held out in turn.
   The control reference, target PCA, and Random Forest are fitted within each
   training fold.

The primary comparator is a training-set condition-by-cell-type delta mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder


TARGET_CONDITIONS = ("IR-24h", "IR-1week")
DUMMY_GENES = {"IR-24h", "IR-1week", "IR_24h", "IR_1week"}
TIME_HOURS = {"ctrl": 0.0, "IR-24h": 24.0, "IR-1week": 168.0}
GROUP_COLUMNS = ["Sample", "condition", "cell_type", "Batch", "Sex"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/Users/zhehaowang/PycharmProjects/scgpt_test/"
            "dev_perturb_adamson-Jun28-15-05"
        ),
        help="Directory containing the saved expression arrays and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/zhehaowang/Documents/New project/outputs/pseudobulk_rf_pca"
        ),
        help="Directory for CSV results.",
    )
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.7)
    parser.add_argument("--top-k-de", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def standardize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["condition"] = (
        metadata["condition"].astype(str).str.replace("_", "-", regex=False)
    )
    for column in GROUP_COLUMNS:
        if column not in metadata:
            raise ValueError(f"Required metadata column is missing: {column}")
    return metadata


def load_data(data_dir: Path):
    train_expression = np.load(data_dir / "train_expression.npy")
    test_expression = np.load(data_dir / "test_expression.npy")
    train_metadata = standardize_metadata(
        pd.read_csv(data_dir / "train_metadata.csv", index_col=0)
    )
    test_metadata = standardize_metadata(
        pd.read_csv(data_dir / "test_metadata.csv", index_col=0)
    )

    with (data_dir / "gene_names.json").open() as handle:
        gene_names_full = np.asarray(json.load(handle), dtype=str)

    if train_expression.shape[1] != len(gene_names_full):
        raise ValueError("Expression columns do not match gene_names.json")
    if len(train_metadata) != len(train_expression):
        raise ValueError("Training metadata and expression rows do not match")
    if len(test_metadata) != len(test_expression):
        raise ValueError("Test metadata and expression rows do not match")

    biological_mask = ~np.isin(gene_names_full, list(DUMMY_GENES))
    gene_names = gene_names_full[biological_mask]
    train_expression = train_expression[:, biological_mask].astype(np.float32)
    test_expression = test_expression[:, biological_mask].astype(np.float32)
    return (
        train_expression,
        test_expression,
        train_metadata,
        test_metadata,
        gene_names,
    )


def aggregate_pseudobulk(expression: np.ndarray, metadata: pd.DataFrame):
    metadata = metadata.reset_index(drop=True).copy()
    rows = []
    means = []
    grouped = metadata.groupby(GROUP_COLUMNS, observed=True, sort=True).indices
    for key, positions in grouped.items():
        positions = np.asarray(positions, dtype=int)
        row = dict(zip(GROUP_COLUMNS, key))
        row["n_cells"] = int(len(positions))
        rows.append(row)
        means.append(expression[positions].mean(axis=0))
    return pd.DataFrame(rows), np.asarray(means, dtype=np.float32)


def build_control_references(
    group_metadata: pd.DataFrame, group_expression: np.ndarray
) -> dict[str, np.ndarray]:
    control_mask = group_metadata["condition"].eq("ctrl").to_numpy()
    if not np.any(control_mask):
        raise ValueError("No control pseudobulks are present in the training fold")

    global_control = group_expression[control_mask].mean(axis=0)
    references = {}
    for cell_type in group_metadata["cell_type"].unique():
        mask = control_mask & group_metadata["cell_type"].eq(cell_type).to_numpy()
        references[str(cell_type)] = (
            group_expression[mask].mean(axis=0) if np.any(mask) else global_control
        )
    return references


def subtract_control_reference(
    group_metadata: pd.DataFrame,
    group_expression: np.ndarray,
    references: dict[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            expression - references[str(cell_type)]
            for expression, cell_type in zip(
                group_expression, group_metadata["cell_type"], strict=True
            )
        ],
        dtype=np.float32,
    )


def add_control_reference(
    group_metadata: pd.DataFrame,
    predicted_delta: np.ndarray,
    references: dict[str, np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            delta + references[str(cell_type)]
            for delta, cell_type in zip(
                predicted_delta, group_metadata["cell_type"], strict=True
            )
        ],
        dtype=np.float32,
    )


def encode_features(
    train_metadata: pd.DataFrame, test_metadata: pd.DataFrame
):
    categorical_columns = ["condition", "cell_type", "Batch", "Sex"]
    train_categories = train_metadata[categorical_columns].fillna("missing").astype(str)
    test_categories = test_metadata[categorical_columns].fillna("missing").astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded = encoder.fit_transform(train_categories)
    test_encoded = encoder.transform(test_categories)
    train_time = train_metadata["condition"].map(TIME_HOURS).to_numpy(float)[:, None]
    test_time = test_metadata["condition"].map(TIME_HOURS).to_numpy(float)[:, None]

    train_features = np.column_stack([train_time, train_encoded])
    test_features = np.column_stack([test_time, test_encoded])
    feature_names = ["time_hours"] + encoder.get_feature_names_out(
        categorical_columns
    ).tolist()
    return train_features, test_features, feature_names


def fit_predict_rf_pca(
    train_metadata: pd.DataFrame,
    train_delta: np.ndarray,
    test_metadata: pd.DataFrame,
    args: argparse.Namespace,
):
    train_features, test_features, feature_names = encode_features(
        train_metadata, test_metadata
    )
    n_components = min(
        args.n_components,
        train_delta.shape[0] - 1,
        train_delta.shape[1],
    )
    if n_components < 1:
        raise ValueError("Not enough pseudobulk observations to fit PCA")

    pca = PCA(n_components=n_components, random_state=args.random_state)
    train_scores = pca.fit_transform(train_delta)
    forest = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=True,
        n_jobs=-1,
        random_state=args.random_state,
    )
    forest.fit(train_features, train_scores)
    predicted_delta = pca.inverse_transform(forest.predict(test_features))
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": forest.feature_importances_}
    )
    diagnostics = {
        "n_train_pseudobulks": len(train_metadata),
        "n_components": n_components,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    return predicted_delta.astype(np.float32), importance, diagnostics


def predict_delta_mean(
    train_metadata: pd.DataFrame,
    train_delta: np.ndarray,
    test_metadata: pd.DataFrame,
    match_columns: Iterable[str],
) -> np.ndarray:
    match_columns = list(match_columns)
    train_frame = train_metadata[match_columns].copy()
    train_frame["_row"] = np.arange(len(train_frame))
    lookup = {}
    for key, positions in train_frame.groupby(
        match_columns, observed=True, sort=False
    ).indices.items():
        if not isinstance(key, tuple):
            key = (key,)
        lookup[key] = train_delta[np.asarray(positions, dtype=int)].mean(axis=0)

    predictions = []
    for _, row in test_metadata.iterrows():
        key = tuple(row[column] for column in match_columns)
        predictions.append(lookup.get(key, np.zeros(train_delta.shape[1])))
    return np.asarray(predictions, dtype=np.float32)


def predict_seen_condition_mean(
    train_metadata: pd.DataFrame,
    train_expression: np.ndarray,
    test_metadata: pd.DataFrame,
) -> np.ndarray:
    lookup = {}
    for condition in train_metadata["condition"].unique():
        mask = train_metadata["condition"].eq(condition).to_numpy()
        weights = train_metadata.loc[mask, "n_cells"].to_numpy(float)
        lookup[condition] = weighted_mean(train_expression[mask], weights)
    fallback = lookup.get("ctrl", train_expression.mean(axis=0))
    return np.asarray(
        [lookup.get(condition, fallback) for condition in test_metadata["condition"]],
        dtype=np.float32,
    )


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(pearsonr(left, right)[0])


def compute_auprc(predicted_delta: np.ndarray, p_values: np.ndarray) -> float:
    labels = (p_values < 0.05).astype(int)
    if labels.min() == labels.max():
        return np.nan
    precision, recall, _ = precision_recall_curve(
        labels, np.abs(predicted_delta)
    )
    return float(auc(recall, precision))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(values, axis=0, weights=weights)


def condition_metrics(
    evaluation: str,
    model: str,
    prediction_metadata: pd.DataFrame,
    predicted_expression: np.ndarray,
    truth_expression: np.ndarray,
    raw_expression: np.ndarray,
    raw_metadata: pd.DataFrame,
    control_mean: np.ndarray,
    top_k: int,
) -> list[dict]:
    rows = []
    raw_control = raw_expression[
        raw_metadata["condition"].eq("ctrl").to_numpy()
    ]
    for condition in TARGET_CONDITIONS:
        mask = prediction_metadata["condition"].eq(condition).to_numpy()
        if not np.any(mask):
            continue
        weights = prediction_metadata.loc[mask, "n_cells"].to_numpy(float)
        predicted_mean = weighted_mean(predicted_expression[mask], weights)
        truth_mean = weighted_mean(truth_expression[mask], weights)
        predicted_delta = predicted_mean - control_mean
        truth_delta = truth_mean - control_mean
        top_indices = np.argsort(np.abs(truth_delta))[-top_k:]

        raw_treated = raw_expression[
            raw_metadata["condition"].eq(condition).to_numpy()
        ]
        p_values = np.asarray(
            [
                ttest_ind(
                    raw_treated[:, gene],
                    raw_control[:, gene],
                    equal_var=False,
                    nan_policy="omit",
                ).pvalue
                for gene in range(raw_expression.shape[1])
            ]
        )
        rows.append(
            {
                "evaluation": evaluation,
                "model": model,
                "condition": condition,
                "pearson": safe_pearson(predicted_mean, truth_mean),
                "pearson_de": safe_pearson(
                    predicted_mean[top_indices], truth_mean[top_indices]
                ),
                "pearson_delta": safe_pearson(predicted_delta, truth_delta),
                "pearson_delta_de": safe_pearson(
                    predicted_delta[top_indices], truth_delta[top_indices]
                ),
                "mse": float(np.mean((predicted_mean - truth_mean) ** 2)),
                "auprc": compute_auprc(predicted_delta, p_values),
                "n_cells": int(weights.sum()),
                "n_pseudobulks": int(mask.sum()),
            }
        )

    average = {"evaluation": evaluation, "model": model, "condition": "average"}
    metric_columns = [
        "pearson",
        "pearson_de",
        "pearson_delta",
        "pearson_delta_de",
        "mse",
        "auprc",
    ]
    for metric in metric_columns:
        average[metric] = float(np.nanmean([row[metric] for row in rows]))
    average["n_cells"] = int(sum(row["n_cells"] for row in rows))
    average["n_pseudobulks"] = int(sum(row["n_pseudobulks"] for row in rows))
    rows.append(average)
    return rows


def group_metrics(
    evaluation: str,
    model: str,
    metadata: pd.DataFrame,
    predicted_expression: np.ndarray,
    truth_expression: np.ndarray,
    control_references: np.ndarray,
    top_k: int,
) -> list[dict]:
    rows = []
    for index, row in metadata.reset_index(drop=True).iterrows():
        if row["condition"] not in TARGET_CONDITIONS:
            continue
        predicted_delta = predicted_expression[index] - control_references[index]
        truth_delta = truth_expression[index] - control_references[index]
        top_indices = np.argsort(np.abs(truth_delta))[-top_k:]
        labels = np.zeros_like(truth_delta, dtype=int)
        labels[top_indices] = 1
        precision, recall, _ = precision_recall_curve(
            labels, np.abs(predicted_delta)
        )
        rows.append(
            {
                "evaluation": evaluation,
                "model": model,
                "Sample": row["Sample"],
                "condition": row["condition"],
                "cell_type": row["cell_type"],
                "n_cells": row["n_cells"],
                "pearson": safe_pearson(
                    predicted_expression[index], truth_expression[index]
                ),
                "pearson_delta": safe_pearson(predicted_delta, truth_delta),
                "pearson_delta_top20": safe_pearson(
                    predicted_delta[top_indices], truth_delta[top_indices]
                ),
                "mse": float(
                    np.mean(
                        (predicted_expression[index] - truth_expression[index]) ** 2
                    )
                ),
                "auprc_top20": float(auc(recall, precision)),
            }
        )
    return rows


def held_out_sample_metrics(
    model: str,
    prediction_metadata: pd.DataFrame,
    predicted_expression: np.ndarray,
    truth_expression: np.ndarray,
    raw_expression: np.ndarray,
    raw_metadata: pd.DataFrame,
    control_mean: np.ndarray,
    top_k: int,
) -> list[dict]:
    rows = []
    raw_control = raw_expression[
        raw_metadata["condition"].eq("ctrl").to_numpy()
    ]
    for sample in prediction_metadata["Sample"].drop_duplicates():
        mask = prediction_metadata["Sample"].eq(sample).to_numpy()
        sample_metadata = prediction_metadata.loc[mask]
        condition = str(sample_metadata["condition"].iloc[0])
        weights = sample_metadata["n_cells"].to_numpy(float)
        predicted_mean = weighted_mean(predicted_expression[mask], weights)
        truth_mean = weighted_mean(truth_expression[mask], weights)
        predicted_delta = predicted_mean - control_mean
        truth_delta = truth_mean - control_mean
        top_indices = np.argsort(np.abs(truth_delta))[-top_k:]

        raw_treated = raw_expression[raw_metadata["Sample"].eq(sample).to_numpy()]
        p_values = np.asarray(
            [
                ttest_ind(
                    raw_treated[:, gene],
                    raw_control[:, gene],
                    equal_var=False,
                    nan_policy="omit",
                ).pvalue
                for gene in range(raw_expression.shape[1])
            ]
        )
        rows.append(
            {
                "evaluation": "leave_one_sample_out",
                "model": model,
                "Sample": sample,
                "condition": condition,
                "pearson": safe_pearson(predicted_mean, truth_mean),
                "pearson_de": safe_pearson(
                    predicted_mean[top_indices], truth_mean[top_indices]
                ),
                "pearson_delta": safe_pearson(predicted_delta, truth_delta),
                "pearson_delta_de": safe_pearson(
                    predicted_delta[top_indices], truth_delta[top_indices]
                ),
                "mse": float(np.mean((predicted_mean - truth_mean) ** 2)),
                "auprc": compute_auprc(predicted_delta, p_values),
                "n_cells": int(weights.sum()),
                "n_pseudobulks": int(mask.sum()),
            }
        )
    return rows


def aggregate_held_out_sample_metrics(sample_rows: list[dict]) -> list[dict]:
    sample_frame = pd.DataFrame(sample_rows)
    metric_columns = [
        "pearson",
        "pearson_de",
        "pearson_delta",
        "pearson_delta_de",
        "mse",
        "auprc",
    ]
    rows = []
    for (evaluation, model), model_frame in sample_frame.groupby(
        ["evaluation", "model"], sort=False
    ):
        for condition in TARGET_CONDITIONS:
            condition_frame = model_frame[model_frame["condition"].eq(condition)]
            row = {
                "evaluation": evaluation,
                "model": model,
                "condition": condition,
                "n_cells": int(condition_frame["n_cells"].sum()),
                "n_pseudobulks": int(condition_frame["n_pseudobulks"].sum()),
                "n_samples": int(len(condition_frame)),
            }
            for metric in metric_columns:
                row[metric] = float(np.nanmean(condition_frame[metric]))
            rows.append(row)

        average = {
            "evaluation": evaluation,
            "model": model,
            "condition": "average",
            "n_cells": int(model_frame["n_cells"].sum()),
            "n_pseudobulks": int(model_frame["n_pseudobulks"].sum()),
            "n_samples": int(len(model_frame)),
        }
        for metric in metric_columns:
            average[metric] = float(np.nanmean(model_frame[metric]))
        rows.append(average)
    return rows


def evaluate_saved_cell_split(
    train_expression: np.ndarray,
    test_expression: np.ndarray,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    args: argparse.Namespace,
):
    train_groups, train_pseudobulk = aggregate_pseudobulk(
        train_expression, train_metadata
    )
    test_groups, test_pseudobulk = aggregate_pseudobulk(test_expression, test_metadata)
    references = build_control_references(train_groups, train_pseudobulk)
    train_delta = subtract_control_reference(
        train_groups, train_pseudobulk, references
    )

    rf_delta, importance, diagnostics = fit_predict_rf_pca(
        train_groups, train_delta, test_groups, args
    )
    predictions = {
        "RF-PCA pseudobulk": add_control_reference(test_groups, rf_delta, references),
        "Condition x cell-type mean": add_control_reference(
            test_groups,
            predict_delta_mean(
                train_groups,
                train_delta,
                test_groups,
                ["condition", "cell_type"],
            ),
            references,
        ),
        "Seen-condition mean": predict_seen_condition_mean(
            train_groups, train_pseudobulk, test_groups
        ),
    }
    control_mean = train_expression[
        train_metadata["condition"].eq("ctrl").to_numpy()
    ].mean(axis=0)
    condition_rows = []
    per_group_rows = []
    reference_matrix = np.asarray(
        [references[str(cell_type)] for cell_type in test_groups["cell_type"]]
    )
    for model, prediction in predictions.items():
        condition_rows.extend(
            condition_metrics(
                "saved_cell_split",
                model,
                test_groups,
                prediction,
                test_pseudobulk,
                test_expression,
                test_metadata,
                control_mean,
                args.top_k_de,
            )
        )
        per_group_rows.extend(
            group_metrics(
                "saved_cell_split",
                model,
                test_groups,
                prediction,
                test_pseudobulk,
                reference_matrix,
                args.top_k_de,
            )
        )

    diagnostics["evaluation"] = "saved_cell_split"
    diagnostics["held_out_sample"] = "none"
    importance["evaluation"] = "saved_cell_split"
    importance["held_out_sample"] = "none"
    return condition_rows, per_group_rows, [diagnostics], [importance], []


def evaluate_leave_one_sample_out(
    expression: np.ndarray,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
):
    groups, pseudobulk = aggregate_pseudobulk(expression, metadata)
    treated_samples = groups.loc[
        groups["condition"].isin(TARGET_CONDITIONS), "Sample"
    ].drop_duplicates()

    all_predictions = {
        "RF-PCA pseudobulk": [],
        "Condition x cell-type mean": [],
        "Seen-condition mean": [],
    }
    held_out_metadata = []
    held_out_truth = []
    held_out_references = []
    diagnostics_rows = []
    importance_frames = []

    for fold_index, sample in enumerate(treated_samples, start=1):
        test_mask = groups["Sample"].eq(sample).to_numpy()
        train_mask = ~test_mask
        train_groups = groups.loc[train_mask].reset_index(drop=True)
        test_groups = groups.loc[test_mask].reset_index(drop=True)
        train_pseudobulk = pseudobulk[train_mask]
        test_pseudobulk = pseudobulk[test_mask]

        references = build_control_references(train_groups, train_pseudobulk)
        train_delta = subtract_control_reference(
            train_groups, train_pseudobulk, references
        )
        rf_delta, importance, diagnostics = fit_predict_rf_pca(
            train_groups, train_delta, test_groups, args
        )
        fold_predictions = {
            "RF-PCA pseudobulk": add_control_reference(
                test_groups, rf_delta, references
            ),
            "Condition x cell-type mean": add_control_reference(
                test_groups,
                predict_delta_mean(
                    train_groups,
                    train_delta,
                    test_groups,
                    ["condition", "cell_type"],
                ),
                references,
            ),
            "Seen-condition mean": predict_seen_condition_mean(
                train_groups, train_pseudobulk, test_groups
            ),
        }
        for model, prediction in fold_predictions.items():
            all_predictions[model].append(prediction)
        held_out_metadata.append(test_groups)
        held_out_truth.append(test_pseudobulk)
        held_out_references.append(
            np.asarray(
                [references[str(cell_type)] for cell_type in test_groups["cell_type"]]
            )
        )

        diagnostics.update(
            {
                "evaluation": "leave_one_sample_out",
                "held_out_sample": sample,
                "fold": fold_index,
            }
        )
        diagnostics_rows.append(diagnostics)
        importance["evaluation"] = "leave_one_sample_out"
        importance["held_out_sample"] = sample
        importance["fold"] = fold_index
        importance_frames.append(importance)

    prediction_metadata = pd.concat(held_out_metadata, ignore_index=True)
    truth_pseudobulk = np.vstack(held_out_truth)
    reference_matrix = np.vstack(held_out_references)
    control_mean = expression[metadata["condition"].eq("ctrl").to_numpy()].mean(axis=0)
    sample_rows = []
    per_group_rows = []
    for model, prediction_parts in all_predictions.items():
        prediction = np.vstack(prediction_parts)
        sample_rows.extend(
            held_out_sample_metrics(
                model,
                prediction_metadata,
                prediction,
                truth_pseudobulk,
                expression,
                metadata,
                control_mean,
                args.top_k_de,
            )
        )
        per_group_rows.extend(
            group_metrics(
                "leave_one_sample_out",
                model,
                prediction_metadata,
                prediction,
                truth_pseudobulk,
                reference_matrix,
                args.top_k_de,
            )
        )
    condition_rows = aggregate_held_out_sample_metrics(sample_rows)
    return (
        condition_rows,
        per_group_rows,
        diagnostics_rows,
        importance_frames,
        sample_rows,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        train_expression,
        test_expression,
        train_metadata,
        test_metadata,
        gene_names,
    ) = load_data(args.data_dir)

    saved_results = evaluate_saved_cell_split(
        train_expression,
        test_expression,
        train_metadata,
        test_metadata,
        args,
    )
    combined_expression = np.vstack([train_expression, test_expression])
    combined_metadata = pd.concat(
        [train_metadata, test_metadata], ignore_index=True
    )
    loso_results = evaluate_leave_one_sample_out(
        combined_expression, combined_metadata, args
    )

    condition_rows = saved_results[0] + loso_results[0]
    group_rows = saved_results[1] + loso_results[1]
    diagnostics_rows = saved_results[2] + loso_results[2]
    importance_frames = saved_results[3] + loso_results[3]
    sample_rows = saved_results[4] + loso_results[4]

    condition_frame = pd.DataFrame(condition_rows)
    group_frame = pd.DataFrame(group_rows)
    sample_frame = pd.DataFrame(sample_rows)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    importance_frame = pd.concat(importance_frames, ignore_index=True)
    average_importance = (
        importance_frame.groupby(["evaluation", "feature"], as_index=False)[
            "importance"
        ]
        .mean()
        .sort_values(["evaluation", "importance"], ascending=[True, False])
    )

    condition_frame.to_csv(
        args.output_dir / "condition_level_metrics.csv", index=False
    )
    group_frame.to_csv(args.output_dir / "pseudobulk_group_metrics.csv", index=False)
    sample_frame.to_csv(
        args.output_dir / "held_out_sample_metrics.csv", index=False
    )
    group_summary = (
        group_frame.groupby(["evaluation", "model"], as_index=False)
        .agg(
            pearson_mean=("pearson", "mean"),
            pearson_sd=("pearson", "std"),
            pearson_delta_mean=("pearson_delta", "mean"),
            pearson_delta_sd=("pearson_delta", "std"),
            pearson_delta_top20_mean=("pearson_delta_top20", "mean"),
            pearson_delta_top20_sd=("pearson_delta_top20", "std"),
            mse_mean=("mse", "mean"),
            mse_sd=("mse", "std"),
            auprc_top20_mean=("auprc_top20", "mean"),
            auprc_top20_sd=("auprc_top20", "std"),
        )
    )
    group_summary.to_csv(
        args.output_dir / "pseudobulk_group_summary.csv", index=False
    )
    diagnostics_frame.to_csv(
        args.output_dir / "fold_diagnostics.csv", index=False
    )
    average_importance.to_csv(
        args.output_dir / "mean_feature_importance.csv", index=False
    )
    pd.DataFrame({"gene": gene_names}).to_csv(
        args.output_dir / "biological_genes.csv", index=False
    )

    display_columns = [
        "evaluation",
        "model",
        "condition",
        "pearson",
        "pearson_de",
        "pearson_delta",
        "pearson_delta_de",
        "mse",
        "auprc",
    ]
    summary = condition_frame.loc[
        condition_frame["condition"].eq("average"), display_columns
    ].copy()
    print("\nAverage across IR-24h and IR-1week")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nMean across sample x cell-type pseudobulks")
    print(
        group_summary.to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\nResults written to: {args.output_dir}")


if __name__ == "__main__":
    main()
