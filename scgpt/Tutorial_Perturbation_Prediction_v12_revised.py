#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scGPT Perturbation Training with Reformulated Local Energy Distance and AUPRC
"""

import json
import os
import sys
import time
import copy
import shutil
from pathlib import Path
import warnings
import argparse
import logging

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, wasserstein_distance, entropy, ttest_ind
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, precision_recall_curve, auc
from torch import nn
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from gears import PertData

try:
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics
except ImportError:
    print("CRITICAL: scgpt not found.")
    sys.exit(1)

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("scGPT_Perturbation")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--load_model', type=str, default='save/scGPT_human')
    parser.add_argument('--data_name', type=str, default='adamson', choices=['adamson', 'norman'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--schedule_interval', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--embsize', type=int, default=512)
    parser.add_argument('--d_hid', type=int, default=512)
    parser.add_argument('--nlayers', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--n_layers_cls', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--no_amp', action='store_true')
    return parser.parse_args()


def _mean_k_smallest(distances, k):
    """Row-wise mean of the k smallest distances, including zero self-distance."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distances must be a 2-D matrix")
    if k < 1 or k > distances.shape[1]:
        raise ValueError(f"Invalid k={k} for matrix with {distances.shape[1]} columns")
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def compute_local_energy_distance(pred, truth, k_neighbors=30, n_sample=400, seed=42):
    """
    Local k-nearest-neighbour distance statistic described in the dissertation.

    P = predicted cells and Q = observed cells. Up to ``n_sample`` cells are
    sampled independently from P and Q without replacement. Distances are
    Euclidean in the full biological-gene expression space; no PCA or further
    standardisation is applied.

    For every predicted cell x_i, the contribution is
        mean distance to Q neighbours - mean distance to P neighbours.
    For every observed cell y_j, the contribution is
        mean distance to P neighbours - mean distance to Q neighbours.

    The queried cell is retained in its own-population neighbour set, so its
    zero self-distance is included. The final statistic is averaged over all
    sampled predicted and observed cells. Lower values indicate greater local
    similarity.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError("Local Energy distance expects two 2-D arrays.")
    if pred.shape[1] != truth.shape[1]:
        raise ValueError(
            f"Local Energy feature mismatch: {pred.shape[1]} versus {truth.shape[1]}."
        )
    if pred.shape[0] < 2 or truth.shape[0] < 2:
        return 0.0

    rng = np.random.default_rng(seed)

    if pred.shape[0] > n_sample:
        pred_idx = rng.choice(pred.shape[0], size=n_sample, replace=False)
        pred_sub = pred[pred_idx]
    else:
        pred_sub = pred

    if truth.shape[0] > n_sample:
        truth_idx = rng.choice(truth.shape[0], size=n_sample, replace=False)
        truth_sub = truth[truth_idx]
    else:
        truth_sub = truth

    n_pred = pred_sub.shape[0]
    n_truth = truth_sub.shape[0]
    k = min(k_neighbors, n_pred - 1, n_truth - 1)
    if k < 1:
        return 0.0

    # Pairwise Euclidean distances in all biological genes.
    d_pp = cdist(pred_sub, pred_sub, metric='euclidean')
    d_pq = cdist(pred_sub, truth_sub, metric='euclidean')
    d_qq = cdist(truth_sub, truth_sub, metric='euclidean')

    # Predicted anchors x_i in P. The P-neighbour calculation includes x_i.
    mean_d_p_for_pred = _mean_k_smallest(d_pp, k)
    mean_d_q_for_pred = _mean_k_smallest(d_pq, k)
    pred_terms = mean_d_q_for_pred - mean_d_p_for_pred

    # Observed anchors y_j in Q. The Q-neighbour calculation includes y_j.
    mean_d_p_for_truth = _mean_k_smallest(d_pq.T, k)
    mean_d_q_for_truth = _mean_k_smallest(d_qq, k)
    truth_terms = mean_d_p_for_truth - mean_d_q_for_truth

    local_e_distance = (
        pred_terms.sum() + truth_terms.sum()
    ) / float(n_pred + n_truth)

    return float(local_e_distance)


def compute_metrics_no_auprc(pred, truth, ctrl_mean, top_k=100, k_neighbors=30, n_sample=400, seed=42):
    """Compute all distribution metrics except AUPRC for one RT condition."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    ctrl_mean = np.asarray(ctrl_mean, dtype=np.float64)

    if pred.ndim != 2 or truth.ndim != 2:
        raise ValueError("pred and truth must be 2-D arrays")
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {pred.shape} versus {truth.shape}")
    if pred.shape[1] != ctrl_mean.shape[0]:
        raise ValueError(
            f"Control mean has {ctrl_mean.shape[0]} genes, expected {pred.shape[1]}."
        )

    n_genes = pred.shape[1]
    pred_mean = pred.mean(axis=0)
    truth_mean = truth.mean(axis=0)
    mse = mean_squared_error(truth_mean, pred_mean)

    pred_delta = pred_mean - ctrl_mean
    truth_delta = truth_mean - ctrl_mean
    if np.std(pred_delta) > 1e-6 and np.std(truth_delta) > 1e-6:
        pcc_delta, _ = pearsonr(pred_delta, truth_delta)
    else:
        pcc_delta = 0.0

    # Standard global energy-distance formula, using at most 1000 aligned cells.
    max_samples = 1000
    rng_global = np.random.default_rng(seed)
    n_cells = pred.shape[0]
    if n_cells > max_samples:
        sample_idx = rng_global.choice(n_cells, size=max_samples, replace=False)
        pred_sample = pred[sample_idx]
        truth_sample = truth[sample_idx]
    else:
        pred_sample = pred
        truth_sample = truth

    cross_dist = cdist(pred_sample, truth_sample, metric='euclidean').mean()
    within_pred = cdist(pred_sample, pred_sample, metric='euclidean').mean()
    within_truth = cdist(truth_sample, truth_sample, metric='euclidean').mean()
    global_e_distance = 2.0 * cross_dist - within_pred - within_truth

    wasser_per_gene = []
    kl_per_gene = []
    bins = 50
    for g in range(n_genes):
        p_g = pred[:, g]
        t_g = truth[:, g]
        wasser_per_gene.append(wasserstein_distance(p_g, t_g))

        min_val = min(p_g.min(), t_g.min())
        max_val = max(p_g.max(), t_g.max())
        if max_val - min_val < 1e-6:
            kl_per_gene.append(0.0)
            continue

        edges = np.linspace(min_val, max_val, bins + 1)
        p_hist, _ = np.histogram(p_g, bins=edges, density=True)
        q_hist, _ = np.histogram(t_g, bins=edges, density=True)
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        kl_per_gene.append(entropy(p_hist, q_hist))

    wasserstein = np.mean(wasser_per_gene)
    kl_divergence = np.mean(kl_per_gene)

    effective_top_k = min(top_k, n_genes)
    top_true_idx = np.argsort(np.abs(truth_delta))[-effective_top_k:]
    top_pred_idx = np.argsort(np.abs(pred_delta))[-effective_top_k:]
    common = len(set(top_true_idx) & set(top_pred_idx))
    common_deg = common / float(effective_top_k)

    local_e_distance = compute_local_energy_distance(
        pred,
        truth,
        k_neighbors=k_neighbors,
        n_sample=n_sample,
        seed=seed,
    )

    return {
        "MSE": float(mse),
        "PCC-delta": float(pcc_delta),
        "E-distance": float(global_e_distance),
        "Wasserstein": float(wasserstein),
        "KL-divergence": float(kl_divergence),
        "Common-DEGs": float(common_deg),
        "Local-E-distance": float(local_e_distance),
    }


def compute_auprc_trapezoidal(pred_cond, truth_cond, eval_ctrl_expr, train_ctrl_mean):
    """
    Gene-level AUPRC using the exact requested formulation.

    Labels are defined by an uncorrected Welch two-sample t-test comparing
    observed irradiated cells against observed control cells in the same
    evaluation split (p < 0.05). Ranking scores are the absolute predicted
    perturbation deltas relative to the training-control mean. The precision-
    recall area is calculated by trapezoidal integration, not by
    ``average_precision_score``.
    """
    pred_cond = np.asarray(pred_cond, dtype=np.float64)
    truth_cond = np.asarray(truth_cond, dtype=np.float64)
    eval_ctrl_expr = np.asarray(eval_ctrl_expr, dtype=np.float64)
    train_ctrl_mean = np.asarray(train_ctrl_mean, dtype=np.float64)

    if pred_cond.ndim != 2 or truth_cond.ndim != 2 or eval_ctrl_expr.ndim != 2:
        raise ValueError("AUPRC inputs must be 2-D arrays.")
    n_genes = pred_cond.shape[1]
    if truth_cond.shape[1] != n_genes or eval_ctrl_expr.shape[1] != n_genes:
        raise ValueError("AUPRC inputs must contain the same genes in the same order.")
    if train_ctrl_mean.shape[0] != n_genes:
        raise ValueError("Training-control mean length does not match the gene dimension.")
    if eval_ctrl_expr.shape[0] == 0:
        raise ValueError("No observed control cells are present in this evaluation split.")

    # Vectorised Welch tests across genes; no multiple-testing correction.
    _, p_values = ttest_ind(
        truth_cond,
        eval_ctrl_expr,
        axis=0,
        equal_var=False,
        nan_policy='omit',
    )
    p_values = np.asarray(p_values, dtype=np.float64)
    p_values[~np.isfinite(p_values)] = 1.0
    labels = (p_values < 0.05).astype(np.int8)

    pred_delta = pred_cond.mean(axis=0) - train_ctrl_mean
    scores = np.abs(pred_delta)

    if np.unique(labels).size < 2:
        logger.warning(
            "AUPRC is undefined because the Welch-test labels contain only one class; "
            "returning 0.0."
        )
        return 0.0, int(labels.sum()), int(n_genes)

    precision, recall, _ = precision_recall_curve(labels, scores)
    # sklearn.metrics.auc applies trapezoidal integration and handles the
    # decreasing recall order returned by precision_recall_curve.
    auprc_value = auc(recall, precision)
    return float(auprc_value), int(labels.sum()), int(n_genes)


def evaluate_full_metrics(loader, model, device, gene_ids, ctrl_mean_full, biological_gene_mask):
    """Evaluate reformulated metrics per RT condition and their arithmetic mean."""
    model.eval()
    all_pred, all_truth, all_pert = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = model.pred_perturb(batch, include_zero_gene="all", gene_ids=gene_ids)
            t = batch.y
            if p.shape[1] != t.shape[1]:
                min_dim = min(p.shape[1], t.shape[1])
                p = p[:, :min_dim]
                t = t[:, :min_dim]
            all_pred.append(p.cpu().numpy())
            all_truth.append(t.cpu().numpy())
            all_pert.extend(batch.pert)

    if not all_pred:
        return {'average': {}}

    pred_all = np.vstack(all_pred)
    truth_all = np.vstack(all_truth)
    pert_cat = np.asarray(all_pert)

    mask_trim = np.asarray(biological_gene_mask[:pred_all.shape[1]], dtype=bool)
    pred_all = pred_all[:, mask_trim]
    truth_all = truth_all[:, mask_trim]

    ctrl_mean_trimmed = np.asarray(ctrl_mean_full[:mask_trim.shape[0]], dtype=np.float64)
    ctrl_mean_bio = ctrl_mean_trimmed[mask_trim]

    # Observed control cells from the same validation/test split are used only
    # to define the Welch-test AUPRC labels.
    eval_ctrl_expr = truth_all[pert_cat == 'ctrl']
    if eval_ctrl_expr.shape[0] == 0:
        raise RuntimeError(
            "The evaluation loader contains no control cells, so Welch-test AUPRC "
            "labels cannot be constructed."
        )

    results = {}
    for cond in ["IR-24h", "IR-1week"]:
        condition_mask = pert_cat == cond
        if not np.any(condition_mask):
            continue

        pred_cond = pred_all[condition_mask]
        truth_cond = truth_all[condition_mask]

        metrics = compute_metrics_no_auprc(
            pred_cond,
            truth_cond,
            ctrl_mean_bio,
            top_k=100,
            k_neighbors=30,
            n_sample=400,
            seed=42,
        )

        auprc_value, n_positive, n_total = compute_auprc_trapezoidal(
            pred_cond,
            truth_cond,
            eval_ctrl_expr,
            ctrl_mean_bio,
        )
        metrics["AUPRC"] = float(auprc_value)
        results[cond] = metrics

        logger.info(
            f"{cond} AUPRC Welch labels: {n_positive}/{n_total} genes positive "
            "(uncorrected p < 0.05)."
        )

    average = {}
    if results:
        keys = results[next(iter(results))].keys()
        for key in keys:
            average[key] = float(np.mean([results[c][key] for c in results]))
    results['average'] = average
    return results


def train(model, loader, optimizer, scheduler, scaler, device, config, epoch, vocab, gene_ids,
          de_gene_ids=None, dummy_vocab_ids=None):
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()
    num_batches = len(loader)

    use_amp = not config.no_amp
    loss_weights = None
    if de_gene_ids is not None:
        loss_weights = torch.ones(len(gene_ids), device=device)
        de_weight_mask = torch.as_tensor(
            np.isin(gene_ids, de_gene_ids), dtype=torch.bool, device=device
        )
        loss_weights[de_weight_mask] = 5.0

    for batch_idx, batch_data in enumerate(loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)
        pert_flags = x[:, 1].long().view(batch_size, -1)
        target_gene_values = batch_data.y
        n_genes = ori_gene_values.shape[1]

        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        if len(input_gene_ids) > 1536:
            temp_mapped = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            is_dummy = torch.isin(temp_mapped, torch.tensor(dummy_vocab_ids, device=device)) if dummy_vocab_ids else torch.zeros_like(temp_mapped, dtype=torch.bool)
            dummy_idx = input_gene_ids[is_dummy]
            bio_idx = input_gene_ids[~is_dummy]
            keep_bio = 1536 - len(dummy_idx)
            sampled_bio = bio_idx[torch.randperm(len(bio_idx), device=device)[:keep_bio]]
            input_gene_ids = torch.cat([dummy_idx, sampled_bio])
            input_gene_ids = input_gene_ids[torch.randperm(len(input_gene_ids), device=device)]

        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(batch_size, 1)

        random_mask = torch.rand_like(input_values) < 0.20
        de_mask = torch.isin(mapped_input_gene_ids, torch.tensor(de_gene_ids, device=device)) if de_gene_ids is not None else torch.zeros_like(input_values, dtype=torch.bool)
        masked_positions = random_mask | de_mask
        if dummy_vocab_ids is not None:
            is_dummy = torch.isin(mapped_input_gene_ids, torch.tensor(dummy_vocab_ids, device=device))
            masked_positions = masked_positions & ~is_dummy

        masked_input_values = input_values.clone()
        masked_input_values[masked_positions] = 0.0

        with torch.cuda.amp.autocast(enabled=use_amp):
            output_dict = model(mapped_input_gene_ids, masked_input_values, input_pert_flags,
                                src_key_padding_mask=torch.zeros_like(input_values, dtype=torch.bool, device=device),
                                CLS=False, CCE=False, MVC=False, ECS=False)
            output_values = output_dict["mlm_output"]
            loss = (output_values - target_values) ** 2
            loss = loss * masked_positions
            if loss_weights is not None:
                loss = loss * loss_weights[input_gene_ids].unsqueeze(0).expand(batch_size, -1)
            mask_sum = masked_positions.sum()
            loss = loss.sum() / mask_sum if mask_sum > 0 else loss.sum() * 0.0
            loss_mse = loss

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()

        if batch_idx % config.log_interval == 0 and batch_idx > 0:
            lr = scheduler.get_last_lr()[0]
            ms = (time.time() - start_time) * 1000 / config.log_interval
            logger.info(f"epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} | lr {lr:.4f} | ms/batch {ms:.2f} | loss {total_loss/config.log_interval:.4f} | mse {total_mse/config.log_interval:.4f}")
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def eval_perturb(loader, model, device, gene_ids):
    model.eval()
    pert_cat, pred, truth = [], [], []
    for batch in loader:
        batch.to(device)
        pert_cat.extend(batch.pert)
        with torch.no_grad():
            p = model.pred_perturb(batch, include_zero_gene="all", gene_ids=gene_ids)
            t = batch.y
            if p.shape[1] != t.shape[1]:
                min_dim = min(p.shape[1], t.shape[1])
                p = p[:, :min_dim]
                t = t[:, :min_dim]
            pred.extend(p.cpu())
            truth.extend(t.cpu())
    return {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(pred).numpy().astype(np.float32),
        "truth": torch.stack(truth).numpy().astype(np.float32),
    }


def evaluate_and_log_metrics(res_dict, biological_gene_mask, ctrl_mean, phase, epoch=None):
    target_conditions = ["IR-24h", "IR-1week"]
    metrics = {"pearson": [], "pearson_de": [], "pearson_delta": [], "pearson_delta_de": []}
    prefix = f"[{phase}]" if epoch is None else f"[Epoch {epoch} - {phase}]"
    logger.info(f"\n{'-'*20} {prefix} PEARSON METRICS {'-'*20}")

    for pert in target_conditions:
        idx = np.where(res_dict["pert_cat"] == pert)[0]
        if len(idx) == 0:
            continue
        pred = res_dict["pred"][idx][:, biological_gene_mask[:res_dict["pred"].shape[1]]]
        truth = res_dict["truth"][idx][:, biological_gene_mask[:res_dict["truth"].shape[1]]]
        pred_mean = pred.mean(axis=0)
        truth_mean = truth.mean(axis=0)
        p = np.corrcoef(pred_mean, truth_mean)[0,1] if len(pred_mean) > 1 else 0.0
        pred_delta = pred_mean - ctrl_mean[:pred.shape[1]]
        truth_delta = truth_mean - ctrl_mean[:truth.shape[1]]
        p_delta = np.corrcoef(pred_delta, truth_delta)[0,1] if len(pred_delta) > 1 else 0.0
        de = np.argsort(np.abs(truth_delta))[-20:]
        p_de = np.corrcoef(pred_mean[de], truth_mean[de])[0,1] if len(de) > 1 else 0.0
        p_delta_de = np.corrcoef(pred_delta[de], truth_delta[de])[0,1] if len(de) > 1 else 0.0
        metrics["pearson"].append(p)
        metrics["pearson_de"].append(p_de)
        metrics["pearson_delta"].append(p_delta)
        metrics["pearson_delta_de"].append(p_delta_de)
        logger.info(f"{pert:<10} -> Pearson: {p:.4f} | DE: {p_de:.4f} | Delta: {p_delta:.4f} | DE_Delta: {p_delta_de:.4f}")

    if metrics["pearson"]:
        avg = {k: np.mean(v) for k, v in metrics.items()}
        logger.info(f"AVERAGE    -> Pearson: {avg['pearson']:.4f} | DE: {avg['pearson_de']:.4f} | Delta: {avg['pearson_delta']:.4f} | DE_Delta: {avg['pearson_delta_de']:.4f}")
        return avg['pearson'], avg['pearson_de'], avg['pearson_delta'], avg['pearson_delta_de']
    return 0.0, 0.0, 0.0, 0.0


def plot_perturbation(query, save_dir, top_genes=20):
    import pandas as pd
    X_train = np.load(f"{save_dir}/train_expression.npy")
    X_test = np.load(f"{save_dir}/test_expression.npy")
    meta_train = pd.read_csv(f"{save_dir}/train_metadata.csv")
    meta_test = pd.read_csv(f"{save_dir}/test_metadata.csv")
    with open(f"{save_dir}/gene_names.json", "r") as f:
        gene_names = np.array(json.load(f))

    dummy = ["IR-24h", "IR-1week", "IR_24h", "IR_1week"]
    mask = ~np.isin(gene_names, dummy)
    gene_names = gene_names[mask]
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]
    ctrl_mean = X_train[meta_train["condition"] == "ctrl"].mean(axis=0)

    truth = X_test[meta_test["condition"] == query]
    res = np.load(f"{save_dir}/test_results.npz")
    pred = res["pred"][res["pert_cat"] == query]

    if pred.shape[1] != len(gene_names):
        min_dim = min(pred.shape[1], len(gene_names))
        pred = pred[:, :min_dim]
        truth = truth[:, :min_dim]
        gene_names = gene_names[:min_dim]
        ctrl_mean = ctrl_mean[:min_dim]

    truth_delta = truth.mean(axis=0) - ctrl_mean
    pred_delta = pred.mean(axis=0) - ctrl_mean
    de_idx = np.argsort(np.abs(truth_delta))[-top_genes:]
    de_idx = de_idx[np.argsort(truth_delta[de_idx])]
    plot_genes = gene_names[de_idx]

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(plot_genes))
    ax.bar(x - 0.17, truth_delta[de_idx], 0.35, label='True Δ', color='steelblue')
    ax.bar(x + 0.17, pred_delta[de_idx], 0.35, label='Predicted Δ', color='coral', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_genes, rotation=90)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel("Change in Expression (over CTRL)")
    ax.set_title(f"Perturbation: {query}")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{query}_barplot.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.boxplot(truth[:, de_idx] - ctrl_mean[de_idx], showfliers=False, medianprops=dict(linewidth=1.6, color='black'))
    ax.scatter(np.arange(1, len(pred_delta[de_idx])+1), pred_delta[de_idx], color='red', s=80, zorder=5)
    ax.axhline(0, linestyle='dashed', color='green', alpha=0.45)
    ax.set_xticks(np.arange(1, len(plot_genes)+1))
    ax.set_xticklabels(plot_genes, rotation=90, fontstyle='italic', fontsize=10)
    ax.set_ylabel("Log Expression Change\n(Perturbed - Control)")
    ax.set_title(f"scGPT vs Truth: {query}\n(Top {top_genes} DE Genes)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{query}_boxplot.png", dpi=150)
    plt.close()
    logger.info(f"Plots saved for {query}")


def main():
    args = parse_args()
    set_seed(42)

    dummy_genes = ["IR-24h", "IR-1week"]
    save_dir = Path(f"{args.save_dir}/dev_perturb_{args.data_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(save_dir / "run.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.info(f"Args: {args}")
    logger.info(
        "Metric definitions: Local-E uses independently sampled P/Q populations "
        "(up to 400 each), k=min(30,nP-1,nQ-1), full biological-gene Euclidean "
        "space, and includes zero self-distance. AUPRC uses uncorrected Welch "
        "p<0.05 labels and trapezoidal PR-curve integration."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Clear cache
    cache_dir = Path(args.data_dir) / args.data_name / "processed"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.data_name)
    adata = pert_data.adata

    # Remove existing dummy genes
    existing = [g for g in dummy_genes if g in adata.var_names]
    if existing:
        adata = adata[:, ~adata.var_names.isin(existing)].copy()
        pert_data.adata = adata

    # Sanitize condition names
    pert_data.adata.obs['condition'] = pert_data.adata.obs['condition'].astype(str).str.replace('_', '-')

    # Update DE dictionaries
    if 'rank_genes_groups_cov_all' in pert_data.adata.uns:
        old_dict = pert_data.adata.uns['rank_genes_groups_cov_all']
        new_dict = {}
        all_keys = list(old_dict.keys())
        for k_cond, v_data in old_dict.items():
            new_cond_base = k_cond.replace('_', '-')
            if new_cond_base == 'ctrl':
                continue
            for suffix in ['_ctrl_dummy', f'_{new_cond_base}_dummy', f'_{k_cond}_dummy']:
                new_dict[f"{new_cond_base}{suffix}"] = v_data
            for k_ref in all_keys:
                if k_ref == 'ctrl' or k_ref == k_cond:
                    continue
                new_ref_base = k_ref.replace('_', '-')
                new_dict[f"{new_cond_base}_{new_ref_base}_dummy"] = v_data
                new_dict[f"{new_cond_base}_{k_ref}_dummy"] = v_data
        pert_data.adata.uns['rank_genes_groups_cov_all'] = new_dict

    if 'top_non_dropout_de_20' in pert_data.adata.uns:
        old_top = pert_data.adata.uns['top_non_dropout_de_20']
        new_top = {}
        for k, v in old_top.items():
            new_key = k.replace('_', '-')
            new_top[new_key] = v
            new_top[k] = v
        pert_data.adata.uns['top_non_dropout_de_20'] = new_top

    if hasattr(pert_data, 'dataset_processed'):
        new_processed = {}
        for k, data_list in pert_data.dataset_processed.items():
            new_key = k.replace('_', '-')
            for data_obj in data_list:
                if hasattr(data_obj, 'pert'):
                    data_obj.pert = data_obj.pert.replace('_', '-')
            new_processed[new_key] = data_list
        pert_data.dataset_processed = new_processed

    # Inject dummy genes
    from anndata import AnnData
    import pandas as pd
    import scipy.sparse
    from scipy.sparse import hstack, csr_matrix

    adata = pert_data.adata
    for cond, token in {'IR-24h': 'IR-24h', 'IR-1week': 'IR-1week'}.items():
        if token in adata.var_names or cond not in adata.obs['condition'].unique():
            continue
        expr = (adata.obs['condition'] == cond).astype(float).values.reshape(-1, 1)
        if scipy.sparse.issparse(adata.X):
            new_X = hstack([adata.X, csr_matrix(expr)], format='csr')
        else:
            new_X = np.hstack([adata.X, expr])
        new_row = pd.DataFrame(index=[token])
        for col in adata.var.columns:
            new_row[col] = token if col == 'gene_name' else 0
        new_var = pd.concat([adata.var, new_row])
        new_adata = AnnData(X=new_X, obs=adata.obs, var=new_var)
        new_adata.uns = adata.uns.copy()
        new_adata.obsm = adata.obsm.copy()
        adata = new_adata
        pert_data.adata = adata

    # Train/val/test split
    from sklearn.model_selection import train_test_split
    obs = pert_data.adata.obs
    idx = np.arange(len(obs))
    train_idx, temp_idx = train_test_split(idx, test_size=0.20, random_state=1, stratify=obs['condition'])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=1, stratify=obs.iloc[temp_idx]['condition'])

    pert_data.adata.obs['split'] = 'train'
    pert_data.adata.obs.iloc[val_idx, pert_data.adata.obs.columns.get_loc('split')] = 'val'
    pert_data.adata.obs.iloc[test_idx, pert_data.adata.obs.columns.get_loc('split')] = 'test'

    pert_data.set2conditions = {
        'train': pert_data.adata.obs[pert_data.adata.obs['split'] == 'train']['condition'].unique().tolist(),
        'val': pert_data.adata.obs[pert_data.adata.obs['split'] == 'val']['condition'].unique().tolist(),
        'test': pert_data.adata.obs[pert_data.adata.obs['split'] == 'test']['condition'].unique().tolist(),
    }
    pert_data.split = 'simulation'
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.eval_batch_size)
    train_loader = pert_data.dataloader['train_loader']
    val_loader = pert_data.dataloader['val_loader']
    test_loader = pert_data.dataloader['test_loader']

    # Control mean
    train_ctrl = (pert_data.adata.obs['split'] == 'train') & (pert_data.adata.obs['condition'] == 'ctrl')
    ctrl_X = pert_data.adata[train_ctrl].X
    ctrl_X = ctrl_X.toarray() if hasattr(ctrl_X, 'toarray') else ctrl_X
    ctrl_mean_full = ctrl_X.mean(axis=0)

    # Load model
    if args.load_model and os.path.exists(args.load_model):
        model_dir = Path(args.load_model)
        vocab = GeneVocab.from_file(model_dir / "vocab.json")
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in vocab:
                vocab.append_token(s)
        genes = pert_data.adata.var["gene_name"].tolist()
        with open(model_dir / "args.json") as f:
            cfg = json.load(f)
        embsize, nhead, d_hid, nlayers, n_layers_cls = cfg["embsize"], cfg["nheads"], cfg["d_hid"], cfg["nlayers"], cfg["n_layers_cls"]
        model_file = model_dir / "best_model.pt"
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + ["<pad>", "<cls>", "<eoc>"], None))
        vocab.set_default_index(vocab["<pad>"])
        embsize, nhead, d_hid, nlayers, n_layers_cls = args.embsize, args.nhead, args.d_hid, args.nlayers, args.n_layers_cls
        model_file = None

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int)
    dummy_vocab_ids = [vocab[g] for g in dummy_genes if g in vocab]

    # The installed scGPT version expects the first five architecture
    # arguments positionally.
    model = TransformerGenerator(
        len(vocab),  # vocabulary size / ntoken
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=args.dropout,
        pad_token="<pad>",
        pad_value=0,
        pert_pad_id=0,
        use_fast_transformer=True,
    )
    
    if model_file and os.path.exists(model_file):
        try:
            sd = torch.load(model_file, map_location=device)
            sd = {k: v for k, v in sd.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(sd, strict=False)
            logger.info("Loaded pretrained weights.")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    # DE genes for loss weighting
    de_ids = []
    if 'rank_genes_groups_cov_all' in pert_data.adata.uns:
        for v in pert_data.adata.uns['rank_genes_groups_cov_all'].values():
            for g in v[:50]:
                if g in vocab:
                    de_ids.append(vocab[g])
    de_gene_ids = np.unique(de_ids)

    biological_gene_mask = ~np.isin(genes, dummy_genes)
    ctrl_mean_bio = np.asarray(ctrl_mean_full[:len(genes)])[biological_gene_mask]

    # Training loop
    best_val_delta = -np.inf
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train(model, train_loader, optimizer, scheduler, scaler, device, args, epoch, vocab, gene_ids,
              de_gene_ids=de_gene_ids, dummy_vocab_ids=dummy_vocab_ids)

        eval_model = model.module if isinstance(model, nn.DataParallel) else model
        val_res = eval_perturb(val_loader, eval_model, device, gene_ids)
        _, _, val_delta, _ = evaluate_and_log_metrics(val_res, biological_gene_mask, ctrl_mean_bio, "Validation", epoch)

        full_val = evaluate_full_metrics(val_loader, eval_model, device, gene_ids, ctrl_mean_full, biological_gene_mask)
        logger.info("Full metrics (validation):")
        for cond in ['IR-24h', 'IR-1week']:
            if cond in full_val:
                m = full_val[cond]
                logger.info(f"{cond:<10} MSE: {m['MSE']:.4f} | PCC-delta: {m['PCC-delta']:.4f} | E-dist: {m['E-distance']:.4f} | Wasserstein: {m['Wasserstein']:.4f} | KL: {m['KL-divergence']:.4f} | Common-DEGs: {m['Common-DEGs']:.4f} | AUPRC: {m['AUPRC']:.4f} | Local-E: {m['Local-E-distance']:.4f}")
        if 'average' in full_val:
            a = full_val['average']
            logger.info(f"AVERAGE    MSE: {a['MSE']:.4f} | PCC-delta: {a['PCC-delta']:.4f} | E-dist: {a['E-distance']:.4f} | Wasserstein: {a['Wasserstein']:.4f} | KL: {a['KL-divergence']:.4f} | Common-DEGs: {a['Common-DEGs']:.4f} | AUPRC: {a['AUPRC']:.4f} | Local-E: {a['Local-E-distance']:.4f}")

        if val_delta > best_val_delta:
            best_val_delta = val_delta
            best_state = copy.deepcopy(eval_model.state_dict())
            logger.info(f"Best model updated (val delta: {val_delta:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

        scheduler.step()
        logger.info(f"Epoch {epoch} done in {time.time()-t0:.2f}s")

    # Final test evaluation
    if best_state is not None:
        eval_model.load_state_dict(best_state)
        model = eval_model
        torch.save(best_state, save_dir / "best_model.pt")

        test_res = eval_perturb(test_loader, model, device, gene_ids)
        np.savez(save_dir / "test_results.npz", **test_res)

        # Save splits
        for name, idx in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
            X = pert_data.adata[idx].X
            X = X.toarray() if hasattr(X, 'toarray') else X
            np.save(save_dir / f"{name}_expression.npy", X)
            pert_data.adata[idx].obs.to_csv(save_dir / f"{name}_metadata.csv")
        with open(save_dir / "gene_names.json", "w") as f:
            json.dump(pert_data.adata.var_names.tolist(), f)

        logger.info("\n" + "="*60)
        logger.info("TEST SET PEARSON METRICS")
        evaluate_and_log_metrics(test_res, biological_gene_mask, ctrl_mean_bio, "Testing")

        full_test = evaluate_full_metrics(test_loader, model, device, gene_ids, ctrl_mean_full, biological_gene_mask)
        logger.info("TEST SET FULL METRICS")
        for cond in ['IR-24h', 'IR-1week']:
            if cond in full_test:
                m = full_test[cond]
                logger.info(f"{cond:<10} MSE: {m['MSE']:.4f} | PCC-delta: {m['PCC-delta']:.4f} | E-dist: {m['E-distance']:.4f} | Wasserstein: {m['Wasserstein']:.4f} | KL: {m['KL-divergence']:.4f} | Common-DEGs: {m['Common-DEGs']:.4f} | AUPRC: {m['AUPRC']:.4f} | Local-E: {m['Local-E-distance']:.4f}")
        if 'average' in full_test:
            a = full_test['average']
            logger.info(f"AVERAGE    MSE: {a['MSE']:.4f} | PCC-delta: {a['PCC-delta']:.4f} | E-dist: {a['E-distance']:.4f} | Wasserstein: {a['Wasserstein']:.4f} | KL: {a['KL-divergence']:.4f} | Common-DEGs: {a['Common-DEGs']:.4f} | AUPRC: {a['AUPRC']:.4f} | Local-E: {a['Local-E-distance']:.4f}")
        logger.info("="*60)

        # Save JSON
        with open(save_dir / "test_metrics_full.json", "w") as f:
            json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in full_test.items()}, f, indent=2)

        # Plots
        for p in ["IR-24h", "IR-1week"]:
            try:
                plot_perturbation(p, save_dir)
            except Exception as e:
                logger.error(f"Plot failed for {p}: {e}")


    logger.info("All done!")


if __name__ == "__main__":
    main()
    print("Completed")