#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention-based GRN analysis for scGenePT MVM full-retrained model.

This version is intended for the post-test full-data checkpoint saved under:

    <run_dir>/full_retrained/final_model_full_finetuned.pt

It uses the saved train/val/test expression matrices and metadata from the same
scGenePT run, rather than re-processing the raw h5ad. This preserves the exact
feature order used during fine-tuning, including the two timepoint pseudo-tokens.

Run from ~/Scratch/grn_scgpt, for example:

    python Tutorial_Attention_GRN_scGenePT_BioAll.py \
        --scgenept_repo ~/Scratch/scGenePT \
        --run_dir ./scGenePT_adamson_Jul11-2155 \
        --model_type scgenept_bioall_gpt_concat \
        --max_cells_per_condition 3000

Notes:
- The output is for GRN/attention interpretation only.
- It should use the full_retrained model, not the parent best_model.pt, if your
  aim is the full-data GRN model.
- By default, it samples up to 1000 cells per condition for speed. Set
  --max_cells_per_condition 0 to use all available cells.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from anndata import AnnData
from tqdm import tqdm


# -----------------------------
# Basic configuration
# -----------------------------
PAD_TOKEN = "<pad>"
GENES_OF_INTEREST = ["NFIB", "NFIA", "FOSL2", "FOS", "JUNB", "JUN", "BIRC6"]

LOGGER = logging.getLogger("scGenePT_Attention_GRN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention GRN for scGenePT full-retrained checkpoint")
    parser.add_argument("--scgenept_repo", type=str, default="~/Scratch/scGenePT",
                        help="Path to the cloned scGenePT repository root.")
    parser.add_argument("--run_dir", type=str, default="./scGenePT_adamson_Jul11-2155",
                        help="Copied scGenePT result folder containing train/val/test arrays and full_retrained/.")
    parser.add_argument("--models_dir", type=str, default=None,
                        help="scGenePT models directory. Default: <scgenept_repo>/models")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path. Default: <run_dir>/full_retrained/final_model_full_finetuned.pt")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model type. Default: read from full_retrained/model_config.json or run_args.json.")
    parser.add_argument("--target_conditions", type=str, default="IR_24h,IR_1week")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: ./results/GRN_scGenePT_<timestamp>")
    parser.add_argument("--layer_idx", type=int, default=11,
                        help="Transformer layer index to analyse. For 12 layers, 11 is the final layer.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for attention extraction. Keep at 1 or 2 to avoid GPU OOM.")
    parser.add_argument("--top_n", type=int, default=30,
                        help="Number of top genes by absolute influence shift to plot.")
    parser.add_argument("--max_cells_per_condition", type=int, default=3000,
                        help="Max cells sampled per condition. 0 means use all cells.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA autocast.")
    parser.add_argument("--drop_dummy_tokens_from_outputs", action="store_true", default=True,
                        help="Exclude IR_24h/IR_1week pseudo-tokens from saved influence matrices/plots.")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")],
    )
    LOGGER.info("Writing log to %s", log_path)


def canonical_condition(x: str) -> str:
    return str(x).replace("-", "_")


def parse_conditions(s: str) -> List[str]:
    return [canonical_condition(x.strip()) for x in s.split(",") if x.strip()]


def load_json(path: Path, default=None):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def load_saved_expression_and_metadata(run_dir: Path) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Load and concatenate train/val/test arrays from the scGenePT result folder."""
    xs = []
    metas = []
    for split in ["train", "val", "test"]:
        x_path = run_dir / f"{split}_expression.npy"
        m_path = run_dir / f"{split}_metadata.csv"
        if not x_path.exists() or not m_path.exists():
            raise FileNotFoundError(f"Missing {x_path} or {m_path}")
        X = np.load(x_path)
        meta = pd.read_csv(m_path, index_col=0)
        meta["original_split"] = split
        xs.append(X.astype(np.float32, copy=False))
        metas.append(meta)

    X_all = np.vstack(xs)
    meta_all = pd.concat(metas, axis=0)
    if "condition" not in meta_all.columns:
        raise ValueError("Metadata must contain a 'condition' column.")
    meta_all["condition"] = meta_all["condition"].map(canonical_condition)

    gene_path = run_dir / "gene_names.json"
    if not gene_path.exists():
        raise FileNotFoundError(f"Missing {gene_path}")
    with open(gene_path, "r", encoding="utf-8") as f:
        genes = json.load(f)
    genes = [str(g) for g in genes]

    if X_all.shape[1] != len(genes):
        raise ValueError(f"Expression columns ({X_all.shape[1]}) != gene_names ({len(genes)})")

    return X_all, meta_all, genes


def resolve_model_type(run_dir: Path, user_model_type: str | None) -> str:
    if user_model_type:
        return user_model_type
    cfg = load_json(run_dir / "full_retrained" / "model_config.json", {}) or {}
    if cfg.get("model_type"):
        return cfg["model_type"]
    run_args = load_json(run_dir / "full_retrained" / "run_args.json", {}) or {}
    if run_args.get("model_type"):
        return run_args["model_type"]
    raise ValueError("Could not infer model_type. Pass --model_type explicitly.")


def import_training_helpers(scgenept_repo: Path):
    """Import the training script helpers so model construction exactly matches training."""
    sys.path.insert(0, str(scgenept_repo))
    sys.path.insert(0, str(scgenept_repo / "models"))
    try:
        import scgenept_gb_timepoint_train_mvm_fullretrain as trainmod
    except Exception as exc:
        raise ImportError(
            "Could not import scgenept_gb_timepoint_train_mvm_fullretrain.py from "
            f"{scgenept_repo}. Copy that script to the scGenePT repo root first."
        ) from exc
    return trainmod


def build_model_from_saved_run(
    trainmod,
    genes: List[str],
    models_dir: Path,
    model_type: str,
    checkpoint: Path,
    target_conditions: List[str],
    device: torch.device,
):
    """Build the scGenePT architecture and load the full-retrained checkpoint."""
    fake_adata = AnnData(X=np.zeros((1, len(genes)), dtype=np.float32))
    fake_adata.var_names = pd.Index(genes)
    fake_adata.var["gene_name"] = genes

    build_args = SimpleNamespace(
        models_dir=str(models_dir),
        model_type=model_type,
        finetuned_dataset="norman",
        checkpoint_path=None,
        init_from="none",  # build architecture only; then load full-retrained weights below
        dropout=0.0,
        use_fast_transformer=False,
    )

    model, vocab, gene_ids_np = trainmod.build_scgenept_model(
        fake_adata,
        build_args,
        target_conditions,
        device,
    )

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    LOGGER.info("Loaded checkpoint: %s", checkpoint)
    LOGGER.info("Missing keys: %d | Unexpected keys: %d", len(missing), len(unexpected))
    if len(missing) > 0:
        LOGGER.info("First missing keys: %s", list(missing)[:8])
    if len(unexpected) > 0:
        LOGGER.info("First unexpected keys: %s", list(unexpected)[:8])

    model.to(device)
    model.eval()
    return model, vocab, np.asarray(gene_ids_np, dtype=np.int64)


def make_sequence_masks(
    genes: List[str],
    gene_ids_np: np.ndarray,
    vocab,
    target_conditions: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Return valid-token mask and biological-output mask."""
    pad_id = int(vocab[PAD_TOKEN])
    genes_arr = np.asarray(genes, dtype=object)
    valid_mask = gene_ids_np != pad_id
    dummy_mask = np.isin(genes_arr, np.asarray(target_conditions, dtype=object))
    biological_mask = valid_mask & (~dummy_mask)

    seq_genes = genes_arr[valid_mask].astype(str).tolist()
    bio_genes = genes_arr[biological_mask].astype(str).tolist()
    return valid_mask, biological_mask[valid_mask], seq_genes, bio_genes


def compute_rank_attention_from_query(attn_module, query: torch.Tensor) -> torch.Tensor:
    """
    Compute raw QK attention scores from the query entering a MultiheadAttention layer,
    then apply the row/column rank normalisation used in the original scGPT GRN script.

    Returns: tensor of shape [batch, seq_len, seq_len], averaged over heads.
    """
    if not hasattr(attn_module, "in_proj_weight"):
        raise AttributeError("Target attention module has no in_proj_weight; use_fast_transformer may be enabled.")

    # PyTorch MultiheadAttention can be batch_first=True or False.
    if getattr(attn_module, "batch_first", False):
        x = query  # [B, L, D]
    else:
        x = query.transpose(0, 1)  # [B, L, D]

    qkv = F.linear(x, attn_module.in_proj_weight, attn_module.in_proj_bias)
    batch_size, seq_len, d3 = qkv.shape
    d_model = d3 // 3
    num_heads = int(attn_module.num_heads)
    head_dim = d_model // num_heads

    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]

    # [B, H, L, L], raw pre-softmax QK scores.
    scores = torch.einsum("bnhd,bmhd->bhnm", q, k)

    # Same two-step rank normalisation as your original Tutorial_Attention_GRN.py.
    scores = scores.reshape(-1, seq_len)
    order = torch.argsort(scores, dim=1)
    rank = torch.argsort(order, dim=1)
    scores = rank.reshape(batch_size, num_heads, seq_len, seq_len) / seq_len

    scores = scores.permute(0, 1, 3, 2).reshape(-1, seq_len)
    order = torch.argsort(scores, dim=1)
    rank = torch.argsort(order, dim=1)
    scores = (rank.reshape(batch_size, num_heads, seq_len, seq_len) / seq_len).permute(0, 1, 3, 2)

    return scores.mean(dim=1)  # [B, L, L]


def extract_attention_batch(
    model,
    mapped_gene_ids: torch.Tensor,
    input_values: torch.Tensor,
    pert_flags: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    layer_idx: int,
    use_amp: bool,
) -> np.ndarray:
    """Run the model once and capture rank-normalised attention at a chosen layer."""
    layers = model.transformer_encoder.layers
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(layers)} layers")

    target_layer = layers[layer_idx]
    attn_module = target_layer.self_attn
    original_forward = attn_module.forward
    captured: Dict[str, torch.Tensor] = {}

    def wrapped_forward(query, key, value, *args, **kwargs):
        # Capture raw-rank attention from the layer input, then let the model run normally.
        captured["attn"] = compute_rank_attention_from_query(attn_module, query).detach()
        return original_forward(query, key, value, *args, **kwargs)

    attn_module.forward = wrapped_forward
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            _ = model(
                mapped_gene_ids,
                input_values,
                pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
    finally:
        attn_module.forward = original_forward

    if "attn" not in captured:
        raise RuntimeError("Failed to capture attention. Check scGenePT transformer layer structure.")
    return captured["attn"].cpu().numpy().astype(np.float32)


def make_pert_flags(
    batch_size: int,
    seq_genes: List[str],
    condition: str,
    target_conditions: List[str],
    device: torch.device,
) -> torch.Tensor:
    flags = torch.zeros((batch_size, len(seq_genes)), dtype=torch.long, device=device)
    if condition != "ctrl":
        if condition not in target_conditions:
            raise ValueError(f"Unknown target condition {condition}")
        try:
            pos = seq_genes.index(condition)
            flags[:, pos] = 1
        except ValueError:
            LOGGER.warning("Condition pseudo-token %s not found in sequence genes; perturb flag remains zero.", condition)
    return flags


def sample_indices_for_condition(meta: pd.DataFrame, condition: str, max_cells: int, seed: int) -> np.ndarray:
    idx = np.where(meta["condition"].values == condition)[0]
    if idx.size == 0:
        raise ValueError(f"No cells found for condition {condition}")
    if max_cells and max_cells > 0 and idx.size > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_cells, replace=False)
    return np.asarray(idx, dtype=np.int64)


def compute_condition_attention(
    X: np.ndarray,
    meta: pd.DataFrame,
    condition: str,
    model,
    vocab,
    gene_ids_np: np.ndarray,
    valid_mask: np.ndarray,
    biological_seq_mask: np.ndarray,
    seq_genes: List[str],
    device: torch.device,
    target_conditions: List[str],
    layer_idx: int,
    batch_size: int,
    max_cells: int,
    seed: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Average attention matrix and influence vector for one condition."""
    idx = sample_indices_for_condition(meta, condition, max_cells, seed)
    LOGGER.info("Computing attention for %s: %d cells", condition, len(idx))

    seq_gene_ids_np = gene_ids_np[valid_mask]
    pad_id = int(vocab[PAD_TOKEN])
    mapped_base = torch.as_tensor(seq_gene_ids_np, dtype=torch.long, device=device)

    n_seq = len(seq_gene_ids_np)
    attn_sum = np.zeros((n_seq, n_seq), dtype=np.float64)
    n_seen = 0

    for start in tqdm(range(0, len(idx), batch_size), desc=f"attention {condition}"):
        batch_idx = idx[start:start + batch_size]
        vals_np = X[batch_idx][:, valid_mask].astype(np.float32, copy=False)
        vals = torch.as_tensor(vals_np, dtype=torch.float32, device=device)
        bsz = vals.shape[0]

        mapped_ids = mapped_base.unsqueeze(0).expand(bsz, -1)
        mask = mapped_ids.eq(pad_id)
        pert_flags = make_pert_flags(bsz, seq_genes, condition, target_conditions, device)

        attn = extract_attention_batch(
            model,
            mapped_ids,
            vals,
            pert_flags,
            mask,
            layer_idx=layer_idx,
            use_amp=use_amp,
        )
        attn_sum += attn.sum(axis=0).astype(np.float64)
        n_seen += bsz
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    attn_mean_seq = (attn_sum / max(n_seen, 1)).astype(np.float32)
    attn_mean_bio = attn_mean_seq[np.ix_(biological_seq_mask, biological_seq_mask)]
    influence = attn_mean_bio.sum(axis=0)
    return attn_mean_bio, influence.astype(np.float32), n_seen


def plot_heatmap(diff_matrix, gene_list, title, output_file, highlight_genes=None):
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(diff_matrix, xticklabels=gene_list, yticklabels=gene_list,
                     cmap="coolwarm", center=0, square=True)
    if highlight_genes:
        for tick in ax.get_xticklabels():
            if tick.get_text() in highlight_genes:
                tick.set_color("red")
                tick.set_weight("bold")
        for tick in ax.get_yticklabels():
            if tick.get_text() in highlight_genes:
                tick.set_color("red")
                tick.set_weight("bold")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_clustermap(diff_matrix, gene_list, title, output_file):
    g = sns.clustermap(diff_matrix, xticklabels=gene_list, yticklabels=gene_list,
                       cmap="coolwarm", center=0, figsize=(14, 12))
    g.ax_heatmap.set_title(title)
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_network(diff_matrix, gene_list, title, output_file, highlight_genes=None):
    highlight_genes = set(highlight_genes or [])
    G = nx.DiGraph()
    n = len(gene_list)
    edges_pool = []
    for i in range(n):
        for j in range(n):
            if i != j and diff_matrix[i, j] > 0:
                edges_pool.append((gene_list[i], gene_list[j], float(diff_matrix[i, j])))

    if not edges_pool:
        LOGGER.info("No positive edges to display for %s", title)
        return

    threshold = np.percentile([e[2] for e in edges_pool], 85)
    for u, v, w in edges_pool:
        if w >= threshold:
            G.add_edge(u, v, weight=w)

    plt.figure(figsize=(16, 14))
    pos = nx.spring_layout(G, k=0.4, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    scaled_widths = [(w / max_w) * 4 for w in weights]
    node_colors = ["lightcoral" if g in highlight_genes else "lightblue" for g in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, width=scaled_widths, alpha=0.5,
                           edge_color="gray", arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    plt.title(f"{title} (Top 15% positive shifts)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def collapse_attention_by_gene(
    attention: np.ndarray,
    gene_codes: np.ndarray,
    gene_counts: np.ndarray,
) -> np.ndarray:
    """Average duplicate feature rows/columns into one matrix per gene symbol.

    If multiple model features share the same displayed gene symbol, the returned
    gene-level entry (i, j) is the mean of all feature-level attention entries
    linking occurrences of gene i to occurrences of gene j.
    """
    attention = np.asarray(attention, dtype=np.float64)
    n_unique = len(gene_counts)

    # First average rows belonging to the same symbol.
    row_sum = np.zeros((n_unique, attention.shape[1]), dtype=np.float64)
    np.add.at(row_sum, gene_codes, attention)
    row_mean = row_sum / gene_counts[:, None]

    # Then average columns belonging to the same symbol.
    col_sum_transposed = np.zeros((n_unique, n_unique), dtype=np.float64)
    np.add.at(col_sum_transposed, gene_codes, row_mean.T)
    collapsed = (col_sum_transposed / gene_counts[:, None]).T
    return collapsed.astype(np.float32)


def save_pair_outputs(
    output_dir: Path,
    label: str,
    ctrl_attn: np.ndarray,
    pert_attn: np.ndarray,
    ctrl_influence: np.ndarray,
    pert_influence: np.ndarray,
    genes: List[str],
    top_n: int,
):
    """Save unique-gene attention outputs and plots.

    Duplicate gene symbols are collapsed by averaging their corresponding
    attention rows and columns. This avoids ambiguous duplicated network nodes
    and the pandas ``cannot reindex on an axis with duplicate labels`` error.
    """
    genes_arr = np.asarray(genes, dtype=str)
    ctrl_attn = np.asarray(ctrl_attn, dtype=np.float32)
    pert_attn = np.asarray(pert_attn, dtype=np.float32)

    expected_shape = (len(genes_arr), len(genes_arr))
    if ctrl_attn.shape != expected_shape or pert_attn.shape != expected_shape:
        raise ValueError(
            f"Attention matrix shape mismatch: genes={len(genes_arr)}, "
            f"ctrl={ctrl_attn.shape}, perturbation={pert_attn.shape}."
        )

    # Preserve first-occurrence ordering while assigning every duplicate symbol
    # to the same output gene.
    gene_codes, unique_genes_index = pd.factorize(genes_arr, sort=False)
    unique_genes = np.asarray(unique_genes_index, dtype=str)
    gene_counts = np.bincount(gene_codes, minlength=len(unique_genes)).astype(np.float64)
    duplicate_counts = pd.Series(genes_arr).value_counts()
    duplicate_counts = duplicate_counts[duplicate_counts > 1]

    canonical_label = canonical_condition(label)
    suffix = (
        "24h" if canonical_label == "IR_24h"
        else "1week" if canonical_label == "IR_1week"
        else canonical_label
    )

    if len(duplicate_counts) > 0:
        LOGGER.warning(
            "Found %d duplicated biological gene symbols (%d extra feature rows). "
            "Collapsing duplicate attention rows/columns by their mean. Examples: %s",
            len(duplicate_counts),
            int(duplicate_counts.sum() - len(duplicate_counts)),
            duplicate_counts.head(10).to_dict(),
        )

        # Retain raw feature-level matrices for provenance/debugging.
        np.savez_compressed(
            output_dir / f"attention_matrices_{suffix}_feature_level.npz",
            ctrl=ctrl_attn,
            ir=pert_attn,
            diff=pert_attn - ctrl_attn,
            genes=genes_arr,
        )

        ctrl_gene_attn = collapse_attention_by_gene(ctrl_attn, gene_codes, gene_counts)
        pert_gene_attn = collapse_attention_by_gene(pert_attn, gene_codes, gene_counts)
    else:
        ctrl_gene_attn = ctrl_attn
        pert_gene_attn = pert_attn

    # Recompute influence from the same unique-gene matrices used for plots.
    ctrl_gene_influence = ctrl_gene_attn.sum(axis=0)
    pert_gene_influence = pert_gene_attn.sum(axis=0)
    diff = pert_gene_influence - ctrl_gene_influence

    df = pd.DataFrame(
        {
            "ctrl": ctrl_gene_influence,
            "ir": pert_gene_influence,
            "diff": diff,
            "abs_diff": np.abs(diff),
            "n_collapsed_features": gene_counts.astype(int),
        },
        index=pd.Index(unique_genes, name="gene"),
    )
    df_ctrl = df[["ctrl", "n_collapsed_features"]].rename(columns={"ctrl": "influence"})
    df_ir = df[["ir", "n_collapsed_features"]].rename(columns={"ir": "influence"})

    df_ctrl.to_csv(output_dir / f"ctrl_influence_{suffix}.csv")
    df_ir.to_csv(output_dir / f"ir_influence_{suffix}.csv")
    df.to_csv(output_dir / f"diff_influence_{suffix}.csv")

    # Rank by position rather than reindexing a potentially duplicated index.
    top_n_effective = min(int(top_n), len(unique_genes))
    top_idx = np.argsort(-np.abs(diff), kind="stable")[:top_n_effective]
    top_abs = df.iloc[top_idx].copy()
    top_abs.to_csv(output_dir / f"top{top_n}_abs_diff_{suffix}.csv")

    # Main matrix output is now one row/column per unique biological gene.
    np.savez_compressed(
        output_dir / f"attention_matrices_{suffix}.npz",
        ctrl=ctrl_gene_attn,
        ir=pert_gene_attn,
        diff=pert_gene_attn - ctrl_gene_attn,
        genes=unique_genes,
        n_collapsed_features=gene_counts.astype(np.int32),
    )

    top_genes = unique_genes[top_idx].tolist()
    sub_diff = (pert_gene_attn - ctrl_gene_attn)[np.ix_(top_idx, top_idx)]
    highlight = [g for g in GENES_OF_INTEREST if g in top_genes]

    plot_heatmap(
        sub_diff,
        top_genes,
        f"scGenePT attention difference ({label} - control), top {top_n_effective} genes",
        output_dir / f"heatmap_{suffix}.png",
        highlight,
    )
    plot_clustermap(
        sub_diff,
        top_genes,
        f"Clustered scGenePT attention difference ({label} - control), top {top_n_effective} genes",
        output_dir / f"clustermap_{suffix}.png",
    )
    plot_network(
        sub_diff,
        top_genes,
        f"Network of positive scGenePT attention shifts ({label} - control), top {top_n_effective} genes",
        output_dir / f"network_{suffix}.png",
        highlight,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path("./results") / f"GRN_scGenePT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_dir.resolve()
    setup_logging(output_dir)

    scgenept_repo = Path(args.scgenept_repo).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else scgenept_repo / "models"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else run_dir / "full_retrained" / "final_model_full_finetuned.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    target_conditions = parse_conditions(args.target_conditions)
    model_type = resolve_model_type(run_dir, args.model_type)

    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("scGenePT repo: %s", scgenept_repo)
    LOGGER.info("models_dir: %s", models_dir)
    LOGGER.info("checkpoint: %s", checkpoint)
    LOGGER.info("model_type: %s", model_type)
    LOGGER.info("target_conditions: %s", target_conditions)

    X, meta, genes = load_saved_expression_and_metadata(run_dir)
    LOGGER.info("Loaded expression matrix %s and metadata %s", X.shape, meta.shape)
    LOGGER.info("Condition counts:\n%s", meta["condition"].value_counts())

    trainmod = import_training_helpers(scgenept_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model, vocab, gene_ids_np = build_model_from_saved_run(
        trainmod,
        genes,
        models_dir,
        model_type,
        checkpoint,
        target_conditions,
        device,
    )

    valid_mask, biological_seq_mask, seq_genes, bio_genes = make_sequence_masks(
        genes, gene_ids_np, vocab, target_conditions
    )
    LOGGER.info("Valid sequence tokens: %d/%d", int(valid_mask.sum()), len(valid_mask))
    LOGGER.info("Biological output genes: %d", len(bio_genes))
    LOGGER.info("Sequence contains dummy tokens: %s", [g for g in target_conditions if g in seq_genes])

    use_amp = (not args.no_amp) and torch.cuda.is_available()

    # Compute control once, then compare against each target condition.
    ctrl_attn, ctrl_influence, ctrl_n = compute_condition_attention(
        X, meta, "ctrl", model, vocab, gene_ids_np, valid_mask, biological_seq_mask,
        seq_genes, device, target_conditions, args.layer_idx, args.batch_size,
        args.max_cells_per_condition, args.seed, use_amp
    )

    summary = {"ctrl_cells_used": int(ctrl_n), "comparisons": {}}
    for cond in target_conditions:
        pert_attn, pert_influence, pert_n = compute_condition_attention(
            X, meta, cond, model, vocab, gene_ids_np, valid_mask, biological_seq_mask,
            seq_genes, device, target_conditions, args.layer_idx, args.batch_size,
            args.max_cells_per_condition, args.seed + 17, use_amp
        )
        save_pair_outputs(
            output_dir,
            cond,
            ctrl_attn,
            pert_attn,
            ctrl_influence,
            pert_influence,
            bio_genes,
            args.top_n,
        )
        summary["comparisons"][cond] = {
            "pert_cells_used": int(pert_n),
            "top_n": int(args.top_n),
            "layer_idx": int(args.layer_idx),
        }
        LOGGER.info("Finished %s vs ctrl", cond)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("All GRN outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
