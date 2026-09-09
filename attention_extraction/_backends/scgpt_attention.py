#!/usr/bin/env python3
"""
Attention-based GRN analysis for a fine-tuned scGPT model.

For each requested irradiation timepoint, this script:
  1. Loads and maps the mouse genes to human gene symbols.
  2. Filters genes to the scGPT vocabulary.
  3. Preprocesses the complete control + irradiated comparison.
  4. Selects highly variable genes using the complete comparison.
  5. Draws a deterministic, approximately balanced sample of cells.
  6. Computes condition-average rank-normalised attention matrices.
  7. Calculates gene influence and IR-minus-control differences.
  8. Saves matrices, tables, plots, sampled-cell metadata, configuration,
     and a complete run.log in a timestamped output directory.

The default cell sample is 3,000 cells per timepoint comparison:
1,500 control and 1,500 irradiated cells when both groups have enough cells.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO, Tuple

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.sparse import issparse
from tqdm import tqdm

from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<cls>", "<eoc>"]
PAD_TOKEN = "<pad>"
PAD_VALUE = -2
N_BINS = 51
N_INPUT_BINS = N_BINS

GENES_OF_INTEREST = [
    "NFIB",
    "NFIA",
    "FOSL2",
    "FOS",
    "JUNB",
    "JUN",
    "BIRC6",
]

TIMEPOINT_CONFIG = {
    "24h": ("ctrl", "IR_24h"),
    "1week": ("ctrl", "IR_1week"),
}

CONDITION_MAP = {
    "pre-IR": "ctrl",
    "24h post-IR": "IR_24h",
    "1 week post-IR": "IR_1week",
}


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

class TeeStream:
    """Write one stream to both the terminal and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self) -> str:
        return getattr(self.streams[0], "encoding", "utf-8")


# -----------------------------------------------------------------------------
# Argument parsing and configuration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compute attention-based scGPT GRN influence for 24 h and 1 week irradiation comparisons."
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=script_dir / "dev_perturb_adamson-Jun28-15-05",
        help="Directory containing best_model.pt.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "save" / "scGPT_human",
        help="Directory containing vocab.json and args.json.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=script_dir / "rna_malignant_counts.h5ad",
        help="Input AnnData file.",
    )
    parser.add_argument(
        "--mouse2human-pkl",
        type=Path,
        default=script_dir / "mouse2human.pkl",
        help="Pickle file containing the mouse-to-human gene mapping.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=script_dir / "results",
        help="Base directory in which the timestamped GRN output directory is created.",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=3000,
        help="Total cells sampled for each control-versus-IR comparison. Use 0 for all cells.",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=3000,
        help="Maximum number of highly variable genes retained.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Attention-computation batch size. Keep at 1 if GPU memory is limited.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=11,
        help="Zero-based transformer layer from which Q and K are extracted.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of genes retained in top-difference tables and plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible cell sampling and network layouts.",
    )
    parser.add_argument(
        "--timepoints",
        nargs="+",
        choices=sorted(TIMEPOINT_CONFIG),
        default=["24h", "1week"],
        help="Timepoint comparisons to run.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution. Without this flag, the script stops if CUDA is unavailable.",
    )

    args = parser.parse_args()

    if args.n_cells < 0:
        parser.error("--n-cells must be 0 or a positive integer.")
    if args.n_hvg <= 0:
        parser.error("--n-hvg must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.layer_idx < 0:
        parser.error("--layer-idx must be non-negative.")
    if args.top_n <= 0:
        parser.error("--top-n must be positive.")

    return args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def make_output_dir(output_base: Path) -> Path:
    output_base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("JOB_ID")
    suffix = f"_job{job_id}" if job_id else ""
    candidate = output_base / f"GRN_{timestamp}{suffix}"

    counter = 1
    while candidate.exists():
        candidate = output_base / f"GRN_{timestamp}{suffix}_{counter}"
        counter += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialisable_config(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    return {
        "model_dir": str(args.model_dir),
        "model_file": str(args.model_dir / "best_model.pt"),
        "data_dir": str(args.data_dir),
        "vocab_file": str(args.data_dir / "vocab.json"),
        "model_config_file": str(args.data_dir / "args.json"),
        "data_file": str(args.data_file),
        "mouse2human_pkl": str(args.mouse2human_pkl),
        "output_dir": str(output_dir),
        "n_cells_per_timepoint": args.n_cells,
        "n_hvg": args.n_hvg,
        "batch_size": args.batch_size,
        "layer_idx": args.layer_idx,
        "top_n": args.top_n,
        "seed": args.seed,
        "timepoints": list(args.timepoints),
        "pad_value": PAD_VALUE,
        "n_bins": N_BINS,
        "genes_of_interest": GENES_OF_INTEREST,
        "job_id": os.environ.get("JOB_ID"),
        "hostname": os.uname().nodename,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def validate_inputs(args: argparse.Namespace) -> None:
    required_files = {
        "model checkpoint": args.model_dir / "best_model.pt",
        "vocabulary": args.data_dir / "vocab.json",
        "model configuration": args.data_dir / "args.json",
        "AnnData input": args.data_file,
        "mouse-to-human mapping": args.mouse2human_pkl,
    }

    missing = [f"{label}: {path}" for label, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input files were not found:\n  " + "\n  ".join(missing))


# -----------------------------------------------------------------------------
# Model and data preparation
# -----------------------------------------------------------------------------

def load_model_and_vocab(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[TransformerModel, GeneVocab, Dict[str, object]]:
    vocab_file = args.data_dir / "vocab.json"
    model_config_file = args.data_dir / "args.json"
    model_file = args.model_dir / "best_model.pt"

    print(f"Loading vocabulary: {vocab_file}")
    vocab = GeneVocab.from_file(vocab_file)
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab.append_token(token)

    with model_config_file.open("r", encoding="utf-8") as handle:
        model_configs = json.load(handle)

    print(f"Loading model configuration: {model_config_file}")
    print(
        "Model architecture: "
        f"embsize={model_configs['embsize']}, "
        f"nheads={model_configs['nheads']}, "
        f"d_hid={model_configs['d_hid']}, "
        f"nlayers={model_configs['nlayers']}"
    )

    model = TransformerModel(
        len(vocab),
        model_configs["embsize"],
        model_configs["nheads"],
        model_configs["d_hid"],
        model_configs["nlayers"],
        vocab=vocab,
        pad_value=PAD_VALUE,
        n_input_bins=N_INPUT_BINS,
        use_fast_transformer=True,
    )

    print(f"Loading checkpoint: {model_file}")
    checkpoint = torch.load(model_file, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint does not contain a valid model state dictionary.")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f"Missing checkpoint keys: {len(incompatible.missing_keys)}")
    print(f"Unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")

    model.to(device)
    model.eval()

    n_layers = len(model.transformer_encoder.layers)
    if args.layer_idx >= n_layers:
        raise IndexError(
            f"Requested layer index {args.layer_idx}, but the loaded model has only "
            f"{n_layers} transformer layers (valid indices: 0-{n_layers - 1})."
        )

    return model, vocab, model_configs


def prepare_base_data(
    data_file: Path,
    mouse2human_pkl: Path,
    vocab: GeneVocab,
) -> sc.AnnData:
    print(f"Reading AnnData: {data_file}")
    adata_raw = sc.read_h5ad(data_file)
    print(f"Raw AnnData shape: {adata_raw.shape}")

    required_obs = ["Timepoint", "Batch"]
    missing_obs = [column for column in required_obs if column not in adata_raw.obs.columns]
    if missing_obs:
        raise KeyError(f"Missing required AnnData .obs columns: {missing_obs}")

    with mouse2human_pkl.open("rb") as handle:
        mouse2human = pickle.load(handle)

    if not isinstance(mouse2human, dict):
        raise TypeError("mouse2human.pkl must contain a dictionary-like mouse-to-human mapping.")

    mouse_genes = adata_raw.var_names.astype(str).tolist()
    valid_mask = np.array([gene in mouse2human for gene in mouse_genes], dtype=bool)
    print(f"Mouse genes with a human mapping: {int(valid_mask.sum())}/{len(valid_mask)}")

    if valid_mask.sum() == 0:
        raise ValueError("No input genes were found in the mouse-to-human mapping.")

    adata = adata_raw[:, valid_mask].copy()
    mapped_human_genes = [str(mouse2human[gene]) for gene in adata.var_names.astype(str)]
    adata.var_names = pd.Index(mapped_human_genes)

    if not adata.var_names.is_unique:
        duplicate_count = int(adata.var_names.duplicated().sum())
        print(f"Mapped human gene symbols contain {duplicate_count} duplicates; making names unique.")
        adata.var_names_make_unique()

    adata.var["gene_name"] = adata.var_names.astype(str)
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    print(f"Genes retained after vocabulary filtering: {adata.n_vars}")

    if adata.n_vars == 0:
        raise ValueError("No mapped genes were present in the model vocabulary.")

    adata.obs["condition"] = adata.obs["Timepoint"].map(CONDITION_MAP)
    adata.obs["str_batch"] = adata.obs["Batch"].astype(str)

    if "GBM_malig_subtypes" in adata.obs.columns:
        adata.obs["cell_type"] = (
            adata.obs["GBM_malig_subtypes"].astype(str).str.replace("-", "_", regex=False)
        )
    else:
        adata.obs["cell_type"] = "unknown"

    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    print("Mapped condition counts:")
    print(adata.obs["condition"].value_counts(dropna=False).to_string())
    return adata


def preprocess_timepoint(
    adata_base: sc.AnnData,
    timepoint: str,
    n_hvg: int,
) -> Tuple[sc.AnnData, str, str]:
    cond_ctrl, cond_ir = TIMEPOINT_CONFIG[timepoint]
    adata = adata_base[adata_base.obs["condition"].isin([cond_ctrl, cond_ir])].copy()

    if adata.n_obs == 0:
        raise ValueError(f"No cells were available for timepoint comparison {timepoint}.")

    print(f"Full subset shape before preprocessing for {timepoint}: {adata.shape}")
    print("Full subset condition counts:")
    print(adata.obs["condition"].value_counts().to_string())

    for condition in (cond_ctrl, cond_ir):
        if int((adata.obs["condition"] == condition).sum()) == 0:
            raise ValueError(f"No cells found for required condition {condition}.")

    preprocessor = Preprocessor(
        use_key=None,
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=False,
        hvg_flavor="seurat_v3",
        binning=N_BINS,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="str_batch")

    n_top_genes = min(n_hvg, adata.n_vars)
    print(f"Selecting up to {n_top_genes} highly variable genes from the full comparison...")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key="str_batch",
        flavor="seurat_v3",
        subset=True,
    )

    print(f"Shape after preprocessing and HVG selection: {adata.shape}")
    return adata, cond_ctrl, cond_ir


# -----------------------------------------------------------------------------
# Cell sampling
# -----------------------------------------------------------------------------

def allocate_balanced_samples(
    available: Dict[str, int],
    conditions: Sequence[str],
    total_requested: int,
) -> Dict[str, int]:
    """Allocate samples as evenly as possible, redistributing shortages."""
    total_available = sum(available[condition] for condition in conditions)
    total_target = min(total_requested, total_available)

    base = total_target // len(conditions)
    allocation = {
        condition: min(base, available[condition])
        for condition in conditions
    }

    remaining = total_target - sum(allocation.values())
    while remaining > 0:
        progressed = False
        for condition in conditions:
            spare = available[condition] - allocation[condition]
            if spare > 0 and remaining > 0:
                allocation[condition] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return allocation


def sample_cells_balanced(
    adata: sc.AnnData,
    conditions: Sequence[str],
    n_cells: int,
    seed: int,
) -> Tuple[sc.AnnData, Dict[str, int], Dict[str, int]]:
    available = {
        condition: int((adata.obs["condition"] == condition).sum())
        for condition in conditions
    }

    if n_cells == 0 or adata.n_obs <= n_cells:
        allocation = available.copy()
        sampled = adata.copy()
        return sampled, available, allocation

    allocation = allocate_balanced_samples(available, conditions, n_cells)
    rng = np.random.default_rng(seed)
    selected_positions: List[int] = []

    condition_array = adata.obs["condition"].to_numpy()
    for condition in conditions:
        positions = np.flatnonzero(condition_array == condition)
        n_select = allocation[condition]
        chosen = rng.choice(positions, size=n_select, replace=False)
        selected_positions.extend(np.sort(chosen).tolist())

    sampled = adata[selected_positions].copy()
    return sampled, available, allocation


def save_sample_metadata(
    adata: sc.AnnData,
    output_file: Path,
) -> None:
    columns = [
        column
        for column in ["condition", "Timepoint", "Batch", "str_batch", "cell_type"]
        if column in adata.obs.columns
    ]
    metadata = adata.obs[columns].copy()
    metadata.insert(0, "cell_id", metadata.index.astype(str))
    metadata.to_csv(output_file, index=False)


# -----------------------------------------------------------------------------
# Attention computation
# -----------------------------------------------------------------------------

def compute_influence(
    adata: sc.AnnData,
    model: TransformerModel,
    vocab: GeneVocab,
    cond_ctrl: str,
    cond_ir: str,
    batch_size: int,
    layer_idx: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    input_layer_key = "X_binned"
    if input_layer_key not in adata.layers:
        raise KeyError(f"Expected preprocessed layer {input_layer_key!r} was not found.")

    all_counts = adata.layers[input_layer_key]
    if issparse(all_counts):
        all_counts = all_counts.toarray()
    all_counts = np.asarray(all_counts)

    genes = adata.var["gene_name"].astype(str).tolist()
    gene_ids = np.asarray(vocab(genes), dtype=np.int64)

    print(
        f"Tokenising {adata.n_obs} cells and {len(genes)} genes "
        f"(sequence length including <cls>: {len(genes) + 1})..."
    )
    tokenized = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=len(genes) + 1,
        vocab=vocab,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        append_cls=True,
        include_zero_gene=True,
    )

    all_gene_ids = torch.as_tensor(tokenized["genes"], dtype=torch.long)
    all_values = torch.as_tensor(tokenized["values"], dtype=torch.float32)
    src_key_padding_mask = all_gene_ids.eq(vocab[PAD_TOKEN])

    condition_ids = adata.obs["condition"].astype(str).to_numpy()
    n_cells = int(all_gene_ids.size(0))
    n_tokens = int(all_gene_ids.size(1))

    sums = {
        cond_ctrl: np.zeros((n_tokens, n_tokens), dtype=np.float32),
        cond_ir: np.zeros((n_tokens, n_tokens), dtype=np.float32),
    }
    counts = {cond_ctrl: 0, cond_ir: 0}

    model.eval()
    autocast_enabled = device.type == "cuda"

    with torch.inference_mode():
        for start in tqdm(
            range(0, n_cells, batch_size),
            desc="Attention batches",
            unit="batch",
            dynamic_ncols=True,
        ):
            end = min(start + batch_size, n_cells)
            current_batch_size = end - start

            input_ids = all_gene_ids[start:end].to(device, non_blocking=True)
            input_values = all_values[start:end].to(device, non_blocking=True)
            padding_mask = src_key_padding_mask[start:end].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                gene_embeddings = model.encoder(input_ids)
                value_embeddings = model.value_encoder(input_values)
                total_embeddings = gene_embeddings + value_embeddings

                for layer in model.transformer_encoder.layers[:layer_idx]:
                    total_embeddings = layer(
                        total_embeddings,
                        src_key_padding_mask=padding_mask,
                    )

                target_layer = model.transformer_encoder.layers[layer_idx]
                self_attention = target_layer.self_attn

                if self_attention.in_proj_weight is None:
                    raise RuntimeError(
                        "The selected attention layer does not expose a combined in_proj_weight. "
                        "This extraction method requires a PyTorch MultiheadAttention-style layer."
                    )

                qkv = F.linear(
                    total_embeddings,
                    self_attention.in_proj_weight,
                    self_attention.in_proj_bias,
                )

                d_model = int(total_embeddings.size(-1))
                n_heads = int(self_attention.num_heads)
                if d_model % n_heads != 0:
                    raise ValueError(
                        f"Embedding dimension {d_model} is not divisible by {n_heads} attention heads."
                    )
                head_dim = d_model // n_heads

                qkv = qkv.reshape(
                    current_batch_size,
                    n_tokens,
                    3,
                    n_heads,
                    head_dim,
                )
                queries = qkv[:, :, 0, :, :]
                keys = qkv[:, :, 1, :, :]

                # Scaling is not required for ranking, because multiplication by a
                # positive constant does not change the order of the scores.
                attention_scores = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    queries,
                    keys,
                )

            # Rank-normalise each row, followed by each column, as in the
            # original analysis implementation.
            attention_scores = attention_scores.reshape(-1, n_tokens)
            row_order = torch.argsort(attention_scores, dim=1)
            row_rank = torch.argsort(row_order, dim=1)
            attention_scores = row_rank.reshape(
                current_batch_size,
                n_heads,
                n_tokens,
                n_tokens,
            ).to(torch.float32) / float(n_tokens)

            attention_scores = attention_scores.permute(0, 1, 3, 2).reshape(-1, n_tokens)
            column_order = torch.argsort(attention_scores, dim=1)
            column_rank = torch.argsort(column_order, dim=1)
            attention_scores = (
                column_rank.reshape(
                    current_batch_size,
                    n_heads,
                    n_tokens,
                    n_tokens,
                ).to(torch.float32)
                / float(n_tokens)
            ).permute(0, 1, 3, 2)

            attention_scores = attention_scores.mean(dim=1)
            attention_numpy = attention_scores.cpu().numpy()

            for batch_position in range(current_batch_size):
                condition = condition_ids[start + batch_position]
                if condition not in sums:
                    raise KeyError(f"Unexpected condition encountered during attention computation: {condition}")
                sums[condition] += attention_numpy[batch_position]
                counts[condition] += 1

            del (
                input_ids,
                input_values,
                padding_mask,
                gene_embeddings,
                value_embeddings,
                total_embeddings,
                qkv,
                queries,
                keys,
                row_order,
                row_rank,
                column_order,
                column_rank,
                attention_scores,
                attention_numpy,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    means: Dict[str, np.ndarray] = {}
    for condition in (cond_ctrl, cond_ir):
        if counts[condition] == 0:
            raise ValueError(f"No cells were processed for condition {condition}.")
        means[condition] = sums[condition] / float(counts[condition])

    print(f"Processed cells by condition: {counts}")

    # Rows are query/source positions and columns are key/target positions.
    # Column sums quantify how strongly each gene is attended to across sources.
    ctrl_influence = means[cond_ctrl].sum(axis=0)[1:]
    ir_influence = means[cond_ir].sum(axis=0)[1:]

    df_ctrl = pd.DataFrame({"influence": ctrl_influence}, index=genes)
    df_ctrl.index.name = "gene"

    df_ir = pd.DataFrame({"influence": ir_influence}, index=genes)
    df_ir.index.name = "gene"

    df_diff = pd.DataFrame(
        {
            "ctrl": ctrl_influence,
            "ir": ir_influence,
            "diff": ir_influence - ctrl_influence,
            "abs_diff": np.abs(ir_influence - ctrl_influence),
        },
        index=genes,
    )
    df_diff.index.name = "gene"

    return df_ctrl, df_ir, df_diff, means[cond_ctrl], means[cond_ir]


# -----------------------------------------------------------------------------
# Output and plotting
# -----------------------------------------------------------------------------

def plot_heatmap(
    difference_matrix: np.ndarray,
    gene_list: Sequence[str],
    title: str,
    output_file: Path,
    highlight_genes: Iterable[str] = (),
) -> None:
    highlight_set = set(highlight_genes)
    plt.figure(figsize=(14, 12))
    axis = sns.heatmap(
        difference_matrix,
        xticklabels=gene_list,
        yticklabels=gene_list,
        cmap="coolwarm",
        center=0,
        square=True,
    )

    for tick in axis.get_xticklabels():
        if tick.get_text() in highlight_set:
            tick.set_color("red")
            tick.set_weight("bold")
    for tick in axis.get_yticklabels():
        if tick.get_text() in highlight_set:
            tick.set_color("red")
            tick.set_weight("bold")

    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_clustermap(
    difference_matrix: np.ndarray,
    gene_list: Sequence[str],
    title: str,
    output_file: Path,
) -> None:
    grid = sns.clustermap(
        difference_matrix,
        xticklabels=gene_list,
        yticklabels=gene_list,
        cmap="coolwarm",
        center=0,
        figsize=(14, 12),
    )
    grid.ax_heatmap.set_title(title)
    grid.fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def plot_network(
    difference_matrix: np.ndarray,
    gene_list: Sequence[str],
    title: str,
    output_file: Path,
    highlight_genes: Iterable[str] = (),
    seed: int = 42,
) -> None:
    graph = nx.DiGraph()
    edges_pool: List[Tuple[str, str, float]] = []

    for row_idx, source_gene in enumerate(gene_list):
        for column_idx, target_gene in enumerate(gene_list):
            weight = float(difference_matrix[row_idx, column_idx])
            if row_idx != column_idx and weight > 0:
                edges_pool.append((source_gene, target_gene, weight))

    if not edges_pool:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No positive attention-difference edges", ha="center", va="center")
        plt.title(title)
        plt.axis("off")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        return

    threshold = float(np.percentile([edge[2] for edge in edges_pool], 85))
    filtered_edges = [edge for edge in edges_pool if edge[2] >= threshold]

    for source, target, weight in filtered_edges:
        graph.add_edge(source, target, weight=weight)

    plt.figure(figsize=(16, 14))
    positions = nx.spring_layout(graph, k=0.4, seed=seed)
    weights = [graph[source][target]["weight"] for source, target in graph.edges()]
    maximum_weight = max(weights) if weights else 1.0
    scaled_widths = [(weight / maximum_weight) * 4.0 for weight in weights]

    highlight_set = set(highlight_genes)
    node_colours = [
        "lightcoral" if gene in highlight_set else "lightblue"
        for gene in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=600,
        node_color=node_colours,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        width=scaled_widths,
        alpha=0.5,
        edge_color="gray",
        arrows=True,
        arrowsize=15,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=10,
        font_weight="bold",
    )

    plt.title(f"{title} (top 15% positive shifts)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_attention_matrices(
    attention_ctrl: np.ndarray,
    attention_ir: np.ndarray,
    genes: Sequence[str],
    timepoint: str,
    output_dir: Path,
) -> Path:
    output_file = output_dir / f"attention_matrices_{timepoint}.npz"
    np.savez_compressed(
        output_file,
        ctrl=attention_ctrl,
        ir=attention_ir,
        diff=attention_ir - attention_ctrl,
        genes=np.asarray(genes, dtype=str),
        token_order=np.asarray(["<cls>", *genes], dtype=str),
    )
    print(f"Saved full average attention matrices: {output_file}")
    return output_file


def process_timepoint(
    timepoint: str,
    adata_base: sc.AnnData,
    model: TransformerModel,
    vocab: GeneVocab,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    print("\n" + "=" * 72)
    print(f"Processing timepoint: {timepoint}")
    print("=" * 72)

    adata_full, cond_ctrl, cond_ir = preprocess_timepoint(
        adata_base,
        timepoint,
        args.n_hvg,
    )

    genes = adata_full.var["gene_name"].astype(str).tolist()
    pd.DataFrame({"gene": genes}).to_csv(
        output_dir / f"gene_order_{timepoint}.csv",
        index=False,
    )

    sampled, available, allocation = sample_cells_balanced(
        adata_full,
        conditions=(cond_ctrl, cond_ir),
        n_cells=args.n_cells,
        seed=args.seed,
    )

    print(f"Available cells before sampling: {available}")
    print(f"Cells selected for attention analysis: {allocation}")
    print(f"Sampled AnnData shape: {sampled.shape}")

    save_sample_metadata(
        sampled,
        output_dir / f"sampled_cells_{timepoint}.csv",
    )

    df_ctrl, df_ir, df_diff, attention_ctrl, attention_ir = compute_influence(
        sampled,
        model,
        vocab,
        cond_ctrl,
        cond_ir,
        args.batch_size,
        args.layer_idx,
        device,
    )

    save_attention_matrices(
        attention_ctrl,
        attention_ir,
        genes,
        timepoint,
        output_dir,
    )

    suffix = f"_{timepoint}"
    df_ctrl.to_csv(output_dir / f"ctrl_influence{suffix}.csv")
    df_ir.to_csv(output_dir / f"ir_influence{suffix}.csv")
    df_diff.to_csv(output_dir / f"diff_influence{suffix}.csv")

    top_n = min(args.top_n, len(df_diff))
    top_absolute = df_diff.sort_values("abs_diff", ascending=False).head(top_n)
    top_absolute.to_csv(output_dir / f"top{top_n}_abs_diff{suffix}.csv")

    difference_without_cls = (attention_ir - attention_ctrl)[1:, 1:]
    top_genes = top_absolute.index.astype(str).tolist()
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    selected_indices = [gene_to_index[gene] for gene in top_genes]
    sub_difference = difference_without_cls[np.ix_(selected_indices, selected_indices)]
    highlighted = [gene for gene in GENES_OF_INTEREST if gene in top_genes]

    plot_heatmap(
        sub_difference,
        top_genes,
        f"Attention difference (IR - control): top {top_n} genes\n{timepoint}",
        output_dir / f"heatmap_{timepoint}.png",
        highlighted,
    )
    plot_clustermap(
        sub_difference,
        top_genes,
        f"Clustered attention difference (IR - control): top {top_n} genes\n{timepoint}",
        output_dir / f"clustermap_{timepoint}.png",
    )
    plot_network(
        sub_difference,
        top_genes,
        f"Positive attention-difference network: top {top_n} genes\n{timepoint}",
        output_dir / f"network_{timepoint}.png",
        highlighted,
        seed=args.seed,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Finished {timepoint}. Outputs saved in: {output_dir}")
    return {
        "timepoint": timepoint,
        "control_condition": cond_ctrl,
        "irradiated_condition": cond_ir,
        "available_cells": available,
        "sampled_cells": allocation,
        "total_sampled_cells": int(sampled.n_obs),
        "genes": int(sampled.n_vars),
        "top_n": top_n,
    }


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def run_analysis(args: argparse.Namespace, output_dir: Path) -> None:
    validate_inputs(args)
    set_reproducibility(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable. This attention analysis is extremely expensive on CPU. "
            "Submit the GPU job or pass --allow-cpu deliberately."
        )

    print(f"Output directory: {output_dir}")
    print(f"Run log: {output_dir / 'run.log'}")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {total_memory:.2f} GiB")

    config = serialisable_config(args, output_dir)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    model, vocab, model_configs = load_model_and_vocab(args, device)
    config["loaded_model_config"] = model_configs
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    adata_base = prepare_base_data(
        args.data_file,
        args.mouse2human_pkl,
        vocab,
    )

    summaries = []
    for timepoint in args.timepoints:
        summaries.append(
            process_timepoint(
                timepoint,
                adata_base,
                model,
                vocab,
                args,
                output_dir,
                device,
            )
        )

    summary = {
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "timepoints": summaries,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nAll requested GRN analyses completed successfully.")
    print(f"Final output directory: {output_dir}")


def main() -> int:
    args = parse_args()

    args.model_dir = resolve_path(args.model_dir)
    args.data_dir = resolve_path(args.data_dir)
    args.data_file = resolve_path(args.data_file)
    args.mouse2human_pkl = resolve_path(args.mouse2human_pkl)
    args.output_base = resolve_path(args.output_base)

    output_dir = make_output_dir(args.output_base)
    log_file = output_dir / "run.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_file.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    exit_code = 0
    try:
        run_analysis(args, output_dir)
    except KeyboardInterrupt:
        exit_code = 130
        print("\nAnalysis interrupted by user.", file=sys.stderr)
    except Exception:
        exit_code = 1
        print("\nGRN analysis failed with the following exception:", file=sys.stderr)
        traceback.print_exc()
        failure_summary = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(output_dir),
        }
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(failure_summary, handle, indent=2)
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()

    if exit_code != 0:
        print(f"GRN analysis exited with status {exit_code}. See: {log_file}", file=original_stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
