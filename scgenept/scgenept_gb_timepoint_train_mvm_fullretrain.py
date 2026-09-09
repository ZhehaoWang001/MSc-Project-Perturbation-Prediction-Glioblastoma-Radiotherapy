#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune scGenePT on GB irradiation perturbation data with two timepoints.
This variant uses scGPT-style masked value modeling (MVM).

Run this from the scGenePT repository root on Myriad, for example:

python scgenept_gb_timepoint_train.py \
  --h5ad_path data/radiotherapyGB/perturb_processed.h5ad \
  --models_dir models \
  --model_type scgenept_bioall_gpt_concat \
  --init_from pretrained \
  --batch_size 32 \
  --eval_batch_size 32

Expected model layout from the scGenePT repository:

models/
  pretrained/scgpt/{args.json,vocab.json,best_model.pt}
  gene_embeddings/

If you explicitly want to warm-start from a CZI model-zoo perturbation checkpoint,
also download the relevant finetuned folder and add, for example:

  --init_from finetuned --finetuned_dataset norman

  finetuned/scgenept_go_c/norman/best_model_gpt3.5_ada_rnd_seed_42_concat.pt

BioAll option:
  --model_type scgenept_bioall_gpt_concat

This is a convenience alias for the official scGenePT combined model type:
  scgenept_ncbi+uniprot_gpt_go_all_gpt_concat
"""

import argparse
import copy
import json
import logging
import os
import random
import shutil
import sys
import time
import types
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.stats import entropy, pearsonr, ttest_ind, wasserstein_distance
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.loader import DataLoader

from gears import PertData

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models"))

try:
    from models.scGenePT import scGenePT
except ImportError:
    from scGenePT import scGenePT

try:
    from utils.data_loading import *  # noqa: F401,F403 - scGenePT exposes constants here.
except ImportError as exc:
    raise ImportError(
        "Could not import scGenePT utils. Run this script from the scGenePT "
        "repository root, or place it inside the cloned repository."
    ) from exc

warnings.filterwarnings("ignore")
matplotlib.rcParams["savefig.transparent"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scGenePT_GB_Timepoint")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(save_dir):
    log_path = save_dir / "run.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    logger.info("Writing run log to %s", log_path)
    return log_path


def add_extra_file_logger(log_path):
    """Add a temporary extra logger, for example full_retrained/run.log."""
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    return file_handler


def remove_extra_file_logger(file_handler):
    root_logger = logging.getLogger()
    root_logger.removeHandler(file_handler)
    file_handler.close()


def to_jsonable(obj):
    """Recursively convert numpy/torch scalars and arrays to JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


# Public/local experiment names accepted by this script.
#
# BioAll is a convenience alias for the official scGenePT combined model type:
#     scgenept_ncbi+uniprot_gpt_go_all_gpt_concat
# which uses NCBI+UniProt gene-text embeddings together with GO_all embeddings.
BIOALL_ALIAS = "scgenept_bioall_gpt_concat"
BIOALL_INTERNAL_MODEL_TYPE = "scgenept_ncbi+uniprot_gpt_go_all_gpt_concat"

MODEL_FILE_BY_TYPE = {
    "scgpt": "best_model_seed_42.pt",
    "scgenept_ncbi_gpt": "best_model_gpt3.5_ada_rnd_seed_42.pt",
    "scgenept_ncbi+uniprot_gpt": "best_model_gpt3.5_ada_rnd_seed_42.pt",
    "scgenept_go_c_gpt_concat": "best_model_gpt3.5_ada_rnd_seed_42_concat.pt",
    "scgenept_go_f_gpt_concat": "best_model_gpt3.5_ada_rnd_seed_42_concat.pt",
    "scgenept_go_p_gpt_concat": "best_model_gpt3.5_ada_rnd_seed_42_concat.pt",
    "scgenept_go_all_gpt_concat": "best_model_gpt3.5_ada_rnd_seed_42_concat.pt",
    BIOALL_ALIAS: "best_model_gpt3.5_ada_rnd_seed_42_concat.pt",
    BIOALL_INTERNAL_MODEL_TYPE: "best_model_gpt3.5_ada_rnd_seed_42_concat.pt",
}

MODEL_DIR_BY_TYPE = {
    "scgpt": "scgpt",
    "scgenept_ncbi_gpt": "scgenept_ncbi",
    "scgenept_ncbi+uniprot_gpt": "scgenept_ncbi+uniprot",
    "scgenept_go_c_gpt_concat": "scgenept_go_c",
    "scgenept_go_f_gpt_concat": "scgenept_go_f",
    "scgenept_go_p_gpt_concat": "scgenept_go_p",
    "scgenept_go_all_gpt_concat": "scgenept_go_all",
    # The official public model zoo may not include a pre-finetuned BioAll folder.
    # This only matters for --init_from finetuned. For your recommended
    # --init_from pretrained run, the base scGPT checkpoint is used instead.
    BIOALL_ALIAS: "scgenept_ncbi+uniprot_gpt_go_all_gpt_concat",
    BIOALL_INTERNAL_MODEL_TYPE: "scgenept_ncbi+uniprot_gpt_go_all_gpt_concat",
}


def canonical_model_type(model_type: str) -> str:
    """Return the model_type string understood by scGenePT utility functions."""
    if model_type == BIOALL_ALIAS:
        return BIOALL_INTERNAL_MODEL_TYPE
    return model_type


def display_model_type(model_type: str) -> str:
    """Human-readable model-type label for logs and saved metadata."""
    if model_type == BIOALL_ALIAS:
        return f"{BIOALL_ALIAS} ({BIOALL_INTERNAL_MODEL_TYPE})"
    return model_type


def parse_args():
    parser = argparse.ArgumentParser(description="scGenePT GB irradiation training")
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="radiotherapyGB")
    parser.add_argument(
        "--h5ad_path",
        type=str,
        default="./data/radiotherapyGB/perturb_processed.h5ad",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="scGenePT models root containing pretrained/, gene_embeddings/, finetuned/.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="scgenept_go_c_gpt_concat",
        choices=sorted(MODEL_FILE_BY_TYPE.keys()),
    )
    parser.add_argument(
        "--finetuned_dataset",
        type=str,
        default="norman",
        help=(
            "Only used with --init_from finetuned. This is the CZI model-zoo "
            "checkpoint subfolder, not your GB dataset name."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint file. Overrides model_type-derived path.",
    )
    parser.add_argument(
        "--init_from",
        choices=["finetuned", "pretrained", "none"],
        default="pretrained",
        help=(
            "pretrained: base scGPT weights for your GB fine-tuning; "
            "finetuned: optional CZI Norman/Adamson model-zoo warm start; "
            "none: random init."
        ),
    )
    parser.add_argument("--target_conditions", type=str, default="IR-24h,IR-1week")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--schedule_interval", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Print training speed/loss every N batches.",
    )
    parser.add_argument(
        "--full_val_interval",
        type=int,
        default=1,
        help="Compute full eight validation metrics every N epochs. Use 0 to disable.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_genes", type=int, default=4096)
    parser.add_argument(
        "--include_zero_gene",
        choices=["all", "batch-wise"],
        default="all",
        help="MVM input gene selection. Use all genes or only genes nonzero in the batch.",
    )
    parser.add_argument(
        "--mvm_probability",
        type=float,
        default=0.20,
        help="Random fraction of input expression values masked for MVM.",
    )
    parser.add_argument(
        "--de_loss_weight",
        type=float,
        default=5.0,
        help="Extra loss weight for DE genes forced into the MVM mask.",
    )
    parser.add_argument(
        "--mean_response_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Auxiliary loss on condition-level mean predicted response. "
            "This directly supports delta metrics; set 0 for pure MVM."
        ),
    )
    parser.add_argument(
        "--no_force_de_mask",
        action="store_true",
        help="Disable always masking top DE genes during MVM.",
    )
    parser.add_argument("--graph_num_samples", type=int, default=5)
    parser.add_argument(
        "--retrain_full",
        action="store_true",
        help=(
            "After final held-out test evaluation, continue fine-tuning the "
            "validation-selected checkpoint on train+val+test. The resulting "
            "checkpoint is saved under full_retrained/ and is intended only "
            "for downstream GRN/embedding/attention analysis, not for reported "
            "test performance."
        ),
    )
    parser.add_argument(
        "--full_retrain_epochs",
        type=int,
        default=5,
        help="Number of post-test full-dataset fine-tuning epochs for GRN analysis.",
    )
    parser.add_argument(
        "--full_retrain_lr_factor",
        type=float,
        default=0.5,
        help="Full-data retraining LR = lr * this factor.",
    )
    parser.add_argument(
        "--full_retrain_schedule_interval",
        type=int,
        default=1,
        help="StepLR schedule interval used during full-data retraining.",
    )
    parser.add_argument(
        "--full_retrain_gamma",
        type=float,
        default=0.9,
        help="StepLR gamma used during full-data retraining.",
    )
    parser.add_argument(
        "--full_retrain_metric_interval",
        type=int,
        default=1,
        help=(
            "Compute and log descriptive full-dataset fit metrics every N full-retrain "
            "epochs. Use 0 to disable. These are resubstitution metrics, not held-out "
            "performance estimates."
        ),
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--use_fast_transformer", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_condition(value):
    """Canonical condition label used by the scGPT comparison pipeline."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().replace("_", "-")


def parse_target_conditions(value):
    return [canonical_condition(x.strip()) for x in value.split(",") if x.strip()]


def ensure_gene_name_column(adata):
    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str)
    adata.var["gene_name"] = adata.var["gene_name"].astype(str)
    return adata


def add_missing_timepoint_genes(adata, target_conditions):
    """Add binary condition pseudo-genes matching the scGPT pipeline.

    Existing underscore- or hyphen-named dummy variants are removed first.
    Duplicate biological feature indices are made unique only at the AnnData
    ``var_names`` level; the original biological symbols in ``gene_name`` are
    preserved for scGenePT embedding lookup and downstream reporting.
    """
    adata = ensure_gene_name_column(adata.copy())

    canonical_targets = [canonical_condition(c) for c in target_conditions]
    target_conditions = list(dict.fromkeys(canonical_targets))
    if len(target_conditions) != len(canonical_targets):
        raise ValueError(
            f"Target conditions are duplicated after canonicalisation: {canonical_targets}"
        )

    # Canonicalisation maps both underscore and hyphen variants to these labels.
    target_set = set(target_conditions)
    gene_name_values = adata.var["gene_name"].astype(str).to_numpy()
    var_name_values = adata.var_names.astype(str).to_numpy()
    gene_name_canonical = np.asarray(
        [canonical_condition(value) for value in gene_name_values],
        dtype=str,
    )
    var_name_canonical = np.asarray(
        [canonical_condition(value) for value in var_name_values],
        dtype=str,
    )

    keep_mask = ~(
        np.isin(gene_name_canonical, list(target_set))
        | np.isin(var_name_canonical, list(target_set))
    )
    removed = int((~keep_mask).sum())
    if removed:
        logger.info("Removing %d existing condition pseudo-gene column(s).", removed)

    adata = adata[:, keep_mask].copy()
    adata = ensure_gene_name_column(adata)

    # Some mapped biological genes have duplicated AnnData indices.  That is
    # unrelated to pseudo-gene duplication.  Make only the internal var index
    # unique while retaining the original symbols in adata.var['gene_name'].
    if not adata.var_names.is_unique:
        duplicate_count = int(adata.var_names.duplicated().sum())
        logger.warning(
            "Found %d duplicate biological var_names after removing condition "
            "tokens; making internal AnnData var_names unique while preserving "
            "the gene_name column.",
            duplicate_count,
        )
        adata.var_names_make_unique()

    obs_conditions = (
        adata.obs["condition"]
        .astype(str)
        .map(canonical_condition)
        .to_numpy()
    )

    condition_columns = []
    for condition in target_conditions:
        values = (obs_conditions == condition).astype(np.float32).reshape(-1, 1)
        if sparse.issparse(adata.X):
            values = sparse.csr_matrix(values)
        condition_columns.append(values)

    if sparse.issparse(adata.X):
        X = sparse.hstack([adata.X] + condition_columns, format="csr")
    else:
        X = np.hstack([np.asarray(adata.X)] + condition_columns)

    var_extra = pd.DataFrame(index=pd.Index(target_conditions, dtype=str))
    for column in adata.var.columns:
        var_extra[column] = ""
    var_extra["gene_name"] = target_conditions

    var = pd.concat([adata.var.copy(), var_extra], axis=0)
    if not var.index.is_unique:
        duplicated = var.index[var.index.duplicated()].astype(str).unique().tolist()
        raise ValueError(
            "Internal feature indices remain duplicated after cleanup: "
            f"{duplicated[:20]}"
        )

    new_adata = sc.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=var,
        uns=copy.deepcopy(adata.uns),
        obsm=copy.deepcopy(adata.obsm),
        obsp=copy.deepcopy(adata.obsp),
    )

    pseudo_counts = (
        new_adata.var["gene_name"]
        .astype(str)
        .map(canonical_condition)
        .value_counts()
    )
    invalid_counts = {
        condition: int(pseudo_counts.get(condition, 0))
        for condition in target_conditions
        if int(pseudo_counts.get(condition, 0)) != 1
    }
    if invalid_counts:
        raise ValueError(
            "Expected exactly one feature for each condition pseudo-gene, but got "
            f"{invalid_counts}."
        )

    logger.info(
        "Added binary condition pseudo-genes %s; final shape=%s.",
        target_conditions,
        new_adata.shape,
    )
    return new_adata

def resolve_checkpoint(models_dir, model_type, finetuned_dataset, checkpoint_path, init_from):
    models_dir = Path(models_dir)
    if checkpoint_path:
        return Path(checkpoint_path)
    if init_from == "none":
        return None
    if init_from == "pretrained":
        return models_dir / "pretrained" / "scgpt" / "best_model.pt"
    model_dir = MODEL_DIR_BY_TYPE[model_type]
    model_file = MODEL_FILE_BY_TYPE[model_type]
    return models_dir / "finetuned" / model_dir / finetuned_dataset / model_file


def load_torch_state(path, device):
    try:
        state = torch.load(path, weights_only=True, map_location=device)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return state


def maybe_convert_flash_attention_keys(state, use_fast_transformer):
    if use_fast_transformer:
        return state
    return {k.replace("Wqkv.", "in_proj_"): v for k, v in state.items()}


def flexible_load_state_dict(model, state):
    """Load exact tensors and expand first-dimension vocab tensors when needed."""
    model_state = model.state_dict()
    merged = {}
    skipped = []

    for raw_key, value in state.items():
        key = raw_key.replace("module.", "")
        if key not in model_state:
            skipped.append((raw_key, tuple(value.shape), "missing key"))
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            merged[key] = value
            continue
        can_expand_dim0 = (
            value.ndim == target.ndim
            and value.ndim > 0
            and value.shape[0] <= target.shape[0]
            and tuple(value.shape[1:]) == tuple(target.shape[1:])
        )
        if can_expand_dim0:
            expanded = target.clone()
            expanded[: value.shape[0]] = value.to(expanded.dtype)
            merged[key] = expanded
        else:
            skipped.append((raw_key, tuple(value.shape), f"target {tuple(target.shape)}"))

    model_state.update(merged)
    model.load_state_dict(model_state, strict=False)
    logger.info("Loaded %d tensors from checkpoint; skipped %d.", len(merged), len(skipped))
    if skipped:
        logger.info("First skipped tensors: %s", skipped[:8])


def build_scgenept_model(adata, args, target_conditions, device):
    models_dir = Path(args.models_dir)
    vocab_file = models_dir / "pretrained" / "scgpt" / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(
            f"Missing {vocab_file}. Download with:\n"
            f"aws s3 sync --no-sign-request "
            f"s3://czi-scgenept-public/models/pretrained/scgpt {models_dir / 'pretrained' / 'scgpt'}"
        )

    embedding_model_type = canonical_model_type(args.model_type)
    if embedding_model_type != args.model_type:
        logger.info(
            "Using BioAll alias: %s -> %s",
            args.model_type,
            embedding_model_type,
        )
    embs_to_include = get_embs_to_include(embedding_model_type)  # noqa: F405
    vocab, _, dataset_genes, _ = match_genes_to_scgpt_vocab_from_adata(  # noqa: F405
        str(vocab_file), adata, SPECIAL_TOKENS  # noqa: F405
    )

    for token in target_conditions:
        if token not in vocab:
            vocab.append_token(token)

    genept_embs, genept_emb_type, genept_emb_dim, _ = initialize_genept_embeddings(  # noqa: F405
        embs_to_include, dataset_genes, vocab, embedding_model_type, str(models_dir) + "/"
    )
    go_embs_to_include, go_emb_type, go_emb_dim, _ = initialize_go_embeddings(  # noqa: F405
        embs_to_include, dataset_genes, vocab, embedding_model_type, str(models_dir) + "/"
    )

    model = scGenePT(
        ntoken=len(vocab),
        d_model=EMBSIZE,  # noqa: F405
        nhead=NHEAD,  # noqa: F405
        d_hid=D_HID,  # noqa: F405
        nlayers=NLAYERS,  # noqa: F405
        nlayers_cls=N_LAYERS_CLS,  # noqa: F405
        n_cls=N_CLS,  # noqa: F405
        vocab=vocab,
        n_perturbagens=2,
        dropout=args.dropout,
        pad_token=PAD_TOKEN,  # noqa: F405
        pad_value=PAD_VALUE,  # noqa: F405
        pert_pad_id=PERT_PAD_ID,  # noqa: F405
        use_fast_transformer=args.use_fast_transformer,
        embs_to_include=embs_to_include,
        genept_embs=genept_embs,
        genept_emb_type=genept_emb_type,
        genept_emb_size=genept_emb_dim,
        go_embs_to_include=go_embs_to_include,
        go_emb_type=go_emb_type,
        go_emb_size=go_emb_dim,
    )

    checkpoint = resolve_checkpoint(
        models_dir,
        args.model_type,
        args.finetuned_dataset,
        args.checkpoint_path,
        args.init_from,
    )
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}\n"
                f"For model-zoo weights, download the folder with:\n"
                f"aws s3 sync --no-sign-request "
                f"s3://czi-scgenept-public/models/finetuned/"
                f"{MODEL_DIR_BY_TYPE[args.model_type]} "
                f"{models_dir / 'finetuned' / MODEL_DIR_BY_TYPE[args.model_type]}"
            )
        logger.info("Loading %s checkpoint from %s", args.init_from, checkpoint)
        state = load_torch_state(checkpoint, device)
        state = maybe_convert_flash_attention_keys(state, args.use_fast_transformer)
        flexible_load_state_dict(model, state)
    else:
        logger.info("Using randomly initialized scGenePT weights.")

    gene_names = adata.var["gene_name"].astype(str).tolist()
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab[PAD_TOKEN] for g in gene_names],  # noqa: F405
        dtype=np.int64,
    )
    logger.info(
        "Tokenized %d genes/timepoint tokens; %d map to <pad>.",
        len(gene_ids),
        int((gene_ids == vocab[PAD_TOKEN]).sum()),  # noqa: F405
    )
    return model, vocab, gene_ids


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
    """Compute the reference local nearest-neighbour energy statistic."""
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

    d_pp = cdist(pred_sub, pred_sub, metric="euclidean")
    d_pq = cdist(pred_sub, truth_sub, metric="euclidean")
    d_qq = cdist(truth_sub, truth_sub, metric="euclidean")

    mean_d_p_for_pred = _mean_k_smallest(d_pp, k)
    mean_d_q_for_pred = _mean_k_smallest(d_pq, k)
    pred_terms = mean_d_q_for_pred - mean_d_p_for_pred

    mean_d_p_for_truth = _mean_k_smallest(d_pq.T, k)
    mean_d_q_for_truth = _mean_k_smallest(d_qq, k)
    truth_terms = mean_d_p_for_truth - mean_d_q_for_truth

    return float(
        (pred_terms.sum() + truth_terms.sum()) / float(n_pred + n_truth)
    )


def compute_metrics_no_auprc(
    pred,
    truth,
    ctrl_mean,
    top_k=100,
    k_neighbors=30,
    n_sample=400,
    seed=42,
):
    """Compute the seven non-AUPRC reference metrics for one condition."""
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

    cross_dist = cdist(pred_sample, truth_sample, metric="euclidean").mean()
    within_pred = cdist(pred_sample, pred_sample, metric="euclidean").mean()
    within_truth = cdist(truth_sample, truth_sample, metric="euclidean").mean()
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
        "Wasserstein": float(np.mean(wasser_per_gene)),
        "KL-divergence": float(np.mean(kl_per_gene)),
        "Common-DEGs": float(common_deg),
        "Local-E-distance": float(local_e_distance),
    }


def compute_auprc_trapezoidal(pred_cond, truth_cond, eval_ctrl_expr, train_ctrl_mean):
    """Compute the reference Welch-label, trapezoidal gene-level AUPRC."""
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

    _, p_values = ttest_ind(
        truth_cond,
        eval_ctrl_expr,
        axis=0,
        equal_var=False,
        nan_policy="omit",
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
    auprc_value = auc(recall, precision)
    return float(auprc_value), int(labels.sum()), int(n_genes)


def compute_full_metrics(pred, truth, ctrl_mean, eval_ctrl_expr, top_k=100):
    """Compute the unified eight metrics using the reference definitions."""
    metrics = compute_metrics_no_auprc(
        pred,
        truth,
        ctrl_mean,
        top_k=top_k,
        k_neighbors=30,
        n_sample=400,
        seed=42,
    )
    auprc_value, n_positive, n_total = compute_auprc_trapezoidal(
        pred,
        truth,
        eval_ctrl_expr,
        ctrl_mean,
    )
    metrics["AUPRC"] = float(auprc_value)
    return metrics, n_positive, n_total


def compute_pearson_metrics(pred, truth, ctrl_mean, top_k=20):
    pred_mean = pred.mean(axis=0)
    truth_mean = truth.mean(axis=0)
    p_all = np.corrcoef(pred_mean, truth_mean)[0, 1] if len(pred_mean) > 1 else 0.0
    pred_delta = pred_mean - ctrl_mean
    truth_delta = truth_mean - ctrl_mean
    p_delta = np.corrcoef(pred_delta, truth_delta)[0, 1] if len(pred_delta) > 1 else 0.0
    top_k = min(top_k, len(truth_delta))
    de_idx = np.argsort(np.abs(truth_delta))[-top_k:]
    p_de = np.corrcoef(pred_mean[de_idx], truth_mean[de_idx])[0, 1] if len(de_idx) > 1 else 0.0
    p_delta_de = (
        np.corrcoef(pred_delta[de_idx], truth_delta[de_idx])[0, 1]
        if len(de_idx) > 1
        else 0.0
    )
    return {
        "pearson": p_all,
        "pearson_de": p_de,
        "pearson_delta": p_delta,
        "pearson_delta_de": p_delta_de,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    args,
    epoch,
    vocab,
    gene_ids,
    de_ids,
    dummy_vocab_ids=None,
):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_response = 0.0
    interval_loss = 0.0
    interval_mse = 0.0
    interval_response = 0.0
    interval_start = time.time()
    pad_token_id = vocab[PAD_TOKEN]  # noqa: F405
    de_ids_tensor = None
    if de_ids is not None and len(de_ids) > 0:
        de_ids_tensor = torch.as_tensor(de_ids, device=device, dtype=torch.long)
    dummy_ids_tensor = None
    if dummy_vocab_ids is not None and len(dummy_vocab_ids) > 0:
        dummy_ids_tensor = torch.as_tensor(dummy_vocab_ids, device=device, dtype=torch.long)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        x = batch.x
        batch_size = len(batch.y)
        ori_vals = x[:, 0].view(batch_size, -1)
        pert_flags = x[:, 1].long().view(len(batch.y), -1)
        target = batch.y
        batch_conditions = [canonical_condition(p) for p in batch.pert]

        n_genes = ori_vals.shape[1]
        if args.include_zero_gene == "all":
            input_gene_pos = torch.arange(n_genes, device=device, dtype=torch.long)
        else:
            input_gene_pos = ori_vals.nonzero()[:, 1].flatten().unique().sort()[0]
            if input_gene_pos.numel() == 0:
                input_gene_pos = torch.arange(n_genes, device=device, dtype=torch.long)

        if input_gene_pos.numel() > args.max_genes:
            mapped_for_sampling = gene_ids[input_gene_pos].to(device)
            if dummy_ids_tensor is not None:
                is_dummy_for_sampling = torch.isin(mapped_for_sampling, dummy_ids_tensor)
            else:
                is_dummy_for_sampling = torch.zeros_like(mapped_for_sampling, dtype=torch.bool)

            dummy_pos = input_gene_pos[is_dummy_for_sampling]
            bio_pos = input_gene_pos[~is_dummy_for_sampling]
            n_keep_bio = max(args.max_genes - dummy_pos.numel(), 0)
            if bio_pos.numel() > n_keep_bio:
                bio_pos = bio_pos[torch.randperm(bio_pos.numel(), device=device)[:n_keep_bio]]
            input_gene_pos = torch.cat([dummy_pos, bio_pos])
            input_gene_pos = input_gene_pos[torch.randperm(input_gene_pos.numel(), device=device)]

        input_vals = ori_vals[:, input_gene_pos]
        input_pert = pert_flags[:, input_gene_pos]
        target_vals = target[:, input_gene_pos]
        mapped_ids_1d = gene_ids[input_gene_pos].to(device)
        mapped_ids = mapped_ids_1d.repeat(batch_size, 1)

        src_key_padding_mask = mapped_ids == pad_token_id
        rand_mask = torch.rand_like(input_vals) < args.mvm_probability
        if de_ids_tensor is not None and not args.no_force_de_mask:
            de_mask = torch.isin(mapped_ids, de_ids_tensor)
        else:
            de_mask = torch.zeros_like(input_vals, dtype=torch.bool)

        masked_pos = rand_mask | de_mask
        masked_pos = masked_pos & (mapped_ids != pad_token_id)
        response_gene_mask = mapped_ids[0] != pad_token_id
        if dummy_ids_tensor is not None:
            is_dummy = torch.isin(mapped_ids, dummy_ids_tensor)
            masked_pos = masked_pos & ~is_dummy
            response_gene_mask = response_gene_mask & ~torch.isin(mapped_ids[0], dummy_ids_tensor)

        masked_vals = input_vals.clone()
        masked_vals[masked_pos] = 0.0

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            out = model(
                mapped_ids,
                masked_vals,
                input_pert,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            sq_error = (out["mlm_output"] - target_vals) ** 2
            masked_sq_error = sq_error * masked_pos
            mask_sum = masked_pos.sum()
            if mask_sum > 0:
                loss_mse = masked_sq_error.sum() / mask_sum
                if de_ids_tensor is not None and args.de_loss_weight != 1.0:
                    loss_weights = torch.ones_like(input_vals)
                    loss_weights[torch.isin(mapped_ids, de_ids_tensor)] = args.de_loss_weight
                    weighted_error = masked_sq_error * loss_weights
                    loss = weighted_error.sum() / ((masked_pos * loss_weights).sum() + 1e-8)
                else:
                    loss = loss_mse
            else:
                loss = masked_sq_error.sum() * 0.0
                loss_mse = loss

            response_loss = loss.new_tensor(0.0)
            response_terms = 0
            if args.mean_response_loss_weight > 0 and response_gene_mask.any():
                output_values = out["mlm_output"]
                for cond in sorted(set(batch_conditions)):
                    cond_mask = torch.as_tensor(
                        [p == cond for p in batch_conditions],
                        device=device,
                        dtype=torch.bool,
                    )
                    if cond_mask.sum() == 0:
                        continue
                    pred_mean = output_values[cond_mask][:, response_gene_mask].mean(dim=0)
                    target_mean = target_vals[cond_mask][:, response_gene_mask].mean(dim=0)
                    response_loss = response_loss + torch.mean((pred_mean - target_mean) ** 2)
                    response_terms += 1
                if response_terms > 0:
                    response_loss = response_loss / response_terms
                    loss = loss + args.mean_response_loss_weight * response_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_value = loss.item()
        mse_value = loss_mse.item()
        response_value = response_loss.item()
        total_loss += loss_value
        total_mse += mse_value
        total_response += response_value
        interval_loss += loss_value
        interval_mse += mse_value
        interval_response += response_value

        if args.log_interval > 0 and batch_idx % args.log_interval == 0:
            elapsed = time.time() - interval_start
            ms_per_batch = elapsed * 1000 / args.log_interval
            cur_loss = interval_loss / args.log_interval
            cur_mse = interval_mse / args.log_interval
            cur_response = interval_response / args.log_interval
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "| epoch %3d | %3d/%3d batches | lr %.6g | ms/batch %.5f | "
                "loss %.5f | mse %.5f | response %.5f |",
                epoch,
                batch_idx,
                len(loader),
                cur_lr,
                ms_per_batch,
                cur_loss,
                cur_mse,
                cur_response,
            )
            interval_loss = 0.0
            interval_mse = 0.0
            interval_response = 0.0
            interval_start = time.time()

    denom = max(len(loader), 1)
    return total_loss / denom, total_mse / denom, total_response / denom


def evaluate(loader, model, device, gene_ids):
    model.eval()
    if torch.is_tensor(gene_ids):
        gene_ids = gene_ids.detach().cpu().numpy()
    else:
        gene_ids = np.asarray(gene_ids)
    all_pred, all_truth, all_pert = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model.pred_perturb(batch, include_zero_gene="all", gene_ids=gene_ids)
            truth = batch.y
            min_dim = min(pred.shape[1], truth.shape[1])
            all_pred.append(pred[:, :min_dim].cpu().numpy().astype(np.float32))
            all_truth.append(truth[:, :min_dim].cpu().numpy().astype(np.float32))
            all_pert.extend([canonical_condition(p) for p in batch.pert])
    return np.vstack(all_pred), np.vstack(all_truth), np.array(all_pert)


def evaluate_per_condition(
    loader,
    model,
    device,
    gene_ids,
    ctrl_mean,
    target_conditions,
    full=False,
    biological_gene_mask=None,
    eval_ctrl_expr=None,
):
    pred_all, truth_all, pert_cat = evaluate(loader, model, device, gene_ids)
    ctrl_mean = np.asarray(ctrl_mean)
    dimension_candidates = [pred_all.shape[1], truth_all.shape[1], len(ctrl_mean)]

    if full:
        if eval_ctrl_expr is None:
            raise ValueError(
                "eval_ctrl_expr is required for full metrics because AUPRC uses "
                "observed controls from the same evaluation split."
            )
        eval_ctrl_expr = np.asarray(eval_ctrl_expr, dtype=np.float64)
        if eval_ctrl_expr.ndim != 2 or eval_ctrl_expr.shape[0] == 0:
            raise ValueError("eval_ctrl_expr must be a non-empty 2-D array.")
        dimension_candidates.append(eval_ctrl_expr.shape[1])

    if biological_gene_mask is not None:
        biological_gene_mask = np.asarray(biological_gene_mask, dtype=bool)
        dimension_candidates.append(len(biological_gene_mask))

    min_dim = min(dimension_candidates)
    pred_all = pred_all[:, :min_dim]
    truth_all = truth_all[:, :min_dim]
    ctrl_mean = ctrl_mean[:min_dim]
    if full:
        eval_ctrl_expr = eval_ctrl_expr[:, :min_dim]
    if biological_gene_mask is not None:
        biological_gene_mask = biological_gene_mask[:min_dim]
        pred_all = pred_all[:, biological_gene_mask]
        truth_all = truth_all[:, biological_gene_mask]
        ctrl_mean = ctrl_mean[biological_gene_mask]
        if full:
            eval_ctrl_expr = eval_ctrl_expr[:, biological_gene_mask]

    results = {}

    for cond in target_conditions:
        mask = pert_cat == cond
        if not np.any(mask):
            logger.warning("No cells found for condition %s during evaluation.", cond)
            continue
        pred = pred_all[mask]
        truth = truth_all[mask]
        if full:
            metrics, n_positive, n_total = compute_full_metrics(
                pred,
                truth,
                ctrl_mean,
                eval_ctrl_expr,
            )
            results[cond] = metrics
            logger.info(
                "%s AUPRC Welch labels: %d/%d genes positive "
                "(uncorrected p < 0.05).",
                cond,
                n_positive,
                n_total,
            )
        else:
            results[cond] = compute_pearson_metrics(pred, truth, ctrl_mean)

    avg = {}
    if results:
        for key in results[next(iter(results))]:
            avg[key] = float(np.mean([results[c][key] for c in results]))
    results["average"] = avg
    return results




def evaluate_all_metric_sets_per_condition(
    loader,
    model,
    device,
    gene_ids,
    ctrl_mean,
    target_conditions,
    biological_gene_mask=None,
    eval_ctrl_expr=None,
):
    """Evaluate once and return both Pearson and full eight-metric dictionaries."""
    pred_all, truth_all, pert_cat = evaluate(loader, model, device, gene_ids)
    ctrl_mean = np.asarray(ctrl_mean)
    if eval_ctrl_expr is None:
        raise ValueError(
            "eval_ctrl_expr is required because AUPRC uses observed controls from "
            "the same evaluation population."
        )
    eval_ctrl_expr = np.asarray(eval_ctrl_expr, dtype=np.float64)
    if eval_ctrl_expr.ndim != 2 or eval_ctrl_expr.shape[0] == 0:
        raise ValueError("eval_ctrl_expr must be a non-empty 2-D array.")

    dimension_candidates = [
        pred_all.shape[1],
        truth_all.shape[1],
        len(ctrl_mean),
        eval_ctrl_expr.shape[1],
    ]
    if biological_gene_mask is not None:
        biological_gene_mask = np.asarray(biological_gene_mask, dtype=bool)
        dimension_candidates.append(len(biological_gene_mask))

    min_dim = min(dimension_candidates)
    pred_all = pred_all[:, :min_dim]
    truth_all = truth_all[:, :min_dim]
    ctrl_mean = ctrl_mean[:min_dim]
    eval_ctrl_expr = eval_ctrl_expr[:, :min_dim]
    if biological_gene_mask is not None:
        biological_gene_mask = biological_gene_mask[:min_dim]
        pred_all = pred_all[:, biological_gene_mask]
        truth_all = truth_all[:, biological_gene_mask]
        ctrl_mean = ctrl_mean[biological_gene_mask]
        eval_ctrl_expr = eval_ctrl_expr[:, biological_gene_mask]

    pearson_results = {}
    full_results = {}
    for cond in target_conditions:
        cond_mask = pert_cat == cond
        if not np.any(cond_mask):
            logger.warning("No cells found for condition %s during evaluation.", cond)
            continue
        pred = pred_all[cond_mask]
        truth = truth_all[cond_mask]
        pearson_results[cond] = compute_pearson_metrics(pred, truth, ctrl_mean)
        metrics, n_positive, n_total = compute_full_metrics(
            pred,
            truth,
            ctrl_mean,
            eval_ctrl_expr,
        )
        full_results[cond] = metrics
        logger.info(
            "%s AUPRC Welch labels: %d/%d genes positive "
            "(uncorrected p < 0.05).",
            cond,
            n_positive,
            n_total,
        )

    for result_dict in (pearson_results, full_results):
        average = {}
        if result_dict:
            first_condition = next(iter(result_dict))
            for key in result_dict[first_condition]:
                average[key] = float(
                    np.mean([result_dict[c][key] for c in result_dict])
                )
        result_dict["average"] = average

    return pearson_results, full_results


def log_pearson_metrics_block(title, metrics, target_conditions):
    """Log per-condition and average Pearson-style metrics."""
    logger.info("%s", title)
    for cond in target_conditions:
        if cond not in metrics:
            continue
        m = metrics[cond]
        logger.info(
            "%-10s -> Pearson: %.4f | DE: %.4f | Delta: %.4f | DE_Delta: %.4f",
            cond,
            m["pearson"],
            m["pearson_de"],
            m["pearson_delta"],
            m["pearson_delta_de"],
        )
    if metrics.get("average"):
        m = metrics["average"]
        logger.info(
            "%-10s -> Pearson: %.4f | DE: %.4f | Delta: %.4f | DE_Delta: %.4f",
            "AVERAGE",
            m["pearson"],
            m["pearson_de"],
            m["pearson_delta"],
            m["pearson_delta_de"],
        )


def log_eight_metrics_block(title, metrics, target_conditions, note=None):
    """Log the eight full perturbation-prediction metrics."""
    logger.info("%s", title)
    if note:
        logger.info("%s", note)
    for cond in target_conditions:
        if cond not in metrics:
            continue
        m = metrics[cond]
        logger.info(
            "%-10s -> MSE: %.4f | PCC-delta: %.4f | E-dist: %.4f | "
            "Wasserstein: %.4f | KL: %.4f | Common-DEGs: %.4f | "
            "AUPRC: %.4f | Local-E: %.4f",
            cond,
            m["MSE"],
            m["PCC-delta"],
            m["E-distance"],
            m["Wasserstein"],
            m["KL-divergence"],
            m["Common-DEGs"],
            m["AUPRC"],
            m["Local-E-distance"],
        )
    if metrics.get("average"):
        m = metrics["average"]
        logger.info(
            "%-10s -> MSE: %.4f | PCC-delta: %.4f | E-dist: %.4f | "
            "Wasserstein: %.4f | KL: %.4f | Common-DEGs: %.4f | "
            "AUPRC: %.4f | Local-E: %.4f",
            "AVERAGE",
            m["MSE"],
            m["PCC-delta"],
            m["E-distance"],
            m["Wasserstein"],
            m["KL-divergence"],
            m["Common-DEGs"],
            m["AUPRC"],
            m["Local-E-distance"],
        )


def plot_perturbation(query, save_dir, top_genes=20):
    """Plot biological top-DE genes only, with strict feature alignment."""
    X_train = np.load(os.path.join(save_dir, "train_expression.npy"))
    X_test = np.load(os.path.join(save_dir, "test_expression.npy"))

    meta_train = pd.read_csv(
        os.path.join(save_dir, "train_metadata.csv"),
        index_col=0,
    )
    meta_test = pd.read_csv(
        os.path.join(save_dir, "test_metadata.csv"),
        index_col=0,
    )

    with open(
        os.path.join(save_dir, "gene_names.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        gene_names_full = np.asarray(json.load(handle), dtype=str)

    dummy_genes = np.asarray(
        ["IR-24h", "IR-1week", "IR_24h", "IR_1week"],
        dtype=str,
    )
    biological_gene_mask = ~np.isin(gene_names_full, dummy_genes)

    if X_train.shape[0] != len(meta_train):
        raise ValueError(
            f"train_expression has {X_train.shape[0]} rows, but "
            f"train_metadata.csv has {len(meta_train)} rows."
        )
    if X_test.shape[0] != len(meta_test):
        raise ValueError(
            f"test_expression has {X_test.shape[0]} rows, but "
            f"test_metadata.csv has {len(meta_test)} rows."
        )
    if X_train.shape[1] != len(gene_names_full):
        raise ValueError(
            f"train_expression has {X_train.shape[1]} columns, "
            f"but gene_names.json has {len(gene_names_full)} entries."
        )
    if X_test.shape[1] != len(gene_names_full):
        raise ValueError(
            f"test_expression has {X_test.shape[1]} columns, "
            f"but gene_names.json has {len(gene_names_full)} entries."
        )

    gene_names = gene_names_full[biological_gene_mask]
    X_train = X_train[:, biological_gene_mask]
    X_test = X_test[:, biological_gene_mask]

    query = canonical_condition(query)
    train_conditions = np.asarray(
        [canonical_condition(x) for x in meta_train["condition"]],
        dtype=str,
    )
    test_conditions = np.asarray(
        [canonical_condition(x) for x in meta_test["condition"]],
        dtype=str,
    )

    ctrl_mask = train_conditions == "ctrl"
    test_mask = test_conditions == query

    if not np.any(ctrl_mask):
        raise ValueError("No training control cells found.")
    if not np.any(test_mask):
        raise ValueError(f"No test cells found for {query}.")

    ctrl_mean = X_train[ctrl_mask].mean(axis=0)
    truth = X_test[test_mask]

    results = np.load(
        os.path.join(save_dir, "test_results.npz"),
        allow_pickle=True,
    )
    preds_all = np.asarray(results["pred"])
    pert_cat = np.asarray(
        [canonical_condition(x) for x in results["pert_cat"]],
        dtype=str,
    )

    if preds_all.shape[0] != len(pert_cat):
        raise ValueError(
            f"Prediction matrix has {preds_all.shape[0]} rows, but pert_cat "
            f"contains {len(pert_cat)} labels."
        )

    if preds_all.shape[1] == len(gene_names_full):
        preds_all = preds_all[:, biological_gene_mask]
    elif preds_all.shape[1] == int(biological_gene_mask.sum()):
        pass
    else:
        raise ValueError(
            f"Prediction matrix has {preds_all.shape[1]} columns; expected "
            f"{len(gene_names_full)} full features or "
            f"{int(biological_gene_mask.sum())} biological genes."
        )

    pred_mask = pert_cat == query
    if not np.any(pred_mask):
        raise ValueError(f"No predictions found for {query}.")
    pred = preds_all[pred_mask]

    if pred.shape[1] != len(gene_names):
        raise ValueError(
            f"Prediction has {pred.shape[1]} biological genes, "
            f"but gene list has {len(gene_names)}."
        )

    truth_delta = truth.mean(axis=0) - ctrl_mean
    pred_delta = pred.mean(axis=0) - ctrl_mean

    top_k = min(int(top_genes), len(gene_names))
    if top_k < 1:
        raise ValueError("No biological genes are available for plotting.")
    top_idx = np.argsort(np.abs(truth_delta))[-top_k:]
    de_idx = top_idx[np.argsort(truth_delta[top_idx])]
    plot_genes = gene_names[de_idx]

    logger.info(
        "Plotting %s using biological genes only: %d/%d retained; top_k=%d.",
        query,
        len(gene_names),
        len(gene_names_full),
        top_k,
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(plot_genes))
    width = 0.35
    ax.bar(
        x - width / 2,
        truth_delta[de_idx],
        width,
        label="True delta",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        pred_delta[de_idx],
        width,
        label="Predicted delta",
        color="coral",
        alpha=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_genes, rotation=90)
    ax.axhline(0, color="black", linestyle="--")
    ax.set_ylabel("Change over control")
    ax.set_title(f"Perturbation: {query}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{query}_barplot.png"),
        dpi=150,
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 5))
    truth_spread = truth[:, de_idx] - ctrl_mean[de_idx]
    ax.boxplot(
        truth_spread,
        showfliers=False,
        medianprops=dict(linewidth=1.6, color="black"),
    )
    ax.scatter(
        np.arange(1, len(plot_genes) + 1),
        pred_delta[de_idx],
        color="red",
        zorder=5,
        s=80,
    )
    ax.axhline(
        0,
        linestyle="dashed",
        color="green",
        alpha=0.45,
    )
    ax.set_xticks(np.arange(1, len(plot_genes) + 1))
    ax.set_xticklabels(
        plot_genes,
        rotation=90,
        fontstyle="italic",
        fontsize=10,
    )
    ax.set_ylabel("Log Expression Change\n(Perturbed - Control)")
    ax.set_title(
        f"scGenePT Prediction vs Truth: {query}\n"
        f"(Top {top_k} DE Genes)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{query}_boxplot.png"),
        dpi=150,
    )
    plt.close()


def main():
    args = parse_args()
    set_seed(42)
    save_dir = Path(args.save_dir) / f"scGenePT_{args.data_name}_{time.strftime('%b%d-%H%M')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(save_dir)
    logger.info("Run arguments: %s", vars(args))

    target_conditions = parse_target_conditions(args.target_conditions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    cache_dir = Path(args.data_dir) / args.data_name / "processed"
    if cache_dir.exists():
        logger.info("Clearing old GEARS cache at %s", cache_dir)
        shutil.rmtree(cache_dir)

    adata = sc.read_h5ad(args.h5ad_path)
    ensure_gene_name_column(adata)
    adata.obs["condition"] = adata.obs["condition"].map(canonical_condition)
    adata = add_missing_timepoint_genes(adata, target_conditions)
    logger.info("Loaded AnnData after pseudo-gene check: %s", adata.shape)
    logger.info("Conditions: %s", sorted(adata.obs["condition"].unique()))

    if "ctrl" not in set(adata.obs["condition"]):
        raise ValueError("No 'ctrl' condition found in adata.obs['condition'].")
    missing_conditions = [c for c in target_conditions if c not in set(adata.obs["condition"])]
    if missing_conditions:
        raise ValueError(f"Target conditions not present in adata: {missing_conditions}")

    pert_data = PertData(args.data_dir)
    pert_data.adata = adata
    pert_data.dataset_path = str(Path(args.h5ad_path).parent)
    pert_data.dataset_name = Path(args.h5ad_path).stem
    pert_data.gene_names = adata.var["gene_name"].astype(str).values
    pert_data.node_map = {g: i for i, g in enumerate(pert_data.gene_names)}
    timepoint_gene_idx = {cond: pert_data.node_map[cond] for cond in target_conditions}

    def timepoint_get_pert_idx(self, pert_category, *unused_args, **unused_kwargs):
        pert_category = canonical_condition(pert_category)
        if pert_category == "ctrl":
            return None
        if pert_category not in timepoint_gene_idx:
            raise KeyError(f"Unknown perturbation condition: {pert_category}")
        return [timepoint_gene_idx[pert_category]]

    pert_data.get_pert_idx = types.MethodType(timepoint_get_pert_idx, pert_data)

    # Exact 80/10/10 split used by the scGPT comparison pipeline:
    # condition-only stratification with random_state=1.
    indices = np.arange(adata.n_obs)
    strata = adata.obs["condition"].astype(str)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=1,
        stratify=strata,
    )
    temp_strata = strata.iloc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=1,
        stratify=temp_strata,
    )

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    adata.obs["split"] = "train"
    adata.obs.iloc[
        val_idx,
        adata.obs.columns.get_loc("split"),
    ] = "val"
    adata.obs.iloc[
        test_idx,
        adata.obs.columns.get_loc("split"),
    ] = "test"

    np.savez(
        save_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    logger.info(
        "Split configuration: global training seed=42, split random_state=1, "
        "stratification=condition only."
    )
    logger.info("Split sizes:\n%s", adata.obs.groupby(["split", "condition"]).size())

    def build_graphs_from_adata(split_adata, split_label):
        split_ctrl = split_adata[split_adata.obs["condition"] == "ctrl"].copy()
        if split_ctrl.n_obs == 0:
            raise ValueError(f"Graph set '{split_label}' has no control cells.")

        old_ctrl = getattr(pert_data, "ctrl_adata", None)
        pert_data.ctrl_adata = split_ctrl
        graphs = []
        for cond in target_conditions:
            n_cells = int((split_adata.obs["condition"] == cond).sum())
            if n_cells == 0:
                logger.info("No %s cells in graph set %s; skipping.", cond, split_label)
                continue
            logger.info("Building graphs for %s/%s: %d cells", split_label, cond, n_cells)
            graphs.extend(
                pert_data.create_cell_graph_dataset(
                    split_adata=split_adata,
                    pert_category=cond,
                    num_samples=args.graph_num_samples,
                )
            )
        if old_ctrl is not None:
            pert_data.ctrl_adata = old_ctrl
        elif hasattr(pert_data, "ctrl_adata"):
            delattr(pert_data, "ctrl_adata")
        return graphs

    def build_split_graphs(split_name):
        split_adata = adata[adata.obs["split"] == split_name].copy()
        return build_graphs_from_adata(split_adata, split_name)

    train_graphs = build_split_graphs("train")
    val_graphs = build_split_graphs("val")
    test_graphs = build_split_graphs("test")
    if not train_graphs or not val_graphs or not test_graphs:
        raise RuntimeError("One or more graph splits are empty; check target conditions and split.")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_graphs, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.eval_batch_size, shuffle=False)

    ctrl_X = adata[(adata.obs["split"] == "train") & (adata.obs["condition"] == "ctrl")].X
    if sparse.issparse(ctrl_X):
        ctrl_X = ctrl_X.toarray()
    ctrl_mean_full = np.asarray(ctrl_X).mean(axis=0)

    def observed_control_expression(split_name):
        """Return observed controls for reference AUPRC labels in one split."""
        control_X = adata[
            (adata.obs["split"] == split_name)
            & (adata.obs["condition"] == "ctrl")
        ].X
        if sparse.issparse(control_X):
            control_X = control_X.toarray()
        control_X = np.asarray(control_X, dtype=np.float64)
        if control_X.ndim != 2 or control_X.shape[0] == 0:
            raise ValueError(
                f"Split '{split_name}' has no observed control cells for AUPRC labels."
            )
        return control_X

    val_ctrl_expr_full = observed_control_expression("val")
    test_ctrl_expr_full = observed_control_expression("test")
    logger.info(
        "Unified metric definitions: Local-E uses independently sampled P/Q "
        "populations (up to 400 each), k=min(30,nP-1,nQ-1), full biological-"
        "gene Euclidean space, and includes zero self-distance. AUPRC uses "
        "same-split observed controls, uncorrected Welch p<0.05 labels, and "
        "trapezoidal PR-curve integration."
    )

    model, vocab, gene_ids_np = build_scgenept_model(adata, args, target_conditions, device)
    gene_ids_torch = torch.as_tensor(gene_ids_np, dtype=torch.long, device=device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    gene_names_list = pert_data.gene_names.tolist()
    dummy_vocab_ids = [vocab[cond] for cond in target_conditions if cond in vocab]
    dummy_gene_variants = np.asarray(
        ["IR-24h", "IR-1week", "IR_24h", "IR_1week"],
        dtype=str,
    )
    biological_gene_mask = ~np.isin(
        np.asarray(gene_names_list, dtype=str),
        dummy_gene_variants,
    )
    logger.info(
        "MVM setup: probability=%.3f, include_zero_gene=%s, force_de_mask=%s, "
        "de_loss_weight=%.2f, biological genes=%d/%d, dummy vocab ids=%s",
        args.mvm_probability,
        args.include_zero_gene,
        not args.no_force_de_mask,
        args.de_loss_weight,
        int(biological_gene_mask.sum()),
        len(biological_gene_mask),
        dummy_vocab_ids,
    )

    de_ids = []
    if "rank_genes_groups_cov_all" in adata.uns:
        for values in adata.uns["rank_genes_groups_cov_all"].values():
            for gene in values[:50]:
                if gene in gene_names_list:
                    idx = gene_names_list.index(gene)
                    de_ids.append(int(gene_ids_np[idx]))
    de_ids = np.unique(de_ids) if de_ids else None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    best_val = -np.inf
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        loss, train_mse, train_response = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args,
            epoch,
            vocab,
            gene_ids_torch,
            de_ids,
            dummy_vocab_ids=dummy_vocab_ids,
        )
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        val_metrics = evaluate_per_condition(
            val_loader,
            raw_model,
            device,
            gene_ids_np,
            ctrl_mean_full,
            target_conditions,
            biological_gene_mask=biological_gene_mask,
        )
        val_score = val_metrics.get("average", {}).get("pearson_delta", -np.inf)

        logger.info("\n%s [Epoch %d - Validation] %s", "-" * 20, epoch, "-" * 20)
        logger.info(
            "Train MVM loss: %.6f | Train MVM mse: %.6f | Train response loss: %.6f",
            loss,
            train_mse,
            train_response,
        )
        for cond in target_conditions:
            if cond in val_metrics:
                m = val_metrics[cond]
                logger.info(
                    "%-10s -> Pearson: %.4f | DE: %.4f | Delta: %.4f | DE_Delta: %.4f",
                    cond,
                    m["pearson"],
                    m["pearson_de"],
                    m["pearson_delta"],
                    m["pearson_delta_de"],
                )
        if val_metrics["average"]:
            avg = val_metrics["average"]
            logger.info(
                "%-10s -> Pearson: %.4f | DE: %.4f | Delta: %.4f | DE_Delta: %.4f",
                "AVERAGE",
                avg["pearson"],
                avg["pearson_de"],
                avg["pearson_delta"],
                avg["pearson_delta_de"],
            )
        logger.info("%s", "-" * 64)

        if args.full_val_interval > 0 and epoch % args.full_val_interval == 0:
            val_full_metrics = evaluate_per_condition(
                val_loader,
                raw_model,
                device,
                gene_ids_np,
                ctrl_mean_full,
                target_conditions,
                full=True,
                biological_gene_mask=biological_gene_mask,
                eval_ctrl_expr=val_ctrl_expr_full,
            )
            log_eight_metrics_block(
                "Full eight metrics on validation set:",
                val_full_metrics,
                target_conditions,
            )
            logger.info("%s", "-" * 64)

        if val_score > best_val:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict(best_state)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    test_metrics = evaluate_per_condition(
        test_loader,
        raw_model,
        device,
        gene_ids_np,
        ctrl_mean_full,
        target_conditions,
        full=True,
        biological_gene_mask=biological_gene_mask,
        eval_ctrl_expr=test_ctrl_expr_full,
    )

    logger.info("\n%s", "=" * 72)
    log_eight_metrics_block(
        "TEST SET FULL EIGHT METRICS",
        test_metrics,
        target_conditions,
    )
    logger.info("%s", "=" * 72)

    torch.save(best_state, save_dir / "best_model.pt")
    pred_test, truth_test, pert_cat = evaluate(test_loader, raw_model, device, gene_ids_np)
    np.savez(save_dir / "test_results.npz", pred=pred_test, truth=truth_test, pert_cat=pert_cat)

    for split in ["train", "val", "test"]:
        idx = adata.obs[adata.obs["split"] == split].index
        X = adata[idx].X
        if sparse.issparse(X):
            X = X.toarray()
        np.save(save_dir / f"{split}_expression.npy", X)
        adata[idx].obs.to_csv(save_dir / f"{split}_metadata.csv")

    with open(save_dir / "gene_names.json", "w", encoding="utf-8") as handle:
        json.dump(pert_data.gene_names.tolist(), handle)

    test_serializable = {
        cond: {k: float(v) for k, v in metrics.items()}
        for cond, metrics in test_metrics.items()
    }
    with open(save_dir / "test_metrics_full.json", "w", encoding="utf-8") as handle:
        json.dump(test_serializable, handle, indent=2)

    for cond in target_conditions:
        try:
            plot_perturbation(cond, save_dir)
        except Exception as exc:
            logger.error("Plotting failed for %s: %s", cond, exc)

    # ------------------------------------------------------------------
    # Optional post-test full-dataset retraining for GRN/interpretability
    # ------------------------------------------------------------------
    if args.retrain_full:
        retrain_dir = save_dir / "full_retrained"
        retrain_dir.mkdir(parents=True, exist_ok=True)
        full_log_handler = add_extra_file_logger(retrain_dir / "run.log")
        try:
            logger.info("\n%s", "=" * 72)
            logger.info("POST-TEST FULL-DATASET RETRAINING FOR GRN ANALYSIS")
            logger.info("%s", "=" * 72)
            logger.info(
                "This checkpoint is trained after held-out test evaluation and must not "
                "be used for reported test performance."
            )
            logger.info("Full retrain output directory: %s", retrain_dir)

            # Preserve the original cell-level split provenance before using all cells.
            split_cols = [c for c in ["condition", "split", "cell_type", "batch"] if c in adata.obs.columns]
            adata.obs[split_cols].to_csv(retrain_dir / "original_split_assignments.csv")

            with open(retrain_dir / "run_args.json", "w", encoding="utf-8") as handle:
                json.dump(to_jsonable(vars(args)), handle, indent=2)
            with open(retrain_dir / "target_conditions.json", "w", encoding="utf-8") as handle:
                json.dump(target_conditions, handle, indent=2)
            with open(retrain_dir / "gene_names.json", "w", encoding="utf-8") as handle:
                json.dump(pert_data.gene_names.tolist(), handle, indent=2)
            with open(retrain_dir / "dummy_genes.json", "w", encoding="utf-8") as handle:
                json.dump(target_conditions, handle, indent=2)
            np.save(retrain_dir / "gene_ids.npy", gene_ids_np)
            np.save(retrain_dir / "biological_gene_mask.npy", biological_gene_mask.astype(bool))

            # Reload the validation-selected model state. This guarantees that the
            # full-data run starts from exactly the model used for test evaluation.
            model.load_state_dict(best_state)
            full_graphs = build_graphs_from_adata(adata.copy(), "full_train_val_test")
            if not full_graphs:
                raise RuntimeError("Full-dataset graph list is empty; cannot retrain for GRN.")
            full_loader = DataLoader(
                full_graphs,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
            # Separate deterministic loader for descriptive fit metrics. It uses the
            # same full dataset and therefore is not a held-out evaluation set.
            full_eval_loader = DataLoader(
                full_graphs,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
            )
            full_ctrl_X = adata[adata.obs["condition"] == "ctrl"].X
            if sparse.issparse(full_ctrl_X):
                full_ctrl_X = full_ctrl_X.toarray()
            full_ctrl_expr = np.asarray(full_ctrl_X, dtype=np.float64)
            ctrl_mean_all_full = full_ctrl_expr.mean(axis=0)

            full_lr = args.lr * args.full_retrain_lr_factor
            optimizer_full = torch.optim.Adam(model.parameters(), lr=full_lr)
            scheduler_full = torch.optim.lr_scheduler.StepLR(
                optimizer_full,
                args.full_retrain_schedule_interval,
                gamma=args.full_retrain_gamma,
            )
            scaler_full = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

            full_history = []
            last_full_fit_epoch = None
            last_full_fit_pearson = None
            last_full_fit_metrics = None
            for full_epoch in range(1, args.full_retrain_epochs + 1):
                epoch_start = time.time()
                loss, train_mse, train_response = train_epoch(
                    model,
                    full_loader,
                    optimizer_full,
                    scaler_full,
                    device,
                    args,
                    full_epoch,
                    vocab,
                    gene_ids_torch,
                    de_ids,
                    dummy_vocab_ids=dummy_vocab_ids,
                )
                scheduler_full.step()
                elapsed = time.time() - epoch_start
                epoch_record = {
                    "epoch": full_epoch,
                    "loss": float(loss),
                    "mvm_mse": float(train_mse),
                    "response_loss": float(train_response),
                    "lr": float(optimizer_full.param_groups[0]["lr"]),
                    "seconds": float(elapsed),
                }
                logger.info(
                    "| full retrain epoch %3d/%3d | time %.2fs | loss %.6f | "
                    "mse %.6f | response %.6f | lr %.6g |",
                    full_epoch,
                    args.full_retrain_epochs,
                    elapsed,
                    loss,
                    train_mse,
                    train_response,
                    optimizer_full.param_groups[0]["lr"],
                )

                if (
                    args.full_retrain_metric_interval > 0
                    and full_epoch % args.full_retrain_metric_interval == 0
                ):
                    raw_full_model = model.module if isinstance(model, nn.DataParallel) else model
                    full_fit_pearson, full_fit_metrics = evaluate_all_metric_sets_per_condition(
                        full_eval_loader,
                        raw_full_model,
                        device,
                        gene_ids_np,
                        ctrl_mean_all_full,
                        target_conditions,
                        biological_gene_mask=biological_gene_mask,
                        eval_ctrl_expr=full_ctrl_expr,
                    )
                    last_full_fit_epoch = full_epoch
                    last_full_fit_pearson = full_fit_pearson
                    last_full_fit_metrics = full_fit_metrics
                    logger.info("\n%s", "-" * 72)
                    log_pearson_metrics_block(
                        f"[Full retrain epoch {full_epoch}] FULL-DATA FIT PEARSON METRICS",
                        full_fit_pearson,
                        target_conditions,
                    )
                    log_eight_metrics_block(
                        f"[Full retrain epoch {full_epoch}] FULL-DATA FIT EIGHT METRICS",
                        full_fit_metrics,
                        target_conditions,
                        note=(
                            "Descriptive resubstitution metrics on train+val+test; "
                            "do not report these as held-out predictive performance."
                        ),
                    )
                    logger.info("%s", "-" * 72)
                    epoch_record["full_data_fit_pearson"] = to_jsonable(full_fit_pearson)
                    epoch_record["full_data_fit_eight_metrics"] = to_jsonable(full_fit_metrics)

                full_history.append(epoch_record)

            # Always retain a final descriptive fit block. Reuse the last
            # interval result when it was computed after the final epoch.
            raw_full_model = model.module if isinstance(model, nn.DataParallel) else model
            if last_full_fit_epoch == args.full_retrain_epochs:
                final_full_fit_pearson = last_full_fit_pearson
                final_full_fit_metrics = last_full_fit_metrics
            else:
                final_full_fit_pearson, final_full_fit_metrics = evaluate_all_metric_sets_per_condition(
                    full_eval_loader,
                    raw_full_model,
                    device,
                    gene_ids_np,
                    ctrl_mean_all_full,
                    target_conditions,
                    biological_gene_mask=biological_gene_mask,
                    eval_ctrl_expr=full_ctrl_expr,
                )
            logger.info("\n%s", "=" * 72)
            log_pearson_metrics_block(
                "FINAL FULL-DATA FIT PEARSON METRICS",
                final_full_fit_pearson,
                target_conditions,
            )
            log_eight_metrics_block(
                "FINAL FULL-DATA FIT EIGHT METRICS",
                final_full_fit_metrics,
                target_conditions,
                note=(
                    "Descriptive resubstitution metrics on train+val+test; "
                    "do not report these as held-out predictive performance."
                ),
            )
            logger.info("%s", "=" * 72)
            with open(retrain_dir / "full_dataset_fit_metrics_final.json", "w", encoding="utf-8") as handle:
                json.dump(
                    to_jsonable(
                        {
                            "pearson_metrics": final_full_fit_pearson,
                            "eight_metrics": final_full_fit_metrics,
                            "warning": (
                                "Resubstitution metrics computed on the full train+val+test "
                                "dataset; not a held-out performance estimate."
                            ),
                        }
                    ),
                    handle,
                    indent=2,
                )

            full_state = copy.deepcopy(
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            )
            torch.save(full_state, retrain_dir / "final_model_full_finetuned.pt")
            torch.save(
                {
                    "model_state_dict": full_state,
                    "source_validation_selected_checkpoint": str(save_dir / "best_model.pt"),
                    "purpose": "post-test full-data model for GRN/embedding/attention analysis only",
                    "not_for_reported_test_performance": True,
                    "model_type": args.model_type,
                    "canonical_model_type": canonical_model_type(args.model_type),
                    "target_conditions": target_conditions,
                    "train_history": full_history,
                    "final_full_data_fit_pearson": to_jsonable(final_full_fit_pearson),
                    "final_full_data_fit_eight_metrics": to_jsonable(final_full_fit_metrics),
                    "args": to_jsonable(vars(args)),
                },
                retrain_dir / "final_model_full_finetuned_checkpoint.pt",
            )
            with open(retrain_dir / "full_retrain_history.json", "w", encoding="utf-8") as handle:
                json.dump(to_jsonable(full_history), handle, indent=2)
            with open(retrain_dir / "model_config.json", "w", encoding="utf-8") as handle:
                json.dump(
                    to_jsonable(
                        {
                            "model_type": args.model_type,
                            "canonical_model_type": canonical_model_type(args.model_type),
                            "init_from": "validation_selected_best_model.pt",
                            "source_best_model": str(save_dir / "best_model.pt"),
                            "n_genes_or_tokens": int(len(gene_ids_np)),
                            "n_biological_genes": int(biological_gene_mask.sum()),
                            "dummy_timepoint_tokens": target_conditions,
                            "mvm_probability": args.mvm_probability,
                            "de_loss_weight": args.de_loss_weight,
                            "mean_response_loss_weight": args.mean_response_loss_weight,
                            "full_retrain_epochs": args.full_retrain_epochs,
                            "full_retrain_lr": full_lr,
                            "full_retrain_metric_interval": args.full_retrain_metric_interval,
                            "final_full_data_fit_metrics_file": "full_dataset_fit_metrics_final.json",
                        }
                    ),
                    handle,
                    indent=2,
                )

            logger.info("Saved full-data GRN checkpoint to %s", retrain_dir / "final_model_full_finetuned.pt")
            logger.info("Saved full-data retrain provenance files to %s", retrain_dir)
            logger.info("%s", "=" * 72)
        finally:
            remove_extra_file_logger(full_log_handler)

    logger.info("All results saved to %s", save_dir)


if __name__ == "__main__":
    main()
