#!/usr/bin/env python
# coding: utf-8

# # Fine-tuned scGPT embedding gene programs and pathway interpretation
# 
# This notebook is deliberately self-contained. All project analysis code is
# shown in the cells below and runs directly in PyCharm; it does not import a
# separate project-specific Python analysis module.
# 
# The notebook answers three distinct questions:
# 
# 1. Which genes are close in the fine-tuned scGPT embedding space?
# 2. Which gene programs and pathways change after IR in the observed data?
# 3. Does the perturbation checkpoint predict the same direction when the exact
#    same control cell is rerun with the 24-hour or one-week perturbation flag?
# 
# The source data are rna_malignant_counts.h5ad. Saved train/test split arrays and
# saved prediction NPZ files are not used as analysis inputs.

# ## Important terminology
# 
# An embedding gene program is what the previous notebook called a module and
# what the scGPT tutorial calls a metagene or gene program. It is a cluster of
# genes with similar learned gene-token embeddings.
# 
# It is not a cell cluster, a curated pathway, a directed transcription-factor
# network, or proof of causal regulation. The embedding graph is therefore best
# used to organise genes into candidate programs. Pathway enrichment supplies the
# biological interpretation.
# 
# The all-gene graph is retained later only as a supplementary record. It uses the
# exact selected Scanpy embedding-neighbour graph and a deterministic force
# layout, not embedding PCA. Because a roughly 2,400-gene graph is intrinsically
# dense, the all-program response atlas and observed-versus-model scatterplots are
# the recommended presentation figures.

# ## 1. Imports, editable paths and settings

# In[1]:


from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
import hashlib
import json
import os
import pickle
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/scgpt_embedding_mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/scgpt_embedding_numba")

import anndata as ad
import gseapy as gp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from IPython.display import display
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import hypergeom, pearsonr, spearmanr, ttest_1samp, ttest_ind
from sklearn.metrics import adjusted_rand_score, silhouette_score
import torch
import torch.nn.functional as torch_f

DUMMY_TOKENS = {"IR_24h", "IR_1week", "IR-24h", "IR-1week", "<pad>", "<cls>", "<eoc>"}

@dataclass
class LouvainAnalysisConfig:
    """Configuration for the notebook's checkpoint-aligned Louvain workflow."""

    project_dir: str = "/Users/zhehaowang/PycharmProjects/scgpt_test"
    run_name: str = "dev_perturb_adamson-Jun28-15-05"
    model_asset_dir: str = "save/scGPT_human"
    output_dir: str = (
        "/Users/zhehaowang/PycharmProjects/scgpt_test/Embedding_GRN_save/"
        "perturbation_pathway_outputs"
    )
    msigdb_dir: str = (
        "/Users/zhehaowang/PycharmProjects/scgpt_test/Embedding_GRN_save/"
        "resources/msigdb_2026_1"
    )
    seed: int = 42
    neighbor_grid: Sequence[int] = field(default_factory=lambda: (10, 15, 20, 30, 50))
    louvain_resolution_grid: Sequence[float] = field(
        default_factory=lambda: (
            0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
            2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0,
        )
    )
    tuning_seeds: Sequence[int] = field(default_factory=lambda: (7, 19, 42, 73, 101))
    min_program_size: int = 5
    min_fraction_genes_in_programs: float = 0.90
    max_largest_program_fraction: float = 0.20
    seed_plateau_tolerance: float = 0.03
    local_plateau_tolerance: float = 0.03
    silhouette_sample_size: int = 1000
    min_gene_set_size: int = 10
    max_gene_set_size: int = 500
    gsea_permutations: int = 500
    counterfactual_device: str = "cpu"
    counterfactual_cells_per_sample: int = 64
    counterfactual_batch_size: int = 2

    @property
    def project_path(self) -> Path:
        return Path(self.project_dir).expanduser().resolve()

    @property
    def run_dir(self) -> Path:
        return self.project_path / self.run_name

    @property
    def asset_dir(self) -> Path:
        return self.project_path / self.model_asset_dir

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def msigdb_path(self) -> Path:
        return Path(self.msigdb_dir).expanduser().resolve()

# Existing preparation/inference functions retain this historical type name.
AnalysisConfig = LouvainAnalysisConfig

PROJECT_DIR = Path("/Users/zhehaowang/PycharmProjects/scgpt_test")
NOTEBOOK_DIR = PROJECT_DIR / "Embedding_GRN_save"
OUTPUT_DIR = NOTEBOOK_DIR / "perturbation_pathway_outputs"
MSIGDB_DIR = NOTEBOOK_DIR / "resources" / "msigdb_2026_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config = LouvainAnalysisConfig(
    project_dir=str(PROJECT_DIR),
    run_name="dev_perturb_adamson-Jun28-15-05",
    model_asset_dir="save/scGPT_human",
    output_dir=str(OUTPUT_DIR),
    msigdb_dir=str(MSIGDB_DIR),
    counterfactual_device="cpu",
    counterfactual_cells_per_sample=64,
    counterfactual_batch_size=2,
    gsea_permutations=500,
    seed=42,
)

required_paths = [
    PROJECT_DIR / "rna_malignant_counts.h5ad",
    PROJECT_DIR / "mouse2human.pkl",
    config.run_dir / "best_model.pt",
    config.run_dir / "gene_names.json",
    config.asset_dir / "args.json",
    config.asset_dir / "vocab.json",
    MSIGDB_DIR / "h.all.v2026.1.Hs.symbols.gmt",
    MSIGDB_DIR / "c2.cp.reactome.v2026.1.Hs.symbols.gmt",
]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, missing_paths)))

display(pd.DataFrame({"setting": [
    "raw H5AD", "checkpoint", "output directory", "control cells per sample",
    "GSEA permutations", "Louvain neighbours", "Louvain resolutions", "tuning seeds"
], "value": [
    str(required_paths[0]), str(required_paths[2]), str(OUTPUT_DIR),
    config.counterfactual_cells_per_sample, config.gsea_permutations,
    list(config.neighbor_grid), list(config.louvain_resolution_grid), list(config.tuning_seeds)
]}))


# ## 2. Load the checkpoint gene axis and gene embeddings
# 
# This is the familiar embedding step from the author's tutorial, with two
# important safeguards: only genes on the trained feature axis are retained, and
# the checkpoint encoder normalisation is applied before cosine similarity.

# In[2]:


def _canonical_condition(values: pd.Series) -> pd.Series:
    mapping = {
        "ctrl": "ctrl",
        "pre-IR": "ctrl",
        "IR-24h": "IR_24h",
        "IR_24h": "IR_24h",
        "24h post-IR": "IR_24h",
        "IR-1week": "IR_1week",
        "IR_1week": "IR_1week",
        "1 week post-IR": "IR_1week",
    }
    out = values.astype(str).map(mapping)
    if out.isna().any():
        unknown = sorted(values[out.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unrecognised condition labels: {unknown}")
    return out

def _bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored
    return result

def _safe_pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3 or np.std(x_arr[valid]) == 0 or np.std(y_arr[valid]) == 0:
        return np.nan, np.nan
    return tuple(float(v) for v in pearsonr(x_arr[valid], y_arr[valid]))

def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3:
        return np.nan, np.nan
    return tuple(float(v) for v in spearmanr(x_arr[valid], y_arr[valid]))

def _read_gmt(path: Path, database: str) -> Dict[str, set[str]]:
    gene_sets: Dict[str, set[str]] = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            term = fields[0]
            genes = {g for g in fields[2:] if g}
            gene_sets[f"{database}::{term}"] = genes
    return gene_sets

def load_gene_sets(
    config: AnalysisConfig,
    universe: set[str],
    include_go: bool,
) -> Dict[str, set[str]]:
    files = {
        "HALLMARK": config.msigdb_path / "h.all.v2026.1.Hs.symbols.gmt",
        "REACTOME": config.msigdb_path / "c2.cp.reactome.v2026.1.Hs.symbols.gmt",
    }
    if include_go:
        files["GO_BP"] = config.msigdb_path / "c5.go.bp.v2026.1.Hs.symbols.gmt"

    combined: Dict[str, set[str]] = {}
    for database, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        combined.update(_read_gmt(path, database))

    filtered = {}
    for term, genes in combined.items():
        eligible = genes & universe
        if config.min_gene_set_size <= len(eligible) <= config.max_gene_set_size:
            filtered[term] = eligible
    return filtered


# In[3]:


def extract_checkpoint_embeddings(config: AnalysisConfig) -> dict:
    checkpoint_path = config.run_dir / "best_model.pt"
    vocab_path = config.asset_dir / "vocab.json"
    gene_names_path = config.run_dir / "gene_names.json"
    mapping_path = config.project_path / "mouse2human.pkl"

    for path in (checkpoint_path, vocab_path, gene_names_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(path)

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    required = {
        "encoder.embedding.weight",
        "encoder.enc_norm.weight",
        "encoder.enc_norm.bias",
    }
    missing = required - set(state)
    if missing:
        raise KeyError(f"Checkpoint lacks required encoder tensors: {sorted(missing)}")

    with vocab_path.open() as handle:
        vocab = json.load(handle)
    with gene_names_path.open() as handle:
        gene_names = json.load(handle)
    with mapping_path.open("rb") as handle:
        mouse_to_human = pickle.load(handle)

    pad_id = vocab.get("<pad>", -1)
    embedding_rows = int(state["encoder.embedding.weight"].shape[0])
    rows = []
    seen_vocab_ids: set[int] = set()
    for feature_index, gene in enumerate(gene_names):
        reason = ""
        vocab_id = vocab.get(gene)
        if gene in DUMMY_TOKENS:
            reason = "dummy_or_special_token"
        elif vocab_id is None:
            reason = "not_in_vocabulary"
        elif int(vocab_id) == int(pad_id):
            reason = "maps_to_pad"
        elif int(vocab_id) >= embedding_rows:
            reason = "vocab_id_outside_checkpoint"
        elif int(vocab_id) in seen_vocab_ids:
            reason = "duplicate_vocab_id"
        else:
            seen_vocab_ids.add(int(vocab_id))

        human_symbol = mouse_to_human.get(gene, gene)
        if isinstance(human_symbol, (list, tuple, set)):
            human_symbol = sorted(str(x) for x in human_symbol)[0] if human_symbol else gene
        rows.append(
            {
                "feature_index": feature_index,
                "model_gene": str(gene),
                "human_symbol": str(human_symbol),
                "vocab_id": np.nan if vocab_id is None else int(vocab_id),
                "eligible_embedding_gene": reason == "",
                "exclusion_reason": reason,
            }
        )

    feature_map = pd.DataFrame(rows)
    eligible = feature_map[feature_map["eligible_embedding_gene"]].copy().reset_index(drop=True)
    # Repeated expression columns share one vocabulary token. Keep one graph
    # node per token, but remember all source columns so downstream expression
    # scores can average them rather than silently select the first copy.
    integer_vocab_ids = feature_map["vocab_id"].fillna(-1).astype(int)
    valid_source = feature_map[
        feature_map["vocab_id"].notna()
        & ~feature_map["model_gene"].isin(DUMMY_TOKENS)
        & integer_vocab_ids.lt(embedding_rows)
        & integer_vocab_ids.ne(int(pad_id))
    ].copy()
    source_indices_by_vocab = (
        valid_source.groupby(valid_source["vocab_id"].astype(int))["feature_index"]
        .apply(lambda values: tuple(sorted(values.astype(int).tolist())))
        .to_dict()
    )
    eligible["source_feature_indices"] = eligible["vocab_id"].astype(int).map(source_indices_by_vocab)
    eligible["source_feature_count"] = eligible["source_feature_indices"].map(len)
    ids = torch.tensor(eligible["vocab_id"].astype(int).to_numpy(), dtype=torch.long)
    raw = state["encoder.embedding.weight"].index_select(0, ids)
    normed = torch_f.layer_norm(
        raw,
        normalized_shape=(raw.shape[1],),
        weight=state["encoder.enc_norm.weight"],
        bias=state["encoder.enc_norm.bias"],
        eps=1e-5,
    )
    normed = torch_f.normalize(normed, p=2, dim=1)
    embeddings = normed.detach().cpu().numpy().astype(np.float32)

    condition_rows = []
    for token in ("IR_24h", "IR_1week"):
        if token not in vocab or int(vocab[token]) >= embedding_rows:
            continue
        token_raw = state["encoder.embedding.weight"][int(vocab[token])].reshape(1, -1)
        token_emb = torch_f.layer_norm(
            token_raw,
            normalized_shape=(token_raw.shape[1],),
            weight=state["encoder.enc_norm.weight"],
            bias=state["encoder.enc_norm.bias"],
            eps=1e-5,
        )
        token_emb = torch_f.normalize(token_emb, p=2, dim=1).detach().cpu().numpy()[0]
        similarities = embeddings @ token_emb
        for idx, similarity in enumerate(similarities):
            condition_rows.append(
                {
                    "condition_token": token,
                    "model_gene": eligible.loc[idx, "model_gene"],
                    "human_symbol": eligible.loc[idx, "human_symbol"],
                    "cosine_similarity": float(similarity),
                }
            )

    condition_similarity = pd.DataFrame(condition_rows)
    audit = pd.DataFrame(
        [
            {"check": "checkpoint_embedding_rows", "value": embedding_rows},
            {"check": "checkpoint_embedding_dimension", "value": int(embeddings.shape[1])},
            {"check": "saved_feature_names", "value": len(gene_names)},
            {"check": "eligible_training_vocab_genes", "value": len(eligible)},
            {
                "check": "eligible_source_expression_columns",
                "value": int(eligible["source_feature_count"].sum()),
            },
            {"check": "excluded_features", "value": int((~feature_map["eligible_embedding_gene"]).sum())},
            {"check": "unique_human_symbols", "value": int(eligible["human_symbol"].nunique())},
        ]
    )
    return {
        "state": state,
        "gene_names": gene_names,
        "feature_map": feature_map,
        "eligible_features": eligible,
        "embeddings": embeddings,
        "condition_similarity": condition_similarity,
        "embedding_audit": audit,
    }


# In[4]:


embedding_result = extract_checkpoint_embeddings(config)
display(embedding_result["embedding_audit"])

print("Embedding matrix:", embedding_result["embeddings"].shape)
print("First model-axis genes:", embedding_result["gene_names"][:8])


# ## 3. Read and prepare rna_malignant_counts.h5ad
# 
# The preparation is kept visible and reproduces the June checkpoint's biological
# axis:
# 
# - map Timepoint to ctrl, IR_24h and IR_1week;
# - keep genes with a human orthologue and detected in at least three cells;
# - normalise each cell to 10,000 counts and apply log1p;
# - select the author's highly-variable columns;
# - assert that all 2,465 biological features match the checkpoint in order;
# - append the two underscore perturbation-token columns as zeros.

# In[5]:


def load_and_preprocess_raw_adata(config: AnalysisConfig, gene_names: Sequence[str]) -> dict:
    """Reproduce the training expression preparation from the raw H5AD.

    This is the same core preparation used in the user's reformatting and GRN
    notebooks: map Timepoint, retain mouse genes with a human orthologue,
    filter mapped genes expressed in at least three cells, normalize each cell
    to 10,000 over that filtered mapped-gene library, log1p, then select the
    author's pre-computed HVGs. Duplicate human orthologues remain separate on
    the model input axis, exactly as they did during June checkpoint training.
    """

    raw_path = config.project_path / "rna_malignant_counts.h5ad"
    mapping_path = config.project_path / "mouse2human.pkl"
    if not raw_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(f"Missing raw input or orthology map: {raw_path}, {mapping_path}")

    with mapping_path.open("rb") as handle:
        mouse_to_human = pickle.load(handle)
    adata = ad.read_h5ad(raw_path)
    if "highly_variable" not in adata.var.columns:
        raise KeyError("Raw adata.var lacks the author's 'highly_variable' annotation")

    raw_gene_names = adata.var_names.astype(str).to_numpy()
    mapped_mask = np.asarray([gene in mouse_to_human for gene in raw_gene_names], dtype=bool)
    mapped_indices = np.flatnonzero(mapped_mask)
    mapped_counts = adata.X[:, mapped_indices]
    if sparse.issparse(mapped_counts):
        mapped_counts = mapped_counts.tocsr().astype(np.float32)
        mapped_counts.data = np.nan_to_num(
            mapped_counts.data, nan=0.0, posinf=1e5, neginf=0.0
        )
        detected_cells = np.asarray((mapped_counts > 0).sum(axis=0)).ravel()
    else:
        mapped_counts = np.nan_to_num(
            np.asarray(mapped_counts, dtype=np.float32), nan=0.0, posinf=1e5, neginf=0.0
        )
        detected_cells = np.count_nonzero(mapped_counts > 0, axis=0)
    expressed_mask_within_mapped = detected_cells >= 3
    filtered_mapped = mapped_counts[:, expressed_mask_within_mapped]
    library_totals = np.asarray(filtered_mapped.sum(axis=1)).ravel().astype(np.float64)
    if np.any(library_totals <= 0):
        raise ValueError("At least one cell has zero counts after mapped-gene filtering")

    author_hvg = adata.var["highly_variable"].astype(bool).to_numpy()
    hvg_raw_indices = np.flatnonzero(author_hvg & mapped_mask)
    hvg_mouse = raw_gene_names[hvg_raw_indices]
    hvg_human = np.asarray([str(mouse_to_human[gene]) for gene in hvg_mouse], dtype=str)
    expected_biological = np.asarray(gene_names[: len(hvg_human)], dtype=str)
    axis_match = bool(np.array_equal(hvg_human, expected_biological))
    if len(hvg_human) != 2465 or not axis_match:
        raise ValueError(
            "Raw-data preparation does not reproduce the checkpoint's 2,465-feature biological axis"
        )

    hvg_counts = adata.X[:, hvg_raw_indices]
    scale = (10000.0 / library_totals).astype(np.float32)
    if sparse.issparse(hvg_counts):
        normalised = hvg_counts.tocsr().astype(np.float32).multiply(scale[:, None]).tocsr()
        normalised.data = np.log1p(normalised.data)
        biological_expression = normalised.toarray().astype(np.float32, copy=False)
    else:
        biological_expression = np.log1p(
            np.asarray(hvg_counts, dtype=np.float32) * scale[:, None]
        ).astype(np.float32, copy=False)

    dummy_values = np.zeros((adata.n_obs, 2), dtype=np.float32)
    expression = np.concatenate([biological_expression, dummy_values], axis=1)
    if list(gene_names[2465:2467]) != ["IR_24h", "IR_1week"]:
        raise ValueError("Checkpoint feature schema lacks the expected underscore condition tokens")

    metadata = adata.obs.copy().reset_index().rename(columns={adata.obs.index.name or "index": "cell_id"})
    if "cell_id" not in metadata.columns:
        metadata = metadata.rename(columns={metadata.columns[0]: "cell_id"})
    metadata["condition_canonical"] = _canonical_condition(metadata["Timepoint"])
    metadata["survival_classification_adata"] = metadata["survival_classification"].astype(str)
    adata_shape = adata.shape
    del adata

    preparation_audit = pd.DataFrame(
        [
            {"check": "source_file", "value": str(raw_path)},
            {"check": "raw_cells", "value": int(adata_shape[0])},
            {"check": "raw_genes", "value": int(adata_shape[1])},
            {"check": "genes_with_human_orthologue", "value": int(mapped_mask.sum())},
            {
                "check": "mapped_genes_expressed_in_at_least_3_cells",
                "value": int(expressed_mask_within_mapped.sum()),
            },
            {"check": "author_hvg_columns_with_mapping", "value": len(hvg_human)},
            {"check": "unique_human_symbols_on_biological_axis", "value": len(set(hvg_human))},
            {"check": "checkpoint_biological_axis_exact_match", "value": axis_match},
            {"check": "final_model_input_features", "value": expression.shape[1]},
            {"check": "normalisation", "value": "mapped genes min_cells=3; target_sum=1e4; log1p"},
        ]
    )
    feature_axis = pd.DataFrame(
        {
            "feature_index": np.arange(len(hvg_human), dtype=int),
            "mouse_gene": hvg_mouse,
            "human_symbol": hvg_human,
        }
    )
    return {
        "metadata": metadata,
        "expression": expression,
        "preparation_audit": preparation_audit,
        "raw_feature_axis": feature_axis,
    }


# In[6]:


aligned = load_and_preprocess_raw_adata(config, embedding_result["gene_names"])
display(aligned["preparation_audit"])

assert aligned["expression"].shape[1] == 2467
assert aligned["preparation_audit"].set_index("check").loc[
    "checkpoint_biological_axis_exact_match", "value"
] in (True, "True")

metadata_preview = aligned["metadata"][[
    "cell_id", "Sample", "Timepoint", "condition_canonical",
    "GBM_malig_subtypes", "survival_classification_adata"
]].head()
display(metadata_preview)


# ## 4. Build embedding gene programs
# 
# This intentionally uses Louvain community detection on Scanpy cosine-neighbour
# graphs. The original tutorial's `resolution=40` was attached to a different
# algorithm and about 15,500 genes; this checkpoint has only 2,439 unique eligible
# tokens, so that number is included in the grid but is not copied automatically.
# 
# For every neighbours/resolution setting, five Louvain seeds are run. The final
# membership for that setting is the medoid seed: the seed with highest mean ARI
# to the other four. Selection then uses:
# 
# 1. broad structural constraints (not a target number of programs);
# 2. the lower-tail seed-ARI plateau;
# 3. agreement with adjacent hyperparameters;
# 4. cosine silhouette within the stable plateau.
# 
# Pathway labels are added only after tuning and therefore cannot influence the
# chosen clustering. No all-gene PCA network is drawn; the later full-graph record
# uses the exact selected kNN topology and is not treated as evidence of clean
# cluster separation.

# In[7]:


def build_embedding_programs_louvain(
    config: LouvainAnalysisConfig,
    embedding_result: dict,
) -> dict:
    """Tune Louvain partitions on Scanpy cosine-neighbour graphs.

    The partition is selected by reproducibility across random seeds, continuity
    across adjacent hyperparameters, and cosine silhouette.  Program count and
    pathway annotations are deliberately excluded from selection.
    """

    import scanpy as sc

    embeddings = np.asarray(embedding_result["embeddings"], dtype=np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    features = embedding_result["eligible_features"].copy().reset_index(drop=True)
    n_nodes = len(features)
    tune_seeds = tuple(sorted(int(seed) for seed in config.tuning_seeds))
    neighbor_grid = tuple(sorted({min(int(k), n_nodes - 1) for k in config.neighbor_grid}))
    resolution_grid = tuple(sorted({float(r) for r in config.louvain_resolution_grid}))

    rng = np.random.default_rng(config.seed)
    silhouette_n = min(int(config.silhouette_sample_size), n_nodes)
    silhouette_indices = np.sort(rng.choice(n_nodes, size=silhouette_n, replace=False))
    sampled_embeddings = embeddings[silhouette_indices]
    sampled_distances = 1.0 - np.clip(sampled_embeddings @ sampled_embeddings.T, -1.0, 1.0)
    np.fill_diagonal(sampled_distances, 0.0)

    def best_program_jaccards(reference: np.ndarray, comparison: np.ndarray) -> dict[str, float]:
        """Best comparison-program Jaccard for every reference program."""

        ref_codes, ref_labels = pd.factorize(reference, sort=True)
        cmp_codes, _ = pd.factorize(comparison, sort=True)
        overlap = np.zeros((ref_codes.max() + 1, cmp_codes.max() + 1), dtype=np.int32)
        np.add.at(overlap, (ref_codes, cmp_codes), 1)
        ref_sizes = overlap.sum(axis=1, keepdims=True)
        cmp_sizes = overlap.sum(axis=0, keepdims=True)
        union = ref_sizes + cmp_sizes - overlap
        jaccard = np.divide(
            overlap,
            union,
            out=np.zeros_like(overlap, dtype=float),
            where=union > 0,
        )
        return {
            str(label): float(jaccard[index].max())
            for index, label in enumerate(ref_labels)
        }

    tuning_rows = []
    partition_runs: dict[tuple[int, float], dict[int, np.ndarray]] = {}
    medoid_partitions: dict[tuple[int, float], np.ndarray] = {}
    medoid_seeds: dict[tuple[int, float], int] = {}
    connectivities_by_k: dict[int, sparse.csr_matrix] = {}

    for n_neighbors in neighbor_grid:
        gdata = ad.AnnData(X=embeddings.copy())
        gdata.obs_names = features["model_gene"].astype(str).tolist()
        sc.pp.neighbors(
            gdata,
            n_neighbors=n_neighbors,
            use_rep="X",
            metric="cosine",
            random_state=config.seed,
        )
        connectivities = gdata.obsp["connectivities"].tocsr().astype(np.float32)
        connectivities_by_k[n_neighbors] = connectivities.copy()
        graph = nx.from_scipy_sparse_array(
            connectivities,
            create_using=nx.Graph,
            edge_attribute="weight",
        )
        graph.remove_edges_from(nx.selfloop_edges(graph))
        upper_edges = sparse.triu(connectivities, k=1).tocoo()
        total_edge_weight = float(upper_edges.data.sum())

        for resolution in resolution_grid:
            runs: dict[int, np.ndarray] = {}
            for seed in tune_seeds:
                communities = nx.algorithms.community.louvain_communities(
                    graph,
                    weight="weight",
                    resolution=resolution,
                    seed=seed,
                )
                communities = sorted(
                    communities,
                    key=lambda community: (-len(community), min(community)),
                )
                labels = np.full(n_nodes, -1, dtype=int)
                for community_index, community in enumerate(communities):
                    labels[np.fromiter(community, dtype=int)] = community_index
                if (labels < 0).any():
                    raise RuntimeError("Louvain left at least one embedding node unassigned")
                runs[seed] = labels.astype(str)
            grid_key = (n_neighbors, resolution)
            partition_runs[grid_key] = runs

            pair_ari = {
                (first, second): float(adjusted_rand_score(runs[first], runs[second]))
                for first, second in combinations(tune_seeds, 2)
            }
            mean_ari_by_seed = {
                seed: float(
                    np.mean(
                        [
                            pair_ari[tuple(sorted((seed, other)))]
                            for other in tune_seeds
                            if other != seed
                        ]
                    )
                )
                for seed in tune_seeds
            }
            medoid_seed = sorted(
                tune_seeds,
                key=lambda seed: (-mean_ari_by_seed[seed], seed),
            )[0]
            medoid_labels = runs[medoid_seed]
            medoid_seeds[grid_key] = medoid_seed
            medoid_partitions[grid_key] = medoid_labels
            counts = pd.Series(medoid_labels).value_counts().to_numpy(dtype=int)

            program_jaccards = []
            for seed in tune_seeds:
                if seed == medoid_seed:
                    continue
                program_jaccards.extend(
                    best_program_jaccards(medoid_labels, runs[seed]).values()
                )

            sampled_labels = medoid_labels[silhouette_indices]
            try:
                cosine_silhouette = float(
                    silhouette_score(sampled_distances, sampled_labels, metric="precomputed")
                )
            except ValueError:
                cosine_silhouette = np.nan

            within_edge_weight = float(
                upper_edges.data[
                    medoid_labels[upper_edges.row] == medoid_labels[upper_edges.col]
                ].sum()
            )
            pairwise_values = np.fromiter(pair_ari.values(), dtype=float)
            coverage_ge_min = float(
                counts[counts >= int(config.min_program_size)].sum() / n_nodes
            )
            tuning_rows.append(
                {
                    "n_neighbors": n_neighbors,
                    "resolution": resolution,
                    "medoid_seed": medoid_seed,
                    "n_programs": int(len(counts)),
                    "min_size": int(counts.min()),
                    "median_size": float(np.median(counts)),
                    "max_size": int(counts.max()),
                    "largest_program_fraction": float(counts.max() / n_nodes),
                    "fraction_genes_in_programs_size_ge_min": coverage_ge_min,
                    "median_pairwise_seed_ari": float(np.median(pairwise_values)),
                    "q10_pairwise_seed_ari": float(np.quantile(pairwise_values, 0.10)),
                    "min_pairwise_seed_ari": float(pairwise_values.min()),
                    "program_jaccard_median": float(np.median(program_jaccards)),
                    "program_jaccard_q10": float(np.quantile(program_jaccards, 0.10)),
                    "cosine_silhouette": cosine_silhouette,
                    "within_graph_weight_fraction_descriptive": (
                        within_edge_weight / total_edge_weight
                        if total_edge_weight > 0
                        else np.nan
                    ),
                }
            )

    tuning = pd.DataFrame(tuning_rows)
    local_plateau_ari = []
    local_neighbor_count = []
    for row in tuning.itertuples(index=False):
        k_index = neighbor_grid.index(int(row.n_neighbors))
        r_index = resolution_grid.index(float(row.resolution))
        adjacent_keys = []
        if k_index > 0:
            adjacent_keys.append((neighbor_grid[k_index - 1], float(row.resolution)))
        if k_index + 1 < len(neighbor_grid):
            adjacent_keys.append((neighbor_grid[k_index + 1], float(row.resolution)))
        if r_index > 0:
            adjacent_keys.append((int(row.n_neighbors), resolution_grid[r_index - 1]))
        if r_index + 1 < len(resolution_grid):
            adjacent_keys.append((int(row.n_neighbors), resolution_grid[r_index + 1]))
        labels = medoid_partitions[(int(row.n_neighbors), float(row.resolution))]
        adjacent_ari = [
            adjusted_rand_score(labels, medoid_partitions[key])
            for key in adjacent_keys
        ]
        local_plateau_ari.append(float(np.median(adjacent_ari)))
        local_neighbor_count.append(len(adjacent_ari))
    tuning["local_plateau_ari"] = local_plateau_ari
    tuning["local_plateau_neighbor_count"] = local_neighbor_count

    tuning["admissible"] = (
        tuning["fraction_genes_in_programs_size_ge_min"].ge(
            config.min_fraction_genes_in_programs
        )
        & tuning["largest_program_fraction"].le(
            config.max_largest_program_fraction
        )
        & tuning["median_size"].ge(config.min_program_size)
        & tuning["cosine_silhouette"].notna()
    )
    candidates = tuning[tuning["admissible"]].copy()
    if candidates.empty:
        raise RuntimeError(
            "No Louvain setting passed the prespecified coverage, largest-program, "
            "median-size and finite-silhouette constraints"
        )

    seed_cutoff = float(candidates["q10_pairwise_seed_ari"].max()) - float(
        config.seed_plateau_tolerance
    )
    seed_plateau = candidates[candidates["q10_pairwise_seed_ari"] >= seed_cutoff].copy()
    local_cutoff = float(seed_plateau["local_plateau_ari"].max()) - float(
        config.local_plateau_tolerance
    )
    local_plateau = seed_plateau[
        seed_plateau["local_plateau_ari"] >= local_cutoff
    ].copy()
    local_plateau["distance_from_k15"] = (local_plateau["n_neighbors"] - 15).abs()
    chosen_row = local_plateau.sort_values(
        [
            "cosine_silhouette",
            "program_jaccard_median",
            "fraction_genes_in_programs_size_ge_min",
            "resolution",
            "distance_from_k15",
            "n_neighbors",
        ],
        ascending=[False, False, False, True, True, True],
    ).iloc[0]
    chosen_neighbors = int(chosen_row["n_neighbors"])
    chosen_resolution = float(chosen_row["resolution"])
    chosen_seed = int(chosen_row["medoid_seed"])
    tuning["on_seed_stability_plateau"] = (
        tuning["admissible"] & tuning["q10_pairwise_seed_ari"].ge(seed_cutoff)
    )
    tuning["on_local_hyperparameter_plateau"] = (
        tuning["on_seed_stability_plateau"]
        & tuning["local_plateau_ari"].ge(local_cutoff)
    )
    tuning["selected"] = (
        tuning["n_neighbors"].eq(chosen_neighbors)
        & tuning["resolution"].eq(chosen_resolution)
    )

    selected_key = (chosen_neighbors, chosen_resolution)
    selected_runs = partition_runs[selected_key]
    selected_labels = medoid_partitions[selected_key]
    selected_connectivities = connectivities_by_k[chosen_neighbors]

    uid_communities = []
    for label in pd.unique(selected_labels):
        indices = np.flatnonzero(selected_labels == label)
        genes = sorted(features.iloc[indices]["model_gene"].astype(str).tolist())
        digest = hashlib.sha256("|".join(genes).encode()).hexdigest()[:12]
        uid_communities.append((f"GP_{digest}", digest, indices))
    uid_communities.sort(key=lambda item: (-len(item[2]), item[0]))

    display_width = max(2, len(str(len(uid_communities))))
    internal_labels = np.empty(n_nodes, dtype=object)
    display_labels = np.empty(n_nodes, dtype=object)
    membership_hashes = np.empty(n_nodes, dtype=object)
    for rank, (program_uid, digest, indices) in enumerate(uid_communities, start=1):
        internal_labels[indices] = program_uid
        display_labels[indices] = f"P{rank:0{display_width}d}"
        membership_hashes[indices] = digest
    features["module_id"] = internal_labels
    features["program_id"] = display_labels
    features["membership_hash"] = membership_hashes

    central_genes = {}
    for program_uid, _, indices in uid_communities:
        within = selected_connectivities[indices][:, indices]
        degree = np.asarray(within.sum(axis=1)).ravel()
        ordered = sorted(
            range(len(indices)),
            key=lambda offset: (
                -degree[offset],
                str(features.iloc[int(indices[offset])]["human_symbol"]),
            ),
        )
        symbols = []
        for offset in ordered:
            symbol = str(features.iloc[int(indices[offset])]["human_symbol"])
            if symbol not in symbols:
                symbols.append(symbol)
            if len(symbols) == 5:
                break
        central_genes[program_uid] = ";".join(symbols)

    selected_program_stability_rows = []
    for program_uid, _, indices in uid_communities:
        reference_label = selected_labels[int(indices[0])]
        values = []
        for seed in tune_seeds:
            if seed == chosen_seed:
                continue
            score = best_program_jaccards(selected_labels, selected_runs[seed])[
                str(reference_label)
            ]
            values.append(score)
            selected_program_stability_rows.append(
                {
                    "module_id": program_uid,
                    "comparison_seed": seed,
                    "best_jaccard": score,
                }
            )

    module_sizes = (
        features.groupby(
            ["module_id", "program_id", "membership_hash"],
            sort=False,
        )
        .agg(
            n_features=("model_gene", "size"),
            n_human_symbols=("human_symbol", "nunique"),
            genes=("model_gene", lambda x: ";".join(sorted(x.astype(str)))),
            human_symbols=("human_symbol", lambda x: ";".join(sorted(set(x.astype(str))))),
        )
        .reset_index()
    )
    module_sizes["central_genes"] = module_sizes["module_id"].map(central_genes)
    program_stability = pd.DataFrame(selected_program_stability_rows)
    stability_summary = (
        program_stability.groupby("module_id", as_index=False)
        .agg(
            median_seed_jaccard=("best_jaccard", "median"),
            min_seed_jaccard=("best_jaccard", "min"),
        )
    )
    module_sizes = module_sizes.merge(stability_summary, on="module_id", how="left")

    upper = sparse.triu(selected_connectivities, k=1).tocoo()
    edges = pd.DataFrame(
        {
            "source_index": upper.row.astype(int),
            "target_index": upper.col.astype(int),
            "connectivity_weight": upper.data.astype(float),
        }
    )
    edges["source_gene"] = features.iloc[edges["source_index"].to_numpy()][
        "model_gene"
    ].to_numpy()
    edges["target_gene"] = features.iloc[edges["target_index"].to_numpy()][
        "model_gene"
    ].to_numpy()
    edges["source_program"] = features.iloc[edges["source_index"].to_numpy()][
        "program_id"
    ].to_numpy()
    edges["target_program"] = features.iloc[edges["target_index"].to_numpy()][
        "program_id"
    ].to_numpy()
    edges["within_program"] = edges["source_program"].eq(edges["target_program"])

    stability_rows = []
    for first, second in combinations(tune_seeds, 2):
        stability_rows.append(
            {
                "seed_1": first,
                "seed_2": second,
                "adjusted_rand_index": float(
                    adjusted_rand_score(selected_runs[first], selected_runs[second])
                ),
            }
        )
    if (
        chosen_row["median_pairwise_seed_ari"] >= 0.80
        and chosen_row["q10_pairwise_seed_ari"] >= 0.70
        and chosen_row["program_jaccard_median"] >= 0.70
        and chosen_row["cosine_silhouette"] > 0
    ):
        stability_interpretation = "stable"
    elif (
        chosen_row["median_pairwise_seed_ari"] >= 0.60
        and chosen_row["q10_pairwise_seed_ari"] >= 0.50
        and chosen_row["cosine_silhouette"] > 0
    ):
        stability_interpretation = "moderate"
    else:
        stability_interpretation = "exploratory"

    selection = pd.DataFrame(
        [
            {
                "selected_n_neighbors": chosen_neighbors,
                "selected_resolution": chosen_resolution,
                "selected_medoid_seed": chosen_seed,
                "selected_n_programs": int(chosen_row["n_programs"]),
                "fraction_genes_in_programs_size_ge_min": float(
                    chosen_row["fraction_genes_in_programs_size_ge_min"]
                ),
                "largest_program_fraction": float(chosen_row["largest_program_fraction"]),
                "median_pairwise_seed_ari": float(chosen_row["median_pairwise_seed_ari"]),
                "q10_pairwise_seed_ari": float(chosen_row["q10_pairwise_seed_ari"]),
                "local_plateau_ari": float(chosen_row["local_plateau_ari"]),
                "program_jaccard_median": float(chosen_row["program_jaccard_median"]),
                "cosine_silhouette": float(chosen_row["cosine_silhouette"]),
                "stability_interpretation": stability_interpretation,
                "stability_interpretation_rule": (
                    "stable/moderate requires the prespecified seed-ARI thresholds and "
                    "positive cosine silhouette; otherwise exploratory"
                ),
                "selection_rule": (
                    "structural admissibility; q10 seed-ARI plateau; adjacent-hyperparameter "
                    "ARI plateau; maximum cosine silhouette; deterministic tie-breaks"
                ),
                "pathway_labels_used_for_tuning": False,
                "target_program_count_used_for_tuning": False,
            }
        ]
    )
    return {
        "nodes": features,
        "edges": edges,
        "module_sizes": module_sizes,
        "tuning": tuning.sort_values(["n_neighbors", "resolution"]).reset_index(drop=True),
        "selection": selection,
        "stability": pd.DataFrame(stability_rows),
        "program_stability": program_stability,
        "chosen_neighbors": chosen_neighbors,
        "chosen_resolution": chosen_resolution,
        "chosen_seed": chosen_seed,
        "connectivities": selected_connectivities,
    }


# In[8]:


def annotate_embedding_programs(
    config: LouvainAnalysisConfig,
    nodes: pd.DataFrame,
) -> dict:
    """Annotate programs by Hallmark/Reactome over-representation.

    This runs only after Louvain tuning, so pathway labels cannot influence the
    selected graph hyperparameters.  Exact membership IDs remain separate from
    the human-readable biological annotation.
    """

    universe = set(nodes["human_symbol"].astype(str))
    gene_sets = load_gene_sets(config, universe=universe, include_go=False)
    M = len(universe)
    rows = []
    for module_id, frame in nodes.groupby("module_id", sort=False):
        program_id = str(frame["program_id"].iloc[0])
        symbols = set(frame["human_symbol"].astype(str))
        N = len(symbols)
        for term, pathway_genes in gene_sets.items():
            overlap = symbols & pathway_genes
            database, pathway = term.split("::", 1)
            rows.append(
                {
                    "module_id": module_id,
                    "program_id": program_id,
                    "database": database,
                    "pathway": pathway,
                    "term": term,
                    "program_size": N,
                    "pathway_size_in_universe": len(pathway_genes),
                    "overlap_size": len(overlap),
                    "overlap_genes": ";".join(sorted(overlap)),
                    "p_value": float(
                        hypergeom.sf(
                            len(overlap) - 1,
                            M,
                            len(pathway_genes),
                            N,
                        )
                    ),
                }
            )
    ora = pd.DataFrame(rows)
    ora["global_fdr"] = _bh_adjust(ora["p_value"])
    ora["program_fdr"] = ora.groupby("module_id", sort=False)["p_value"].transform(
        lambda values: _bh_adjust(values)
    )
    ora = ora.sort_values(
        ["global_fdr", "program_fdr", "p_value", "overlap_size"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    annotation_rows = []
    for module_id, frame in ora.groupby("module_id", sort=False):
        significant = frame[
            frame["program_fdr"].lt(0.05) & frame["overlap_size"].ge(3)
        ]
        best = significant.iloc[0] if not significant.empty else frame.iloc[0]
        is_significant = not significant.empty
        readable = str(best["pathway"]).removeprefix("HALLMARK_").removeprefix(
            "REACTOME_"
        ).replace("_", " ").title()
        annotation_rows.append(
            {
                "module_id": module_id,
                "program_id": str(best["program_id"]),
                "annotation": readable if is_significant else "Unannotated",
                "annotation_is_program_fdr_significant": bool(is_significant),
                "annotation_database": str(best["database"]) if is_significant else "",
                "annotation_term": str(best["term"]) if is_significant else "",
                "annotation_program_fdr": (
                    float(best["program_fdr"]) if is_significant else np.nan
                ),
                "annotation_global_fdr": (
                    float(best["global_fdr"]) if is_significant else np.nan
                ),
                "annotation_overlap_size": int(best["overlap_size"]) if is_significant else 0,
                "annotation_fdr_scope": (
                    "BH across tested pathways within this program"
                    if is_significant
                    else ""
                ),
            }
        )
    return {
        "program_ora": ora,
        "program_annotations": pd.DataFrame(annotation_rows),
        "ora_gene_set_count": len(gene_sets),
    }


# In[9]:


modules = build_embedding_programs_louvain(config, embedding_result)
program_annotation = annotate_embedding_programs(config, modules["nodes"])
modules["module_sizes"] = modules["module_sizes"].merge(
    program_annotation["program_annotations"].drop(columns="program_id"),
    on="module_id",
    how="left",
)

program_table = (
    modules["module_sizes"]
    .rename(columns={
        "n_features": "n_embedding_genes",
        "n_human_symbols": "n_unique_symbols",
    })
    .sort_values("n_embedding_genes", ascending=False)
    .reset_index(drop=True)
)
program_table["representative_genes"] = program_table["central_genes"].map(
    lambda value: ", ".join(str(value).split(";")[:5])
)
program_table["display_label"] = program_table.apply(
    lambda row: (
        f"{row['program_id']} — {row['annotation']} "
        f"({', '.join(str(row['central_genes']).split(';')[:3])})"
    ),
    axis=1,
)

selection = modules["selection"].iloc[0]
print(f"Selected k={modules['chosen_neighbors']}, resolution={modules['chosen_resolution']:g}, "
      f"medoid seed={modules['chosen_seed']}; {len(program_table)} programs.")
print("Stability interpretation:", selection["stability_interpretation"])
if selection["cosine_silhouette"] <= 0:
    print(
        "The selected silhouette is not positive: the embedding is not cleanly separated "
        "into discrete programs. Treat all program-level results as exploratory."
    )
display(modules["selection"])
display(program_table[[
    "program_id", "module_id", "n_embedding_genes", "n_unique_symbols",
    "annotation", "representative_genes", "median_seed_jaccard"
]].head(20))
print(
    "Programs with a provisional Hallmark/Reactome label: ",
    int(program_table["annotation"].ne("Unannotated").sum()),
    "/", len(program_table),
    "(BH FDR < 0.05 within program and overlap ≥ 3; global FDR is retained in the ORA table)."
)
display(modules["stability"])


# In[10]:


tuning = modules["tuning"].copy()
selected = tuning[tuning["selected"]].iloc[0]
palette = dict(zip(sorted(tuning["n_neighbors"].unique()), sns.color_palette("viridis", tuning["n_neighbors"].nunique())))

fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True)
for n_neighbors, frame in tuning.groupby("n_neighbors", sort=True):
    frame = frame.sort_values("resolution")
    color = palette[n_neighbors]
    axes[0].plot(frame["resolution"], frame["n_programs"], marker="o", ms=3,
                 color=color, label=f"k={n_neighbors}")
    axes[1].plot(frame["resolution"], frame["median_pairwise_seed_ari"], marker="o", ms=3,
                 color=color)
    axes[1].plot(frame["resolution"], frame["local_plateau_ari"], linestyle="--",
                 linewidth=1, alpha=0.65, color=color)
    axes[2].plot(frame["resolution"], frame["cosine_silhouette"], marker="o", ms=3,
                 color=color)

for ax, column in zip(
    axes,
    ("n_programs", "median_pairwise_seed_ari", "cosine_silhouette"),
):
    ax.scatter(selected["resolution"], selected[column], marker="*", s=210,
               color="#c43c39", edgecolor="white", linewidth=0.7, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Louvain resolution")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Number of programs")
axes[0].set_title("A  Partition granularity")
axes[1].set_ylabel("Adjusted Rand index")
axes[1].set_title("B  Seed stability (solid) / local plateau (dashed)")
axes[2].set_ylabel("Cosine silhouette")
axes[2].set_title("C  Embedding separation")
k_handles, k_labels = axes[0].get_legend_handles_labels()
metric_handles = [
    plt.Line2D([0], [0], color="#666666", linestyle="-",
               label="Median across-seed ARI"),
    plt.Line2D([0], [0], color="#666666", linestyle="--",
               label="Adjacent-setting ARI"),
    plt.Line2D([0], [0], marker="*", linestyle="none", markersize=13,
               markerfacecolor="#c43c39", markeredgecolor="white",
               label="Selected setting"),
]
fig.legend(
    k_handles + metric_handles,
    k_labels + [handle.get_label() for handle in metric_handles],
    frameon=False, loc="lower center", ncol=4, fontsize=9,
    bbox_to_anchor=(0.5, 0.01),
)
fig.suptitle(
    f"Louvain tuning: selected k={modules['chosen_neighbors']}, r={modules['chosen_resolution']:g}; "
    f"{int(selected['n_programs'])} programs; median seed ARI={selected['median_pairwise_seed_ari']:.2f}",
    y=0.99,
)
fig.tight_layout(rect=(0, 0.20, 1, 0.92))
fig.savefig(OUTPUT_DIR / "louvain_tuning_diagnostic.png", dpi=220, bbox_inches="tight")
plt.show()


# ### How to read the Louvain tuning diagnostic
# 
# Panel A shows clustering granularity: higher resolution generally divides the
# same embedding graph into more, smaller gene programs. Panel B asks whether
# that partition is reproducible. The solid line is agreement among five random
# Louvain seeds; the dashed line is agreement with nearby tuning settings. Panel C
# is cosine silhouette, which asks whether genes assigned to one program are
# actually more similar to one another than to neighbouring programs. The red
# star is the selected setting in every panel. These are clustering diagnostics,
# not biological significance tests or pathway p-values. In this run the selected
# silhouette is negative, so the embedding does not support cleanly separated
# discrete programs; program-level findings must remain exploratory even though
# seed agreement is reasonably high.
# 
# `program_id` is the short figure label (`P001`, `P002`, ...). `module_id` is the
# exact membership provenance key (`GP_` plus a SHA-256 fingerprint); it is kept
# in result tables, not used as a biological name. Provisional annotations use
# BH-adjusted enrichment across pathways within each program, require at least
# three overlapping genes, and retain the stricter global FDR in the ORA table.
# Ranked pathway GSEA is therefore the primary biological result.

# ### Provisional pathway map of embedding gene programs
# 
# This sparse dot plot provides the compact program-by-pathway overview shown in
# many embedding-GRN tutorials, but with the statistical scope made explicit. It
# shows every term with within-program BH FDR < 0.05 and at least three
# overlapping genes. Colour is `−log10(within-program FDR)` and dot area is the
# fraction of genes in that program overlapping the pathway. The plotted overlaps
# contain only three to five genes, and none of the terms passes the stricter
# global correction across all 141,087 program–pathway tests. The map is therefore
# a hypothesis-generating annotation rather than a set of confirmed pathway
# modules or radiation-responsive programs.

# In[11]:


program_pathway_dotplot_data = (
    program_annotation["program_ora"]
    .query("program_fdr < 0.05 and overlap_size >= 3")
    .sort_values(["program_id", "program_fdr", "p_value", "pathway"])
    .copy()
)

annotation_lookup = (
    program_annotation["program_annotations"]
    .set_index("program_id")["annotation"]
    .to_dict()
)
program_pathway_dotplot_data["program_annotation"] = (
    program_pathway_dotplot_data["program_id"].map(annotation_lookup)
)
program_pathway_dotplot_data["program_overlap_fraction"] = (
    program_pathway_dotplot_data["overlap_size"]
    / program_pathway_dotplot_data["program_size"]
)
program_pathway_dotplot_data["minus_log10_program_fdr"] = -np.log10(
    program_pathway_dotplot_data["program_fdr"].clip(lower=np.finfo(float).tiny)
)

preferred_program_order = [
    "P013", "P162", "P311",
    "P045", "P023", "P137", "P003", "P174",
    "P304", "P288",
]
present_programs = set(program_pathway_dotplot_data["program_id"])
program_order = [p for p in preferred_program_order if p in present_programs]
program_order += sorted(present_programs - set(program_order))
program_position = {program_id: index for index, program_id in enumerate(program_order)}
program_pathway_dotplot_data["program_position"] = (
    program_pathway_dotplot_data["program_id"].map(program_position)
)

preferred_pathway_order = [
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_COMPLEMENT",
    "REACTOME_NEUTROPHIL_DEGRANULATION",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "REACTOME_CELL_CYCLE",
    "REACTOME_CELL_CYCLE_MITOTIC",
    "REACTOME_CELL_CYCLE_CHECKPOINTS",
    "REACTOME_FACTORS_INVOLVED_IN_MEGAKARYOCYTE_DEVELOPMENT_AND_PLATELET_PRODUCTION",
    "REACTOME_RHO_GTPASE_EFFECTORS",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_MYOGENESIS",
    "REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION",
    "REACTOME_SIGNALING_BY_RECEPTOR_TYROSINE_KINASES",
    "REACTOME_SENSORY_PERCEPTION",
]
present_pathways = set(program_pathway_dotplot_data["pathway"])
pathway_order = [p for p in preferred_pathway_order if p in present_pathways]
pathway_order += sorted(present_pathways - set(pathway_order))
pathway_position = {pathway: index for index, pathway in enumerate(pathway_order)}
program_pathway_dotplot_data["pathway_position"] = (
    program_pathway_dotplot_data["pathway"].map(pathway_position)
)

def concise_program_annotation(annotation):
    replacements = {
        "G2M Checkpoint": "G2M",
        "Cell Cycle": "Cell cycle",
        "Cell Cycle Checkpoints": "Checkpoints",
        "E2F Targets": "E2F",
        "Interferon Alpha Response": "IFN-α",
        "Epithelial Mesenchymal Transition": "EMT",
        "Complement": "Complement",
        "Sensory Perception": "Sensory",
    }
    return replacements.get(str(annotation), str(annotation))

def concise_program_pathway(pathway):
    pathway = str(pathway)
    database_prefix = "H:" if pathway.startswith("HALLMARK_") else "R:"
    text = pathway.removeprefix("HALLMARK_").removeprefix("REACTOME_")
    replacements = {
        "INTERFERON_ALPHA_RESPONSE": "Interferon-alpha response",
        "G2M_CHECKPOINT": "G2/M checkpoint",
        "E2F_TARGETS": "E2F targets",
        "CELL_CYCLE": "Cell cycle",
        "CELL_CYCLE_MITOTIC": "Mitotic cell cycle",
        "CELL_CYCLE_CHECKPOINTS": "Cell-cycle checkpoints",
        "FACTORS_INVOLVED_IN_MEGAKARYOCYTE_DEVELOPMENT_AND_PLATELET_PRODUCTION":
            "Megakaryocyte development &\nplatelet production",
        "RHO_GTPASE_EFFECTORS": "Rho GTPase effectors",
        "EPITHELIAL_MESENCHYMAL_TRANSITION": "Epithelial–mesenchymal transition",
        "EXTRACELLULAR_MATRIX_ORGANIZATION": "Extracellular-matrix organisation",
        "SIGNALING_BY_RECEPTOR_TYROSINE_KINASES":
            "Receptor tyrosine-kinase signalling",
        "NEUTROPHIL_DEGRANULATION": "Neutrophil degranulation",
        "SENSORY_PERCEPTION": "Sensory perception",
    }
    label = replacements.get(text, text.replace("_", " ").title())
    return f"{database_prefix} {label}"

x_values = program_pathway_dotplot_data["program_position"].to_numpy()
y_values = program_pathway_dotplot_data["pathway_position"].to_numpy()
overlap_fraction = program_pathway_dotplot_data["program_overlap_fraction"].to_numpy()
dot_sizes = 55 + 620 * overlap_fraction ** 1.6

fig, ax = plt.subplots(figsize=(14.5, 9.2))
dots = ax.scatter(
    x_values,
    y_values,
    s=dot_sizes,
    c=program_pathway_dotplot_data["minus_log10_program_fdr"],
    cmap="magma",
    edgecolor="#303030",
    linewidth=0.8,
    alpha=0.92,
    zorder=3,
)
ax.set_xticks(
    np.arange(len(program_order)),
    [
        f"{program_id}\n{concise_program_annotation(annotation_lookup[program_id])}"
        for program_id in program_order
    ],
)
ax.set_yticks(
    np.arange(len(pathway_order)),
    [concise_program_pathway(pathway) for pathway in pathway_order],
)
ax.invert_yaxis()
ax.set_xlabel("scGPT embedding gene program")
ax.set_ylabel("Provisional enriched pathway")
ax.set_axisbelow(True)
ax.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.24)
for boundary in (2.5, 7.5, 8.5):
    ax.axvline(boundary, color="#666666", linewidth=1.0, alpha=0.24)
for boundary in (2.5, 9.5, 13.5):
    ax.axhline(boundary, color="#666666", linewidth=1.0, alpha=0.24)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=10)
fig.suptitle(
    "Exploratory functional annotation of embedding-derived gene programs",
    y=0.965,
    fontsize=17,
    fontweight="semibold",
)
ax.set_title(
    "Shown: within-program BH FDR < 0.05 and overlap ≥ 3 genes; "
    f"no pair survives global FDR (minimum q = {program_annotation['program_ora']['global_fdr'].min():.3f})",
    fontsize=10,
    color="#555555",
    pad=14,
)

# Both visual keys are placed below the plot for a clean presentation panel.
colourbar_axis = fig.add_axes([0.18, 0.075, 0.30, 0.025])
colourbar = fig.colorbar(dots, cax=colourbar_axis, orientation="horizontal")
colourbar.set_label("Within-program evidence, −log10(q)", fontsize=9)
colourbar.ax.tick_params(labelsize=8)

size_reference = [0.4, 0.6, 0.8]
size_handles = [
    plt.Line2D(
        [0], [0], marker="o", linestyle="none",
        markerfacecolor="#777777", markeredgecolor="#303030",
        markersize=np.sqrt(55 + 620 * fraction ** 1.6) / 1.55,
        label=f"{fraction:.0%}",
    )
    for fraction in size_reference
]
fig.legend(
    handles=size_handles,
    title="Program coverage (overlap/program genes)",
    frameon=False,
    loc="lower center",
    bbox_to_anchor=(0.73, 0.045),
    ncol=3,
    columnspacing=1.5,
    handletextpad=0.5,
    fontsize=9,
    title_fontsize=9,
)
fig.subplots_adjust(left=0.36, right=0.98, top=0.86, bottom=0.20)
fig.savefig(
    OUTPUT_DIR / "embedding_gene_program_pathway_map.png",
    dpi=240,
    bbox_inches="tight",
)
plt.show()


# ## 5. Run matched perturbation-model counterfactuals
# 
# The same real control cell is evaluated three times. The input expression
# vector, gene order and starting cell identity do not change; only the
# perturbation flag changes:
# 
# model effect = prediction(same control cell, IR flag)
#              - prediction(same control cell, control flag)
# 
# This is called a **model counterfactual** because it asks what the checkpoint
# predicts for the same baseline cell under a condition label it did not actually
# receive. It holds the input cell and sampled control composition fixed, which
# makes it cleaner than subtracting unrelated treated and control cells for the
# model comparison. It is still a model-generated conditional contrast: it is
# not an observed treated-minus-control value, a longitudinal measurement of one
# cell, a patient-level forecast, or proof of a causal treatment effect. It is not
# obtained by creating separate 24-hour and one-week embedding networks.

# In[12]:


def run_matched_counterfactual_predictions(
    config: AnalysisConfig,
    aligned: dict,
    gene_names: Sequence[str],
) -> dict:
    """Apply every perturbation flag to the same four control-sample inputs.

    GEARS training graphs pair treated targets with randomly sampled control
    cells.  Those saved predictions are therefore not linked counterfactuals
    for the target cell metadata.  For biological interpretation, this
    function instead selects a balanced, deterministic pool of real control
    cells and applies ctrl, IR_24h and IR_1week flags to every identical input.
    The resulting within-cell differences are aggregated to the four source
    samples and are the direct model contrasts.
    """

    os.environ.setdefault("NUMBA_CACHE_DIR", str(config.output_path / ".numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(config.output_path / ".mpl_cache"))
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    with (config.asset_dir / "args.json").open() as handle:
        model_config = json.load(handle)
    vocab = GeneVocab.from_file(config.asset_dir / "vocab.json")
    vocab.set_default_index(vocab["<pad>"])

    checkpoint = torch.load(config.run_dir / "best_model.pt", map_location="cpu")
    n_model_features = int(aligned["expression"].shape[1])
    model_features = list(gene_names[:n_model_features])
    expected_tokens = {"IR_24h", "IR_1week"}
    if not expected_tokens.issubset(model_features):
        raise ValueError("The prediction feature axis lacks the two underscore perturbation tokens")
    if set(model_features[-2:]) != expected_tokens:
        raise ValueError("Perturbation tokens are not the final two model input features")

    model = TransformerGenerator(
        len(vocab),
        int(model_config["embsize"]),
        int(model_config["nheads"]),
        int(model_config["d_hid"]),
        int(model_config["nlayers"]),
        nlayers_cls=int(model_config["n_layers_cls"]),
        n_cls=1,
        vocab=vocab,
        dropout=0.0,
        pad_token="<pad>",
        pad_value=0,
        pert_pad_id=0,
        use_fast_transformer=False,
    )
    model.load_state_dict(checkpoint, strict=True)

    requested_device = config.counterfactual_device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    if requested_device == "mps" and not torch.backends.mps.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = model.to(device).eval()

    metadata = aligned["metadata"]
    expression = aligned["expression"][:, :n_model_features]
    ctrl_mask = metadata["condition_canonical"].eq("ctrl")
    control_samples = sorted(metadata.loc[ctrl_mask, "Sample"].astype(str).unique().tolist())
    rng = np.random.default_rng(config.seed)
    selected_indices = []
    input_sample_labels = []
    input_cell_ids = []
    audit_rows = []
    for sample in control_samples:
        mask = ctrl_mask & metadata["Sample"].astype(str).eq(sample)
        available = np.flatnonzero(mask.to_numpy())
        n_selected = min(config.counterfactual_cells_per_sample, len(available))
        chosen = np.sort(rng.choice(available, size=n_selected, replace=False))
        selected_indices.extend(chosen.tolist())
        input_sample_labels.extend([sample] * n_selected)
        input_cell_ids.extend(metadata.iloc[chosen]["cell_id"].astype(str).tolist())
        audit_rows.append(
            {
                "control_input_sample": sample,
                "available_control_cells": int(mask.sum()),
                "selected_control_cells": n_selected,
                "input_representation": "real normalized log1p control cells",
            }
        )
    selected_indices = np.asarray(selected_indices, dtype=int)
    input_profiles = expression[selected_indices].astype(np.float32, copy=True)

    base_mapped_gene_ids = torch.tensor(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in model_features],
        dtype=torch.long,
        device=device,
    )

    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for condition, token in (
            ("ctrl", None),
            ("IR_24h", "IR_24h"),
            ("IR_1week", "IR_1week"),
        ):
            output_parts = []
            for start in range(0, len(input_profiles), config.counterfactual_batch_size):
                stop = min(start + config.counterfactual_batch_size, len(input_profiles))
                values = torch.from_numpy(input_profiles[start:stop]).to(device)
                mapped_gene_ids = base_mapped_gene_ids.repeat(stop - start, 1)
                padding_mask = torch.zeros_like(values, dtype=torch.bool, device=device)
                flags = torch.zeros_like(values, dtype=torch.long, device=device)
                if token is not None:
                    flags[:, model_features.index(token)] = 1
                output = model(
                    mapped_gene_ids,
                    values,
                    flags,
                    src_key_padding_mask=padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=True,
                )["mlm_output"]
                output_parts.append(output.float().cpu().numpy().astype(np.float32))
            predictions[condition] = np.concatenate(output_parts, axis=0)

    inference_audit = pd.DataFrame(audit_rows)
    provenance = pd.DataFrame(
        [
            {"check": "counterfactual_input_mode", "value": "balanced_real_control_cells"},
            {"check": "matched_control_inputs", "value": len(input_profiles)},
            {"check": "biological_source_samples", "value": len(control_samples)},
            {"check": "features_per_input", "value": n_model_features},
            {"check": "inference_device", "value": str(device)},
            {
                "check": "contrast_definition",
                "value": "prediction(same control input, perturbation flag) - prediction(same input, ctrl flag)",
            },
        ]
    )
    return {
        "control_samples": control_samples,
        "input_sample_labels": input_sample_labels,
        "input_cell_ids": input_cell_ids,
        "input_profiles": input_profiles,
        "predictions": predictions,
        "input_audit": inference_audit,
        "provenance": provenance,
        "model_features": model_features,
    }


# In[13]:


counterfactual = run_matched_counterfactual_predictions(
    config, aligned, embedding_result["gene_names"]
)
display(counterfactual["input_audit"])
display(counterfactual["provenance"])


# ## 6. Optional diagnostic: compare gene-program effects
# 
# For each gene program, expression is standardised using control cells and then
# averaged across member genes. Observed effects use sample-balanced
# treated-minus-control differences. Model effects use the matched counterfactual
# differences above. Because exact embedding-program membership is seed-sensitive
# and programs may mix several biological processes, this section is a model
# diagnostic rather than the primary biological interpretation.

# In[14]:


def compute_module_and_gene_effects(
    config: AnalysisConfig,
    aligned: dict,
    modules: dict,
    counterfactual: dict,
) -> dict:
    metadata = aligned["metadata"].copy()
    expression = aligned["expression"]
    nodes = modules["nodes"].copy()
    module_ids = modules["module_sizes"]["module_id"].tolist()
    source_feature_groups = [tuple(int(i) for i in values) for values in nodes["source_feature_indices"]]

    def collapse_source_features(matrix: np.ndarray) -> np.ndarray:
        collapsed = np.empty((matrix.shape[0], len(source_feature_groups)), dtype=np.float32)
        for column, indices in enumerate(source_feature_groups):
            collapsed[:, column] = matrix[:, indices].mean(axis=1)
        return collapsed

    eligible_expression = collapse_source_features(expression)
    counterfactual_expression = {
        condition: collapse_source_features(values)
        for condition, values in counterfactual["predictions"].items()
    }

    ctrl_mask = metadata["condition_canonical"].eq("ctrl").to_numpy()
    control_values = eligible_expression[ctrl_mask]
    control_mean = control_values.mean(axis=0, dtype=np.float64)
    control_sd = control_values.std(axis=0, dtype=np.float64)
    control_sd[control_sd < 1e-6] = 1.0

    observed_scores = np.empty((len(metadata), len(module_ids)), dtype=np.float32)
    counterfactual_module_scores = {
        condition: np.empty((len(values), len(module_ids)), dtype=np.float32)
        for condition, values in counterfactual_expression.items()
    }
    for module_index, module_id in enumerate(module_ids):
        member_rows = np.where(nodes["module_id"].to_numpy() == module_id)[0]
        member_mean = control_mean[member_rows]
        member_sd = control_sd[member_rows]
        observed_scores[:, module_index] = (
            (eligible_expression[:, member_rows] - member_mean) / member_sd
        ).mean(axis=1)
        for condition, values in counterfactual_expression.items():
            counterfactual_module_scores[condition][:, module_index] = (
                (values[:, member_rows] - member_mean) / member_sd
            ).mean(axis=1)

    metadata_columns = [
        "cell_id",
        "Sample",
        "Batch",
        "Sex",
        "Timepoint",
        "GBM_malig_subtypes",
        "survival_classification_adata",
        "condition_canonical",
    ]
    obs_score_df = pd.DataFrame(observed_scores, columns=module_ids)
    score_metadata = metadata[metadata_columns].reset_index(drop=True)
    obs_with_meta = pd.concat([score_metadata, obs_score_df], axis=1)

    sample_obs = obs_with_meta.groupby(["Sample", "condition_canonical"], observed=True)[module_ids].mean()
    sample_pred_parts = []
    for condition, scores in counterfactual_module_scores.items():
        frame = pd.DataFrame(scores, columns=module_ids)
        frame.insert(0, "condition", condition)
        frame.insert(0, "control_input_sample", counterfactual["input_sample_labels"])
        sample_pred_parts.append(
            frame.groupby(["control_input_sample", "condition"], as_index=False)[module_ids].mean()
        )
    sample_pred = pd.concat(sample_pred_parts, ignore_index=True)

    effect_rows = []
    for condition in ("IR_24h", "IR_1week"):
        obs_condition = sample_obs.xs(condition, level="condition_canonical")
        obs_control = sample_obs.xs("ctrl", level="condition_canonical")
        pred_condition = counterfactual_module_scores[condition]
        pred_control = counterfactual_module_scores["ctrl"]
        for module_id in module_ids:
            module_index = module_ids.index(module_id)
            obs_delta = float(obs_condition[module_id].mean() - obs_control[module_id].mean())
            paired_cell_deltas = pred_condition[:, module_index] - pred_control[:, module_index]
            paired_predicted_deltas = (
                pd.DataFrame(
                    {
                        "control_input_sample": counterfactual["input_sample_labels"],
                        "delta": paired_cell_deltas,
                    }
                )
                .groupby("control_input_sample", sort=True)["delta"]
                .mean()
                .to_numpy()
            )
            pred_delta = float(paired_predicted_deltas.mean())
            observed_test = ttest_ind(
                obs_condition[module_id].to_numpy(),
                obs_control[module_id].to_numpy(),
                equal_var=False,
                nan_policy="omit",
            )
            predicted_test = ttest_1samp(paired_predicted_deltas, popmean=0.0, nan_policy="omit")
            effect_rows.append(
                {
                    "condition": condition,
                    "module_id": module_id,
                    "observed_delta": obs_delta,
                    "predicted_counterfactual_delta": pred_delta,
                    "predicted_counterfactual_sd": float(np.std(paired_predicted_deltas, ddof=1)),
                    "observed_welch_p": float(observed_test.pvalue),
                    "predicted_paired_input_p": float(predicted_test.pvalue),
                    "n_condition_samples": int(len(obs_condition)),
                    "n_control_samples": int(len(obs_control)),
                    "n_matched_control_inputs": int(len(paired_predicted_deltas)),
                }
            )
    module_effects = pd.DataFrame(effect_rows)
    module_effects["observed_welch_fdr"] = module_effects.groupby("condition")["observed_welch_p"].transform(
        lambda values: _bh_adjust(values)
    )
    module_effects["predicted_paired_input_fdr"] = module_effects.groupby("condition")[
        "predicted_paired_input_p"
    ].transform(lambda values: _bh_adjust(values))
    module_effects = module_effects.merge(modules["module_sizes"], on="module_id", how="left")

    recovery_rows = []
    for condition, frame in module_effects.groupby("condition"):
        pearson_r, pearson_p = _safe_pearson(
            frame["observed_delta"], frame["predicted_counterfactual_delta"]
        )
        spearman_r, spearman_p = _safe_spearman(
            frame["observed_delta"], frame["predicted_counterfactual_delta"]
        )
        sign_match = float(
            np.mean(
                np.sign(frame["observed_delta"].to_numpy())
                == np.sign(frame["predicted_counterfactual_delta"].to_numpy())
            )
        )
        recovery_rows.append(
            {
                "condition": condition,
                "n_modules": len(frame),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "sign_concordance": sign_match,
            }
        )
    recovery = pd.DataFrame(recovery_rows)

    sample_obs_out = sample_obs.reset_index()
    sample_pred_out = sample_pred.reset_index(drop=True)

    sample_gene_obs_parts = []
    for (sample, condition), row_indices in metadata.groupby(["Sample", "condition_canonical"], observed=True).groups.items():
        idx = np.asarray(list(row_indices), dtype=int)
        sample_gene_obs_parts.append((sample, condition, eligible_expression[idx].mean(axis=0)))

    def _condition_gene_mean(parts: list[tuple[str, str, np.ndarray]], condition: str) -> np.ndarray:
        values = [value for _, cond, value in parts if cond == condition]
        if not values:
            raise ValueError(f"No sample-level gene values for {condition}")
        return np.stack(values).mean(axis=0)

    obs_ctrl_gene = _condition_gene_mean(sample_gene_obs_parts, "ctrl")
    rank_frames = []
    for condition in ("IR_24h", "IR_1week"):
        obs_delta = _condition_gene_mean(sample_gene_obs_parts, condition) - obs_ctrl_gene
        paired_cell_delta = (
            counterfactual_expression[condition] - counterfactual_expression["ctrl"]
        )
        paired_sample_delta = (
            pd.DataFrame(paired_cell_delta)
            .assign(control_input_sample=counterfactual["input_sample_labels"])
            .groupby("control_input_sample", sort=True)
            .mean()
            .to_numpy()
        )
        predicted_delta = paired_sample_delta.mean(axis=0)
        frame = nodes[
            [
                "feature_index",
                "source_feature_indices",
                "source_feature_count",
                "model_gene",
                "human_symbol",
                "module_id",
            ]
        ].copy()
        frame["condition"] = condition
        frame["observed_delta"] = obs_delta
        frame["predicted_counterfactual_delta"] = predicted_delta
        frame["predicted_counterfactual_sd"] = paired_sample_delta.std(axis=0, ddof=1)
        rank_frames.append(frame)
    feature_ranks = pd.concat(rank_frames, ignore_index=True)
    gene_ranks = (
        feature_ranks.groupby(["condition", "human_symbol"], as_index=False)
        .agg(
            observed_delta=("observed_delta", "mean"),
            predicted_counterfactual_delta=("predicted_counterfactual_delta", "mean"),
            predicted_counterfactual_sd=("predicted_counterfactual_sd", "mean"),
            n_collapsed_features=("source_feature_count", "sum"),
            model_genes=("model_gene", lambda x: ";".join(sorted(set(x.astype(str))))),
        )
    )
    return {
        "observed_scores": observed_scores,
        "counterfactual_module_scores": counterfactual_module_scores,
        "score_metadata": score_metadata,
        "sample_module_scores_observed": sample_obs_out,
        "sample_module_scores_counterfactual": sample_pred_out,
        "module_effects": module_effects,
        "module_recovery": recovery,
        "feature_ranks": feature_ranks,
        "gene_ranks": gene_ranks,
    }


# In[15]:


effects = compute_module_and_gene_effects(
    config, aligned, modules, counterfactual
)

agreement_summary = effects["module_recovery"].rename(columns={
    "n_modules": "n_gene_programs",
    "pearson_r": "pearson_observed_vs_model",
    "spearman_r": "spearman_observed_vs_model",
})
magnitude_ratio = (
    effects["module_effects"]
    .groupby("condition")
    .apply(
        lambda frame: frame["predicted_counterfactual_delta"].abs().mean()
        / frame["observed_delta"].abs().mean()
    )
    .rename("mean_absolute_model_to_observed_ratio")
    .reset_index()
)
agreement_summary = agreement_summary.merge(magnitude_ratio, on="condition", how="left")
display(agreement_summary[[
    "condition", "n_gene_programs", "pearson_observed_vs_model",
    "spearman_observed_vs_model", "mean_absolute_model_to_observed_ratio"
]])


# The table above is what the previous module recovery plot meant. Recovery only
# meant agreement between observed and model-predicted gene-program changes. It
# did not mean recovery of a regulatory network. The grouped bars below make that
# comparison more directly.

# In[16]:


program_effects = effects["module_effects"].copy()
observed_sample_scores = effects["sample_module_scores_observed"]
predicted_sample_scores = effects["sample_module_scores_counterfactual"]

def compact_program_label(row):
    genes = str(row["central_genes"]).split(";")[:3]
    return f"{row['program_id']} — {row['annotation']} ({', '.join(genes)})"

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
for ax, condition, title in zip(
    axes,
    ("IR_24h", "IR_1week"),
    ("24 hours minus control", "1 week minus control"),
):
    frame = program_effects[program_effects["condition"] == condition].copy()
    frame["absolute_observed_effect"] = frame["observed_delta"].abs()
    frame = (
        frame.nlargest(10, "absolute_observed_effect")
        .sort_values("observed_delta")
        .reset_index(drop=True)
    )
    observed_sd = []
    predicted_sd = []
    for module_id in frame["module_id"]:
        control_mean = observed_sample_scores.loc[
            observed_sample_scores["condition_canonical"].eq("ctrl"), module_id
        ].mean()
        independent_treated_deltas = (
            observed_sample_scores.loc[
                observed_sample_scores["condition_canonical"].eq(condition), module_id
            ]
            - control_mean
        )
        observed_sd.append(independent_treated_deltas.std(ddof=1))
        predicted_wide = predicted_sample_scores.pivot(
            index="control_input_sample", columns="condition", values=module_id
        )
        predicted_sd.append(
            (predicted_wide[condition] - predicted_wide["ctrl"]).std(ddof=1)
        )
    labels = frame.apply(compact_program_label, axis=1)
    y = np.arange(len(frame))
    ax.barh(
        y - 0.19, frame["observed_delta"], height=0.36,
        xerr=observed_sd, error_kw={"linewidth": 0.8, "capsize": 2},
        color="#3b6ea8", label="Observed"
    )
    ax.barh(
        y + 0.19, frame["predicted_counterfactual_delta"], height=0.36,
        xerr=predicted_sd, error_kw={"linewidth": 0.8, "capsize": 2},
        color="#d58a35", label="Model counterfactual"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_title(title)
    ax.set_xlabel("Gene-program score change")
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle(
    "Diagnostic: embedding gene-program changes (mean ± SD across four sample-level contrasts)",
    y=0.99,
)
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
fig.savefig(OUTPUT_DIR / "gene_program_observed_vs_model.png", dpi=220, bbox_inches="tight")
plt.show()


# ### How to read the gene-program comparison
# 
# Blue bars are observed sample-balanced treated-minus-control changes; orange
# bars are changes predicted from matched control-cell counterfactuals. A bar on
# the same side of zero means the observed data and model agree on direction;
# similar bar length means they also agree on approximate magnitude. Error bars
# are mean ± one standard deviation across four sample-level contrasts. They show
# between-sample variability, not a 95% confidence interval and not a p-value.
# Only the ten programs with the largest absolute observed change at each time
# are plotted, so the figure is a focused recovery diagnostic rather than a
# complete or statistically selected program catalogue. It tests recovery of
# program scores, not recovery of regulatory edges.

# ### Complete all-program presentation figures
# 
# The selected-bar plot above is useful for individual examples but does not show
# the entire partition. The following recommended figures retain all 393
# programs:
# 
# 1. a four-column response atlas containing observed and model effects at both
#    time points;
# 2. two observed-versus-model scatterplots containing every program; and
# 3. a program-size and seed-membership stability diagnostic.
# 
# The heatmap displays raw program-score differences on one common zero-centred
# scale. Ward clustering of the four standardized response columns is used only
# to order the rows for visual inspection; it does not replace or retune the
# Louvain membership. Open circles mark within-run program-effect FDR below 0.05.
# Observed and model FDRs come from different tests and should not be compared as
# if they had the same sampling null.

# In[ ]:


SEED = config.seed

def _save(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})

def _effect_matrix(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    program_ids = sorted(effects["program_id"].astype(str).unique())
    values = pd.DataFrame(index=program_ids)
    fdr = pd.DataFrame(index=program_ids)
    mapping = (
        ("IR_24h", "observed_delta", "observed_welch_fdr", "Observed 24h"),
        (
            "IR_24h",
            "predicted_counterfactual_delta",
            "predicted_paired_input_fdr",
            "Model 24h",
        ),
        ("IR_1week", "observed_delta", "observed_welch_fdr", "Observed 1 week"),
        (
            "IR_1week",
            "predicted_counterfactual_delta",
            "predicted_paired_input_fdr",
            "Model 1 week",
        ),
    )
    for condition, value_column, fdr_column, label in mapping:
        frame = effects.loc[effects["condition"].eq(condition)].set_index("program_id")
        values[label] = frame[value_column].reindex(program_ids).astype(float)
        fdr[label] = frame[fdr_column].reindex(program_ids).astype(float)
    values.index.name = "program_id"
    fdr.index.name = "program_id"
    return values, fdr

def _program_metadata(effects: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "program_id",
        "module_id",
        "n_features",
        "n_human_symbols",
        "central_genes",
        "median_seed_jaccard",
        "min_seed_jaccard",
        "annotation",
        "annotation_is_program_fdr_significant",
        "annotation_program_fdr",
        "annotation_global_fdr",
    ]
    available = [column for column in columns if column in effects.columns]
    metadata = effects[available].drop_duplicates("program_id").set_index("program_id")
    metadata["annotation_is_program_fdr_significant"] = _as_bool(
        metadata["annotation_is_program_fdr_significant"]
    )
    return metadata

def make_all_program_atlas(
    effects: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, fdr = _effect_matrix(effects)
    metadata = _program_metadata(effects).reindex(values.index)
    matrix = values.to_numpy(dtype=float)
    column_sd = np.nanstd(matrix, axis=0, ddof=1)
    scaled = (matrix - np.nanmean(matrix, axis=0)) / np.where(column_sd > 0, column_sd, 1)
    row_order = leaves_list(linkage(scaled, method="ward", metric="euclidean"))
    ordered_values = values.iloc[row_order]
    ordered_fdr = fdr.reindex(ordered_values.index)
    ordered_metadata = metadata.reindex(ordered_values.index)

    atlas_export = ordered_metadata.join(ordered_values).join(
        ordered_fdr.add_suffix(" FDR")
    )
    atlas_export.insert(0, "display_order", np.arange(1, len(atlas_export) + 1))
    atlas_export.to_csv(output_dir / "all_program_response_atlas_data.csv")

    effect_limit = float(np.nanmax(np.abs(ordered_values.to_numpy(dtype=float))))
    effect_limit = max(effect_limit, 1e-9)
    sizes = ordered_metadata["n_features"].astype(float).to_numpy()
    stability = ordered_metadata["median_seed_jaccard"].astype(float).to_numpy()
    annotated = ordered_metadata["annotation_is_program_fdr_significant"].astype(bool).to_numpy()

    fig = plt.figure(figsize=(12.8, 15.0))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(7.2, 0.48, 0.48, 0.34),
        left=0.20,
        right=0.96,
        bottom=0.12,
        top=0.91,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(
        ordered_values.to_numpy(dtype=float),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=Normalize(-effect_limit, effect_limit),
    )
    significant_y, significant_x = np.where(ordered_fdr.to_numpy(dtype=float) < 0.05)
    ax.scatter(
        significant_x,
        significant_y,
        s=5.5,
        facecolors="none",
        edgecolors="#171717",
        linewidths=0.35,
    )
    ax.set_xticks(np.arange(len(ordered_values.columns)), ordered_values.columns)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    label_positions = np.flatnonzero(annotated).astype(float)
    displayed_positions = label_positions.copy()
    minimum_gap = 7.0
    for index in range(1, len(displayed_positions)):
        displayed_positions[index] = max(
            displayed_positions[index], displayed_positions[index - 1] + minimum_gap
        )
    overflow = displayed_positions[-1] - (len(ordered_values) - 1)
    if overflow > 0:
        displayed_positions -= overflow
        for index in range(len(displayed_positions) - 2, -1, -1):
            displayed_positions[index] = min(
                displayed_positions[index], displayed_positions[index + 1] - minimum_gap
            )
    ax.set_yticks([])
    y_transform = ax.get_yaxis_transform()
    for true_position, displayed_position in zip(label_positions, displayed_positions):
        program_id = ordered_values.index[int(true_position)]
        annotation = ordered_metadata.iloc[int(true_position)]["annotation"]
        ax.plot(
            [-0.006, -0.024],
            [true_position, displayed_position],
            transform=y_transform,
            color="#777777",
            linewidth=0.45,
            clip_on=False,
        )
        ax.text(
            -0.030,
            displayed_position,
            f"{program_id} — {annotation}",
            transform=y_transform,
            ha="right",
            va="center",
            fontsize=7.3,
            clip_on=False,
        )
    ax.set_ylabel("All 393 Louvain embedding programs\n(only provisional pathway labels shown)")
    ax.set_title("A  Complete program-response atlas", loc="left", fontweight="bold")
    ax.grid(False)

    side_axes = [fig.add_subplot(grid[0, index]) for index in range(1, 4)]
    side_axes[0].imshow(
        sizes[:, None],
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        norm=Normalize(float(np.nanmin(sizes)), float(np.nanmax(sizes))),
    )
    side_axes[1].imshow(
        stability[:, None],
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=Normalize(0, 1),
    )
    side_axes[2].imshow(
        annotated.astype(float)[:, None],
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#d7d7d7", "#d58a35"]),
        norm=Normalize(0, 1),
    )
    for side_ax, label in zip(
        side_axes,
        (
            f"Genes\n{int(np.nanmin(sizes))}–{int(np.nanmax(sizes))}",
            "Seed\nJaccard",
            "ORA\nlabel",
        ),
    ):
        side_ax.set_xticks([0], [label], fontsize=7.5)
        side_ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
        side_ax.set_yticks([])
        side_ax.grid(False)
        for spine in side_ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(
        image,
        ax=[ax, *side_axes],
        orientation="horizontal",
        fraction=0.032,
        pad=0.055,
        aspect=48,
    )
    cbar.set_label("Raw gene-program score difference (red = increased; blue = decreased)")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#171717",
                markersize=5,
                label="Within-run program-effect FDR < 0.05",
            )
        ],
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "All embedding-derived programs across observed and model contrasts\n"
        "row order uses Ward clustering of the four standardized response columns for display only",
        y=0.985,
    )
    _save(fig, output_dir / "all_program_response_atlas", dpi=300)
    return atlas_export, metadata

def make_all_program_scatter(effects: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metadata = _program_metadata(effects)
    summary_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    for ax, condition, title in zip(
        axes,
        ("IR_24h", "IR_1week"),
        ("24 hours minus control", "1 week minus control"),
    ):
        frame = effects.loc[effects["condition"].eq(condition)].copy()
        frame["annotated"] = _as_bool(frame["annotation_is_program_fdr_significant"])
        frame["any_effect_fdr"] = (
            frame["observed_welch_fdr"].lt(0.05)
            | frame["predicted_paired_input_fdr"].lt(0.05)
        )
        x = frame["observed_delta"].to_numpy(dtype=float)
        y = frame["predicted_counterfactual_delta"].to_numpy(dtype=float)
        sizes = 13 + 4.2 * frame["n_features"].to_numpy(dtype=float)
        limit = float(np.nanmax(np.abs(np.r_[x, y]))) * 1.08
        limit = max(limit, 1e-6)

        for is_annotated, color, zorder in (
            (False, "#7e8792", 2),
            (True, "#d58a35", 4),
        ):
            for is_significant, linewidth in ((False, 0.0), (True, 0.75)):
                mask = frame["annotated"].eq(is_annotated) & frame["any_effect_fdr"].eq(
                    is_significant
                )
                ax.scatter(
                    x[mask],
                    y[mask],
                    s=sizes[mask],
                    color=color,
                    alpha=0.52 if not is_annotated else 0.86,
                    edgecolor="#111111" if is_significant else "none",
                    linewidth=linewidth,
                    zorder=zorder,
                )

        ax.plot([-limit, limit], [-limit, limit], color="#333333", linewidth=1, linestyle="--")
        ax.axhline(0, color="#b0b0b0", linewidth=0.65)
        ax.axvline(0, color="#b0b0b0", linewidth=0.65)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed program-score difference")
        ax.set_ylabel("Model counterfactual difference")

        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
        sign = float((np.sign(x) == np.sign(y)).mean())
        attenuation = float(np.mean(np.abs(y)) / max(np.mean(np.abs(x)), 1e-12))
        ax.set_title(title)
        ax.text(
            0.025,
            0.975,
            f"Pearson r={pearson:.2f}\nSpearman ρ={spearman:.2f}\n"
            f"Same sign={sign:.1%}\n|model|/|observed|={attenuation:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.86},
        )
        summary_rows.append(
            {
                "condition": condition,
                "n_programs": len(frame),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "sign_concordance": sign,
                "mean_absolute_model_over_observed": attenuation,
            }
        )

        key_programs = {"P013", "P162", "P003", "P045", "P174", "P304", "P311"}
        annotated_frame = frame.loc[frame["program_id"].isin(key_programs)].copy()
        label_offsets = {
            "P013": (4, -12),
            "P162": (4, 5),
            "P003": (-27, 5),
            "P045": (-27, -12),
            "P174": (4, -14),
            "P304": (4, 5),
            "P311": (4, 5),
        }
        for row in annotated_frame.itertuples(index=False):
            offset = label_offsets.get(str(row.program_id), (3, 3))
            ax.annotate(
                str(row.program_id),
                (float(row.observed_delta), float(row.predicted_counterfactual_delta)),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.4,
                fontweight="bold",
                color="#4a3217",
            )
        ax.grid(alpha=0.16)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#7e8792",
            markeredgecolor="none", markersize=7, label="Unannotated program"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#d58a35",
            markeredgecolor="none", markersize=7, label="Provisional pathway annotation"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#b5b5b5",
            markeredgecolor="#111111", markersize=7, label="Either effect FDR < 0.05"
        ),
        Line2D(
            [0], [0], color="#333333", linestyle="--", label="Observed = model"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Observed versus model-predicted changes for every Louvain gene program",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _save(fig, output_dir / "all_program_observed_vs_model_scatter", dpi=300)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "all_program_observed_vs_model_metrics.csv", index=False)
    return summary

def make_program_size_stability(effects: pd.DataFrame, selection: pd.DataFrame, output_dir: Path) -> None:
    metadata = _program_metadata(effects).reset_index()
    size_counts = metadata["n_features"].value_counts().sort_index()
    median_jaccard = float(metadata["median_seed_jaccard"].median())
    stable_fraction = float(metadata["median_seed_jaccard"].ge(0.70).mean())
    silhouette = float(selection.iloc[0]["cosine_silhouette"])
    interpretation = str(selection.iloc[0]["stability_interpretation"])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    axes[0].bar(
        size_counts.index.astype(int),
        size_counts.to_numpy(dtype=int),
        color="#557da9",
        width=0.72,
    )
    for size, count in size_counts.items():
        axes[0].text(size, count + 2, str(int(count)), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(size_counts.index.astype(int))
    axes[0].set_xlabel("Genes per program")
    axes[0].set_ylabel("Number of programs")
    axes[0].set_title("A  Program-size distribution")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].grid(axis="x", visible=False)

    rng = np.random.default_rng(SEED)
    jittered_size = metadata["n_features"].to_numpy(dtype=float) + rng.normal(
        0, 0.065, len(metadata)
    )
    annotated = metadata["annotation_is_program_fdr_significant"].astype(bool).to_numpy()
    axes[1].scatter(
        jittered_size[~annotated],
        metadata.loc[~annotated, "median_seed_jaccard"],
        s=19,
        color="#7e8792",
        alpha=0.48,
        edgecolor="none",
    )
    axes[1].scatter(
        jittered_size[annotated],
        metadata.loc[annotated, "median_seed_jaccard"],
        s=42,
        color="#d58a35",
        alpha=0.90,
        edgecolor="#333333",
        linewidth=0.4,
    )
    axes[1].axhline(0.70, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_xticks(size_counts.index.astype(int))
    axes[1].set_ylim(-0.02, 1.03)
    axes[1].set_xlabel("Genes per program")
    axes[1].set_ylabel("Median best-match Jaccard across Louvain seeds")
    axes[1].set_title(
        f"B  Membership stability\nmedian={median_jaccard:.2f}; "
        f"{stable_fraction:.1%} at Jaccard ≥ 0.70"
    )
    axes[1].grid(alpha=0.18)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#7e8792",
            markeredgecolor="none", markersize=7, label="Unannotated program"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#d58a35",
            markeredgecolor="#333333", markersize=7, label="Provisional pathway annotation"
        ),
        Line2D([0], [0], color="#333333", linestyle="--", label="Jaccard = 0.70"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        f"Louvain program granularity and robustness: {len(metadata)} programs; "
        f"cosine silhouette={silhouette:.3f} ({interpretation})",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    _save(fig, output_dir / "program_size_stability_diagnostic", dpi=300)

all_program_atlas_data, all_program_metadata = make_all_program_atlas(
    program_effects, OUTPUT_DIR
)
all_program_recovery_metrics = make_all_program_scatter(
    program_effects, OUTPUT_DIR
)
make_program_size_stability(
    program_effects, modules["selection"], OUTPUT_DIR
)

display(all_program_recovery_metrics)
display(all_program_atlas_data.head(12))


# ### How to use the complete program figures
# 
# The response atlas is the preferred full-cluster overview: every row is one
# fixed Louvain program and the same program is followed through all four
# contrasts. The scatterplots quantify descriptive model recovery across all
# programs; distance from the diagonal is disagreement in effect magnitude, not a
# confidence interval or a formal observed-versus-model test. The size/stability
# figure separates granularity from robustness: Jaccard agreement across Louvain
# seeds measures membership reproducibility, whereas the negative cosine
# silhouette still means the programs are not cleanly separated in embedding
# space. Program annotations remain provisional because none survives the global
# program-by-pathway correction.

# ## 7. Ranked pathway enrichment
# 
# Preranked GSEA is run on all gene-level changes. This is the primary biological
# interpretation because it does not depend on how Louvain divided genes into
# programs.
# 
# For each pathway:
# 
# - positive NES means pathway genes tend to increase after IR;
# - negative NES means pathway genes tend to decrease after IR;
# - FDR is the multiple-testing adjusted significance value;
# - observed and model columns answer whether the checkpoint predicts the same
#   pathway direction.

# In[17]:


def run_ranked_gsea(config: AnalysisConfig, gene_ranks: pd.DataFrame) -> dict:
    universe = set(gene_ranks["human_symbol"].astype(str))
    gene_sets = load_gene_sets(config, universe=universe, include_go=False)
    result_frames = []
    for condition in ("IR_24h", "IR_1week"):
        condition_frame = gene_ranks[gene_ranks["condition"] == condition]
        for representation, value_column in (
            ("observed", "observed_delta"),
            ("predicted_counterfactual", "predicted_counterfactual_delta"),
        ):
            rank = (
                condition_frame[["human_symbol", value_column]]
                .dropna()
                .sort_values(value_column, ascending=False)
            )
            rank.columns = ["gene", "score"]
            pre = gp.prerank(
                rnk=rank,
                gene_sets=gene_sets,
                min_size=config.min_gene_set_size,
                max_size=config.max_gene_set_size,
                permutation_num=config.gsea_permutations,
                outdir=None,
                seed=config.seed,
                threads=1,
                verbose=False,
            )
            res = pre.res2d.copy().reset_index(drop=True)
            if "Term" not in res.columns:
                first = res.columns[0]
                res = res.rename(columns={first: "Term"})
            res["condition"] = condition
            res["representation"] = representation
            result_frames.append(res)

    gsea = pd.concat(result_frames, ignore_index=True)
    numeric_columns = ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]
    for column in numeric_columns:
        if column in gsea.columns:
            gsea[column] = pd.to_numeric(gsea[column], errors="coerce")
    gsea["database"] = gsea["Term"].astype(str).str.split("::", n=1).str[0]
    gsea["pathway"] = gsea["Term"].astype(str).str.split("::", n=1).str[-1]

    recovery_rows = []
    recovery_metrics = []
    for condition in ("IR_24h", "IR_1week"):
        observed = gsea[(gsea["condition"] == condition) & (gsea["representation"] == "observed")]
        predicted = gsea[
            (gsea["condition"] == condition)
            & (gsea["representation"] == "predicted_counterfactual")
        ]
        merged = observed[["Term", "database", "pathway", "NES", "FDR q-val"]].merge(
            predicted[["Term", "NES", "FDR q-val"]],
            on="Term",
            suffixes=("_observed", "_predicted"),
        )
        merged["condition"] = condition
        merged["sign_concordant"] = np.sign(merged["NES_observed"]) == np.sign(merged["NES_predicted"])
        recovery_rows.append(merged)
        pearson_r, pearson_p = _safe_pearson(merged["NES_observed"], merged["NES_predicted"])
        spearman_r, spearman_p = _safe_spearman(merged["NES_observed"], merged["NES_predicted"])
        recovery_metrics.append(
            {
                "condition": condition,
                "n_pathways": len(merged),
                "pearson_nes": pearson_r,
                "pearson_p": pearson_p,
                "spearman_nes": spearman_r,
                "spearman_p": spearman_p,
                "sign_concordance": float(merged["sign_concordant"].mean()),
            }
        )
    pathway_recovery = pd.concat(recovery_rows, ignore_index=True)
    return {
        "gsea": gsea,
        "pathway_recovery": pathway_recovery,
        "pathway_recovery_metrics": pd.DataFrame(recovery_metrics),
        "gsea_gene_set_count": len(gene_sets),
    }


# In[18]:


pathways = run_ranked_gsea(config, effects["gene_ranks"])
display(pathways["pathway_recovery_metrics"].rename(columns={
    "pearson_nes": "pearson_observed_vs_model_NES",
    "spearman_nes": "spearman_observed_vs_model_NES",
    "sign_concordance": "fraction_same_direction",
}))

pathway_comparison = pathways["pathway_recovery"].copy()
pathway_comparison["absolute_observed_NES"] = pathway_comparison["NES_observed"].abs()

top_pathways_by_condition = {}
for condition in ("IR_24h", "IR_1week"):
    top = (
        pathway_comparison[
            (pathway_comparison["condition"] == condition)
            & (pathway_comparison["database"] == "HALLMARK")
        ]
        .sort_values(
            ["FDR q-val_observed", "absolute_observed_NES"],
            ascending=[True, False],
        )
        .head(10)
        .copy()
    )
    top_pathways_by_condition[condition] = top
    print("\n", condition)
    display(top[[
        "database", "pathway", "NES_observed", "FDR q-val_observed",
        "NES_predicted", "FDR q-val_predicted", "sign_concordant"
    ]])


# In[19]:


def concise_pathway_name(pathway):
    text = str(pathway)
    text = text.removeprefix("HALLMARK_").removeprefix("REACTOME_")
    text = text.replace("_", " ").title()
    return text if len(text) <= 55 else text[:52] + "..."

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, condition, title in zip(
    axes,
    ("IR_24h", "IR_1week"),
    ("24-hour pathways", "1-week pathways"),
):
    frame = top_pathways_by_condition[condition].sort_values("NES_observed")
    y = np.arange(len(frame))
    ax.hlines(
        y, frame["NES_observed"], frame["NES_predicted"],
        color="#b5b5b5", linewidth=1.4
    )
    for values, fdr, marker, color, label in (
        (frame["NES_observed"], frame["FDR q-val_observed"], "o", "#3b6ea8", "Observed"),
        (frame["NES_predicted"], frame["FDR q-val_predicted"], "s", "#d58a35", "Model"),
    ):
        significant = fdr < 0.05
        ax.scatter(
            values[significant], y[significant], s=65, marker=marker,
            facecolors=color, edgecolors=color, label=label, zorder=3
        )
        ax.scatter(
            values[~significant], y[~significant], s=65, marker=marker,
            facecolors="none", edgecolors=color, zorder=3
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, [concise_pathway_name(x) for x in frame["pathway"]])
    ax.set_xlabel("Normalised enrichment score (NES)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)
legend_handles = [
    plt.Line2D([], [], linestyle="none", marker="o", markersize=8,
               markerfacecolor="#3b6ea8", markeredgecolor="#3b6ea8",
               label="Observed"),
    plt.Line2D([], [], linestyle="none", marker="s", markersize=8,
               markerfacecolor="#d58a35", markeredgecolor="#d58a35",
               label="Model counterfactual"),
    plt.Line2D([], [], linestyle="none", marker="o", markersize=8,
               markerfacecolor="#666666", markeredgecolor="#666666",
               label="FDR q < 0.05 (filled)"),
    plt.Line2D([], [], linestyle="none", marker="o", markersize=8,
               markerfacecolor="none", markeredgecolor="#666666",
               label="FDR q ≥ 0.05 (hollow)"),
]
fig.legend(
    handles=legend_handles, frameon=False, loc="lower center", ncol=4,
    bbox_to_anchor=(0.5, 0.01),
)
fig.suptitle(
    "Observed versus model-predicted Hallmark pathway direction",
    y=0.99,
)
fig.tight_layout(rect=(0, 0.10, 1, 0.94))
fig.savefig(OUTPUT_DIR / "pathway_observed_vs_model.png", dpi=220, bbox_inches="tight")
plt.show()


# ### How to read the pathway comparison
# 
# Each row is a Hallmark pathway. Position on the x-axis is the normalised
# enrichment score (NES): values to the right indicate coordinated increase after
# IR and values to the left indicate coordinated decrease. Circles are observed
# data and squares are model counterfactuals. Filled markers passed pathway-level
# FDR < 0.05; hollow markers did not. The grey segment merely helps compare the
# two estimates for the same pathway. It is not an uncertainty interval, a paired
# test or evidence that the two estimates are causally linked. This is the main
# biological summary because it uses all ranked genes and does not depend on the
# exploratory Louvain partition.

# ### Which genes drive those pathway results?
# 
# The compact heatmap below is different from the removed pathway heatmap. Its
# rows are actual leading-edge genes from the selected Hallmark pathways, and its
# columns are gene-level treated-minus-control effects.

# In[20]:


acute_terms = set(
    top_pathways_by_condition["IR_24h"]
    .query("NES_observed > 0")
    .head(3)["Term"]
)
late_suppressed_terms = set(
    top_pathways_by_condition["IR_1week"]
    .query("NES_observed < 0")
    .head(3)["Term"]
)

def leading_genes_for_terms(terms, condition):
    text_values = pathways["gsea"].loc[
        (pathways["gsea"]["representation"] == "observed")
        & (pathways["gsea"]["condition"] == condition)
        & pathways["gsea"]["Term"].isin(terms),
        "Lead_genes",
    ].dropna()
    genes = set()
    for text in text_values:
        genes.update(str(text).split(";"))
    return genes

acute_candidates = leading_genes_for_terms(acute_terms, "IR_24h")
late_candidates = leading_genes_for_terms(late_suppressed_terms, "IR_1week")
gene_rank_table = effects["gene_ranks"].copy()
acute_order = (
    gene_rank_table[
        (gene_rank_table["condition"] == "IR_24h")
        & gene_rank_table["human_symbol"].isin(acute_candidates)
    ]
    .nlargest(6, "observed_delta")["human_symbol"]
    .tolist()
)
late_order = (
    gene_rank_table[
        (gene_rank_table["condition"] == "IR_1week")
        & gene_rank_table["human_symbol"].isin(late_candidates)
    ]
    .nsmallest(6, "observed_delta")["human_symbol"]
    .tolist()
)
driver_order = list(dict.fromkeys(acute_order + late_order))
driver_ranks = gene_rank_table[
    gene_rank_table["human_symbol"].isin(driver_order)
].copy()

driver_rows = []
for row in driver_ranks[driver_ranks["human_symbol"].isin(driver_order)].itertuples(index=False):
    driver_rows.extend([
        {"gene": row.human_symbol, "column": f"{row.condition} | observed", "delta": row.observed_delta},
        {"gene": row.human_symbol, "column": f"{row.condition} | model", "delta": row.predicted_counterfactual_delta},
    ])
driver_matrix = pd.DataFrame(driver_rows).pivot(
    index="gene", columns="column", values="delta"
).reindex(
    index=driver_order,
    columns=[
        "IR_24h | observed", "IR_24h | model",
        "IR_1week | observed", "IR_1week | model",
    ],
)

fig, ax = plt.subplots(figsize=(9, 7.3))
sns.heatmap(
    driver_matrix,
    cmap="vlag",
    center=0,
    linewidths=0.3,
    cbar_kws={
        "label": "Gene-level effect",
        "orientation": "horizontal",
        "pad": 0.15,
        "shrink": 0.72,
        "aspect": 35,
    },
    ax=ax,
)
ax.set_title("Leading-edge genes behind the selected Hallmark pathways")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "leading_edge_gene_effects.png", dpi=220, bbox_inches="tight")
plt.show()


# ### How to read the leading-edge heatmap
# 
# Rows are genes that contribute most strongly to the selected pathway enrichment
# scores, not all genes in those pathways. Each cell is a gene-level change:
# red/positive means higher after IR and blue/negative means lower after IR. The
# four columns separate time point and representation, allowing direct visual
# comparison of observed and model direction. The colour key is placed below the
# heatmap. These values are effect sizes on the analysed expression scale; they
# are not expression abundance, per-gene p-values or proof that a gene regulates
# another gene. The heatmap is useful for identifying which genes make the
# pathway-level NES biologically interpretable.

# ## 8. Sample-level pathway consistency
# 
# Aggregate GSEA establishes the biological signal; this figure checks whether
# it is visible across biological sample units. The observed IDs (`24_1`, etc.)
# are not paired with the control IDs (`Ctrl_1`, etc.): their Batch/Sex metadata
# also rule out suffix-based pairing. Therefore each treated pseudobulk is
# contrasted with the equal-weight mean of the four control pseudobulks.
# 
# The model points remain matched counterfactuals within each control-input
# sample. Observed and model points are deliberately not connected or subjected
# to a paired test. The four pathways were chosen from the aggregate results, so
# this is a descriptive consistency view rather than an independent discovery.

# In[21]:


def compute_sample_gene_deltas(
    aligned: dict,
    modules: dict,
    counterfactual: dict,
) -> dict:
    """Build independent sample-level observed and model gene contrasts.

    Treated sample IDs are condition-specific and are not paired with control
    IDs.  Each observed treated pseudobulk is therefore contrasted with the
    equal-weight mean of the four control pseudobulks.  Model effects remain
    paired within each real control-input sample.
    """

    metadata = aligned["metadata"].reset_index(drop=True).copy()
    nodes = modules["nodes"].reset_index(drop=True).copy()
    source_feature_groups = [
        tuple(int(index) for index in values)
        for values in nodes["source_feature_indices"]
    ]

    def collapse_source_features(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        collapsed = np.empty((matrix.shape[0], len(source_feature_groups)), dtype=np.float32)
        for column, indices in enumerate(source_feature_groups):
            collapsed[:, column] = matrix[:, indices].mean(axis=1)
        return collapsed

    symbols = nodes["human_symbol"].astype(str).to_numpy()

    def symbol_delta_frame(delta: np.ndarray) -> pd.DataFrame:
        return (
            pd.DataFrame({"human_symbol": symbols, "delta": np.asarray(delta, dtype=float)})
            .groupby("human_symbol", as_index=False, sort=True)["delta"]
            .mean()
        )

    observed_expression = collapse_source_features(aligned["expression"])
    sample_pseudobulks = {}
    for (sample_id, condition), positions in metadata.groupby(
        ["Sample", "condition_canonical"], observed=True, sort=True
    ).indices.items():
        sample_pseudobulks[(str(sample_id), str(condition))] = observed_expression[
            np.asarray(positions, dtype=int)
        ].mean(axis=0)

    control_keys = sorted(
        key for key in sample_pseudobulks if key[1] == "ctrl"
    )
    if len(control_keys) < 2:
        raise ValueError("At least two control pseudobulks are required")
    equal_weight_control_mean = np.stack(
        [sample_pseudobulks[key] for key in control_keys]
    ).mean(axis=0)

    result_parts = []
    for condition in ("IR_24h", "IR_1week"):
        treated_keys = sorted(
            key for key in sample_pseudobulks if key[1] == condition
        )
        for sample_id, _ in treated_keys:
            frame = symbol_delta_frame(
                sample_pseudobulks[(sample_id, condition)] - equal_weight_control_mean
            )
            frame.insert(0, "sample_id", sample_id)
            frame.insert(0, "representation", "observed")
            frame.insert(0, "condition", condition)
            frame["reference"] = "equal-weight mean of control pseudobulks"
            result_parts.append(frame)

    collapsed_predictions = {
        condition: collapse_source_features(matrix)
        for condition, matrix in counterfactual["predictions"].items()
    }
    input_samples = np.asarray(counterfactual["input_sample_labels"], dtype=str)
    for condition in ("IR_24h", "IR_1week"):
        paired_cell_delta = (
            collapsed_predictions[condition] - collapsed_predictions["ctrl"]
        )
        for sample_id in sorted(np.unique(input_samples)):
            frame = symbol_delta_frame(
                paired_cell_delta[input_samples == sample_id].mean(axis=0)
            )
            frame.insert(0, "sample_id", sample_id)
            frame.insert(0, "representation", "model_counterfactual")
            frame.insert(0, "condition", condition)
            frame["reference"] = "same control input with control flag"
            result_parts.append(frame)

    sample_gene_deltas = pd.concat(result_parts, ignore_index=True)
    audit = pd.DataFrame(
        [
            {
                "contrast_type": "observed",
                "replicate_unit": "treated sample pseudobulk",
                "n_per_condition": int(
                    sample_gene_deltas.query("representation == 'observed'")
                    .groupby("condition")["sample_id"]
                    .nunique()
                    .min()
                ),
                "reference": "equal-weight mean of independent control pseudobulks",
                "paired_across_conditions": False,
            },
            {
                "contrast_type": "model_counterfactual",
                "replicate_unit": "control input sample",
                "n_per_condition": int(
                    sample_gene_deltas.query("representation == 'model_counterfactual'")
                    .groupby("condition")["sample_id"]
                    .nunique()
                    .min()
                ),
                "reference": "same cell prediction with control flag",
                "paired_across_conditions": True,
            },
        ]
    )
    return {
        "sample_gene_deltas": sample_gene_deltas,
        "sample_contrast_audit": audit,
    }


# In[22]:


sample_contrasts = compute_sample_gene_deltas(aligned, modules, counterfactual)
sample_gene_deltas = sample_contrasts["sample_gene_deltas"]
display(sample_contrasts["sample_contrast_audit"])

target_pathway_names = [
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
]
sample_universe = set(sample_gene_deltas["human_symbol"].astype(str))
all_sample_sets = load_gene_sets(config, universe=sample_universe, include_go=False)
target_gene_sets = {
    term: genes for term, genes in all_sample_sets.items()
    if term.split("::", 1)[-1] in target_pathway_names
}
missing_target_sets = set(target_pathway_names) - {
    term.split("::", 1)[-1] for term in target_gene_sets
}
if missing_target_sets:
    raise KeyError(f"Missing target Hallmark sets: {sorted(missing_target_sets)}")

sample_gsea_parts = []
for (condition, representation, sample_id), frame in sample_gene_deltas.groupby(
    ["condition", "representation", "sample_id"], sort=True
):
    rank = frame[["human_symbol", "delta"]].sort_values("delta", ascending=False)
    rank.columns = ["gene", "score"]
    pre = gp.prerank(
        rnk=rank,
        gene_sets=target_gene_sets,
        min_size=config.min_gene_set_size,
        max_size=config.max_gene_set_size,
        permutation_num=max(1000, config.gsea_permutations),
        outdir=None,
        seed=config.seed,
        threads=1,
        verbose=False,
    )
    result = pre.res2d.copy().reset_index(drop=True)
    if "Term" not in result.columns:
        result = result.rename(columns={result.columns[0]: "Term"})
    result["condition"] = condition
    result["representation"] = representation
    result["sample_id"] = sample_id
    sample_gsea_parts.append(result)

sample_pathway_effects = pd.concat(sample_gsea_parts, ignore_index=True)
for column in ("ES", "NES", "NOM p-val", "FDR q-val"):
    if column in sample_pathway_effects:
        sample_pathway_effects[column] = pd.to_numeric(
            sample_pathway_effects[column], errors="coerce"
        )
sample_pathway_effects["pathway"] = (
    sample_pathway_effects["Term"].astype(str).str.split("::", n=1).str[-1]
)
display(sample_pathway_effects[[
    "condition", "representation", "sample_id", "pathway", "NES"
]].head(12))


# In[23]:


pathway_order = target_pathway_names
pathway_labels = {
    "HALLMARK_INTERFERON_ALPHA_RESPONSE": "Interferon alpha response",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": "Interferon gamma response",
    "HALLMARK_E2F_TARGETS": "E2F targets",
    "HALLMARK_G2M_CHECKPOINT": "G2M checkpoint",
}
representation_style = {
    "observed": {"color": "#3b6ea8", "marker": "o", "offset": -0.16, "label": "Observed samples"},
    "model_counterfactual": {"color": "#d58a35", "marker": "s", "offset": 0.16, "label": "Model counterfactuals"},
}

finite_nes = sample_pathway_effects["NES"].dropna().to_numpy()
shared_limit = max(1.0, float(np.max(np.abs(finite_nes))) * 1.08)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), sharex=True, sharey=True)
for ax, condition, title in zip(
    axes, ("IR_24h", "IR_1week"), ("24 hours", "1 week")
):
    condition_frame = sample_pathway_effects[
        sample_pathway_effects["condition"].eq(condition)
    ]
    for representation, style in representation_style.items():
        rep = condition_frame[condition_frame["representation"].eq(representation)]
        for row_index, pathway in enumerate(pathway_order):
            values = rep.loc[rep["pathway"].eq(pathway), "NES"].dropna().to_numpy()
            jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.zeros(len(values))
            y_values = row_index + style["offset"] + jitter
            ax.scatter(values, y_values, s=48, marker=style["marker"],
                       facecolor=style["color"], edgecolor="white", linewidth=0.5,
                       alpha=0.82, zorder=3)
            if len(values):
                ax.scatter(np.mean(values), row_index + style["offset"], s=145,
                           marker=style["marker"], facecolor=style["color"],
                           edgecolor="black", linewidth=1.0, zorder=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-shared_limit, shared_limit)
    ax.set_title(title)
    ax.set_xlabel("Sample-level normalised enrichment score")
    ax.grid(axis="x", alpha=0.2)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(
    np.arange(len(pathway_order)), [pathway_labels[pathway] for pathway in pathway_order]
)
axes[0].invert_yaxis()
legend_handles = [
    plt.Line2D([0], [0], marker=style["marker"], color="none",
               markerfacecolor=style["color"], markeredgecolor="black",
               markersize=9, label=style["label"])
    for style in representation_style.values()
]
fig.legend(
    handles=legend_handles, frameon=False, loc="lower center", ncol=2,
    bbox_to_anchor=(0.5, 0.01),
)
fig.suptitle(
    "Pathway effects across four sample-level contrasts per representation\n"
    "large marker = mean; observed and model samples are not paired",
    y=0.99,
)
fig.tight_layout(rect=(0, 0.11, 1, 0.91))
fig.savefig(OUTPUT_DIR / "sample_level_pathway_effects.png", dpi=220, bbox_inches="tight")
plt.show()


# ### How to read the sample-level pathway plot
# 
# For each pathway, the four small markers are the four biological sample-level
# NES values and the large outlined marker is their mean. The spread of the small
# markers reveals heterogeneity that is hidden by aggregate GSEA. A pathway whose
# points remain on one side of zero is directionally consistent across samples;
# points crossing zero indicate sample-dependent response. Observed circles and
# model squares are not paired to one another, so no connecting lines or paired
# test are used. Because these four pathways were selected from the aggregate
# analysis, this is descriptive support for robustness, not a second independent
# significance test.

# ### Optional focused embedding view
# 
# A small cell-cycle graph is shown only if the selected E2F/G2M program has
# median seed-matching Jaccard at least 0.70. Nodes are leading-edge genes and
# edges are undirected embedding proximity. They are not regulatory edges,
# directions or causal claims.

# In[24]:


focused_network_created = False
focused_network_nodes = pd.DataFrame()
focused_network_edges = pd.DataFrame()

cell_cycle_ora = (
    program_annotation["program_ora"]
    .query("pathway in ['HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT'] and program_fdr < 0.05 and overlap_size >= 3")
    .merge(
        modules["module_sizes"][["module_id", "program_id", "annotation", "median_seed_jaccard"]],
        on=["module_id", "program_id"], how="left"
    )
    .sort_values(["program_fdr", "overlap_size"], ascending=[True, False])
)
if not cell_cycle_ora.empty and cell_cycle_ora.iloc[0]["median_seed_jaccard"] >= 0.70:
    focus = cell_cycle_ora.iloc[0]
    focus_module = focus["module_id"]
    cell_cycle_terms = {
        "HALLMARK::HALLMARK_E2F_TARGETS",
        "HALLMARK::HALLMARK_G2M_CHECKPOINT",
    }
    lead_text = pathways["gsea"].loc[
        pathways["gsea"]["condition"].eq("IR_1week")
        & pathways["gsea"]["representation"].eq("observed")
        & pathways["gsea"]["Term"].isin(cell_cycle_terms),
        "Lead_genes",
    ].dropna()
    leading = set()
    for value in lead_text:
        leading.update(str(value).split(";"))
    member_nodes = modules["nodes"].query("module_id == @focus_module").copy()
    member_nodes["embedding_index"] = member_nodes.index.astype(int)
    member_nodes = member_nodes[member_nodes["human_symbol"].isin(leading)]
    one_week_rank = effects["gene_ranks"].query("condition == 'IR_1week'")[[
        "human_symbol", "observed_delta", "predicted_counterfactual_delta"
    ]]
    member_nodes = (
        member_nodes.merge(one_week_rank, on="human_symbol", how="inner")
        .assign(abs_observed=lambda frame: frame["observed_delta"].abs())
        .sort_values(["abs_observed", "human_symbol"], ascending=[False, True])
        .drop_duplicates("human_symbol")
        .head(18)
    )
    selected_indices = member_nodes["embedding_index"].to_numpy(dtype=int)
    if len(selected_indices) >= 6:
        selected_embeddings = embedding_result["embeddings"][selected_indices]
        cosine = selected_embeddings @ selected_embeddings.T
        edge_pairs = set()
        for source in range(len(selected_indices)):
            targets = np.argsort(cosine[source])[::-1]
            targets = [target for target in targets if target != source][:2]
            edge_pairs.update(tuple(sorted((source, target))) for target in targets)
        focused_network_edges = pd.DataFrame([
            {
                "source": str(member_nodes.iloc[source]["human_symbol"]),
                "target": str(member_nodes.iloc[target]["human_symbol"]),
                "cosine_similarity": float(cosine[source, target]),
            }
            for source, target in sorted(edge_pairs)
        ])
        focused_network_nodes = member_nodes[[
            "human_symbol", "observed_delta", "predicted_counterfactual_delta"
        ]].copy()
        graph = nx.from_pandas_edgelist(
            focused_network_edges, "source", "target", edge_attr="cosine_similarity"
        )
        layout = nx.spring_layout(graph, seed=config.seed, weight="cosine_similarity", k=0.8)
        limit = float(np.max(np.abs(focused_network_nodes[[
            "observed_delta", "predicted_counterfactual_delta"
        ]].to_numpy())))
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        values_by_gene = focused_network_nodes.set_index("human_symbol")
        for ax, column, title in zip(
            axes,
            ("observed_delta", "predicted_counterfactual_delta"),
            ("Observed 1-week effect", "Model 1-week counterfactual"),
        ):
            nx.draw_networkx_edges(
                graph, layout, ax=ax, edge_color="#a8a8a8", alpha=0.55,
                width=[0.6 + 2.0 * graph[u][v]["cosine_similarity"] for u, v in graph.edges]
            )
            draw = nx.draw_networkx_nodes(
                graph, layout, ax=ax, node_size=620,
                node_color=[values_by_gene.loc[node, column] for node in graph.nodes],
                cmap="coolwarm", vmin=-limit, vmax=limit,
                edgecolors="white", linewidths=0.8,
            )
            nx.draw_networkx_labels(graph, layout, ax=ax, font_size=8)
            ax.set_title(title)
            ax.axis("off")
        fig.colorbar(
            draw, ax=axes, orientation="horizontal", fraction=0.055, pad=0.07,
            aspect=40, label="Gene-level change",
        )
        fig.suptitle(
            f"{focus['program_id']} — cell-cycle embedding program\n"
            "edges show embedding proximity, not regulation",
            y=0.98,
        )
        fig.subplots_adjust(bottom=0.15, top=0.84)
        fig.savefig(OUTPUT_DIR / "focused_cell_cycle_embedding_network.png", dpi=220, bbox_inches="tight")
        plt.show()
        focused_network_created = True
    else:
        print(
            "Focused network omitted: the stable cell-cycle program contained fewer "
            "than six aggregate leading-edge genes."
        )
else:
    print("Focused network omitted: no significant E2F/G2M program met median seed Jaccard ≥ 0.70.")


# ## 9. Recommended NetworkX maps
# 
# These maps use one fixed embedding geometry. Node colours change between
# observed and model contrasts, but node positions and edges do not. Consequently
# they visualise differential activation on a learned representation network;
# they do not claim condition-specific rewiring, edge direction or causal gene
# regulation.
# 
# Five complementary views are produced:
# 
# 1. the complete 2,439-gene selected embedding-neighbour graph, retained only as
#    an auditable supplementary record;
# 2. a 393-program backbone for the global response;
# 3. a focused interferon leading-edge gene network;
# 4. a focused late E2F/G2M network; and
# 5. a program-to-pathway bipartite annotation network.

# In[25]:


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import textwrap

network_nodes = modules["nodes"].copy().reset_index(drop=True)
network_nodes["embedding_index"] = np.arange(len(network_nodes), dtype=int)
network_embeddings = np.asarray(embedding_result["embeddings"], dtype=np.float32)
network_embeddings /= np.maximum(
    np.linalg.norm(network_embeddings, axis=1, keepdims=True), 1e-12
)

gene_effect_wide = (
    effects["gene_ranks"]
    .pivot_table(
        index="human_symbol",
        columns="condition",
        values=["observed_delta", "predicted_counterfactual_delta"],
        aggfunc="mean",
    )
)
gene_effect_wide.columns = [
    f"{'observed' if representation == 'observed_delta' else 'model'}_"
    f"{'24h' if condition == 'IR_24h' else '1week'}"
    for representation, condition in gene_effect_wide.columns
]
gene_effect_wide = gene_effect_wide.reset_index()

network_panel_specs = [
    ("observed_24h", "Observed 24h − control"),
    ("model_24h", "Model flag: 24h − control"),
    ("observed_1week", "Observed 1 week − control"),
    ("model_1week", "Model flag: 1 week − control"),
]

def _normalise_rows(values):
    values = np.asarray(values, dtype=float)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

def _edge_widths(graph, attribute="cosine_similarity", low=0.45, high=2.4):
    values = np.array([graph[u][v][attribute] for u, v in graph.edges], dtype=float)
    if len(values) == 0:
        return []
    if np.ptp(values) < 1e-12:
        return np.full(len(values), (low + high) / 2)
    scaled = (values - values.min()) / np.ptp(values)
    return low + (high - low) * scaled

def _leading_edge_symbols(terms, conditions=None, representations=None):
    selected = pathways["gsea"].loc[pathways["gsea"]["Term"].isin(terms)].copy()
    if conditions is not None:
        selected = selected[selected["condition"].isin(conditions)]
    if representations is not None:
        selected = selected[selected["representation"].isin(representations)]
    genes = set()
    for value in selected["Lead_genes"].dropna():
        genes.update(gene for gene in str(value).split(";") if gene)
    return genes

def _select_focus_nodes(
    leading_genes, score_columns, required=(), n_genes=24,
    selection_rule="max_absolute",
):
    candidates = (
        network_nodes[["embedding_index", "human_symbol", "program_id"]]
        .merge(gene_effect_wide, on="human_symbol", how="left")
    )
    candidates = candidates[candidates["human_symbol"].isin(leading_genes)].copy()
    if selection_rule == "mean_positive":
        candidates["selection_score"] = (
            candidates[list(score_columns)].mean(axis=1).fillna(-np.inf)
        )
        candidates = candidates[candidates["selection_score"] > 0]
        ascending = False
    elif selection_rule == "most_negative":
        candidates["selection_score"] = (
            candidates[list(score_columns)].mean(axis=1).fillna(np.inf)
        )
        ascending = True
    else:
        candidates["selection_score"] = (
            candidates[list(score_columns)].abs().max(axis=1).fillna(0.0)
        )
        ascending = False
    candidates = candidates.sort_values(
        ["selection_score", "human_symbol"], ascending=[ascending, True]
    ).drop_duplicates("human_symbol")
    required_available = [
        gene for gene in required if gene in set(candidates["human_symbol"])
    ]
    ordered = required_available + [
        gene for gene in candidates["human_symbol"] if gene not in required_available
    ]
    ordered = ordered[:n_genes]
    order_lookup = {gene: index for index, gene in enumerate(ordered)}
    return (
        candidates[candidates["human_symbol"].isin(ordered)]
        .assign(_order=lambda frame: frame["human_symbol"].map(order_lookup))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

def _focused_similarity_graph(node_frame, nearest_per_gene=2):
    symbols = node_frame["human_symbol"].tolist()
    indices = node_frame["embedding_index"].to_numpy(dtype=int)
    cosine = network_embeddings[indices] @ network_embeddings[indices].T
    complete = nx.Graph()
    complete.add_nodes_from(symbols)
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            similarity = float(cosine[first, second])
            if similarity > 0:
                complete.add_edge(
                    symbols[first], symbols[second],
                    cosine_similarity=similarity,
                    layout_weight=similarity,
                )
    graph = nx.maximum_spanning_tree(complete, weight="cosine_similarity")
    graph.add_nodes_from(symbols)
    for source in range(len(symbols)):
        neighbours = [
            target for target in np.argsort(cosine[source])[::-1]
            if target != source and cosine[source, target] > 0
        ][:nearest_per_gene]
        for target in neighbours:
            similarity = float(cosine[source, target])
            graph.add_edge(
                symbols[source], symbols[target],
                cosine_similarity=similarity,
                layout_weight=similarity,
            )
    return graph

def _draw_focused_network(node_frame, graph, panel_specs, title, output_name):
    layout = nx.spring_layout(
        graph, seed=config.seed, weight="layout_weight", k=0.72, iterations=500
    )
    value_table = node_frame.set_index("human_symbol")
    columns = [column for column, _ in panel_specs]
    limit = float(np.nanmax(np.abs(value_table[columns].to_numpy(dtype=float))))
    limit = max(limit, 1e-6)
    n_panels = len(panel_specs)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 6.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (column, panel_title) in zip(axes, panel_specs):
        nx.draw_networkx_edges(
            graph, layout, ax=ax, edge_color="#8f8f8f", alpha=0.42,
            width=_edge_widths(graph),
        )
        values = np.array([value_table.loc[node, column] for node in graph.nodes])
        nx.draw_networkx_nodes(
            graph, layout, ax=ax, node_size=840, node_color=values,
            cmap="RdBu_r", vmin=-limit, vmax=limit,
            edgecolors="#f4f4f4", linewidths=1.1,
        )
        nx.draw_networkx_labels(
            graph, layout, ax=ax, font_size=8, font_weight="bold"
        )
        ax.set_title(panel_title)
        ax.axis("off")
    for ax in axes[n_panels:]:
        ax.axis("off")
    mapper = ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r")
    mapper.set_array([])
    cbar = fig.colorbar(
        mapper, ax=axes[:n_panels].tolist(), orientation="horizontal",
        fraction=0.035, pad=0.055, aspect=48,
    )
    cbar.set_label("Gene-level expression change (red = increased; blue = decreased)")
    fig.suptitle(
        title + "\nfixed layout; edges show scGPT embedding similarity, not regulation",
        y=0.99,
    )
    fig.subplots_adjust(bottom=0.12, top=0.88, wspace=0.02, hspace=0.10)
    fig.savefig(OUTPUT_DIR / output_name, dpi=240, bbox_inches="tight")
    plt.show()

    node_export = node_frame.copy()
    node_export["layout_x"] = node_export["human_symbol"].map(
        {node: float(layout[node][0]) for node in graph.nodes}
    )
    node_export["layout_y"] = node_export["human_symbol"].map(
        {node: float(layout[node][1]) for node in graph.nodes}
    )
    edge_export = pd.DataFrame([
        {
            "source": source,
            "target": target,
            "cosine_similarity": float(data["cosine_similarity"]),
            "backbone_or_nearest_neighbour": True,
        }
        for source, target, data in graph.edges(data=True)
    ])
    return node_export, edge_export


# ### 9.1 Full gene-level embedding-neighbour network — record only
# 
# This panel retains every eligible checkpoint gene and every edge in the exact
# selected Scanpy kNN connectivity graph. The present analysis contains 2,439
# nodes: integer node indices run from 0 through 2,438, so 2,438 is the maximum
# index rather than the node count. The layout is deterministic ForceAtlas2 and
# is used only to make the complete graph inspectable. It can visually exaggerate
# separation and must not be read as a causal, directed or condition-specific GRN.
# 
# The PNG/PDF are visual records. The node table, complete edge table and GraphML
# file are the auditable outputs for Cytoscape, Gephi or later replotting.

# In[ ]:


full_gene_graph = nx.Graph()
full_gene_graph.add_nodes_from(range(len(network_nodes)))
for row in modules["edges"].itertuples(index=False):
    full_gene_graph.add_edge(
        int(row.source_index),
        int(row.target_index),
        weight=float(row.connectivity_weight),
    )

if full_gene_graph.number_of_nodes() != len(network_nodes):
    raise RuntimeError("The full NetworkX graph lost an eligible checkpoint gene")
if nx.number_connected_components(full_gene_graph) != 1:
    raise RuntimeError("The selected embedding-neighbour graph is unexpectedly disconnected")

initial_gene_layout = nx.random_layout(
    full_gene_graph, seed=config.seed, dim=2
)
with warnings.catch_warnings():
    # NetworkX ForceAtlas2 can transiently divide by a zero inter-node distance;
    # the final coordinates are checked explicitly below.
    warnings.filterwarnings(
        "ignore", message="invalid value encountered in divide", category=RuntimeWarning
    )
    full_gene_layout = nx.forceatlas2_layout(
        full_gene_graph,
        pos=initial_gene_layout,
        max_iter=180,
        jitter_tolerance=1.0,
        scaling_ratio=2.0,
        gravity=1.0,
        weight="weight",
        dissuade_hubs=True,
        linlog=True,
        seed=config.seed,
    )

full_gene_coordinates = np.asarray(
    [full_gene_layout[index] for index in range(len(network_nodes))], dtype=float
)
if not np.isfinite(full_gene_coordinates).all():
    raise RuntimeError("The full-gene ForceAtlas2 layout contains non-finite coordinates")

full_gene_network_nodes = network_nodes.copy()
full_gene_network_nodes["network_index"] = np.arange(len(network_nodes), dtype=int)
full_gene_network_nodes["layout_x"] = full_gene_coordinates[:, 0]
full_gene_network_nodes["layout_y"] = full_gene_coordinates[:, 1]
full_gene_network_nodes = full_gene_network_nodes.merge(
    gene_effect_wide, on="human_symbol", how="left"
)
full_gene_network_edges = modules["edges"].copy()

full_gene_network_nodes.to_csv(
    OUTPUT_DIR / "full_embedding_gene_network_nodes.csv", index=False
)
full_gene_network_edges.to_csv(
    OUTPUT_DIR / "full_embedding_gene_network_edges.csv", index=False
)

full_gene_graphml = nx.Graph()
for row in full_gene_network_nodes.itertuples(index=False):
    attributes = {
        "model_gene": str(row.model_gene),
        "human_symbol": str(row.human_symbol),
        "program_id": str(row.program_id),
        "module_id": str(row.module_id),
        "layout_x": float(row.layout_x),
        "layout_y": float(row.layout_y),
    }
    for column in ("observed_24h", "model_24h", "observed_1week", "model_1week"):
        value = getattr(row, column, np.nan)
        attributes[column] = float(value) if pd.notna(value) else float("nan")
    full_gene_graphml.add_node(int(row.network_index), **attributes)
for row in full_gene_network_edges.itertuples(index=False):
    full_gene_graphml.add_edge(
        int(row.source_index),
        int(row.target_index),
        connectivity_weight=float(row.connectivity_weight),
        within_program=bool(row.within_program),
    )
nx.write_graphml(
    full_gene_graphml,
    OUTPUT_DIR / "full_embedding_gene_network.graphml.gz",
    infer_numeric_types=True,
)

full_gene_segments = [
    [
        full_gene_coordinates[int(row.source_index)],
        full_gene_coordinates[int(row.target_index)],
    ]
    for row in full_gene_network_edges.itertuples(index=False)
]
program_numbers = (
    network_nodes["program_id"].astype(str).str.removeprefix("P").astype(int)
)

fig, ax = plt.subplots(figsize=(16.5, 15.5))
ax.add_collection(
    LineCollection(
        full_gene_segments,
        colors=[(0.34, 0.36, 0.39, 0.045)],
        linewidths=0.12,
        rasterized=True,
        zorder=1,
    )
)
full_gene_draw = ax.scatter(
    full_gene_coordinates[:, 0],
    full_gene_coordinates[:, 1],
    c=program_numbers,
    cmap="turbo",
    s=7.2,
    alpha=0.88,
    linewidths=0,
    rasterized=True,
    zorder=2,
)
ax.autoscale_view()
ax.margins(0.03)
ax.axis("off")
selected_setting = modules["selection"].iloc[0]
ax.set_title(
    "Full fine-tuned scGPT embedding-neighbour network — record view\n"
    f"{len(network_nodes):,} genes; {len(full_gene_network_edges):,} kNN connectivity edges; "
    f"{network_nodes['program_id'].nunique()} Louvain programs; "
    f"k={modules['chosen_neighbors']}, resolution={modules['chosen_resolution']:g}",
    pad=18,
)
full_gene_colorbar = fig.colorbar(
    full_gene_draw,
    ax=ax,
    orientation="horizontal",
    fraction=0.025,
    pad=0.025,
    aspect=55,
)
program_ticks = np.array([1, 100, 200, 300, int(program_numbers.max())])
full_gene_colorbar.set_ticks(program_ticks)
full_gene_colorbar.set_ticklabels([f"P{value:03d}" for value in program_ticks])
full_gene_colorbar.set_label(
    "Louvain program identifier (colour is an identifier, not a biological scale)"
)
fig.text(
    0.5,
    0.012,
    "ForceAtlas2 layout; edges are undirected Scanpy embedding connectivities, not regulation. "
    f"Cosine silhouette={float(selected_setting['cosine_silhouette']):.3f}; "
    "partition is exploratory.",
    ha="center",
    va="bottom",
    fontsize=9,
)
fig.subplots_adjust(bottom=0.09, top=0.93)
fig.savefig(
    OUTPUT_DIR / "full_embedding_gene_network_record.png",
    dpi=360,
    bbox_inches="tight",
)
fig.savefig(
    OUTPUT_DIR / "full_embedding_gene_network_record.pdf",
    bbox_inches="tight",
)
plt.show()

print(
    f"Full record graph: {full_gene_graph.number_of_nodes():,} nodes, "
    f"{full_gene_graph.number_of_edges():,} edges, "
    f"{nx.number_connected_components(full_gene_graph)} connected component, "
    f"{len(list(nx.isolates(full_gene_graph)))} isolates."
)


# ### 9.2 Program-level activation backbone
# 
# Each node is one of the 393 fixed Louvain programs. Gene-level k-neighbour
# edges crossing between programs are aggregated into a weighted quotient graph
# and normalised by the geometric mean of the two programs' total graph strength.
# The display retains a maximum-weight spanning forest plus each program's two
# strongest normalised neighbours. Node size represents program size and colour
# represents the program-score contrast. Identical positions make observed/model
# differences comparable without suggesting that the network itself changed.

# In[26]:


program_ids = sorted(network_nodes["program_id"].unique())
node_strength = np.asarray(modules["connectivities"].sum(axis=1)).ravel()
program_strength = (
    network_nodes.assign(node_strength=node_strength)
    .groupby("program_id")["node_strength"].sum()
    .to_dict()
)
cross_program_edges = modules["edges"].loc[~modules["edges"]["within_program"]].copy()
cross_program_edges["program_a"] = np.where(
    cross_program_edges["source_program"] < cross_program_edges["target_program"],
    cross_program_edges["source_program"], cross_program_edges["target_program"],
)
cross_program_edges["program_b"] = np.where(
    cross_program_edges["source_program"] < cross_program_edges["target_program"],
    cross_program_edges["target_program"], cross_program_edges["source_program"],
)
program_pair_edges = (
    cross_program_edges.groupby(["program_a", "program_b"], as_index=False)
    .agg(
        cross_weight=("connectivity_weight", "sum"),
        n_gene_edges=("connectivity_weight", "size"),
    )
)
program_pair_edges["normalised_meta_weight"] = [
    float(row.cross_weight) / max(
        np.sqrt(program_strength[row.program_a] * program_strength[row.program_b]),
        1e-12,
    )
    for row in program_pair_edges.itertuples(index=False)
]

complete_program_graph = nx.Graph()
complete_program_graph.add_nodes_from(program_ids)
for row in program_pair_edges.itertuples(index=False):
    complete_program_graph.add_edge(
        row.program_a, row.program_b,
        meta_weight=float(row.normalised_meta_weight),
        cross_weight=float(row.cross_weight),
        n_gene_edges=int(row.n_gene_edges),
        layout_weight=float(row.normalised_meta_weight),
    )

program_graph = nx.maximum_spanning_tree(
    complete_program_graph, weight="meta_weight"
)
program_graph.add_nodes_from(program_ids)
for source in program_ids:
    strongest = sorted(
        complete_program_graph.edges(source, data=True),
        key=lambda edge: (-edge[2]["meta_weight"], edge[1]),
    )[:2]
    for first, second, data in strongest:
        program_graph.add_edge(first, second, **data)

program_layout = nx.spring_layout(
    program_graph, seed=config.seed, weight="layout_weight",
    k=0.16, iterations=500,
)
program_size_lookup = (
    modules["module_sizes"].set_index("program_id")["n_features"].to_dict()
)
program_annotation_lookup = (
    program_table.set_index("program_id")["annotation"].to_dict()
)
program_effect_wide = (
    program_effects.pivot_table(
        index="program_id", columns="condition",
        values=["observed_delta", "predicted_counterfactual_delta"],
        aggfunc="mean",
    )
)
program_effect_wide.columns = [
    f"{'observed' if representation == 'observed_delta' else 'model'}_"
    f"{'24h' if condition == 'IR_24h' else '1week'}"
    for representation, condition in program_effect_wide.columns
]
program_effect_wide = program_effect_wide.reindex(program_ids)
program_limit = float(np.nanmax(np.abs(program_effect_wide.to_numpy(dtype=float))))
program_limit = max(program_limit, 1e-6)

key_programs = [
    program_id for program_id in
    ["P013", "P162", "P003", "P045", "P174", "P304", "P311"]
    if program_id in program_graph
]
key_labels = {
    program_id: f"{program_id}\n{program_annotation_lookup.get(program_id, '')}"
    for program_id in key_programs
}
program_edge_widths = _edge_widths(
    program_graph, attribute="meta_weight", low=0.18, high=1.35
)

fig, axes = plt.subplots(2, 2, figsize=(17.5, 14.0))
for ax, (column, panel_title) in zip(axes.ravel(), network_panel_specs):
    nx.draw_networkx_edges(
        program_graph, program_layout, ax=ax, edge_color="#9a9a9a",
        alpha=0.20, width=program_edge_widths,
    )
    values = program_effect_wide[column].to_dict()
    nx.draw_networkx_nodes(
        program_graph, program_layout, ax=ax,
        node_size=[22 + 6.5 * program_size_lookup[node] for node in program_graph.nodes],
        node_color=[values.get(node, np.nan) for node in program_graph.nodes],
        cmap="RdBu_r", vmin=-program_limit, vmax=program_limit,
        edgecolors="#eeeeee", linewidths=0.35,
    )
    nx.draw_networkx_labels(
        program_graph, program_layout, labels=key_labels, ax=ax,
        font_size=7.5, font_weight="bold",
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.74},
    )
    ax.set_title(panel_title)
    ax.axis("off")

program_mapper = ScalarMappable(
    norm=Normalize(-program_limit, program_limit), cmap="RdBu_r"
)
program_mapper.set_array([])
program_cbar = fig.colorbar(
    program_mapper, ax=axes.ravel().tolist(), orientation="horizontal",
    fraction=0.025, pad=0.045, aspect=55,
)
program_cbar.set_label(
    "Gene-program score difference (red = increased; blue = decreased)"
)
size_handles = [
    Line2D(
        [0], [0], marker="o", linestyle="none", color="none",
        markerfacecolor="#808080", markeredgecolor="#eeeeee",
        markersize=np.sqrt(22 + 6.5 * size), label=f"{size} genes",
    )
    for size in (3, 6, 12)
]
fig.legend(
    handles=size_handles, frameon=False, loc="lower center", ncol=3,
    bbox_to_anchor=(0.5, 0.005),
)
fig.suptitle(
    "Global scGPT embedding-program activation backbone\n"
    "fixed quotient-graph topology; labelled pathways are exploratory annotations",
    y=0.995,
)
fig.subplots_adjust(bottom=0.12, top=0.91, wspace=0.01, hspace=0.03)
fig.savefig(
    OUTPUT_DIR / "program_activation_backbone_network.png",
    dpi=240, bbox_inches="tight",
)
plt.show()

program_meta_network_nodes = pd.DataFrame({
    "program_id": program_ids,
    "layout_x": [float(program_layout[node][0]) for node in program_ids],
    "layout_y": [float(program_layout[node][1]) for node in program_ids],
    "n_genes": [int(program_size_lookup[node]) for node in program_ids],
    "annotation": [program_annotation_lookup.get(node, "Unannotated") for node in program_ids],
}).merge(program_effect_wide.reset_index(), on="program_id", how="left")
program_meta_network_edges = pd.DataFrame([
    {
        "source_program": source,
        "target_program": target,
        "normalised_meta_weight": float(data["meta_weight"]),
        "cross_program_connectivity_weight": float(data["cross_weight"]),
        "n_cross_program_gene_edges": int(data["n_gene_edges"]),
        "backbone_or_top2_neighbour": True,
    }
    for source, target, data in program_graph.edges(data=True)
])


# ### 9.3 Interferon leading-edge similarity network
# 
# The nodes are the 24 strongest positive genes from the observed aggregate
# Hallmark interferon-alpha and interferon-gamma leading edges, ranked by their
# mean observed effect across 24 hours and one week. Model agreement is not used
# to select genes. Enrichment testing remains the pathway-level GSEA. The same
# nodes, edges and coordinates are reused in all four panels.

# In[27]:


interferon_terms = {
    "HALLMARK::HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK::HALLMARK_INTERFERON_GAMMA_RESPONSE",
}
interferon_leading = _leading_edge_symbols(
    interferon_terms, representations=["observed"]
)
interferon_nodes = _select_focus_nodes(
    interferon_leading,
    score_columns=["observed_24h", "observed_1week"],
    required=("CXCL10", "ISG15", "GBP2", "LY6E", "RSAD2", "SOCS1"),
    n_genes=24,
    selection_rule="mean_positive",
)
interferon_graph = _focused_similarity_graph(interferon_nodes, nearest_per_gene=2)
interferon_network_nodes, interferon_network_edges = _draw_focused_network(
    interferon_nodes,
    interferon_graph,
    network_panel_specs,
    "Interferon-response leading-edge embedding network",
    "interferon_embedding_similarity_network.png",
)
display(interferon_network_nodes[[
    "human_symbol", "program_id", "observed_24h", "model_24h",
    "observed_1week", "model_1week",
]])


# ### 9.4 Late E2F/G2M similarity network
# 
# This focused four-panel map uses the 24 most negatively changed genes from the
# observed one-week Hallmark E2F-target and G2/M-checkpoint leading edges. Model
# agreement is not used for gene selection. The fixed panels make the transition
# from the weaker 24-hour state to late suppression directly visible; the map is
# not intended to identify regulatory hubs.

# In[28]:


cell_cycle_terms = {
    "HALLMARK::HALLMARK_E2F_TARGETS",
    "HALLMARK::HALLMARK_G2M_CHECKPOINT",
}
cell_cycle_leading = _leading_edge_symbols(
    cell_cycle_terms, conditions=["IR_1week"], representations=["observed"]
)
cell_cycle_nodes = _select_focus_nodes(
    cell_cycle_leading,
    score_columns=["observed_1week"],
    required=("TOP2A", "LMNB1", "HMGB2", "SMC4", "MKI67", "CENPA"),
    n_genes=24,
    selection_rule="most_negative",
)
cell_cycle_graph = _focused_similarity_graph(cell_cycle_nodes, nearest_per_gene=2)
cell_cycle_network_nodes, cell_cycle_network_edges = _draw_focused_network(
    cell_cycle_nodes,
    cell_cycle_graph,
    network_panel_specs,
    "Late E2F/G2M leading-edge embedding network",
    "cell_cycle_embedding_similarity_network.png",
)
display(cell_cycle_network_nodes[[
    "human_symbol", "program_id", "observed_1week", "model_1week"
]])


# ### 9.5 Program–pathway bipartite annotation network
# 
# This supplementary network links selected interpretable programs to at most
# their two strongest pathway overlaps meeting within-program FDR < 0.05 and at
# least three overlapping genes. Edge width is overlap size. The legend and title
# explicitly retain the crucial limitation that no displayed association passed
# the global correction across all 141,087 program–pathway tests.

# In[29]:


bipartite_programs = [
    program_id for program_id in
    ["P013", "P162", "P003", "P045", "P174", "P304", "P311"]
    if program_id in set(program_annotation["program_ora"]["program_id"])
]
bipartite_edges = (
    program_annotation["program_ora"]
    .query(
        "program_id in @bipartite_programs and program_fdr < 0.05 and overlap_size >= 3"
    )
    .sort_values(
        ["program_id", "program_fdr", "overlap_size", "pathway"],
        ascending=[True, True, False, True],
    )
    .groupby("program_id", group_keys=False)
    .head(2)
    .reset_index(drop=True)
)

def _short_pathway_label(pathway, database):
    label = str(pathway)
    for prefix in ("HALLMARK_", "REACTOME_"):
        label = label.removeprefix(prefix)
    label = label.replace("_", " ").title()
    prefix = "H" if database == "HALLMARK" else "R"
    return prefix + ": " + textwrap.fill(label, width=30)

bipartite_graph = nx.Graph()
for row in bipartite_edges.itertuples(index=False):
    pathway_node = f"PATH::{row.database}::{row.pathway}"
    bipartite_graph.add_node(row.program_id, kind="program")
    bipartite_graph.add_node(
        pathway_node, kind="pathway",
        display_label=_short_pathway_label(row.pathway, row.database),
    )
    bipartite_graph.add_edge(
        row.program_id, pathway_node,
        overlap_size=int(row.overlap_size),
        program_fdr=float(row.program_fdr),
        global_fdr=float(row.global_fdr),
        overlap_genes=str(row.overlap_genes),
    )

left_nodes = sorted(
    [node for node, data in bipartite_graph.nodes(data=True) if data["kind"] == "program"]
)
right_nodes = sorted(
    [node for node, data in bipartite_graph.nodes(data=True) if data["kind"] == "pathway"],
    key=lambda node: bipartite_graph.nodes[node]["display_label"],
)
left_y = np.linspace(1.0, 0.0, len(left_nodes)) if left_nodes else np.array([])
right_y = np.linspace(1.0, 0.0, len(right_nodes)) if right_nodes else np.array([])
bipartite_layout = {
    **{node: np.array([0.0, y]) for node, y in zip(left_nodes, left_y)},
    **{node: np.array([1.0, y]) for node, y in zip(right_nodes, right_y)},
}

fig, ax = plt.subplots(figsize=(14.5, 9.5))
nx.draw_networkx_edges(
    bipartite_graph, bipartite_layout, ax=ax,
    width=[0.8 + 0.85 * bipartite_graph[u][v]["overlap_size"] for u, v in bipartite_graph.edges],
    edge_color="#8d8d8d", alpha=0.52,
)
nx.draw_networkx_nodes(
    bipartite_graph, bipartite_layout, nodelist=left_nodes,
    node_shape="s", node_size=1050, node_color="#4c78a8",
    edgecolors="white", linewidths=1.0, ax=ax,
)
nx.draw_networkx_nodes(
    bipartite_graph, bipartite_layout, nodelist=right_nodes,
    node_shape="o", node_size=780, node_color="#e2b84b",
    edgecolors="white", linewidths=1.0, ax=ax,
)
for node in left_nodes:
    label = f"{node} — {program_annotation_lookup.get(node, 'Unannotated')}"
    ax.text(-0.08, bipartite_layout[node][1], label, ha="right", va="center", fontsize=10)
for node in right_nodes:
    ax.text(
        1.08, bipartite_layout[node][1],
        bipartite_graph.nodes[node]["display_label"],
        ha="left", va="center", fontsize=9.5,
    )
legend_handles = [
    Line2D([0], [0], marker="s", linestyle="none", color="none",
           markerfacecolor="#4c78a8", markersize=10, label="Embedding program"),
    Line2D([0], [0], marker="o", linestyle="none", color="none",
           markerfacecolor="#e2b84b", markersize=10, label="Hallmark/Reactome term"),
    Line2D([0], [0], color="#8d8d8d", linewidth=3.4,
           label="Thicker edge = more overlapping genes"),
]
fig.legend(
    handles=legend_handles, frameon=False, loc="lower center", ncol=3,
    bbox_to_anchor=(0.5, 0.01),
)
ax.set_xlim(-0.85, 1.85)
ax.set_ylim(-0.08, 1.08)
ax.axis("off")
fig.suptitle(
    "Exploratory program–pathway overlap network\n"
    "within-program FDR < 0.05 and overlap ≥3; none pass global FDR",
    y=0.98,
)
fig.subplots_adjust(bottom=0.11, top=0.88)
fig.savefig(
    OUTPUT_DIR / "program_pathway_bipartite_network.png",
    dpi=240, bbox_inches="tight",
)
plt.show()

bipartite_network_nodes = pd.DataFrame([
    {
        "node": node,
        "kind": data["kind"],
        "display_label": data.get("display_label", node),
        "layout_x": float(bipartite_layout[node][0]),
        "layout_y": float(bipartite_layout[node][1]),
    }
    for node, data in bipartite_graph.nodes(data=True)
])
bipartite_network_edges = bipartite_edges.copy()


# ### Interpretation of the network figures
# 
# The program backbone is an overview of where differential program activation
# appears within the fixed embedding representation. The interferon map makes the
# acute and persistent malignant-cell interferon response visible at gene level,
# while the E2F/G2M map shows the later coordinated loss of cycling-associated
# expression. Agreement of node colours between observed and model panels is
# model recovery of conditional expression direction, not independent validation.
# 
# The program–pathway map is an annotation aid rather than confirmatory evidence.
# Its links passed correction only within individual programs; none survived the
# single global correction across all program–pathway combinations. Ranked GSEA
# therefore remains the primary pathway-level biological result.

# ## 10. Audit survival_classification

# In[30]:


def audit_and_analyse_survival(
    aligned: dict,
    effect_result: dict,
    module_ids: Sequence[str],
) -> dict:
    metadata = aligned["metadata"].copy()
    survival_raw = metadata["survival_classification_adata"].astype(str)
    metadata["survival_status"] = survival_raw.replace({"NA": pd.NA, "": pd.NA, "nan": pd.NA})

    unclassified_label = "Unclassified (original NA)"
    condition_audit = (
        metadata.assign(
            survival_display=metadata["survival_status"].fillna(unclassified_label)
        )
        .groupby(["condition_canonical", "survival_display"], observed=True)
        .size()
        .rename("cells")
        .reset_index()
    )
    sample_audit = (
        metadata.assign(
            survival_display=metadata["survival_status"].fillna(unclassified_label)
        )
        .groupby(["Sample", "condition_canonical", "survival_display"], observed=True)
        .size()
        .rename("cells")
        .reset_index()
    )
    subtype_audit = (
        metadata.dropna(subset=["survival_status"])
        .groupby(["condition_canonical", "GBM_malig_subtypes", "survival_status"], observed=True)
        .size()
        .rename("cells")
        .reset_index()
    )

    association_frames = []
    # Survival labels belong to observed target cells. The perturbation model
    # was not trained on this label and its counterfactual inputs have no
    # target-cell survival identity, so no predicted-survival analysis is run.
    for representation, scores in (("observed", effect_result["observed_scores"]),):
        score_frame = pd.DataFrame(scores, columns=module_ids)
        analysis_frame = pd.concat(
            [
                metadata[[
                    "Sample",
                    "condition_canonical",
                    "GBM_malig_subtypes",
                    "survival_status",
                ]].reset_index(drop=True),
                score_frame,
            ],
            axis=1,
        ).dropna(subset=["survival_status"])

        for condition in ("ctrl", "IR_24h"):
            subset = analysis_frame[analysis_frame["condition_canonical"] == condition]
            grouped = (
                subset.groupby(
                    ["Sample", "GBM_malig_subtypes", "survival_status"],
                    observed=True,
                )[list(module_ids)]
                .mean()
                .reset_index()
            )
            dying = grouped[grouped["survival_status"] == "dying"].set_index(
                ["Sample", "GBM_malig_subtypes"]
            )[list(module_ids)]
            surviving = grouped[grouped["survival_status"] == "surviving"].set_index(
                ["Sample", "GBM_malig_subtypes"]
            )[list(module_ids)]
            common = dying.index.intersection(surviving.index)
            subtype_differences = dying.loc[common] - surviving.loc[common]
            sample_differences = subtype_differences.groupby(level="Sample", observed=True).mean()
            for module_id in module_ids:
                values = sample_differences[module_id].dropna().to_numpy(dtype=float)
                test = ttest_1samp(values, popmean=0.0, nan_policy="omit") if len(values) >= 2 else None
                association_frames.append(
                    {
                        "representation": representation,
                        "condition": condition,
                        "module_id": module_id,
                        "mean_dying_minus_surviving": float(np.mean(values)) if len(values) else np.nan,
                        "sd_across_samples": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                        "n_samples": int(len(values)),
                        "n_sample_subtype_pairs": int(len(common)),
                        "one_sample_t_p": float(test.pvalue) if test is not None else np.nan,
                    }
                )
    survival_effects = pd.DataFrame(association_frames)
    survival_effects["fdr_within_condition_representation"] = survival_effects.groupby(
        ["representation", "condition"]
    )["one_sample_t_p"].transform(lambda x: _bh_adjust(x))
    return {
        "survival_condition_audit": condition_audit,
        "survival_sample_audit": sample_audit,
        "survival_subtype_audit": subtype_audit,
        "survival_module_effects": survival_effects,
    }


# In[31]:


module_ids = modules["module_sizes"]["module_id"].tolist()
survival = audit_and_analyse_survival(aligned, effects, module_ids)

survival_counts = survival["survival_condition_audit"].pivot_table(
    index="condition_canonical",
    columns="survival_display",
    values="cells",
    fill_value=0,
)
display(survival_counts)

print(
    "survival_classification is a per-cell fate/state label, not patient "
    "time-to-event survival. One-week cells contain no dying/surviving labels, "
    "so no one-week comparison is estimable."
)

top_fate_programs = (
    survival["survival_module_effects"]
    .query("condition == 'IR_24h' and representation == 'observed'")
    .merge(modules["module_sizes"], on="module_id", how="left")
    .assign(
        absolute_effect=lambda x: x["mean_dying_minus_surviving"].abs(),
        representative_genes=lambda x: x["central_genes"].map(
            lambda value: ", ".join(str(value).split(";")[:6])
        ),
    )
    .sort_values("absolute_effect", ascending=False)
    .head(12)
)
display(top_fate_programs[[
    "program_id", "annotation", "representative_genes", "mean_dying_minus_surviving",
    "n_samples", "n_sample_subtype_pairs",
    "fdr_within_condition_representation"
]])


# In[32]:


sample_survival = survival["survival_sample_audit"].copy()
condition_order = ["ctrl", "IR_24h", "IR_1week"]
condition_titles = {"ctrl": "Control", "IR_24h": "24 hours", "IR_1week": "1 week"}
status_order = ["surviving", "dying", "Unclassified (original NA)"]
status_colors = {
    "surviving": "#4c9f70",
    "dying": "#c75d5d",
    "Unclassified (original NA)": "#b8b8b8",
}

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), sharey=True)
for ax, condition in zip(axes, condition_order):
    frame = sample_survival[sample_survival["condition_canonical"].eq(condition)]
    counts = frame.pivot_table(
        index="Sample", columns="survival_display", values="cells", fill_value=0
    ).reindex(columns=status_order, fill_value=0)
    counts = counts.sort_index()
    totals = counts.sum(axis=1)
    proportions = counts.div(totals, axis=0)
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for status in status_order:
        values = proportions[status].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=status_colors[status], width=0.72,
               label=status)
        for index, value in enumerate(values):
            if value >= 0.12:
                ax.text(index, bottom[index] + value / 2, f"{value:.0%}",
                        ha="center", va="center", fontsize=8,
                        color="white" if status != "Unclassified (original NA)" else "#333333")
        bottom += values
    ax.set_xticks(x, [f"{sample}\n(n={int(totals.loc[sample]):,})" for sample in counts.index])
    ax.set_title(condition_titles[condition])
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", visible=False)
    if condition == "IR_1week":
        ax.text(0.5, 1.05, "No 1-week fate comparison available",
                transform=ax.transAxes, ha="center", fontsize=9, color="#555555")
axes[0].set_ylabel("Fraction of cells within sample")
handles, labels = axes[0].get_legend_handles_labels()
display_labels = [
    "Surviving", "Dying", "Unclassified (original NA)"
]
fig.legend(handles, display_labels, frameon=False, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle("survival_classification composition by biological sample", y=0.99)
fig.tight_layout(rect=(0, 0.13, 1, 0.90))
fig.savefig(OUTPUT_DIR / "survival_classification_composition.png", dpi=220, bbox_inches="tight")
plt.show()


# ### How to read the survival-classification composition
# 
# Each bar is one biological sample and is normalised to 100%; colours show the
# fractions labelled surviving, dying or originally unclassified. The sample size
# under each bar matters because the samples are strongly unequal in cell count.
# This figure describes annotation composition only. It does not test whether IR
# changes survival, because sample composition, subtype composition and missing
# labels can all differ. All one-week cells are unclassified, so no one-week fate
# comparison is available.
# 
# The fate-state program table is exploratory. Dying-minus-surviving program
# scores are computed within sample and GBM subtype and then averaged across the
# four samples. No Kaplan–Meier or Cox analysis is used, and no predicted survival
# analysis is claimed because `survival_classification` is a per-cell fate/state
# label, was not a model input, and is not patient time-to-event survival.

# ## 11. Save transparent result tables

# In[33]:


program_table.to_csv(OUTPUT_DIR / "embedding_gene_programs.csv", index=False)
program_table[[
    "program_id", "module_id", "membership_hash", "n_embedding_genes",
    "central_genes", "annotation", "annotation_program_fdr",
    "annotation_global_fdr", "median_seed_jaccard"
]].to_csv(OUTPUT_DIR / "program_id_crosswalk.csv", index=False)
modules["nodes"].to_csv(
    OUTPUT_DIR / "embedding_gene_program_membership.csv", index=False
)
program_effects.to_csv(
    OUTPUT_DIR / "gene_program_effects_observed_vs_model.csv", index=False
)
modules["tuning"].to_csv(OUTPUT_DIR / "louvain_tuning_metrics.csv", index=False)
modules["selection"].to_csv(OUTPUT_DIR / "louvain_selected_setting.csv", index=False)
modules["stability"].to_csv(OUTPUT_DIR / "louvain_pairwise_seed_stability.csv", index=False)
modules["program_stability"].to_csv(OUTPUT_DIR / "louvain_program_stability.csv", index=False)
program_annotation["program_annotations"].to_csv(
    OUTPUT_DIR / "embedding_gene_program_annotations.csv", index=False
)
program_annotation["program_ora"].to_csv(
    OUTPUT_DIR / "embedding_gene_program_ora.csv", index=False
)
program_pathway_dotplot_data.to_csv(
    OUTPUT_DIR / "embedding_gene_program_pathway_map_data.csv", index=False
)
effects["gene_ranks"].to_csv(OUTPUT_DIR / "gene_effect_rankings.csv", index=False)
pathways["gsea"].to_csv(OUTPUT_DIR / "pathway_gsea_all.csv", index=False)
pathway_comparison.to_csv(
    OUTPUT_DIR / "pathway_observed_vs_model.csv", index=False
)
sample_gene_deltas.to_csv(OUTPUT_DIR / "sample_gene_effect_rankings.csv", index=False)
sample_pathway_effects.to_csv(OUTPUT_DIR / "sample_level_pathway_effects.csv", index=False)
program_meta_network_nodes.to_csv(
    OUTPUT_DIR / "program_meta_network_nodes.csv", index=False
)
program_meta_network_edges.to_csv(
    OUTPUT_DIR / "program_meta_network_edges.csv", index=False
)
interferon_network_nodes.to_csv(
    OUTPUT_DIR / "interferon_network_nodes.csv", index=False
)
interferon_network_edges.to_csv(
    OUTPUT_DIR / "interferon_network_edges.csv", index=False
)
cell_cycle_network_nodes.to_csv(
    OUTPUT_DIR / "cell_cycle_network_nodes.csv", index=False
)
cell_cycle_network_edges.to_csv(
    OUTPUT_DIR / "cell_cycle_network_edges.csv", index=False
)
bipartite_network_nodes.to_csv(
    OUTPUT_DIR / "program_pathway_bipartite_nodes.csv", index=False
)
bipartite_network_edges.to_csv(
    OUTPUT_DIR / "program_pathway_bipartite_edges.csv", index=False
)
survival["survival_condition_audit"].to_csv(
    OUTPUT_DIR / "survival_classification_counts.csv", index=False
)
survival["survival_sample_audit"].to_csv(
    OUTPUT_DIR / "survival_classification_by_sample.csv", index=False
)
top_fate_programs.to_csv(
    OUTPUT_DIR / "survival_24h_exploratory_programs.csv", index=False
)
if focused_network_created:
    focused_network_nodes.to_csv(OUTPUT_DIR / "focused_network_nodes.csv", index=False)
    focused_network_edges.to_csv(OUTPUT_DIR / "focused_network_edges.csv", index=False)

saved_files = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file())
print("Saved outputs:")
for name in saved_files:
    print(" -", name)


# ## Interpretation limits
# 
# - Embedding proximity is an undirected representation association, not a causal
#   gene-regulatory edge.
# - Seed stability and cosine separation are reported separately; a reproducible
#   fine partition can still have poor separation. Ranked pathway GSEA remains
#   the primary biological result.
# - The model comparison is post-hoc and in-sample because the June checkpoint's
#   original loaders were not genuinely independent.
# - Observed contrasts include native treated-cell composition, whereas the
#   counterfactual contrast intentionally holds the control-cell composition
#   fixed.
# - Treated and control sample IDs are condition-specific and not demonstrably
#   paired; sample-level observed contrasts use an equal-weight control reference.
# - survival_classification is a cellular state label and is unavailable for the
#   one-week comparison.
