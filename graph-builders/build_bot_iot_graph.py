# ╔══════════════════════════════════════════════════════════════════════════╗
# ║   UNSW IoT Botnet — Graph Builder for GNN Edge Classification           ║
# ║   Architectures: GraphSAGE · GAT · GCN · GGNN                           ║
# ║   Task: Edge-level Multi-Class Classification (category)                 ║
# ║   5-Class build: DDoS · DoS · Normal · Reconnaissance · Theft           ║
# ║                                                                          ║
# ║   Set the two paths in Section 0 below, then run the script.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 ── CONFIGURE PATHS  (only thing you need to change)
# ═══════════════════════════════════════════════════════════════════════════

TRAIN_PATH = "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv"
TEST_PATH  = "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv"
SAVE_DIR   = "graph_data"

# ── Classes to EXCLUDE before any processing ──────────────────────────────
# All classes are included. Add class names here if you ever need to exclude any.
EXCLUDED_CLASSES = set()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 ── INSTALL DEPENDENCIES  (safe to re-run)
# ═══════════════════════════════════════════════════════════════════════════

import subprocess, sys

def _pip(pkg, extra_args=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q", pkg]
    if extra_args:
        cmd += extra_args
    subprocess.check_call(cmd)

print("=" * 60)
print("Checking dependencies...")

try:
    import sklearn
    print(f"  scikit-learn {sklearn.__version__}  ✓")
except ImportError:
    print("  Installing scikit-learn...")
    _pip("scikit-learn")

try:
    import torch
    print(f"  torch        {torch.__version__}  ✓")
except ImportError:
    print("  Installing PyTorch (CPU)...")
    print("  NOTE: For GPU support install PyTorch manually from https://pytorch.org")
    _pip("torch", ["--index-url", "https://download.pytorch.org/whl/cpu"])
    import torch

try:
    import torch_geometric
    print(f"  torch-geometric {torch_geometric.__version__}  ✓")
except ImportError:
    print("  Installing torch-geometric...")
    _pip("torch_geometric")
    try:
        _pip(
            "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv",
            ["-f", f"https://data.pyg.org/whl/torch-{torch.__version__}+cpu.html"],
        )
    except Exception:
        pass

print("All dependencies ready.")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 ── IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 ── FEATURE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

EDGE_NUMERIC = [
    "sport", "dport", "seq", "stddev",
    "N_IN_Conn_P_SrcIP", "min", "state_number",
    "mean", "N_IN_Conn_P_DstIP", "max",
]
EDGE_CATEGORICAL = ["proto"]

NODE_AGG_COLS = ["srate", "drate"]

TARGET_COL = "category"

REQUIRED_COLS = (
    ["saddr", "daddr", "attack", TARGET_COL]
    + EDGE_NUMERIC
    + EDGE_CATEGORICAL
    + NODE_AGG_COLS
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 ── NUMERIC PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def parse_mixed_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series that may contain:
      - normal numbers
      - numeric strings
      - hex strings like '0x0303'
      - missing values / junk strings

    Returns float64 with NaN for unparseable values.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s_str = s.astype(str).str.strip()
    s_str = s_str.str.replace(",", "", regex=False)
    s_str = s_str.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})

    s_low = s_str.str.lower()
    hex_mask = s_low.str.match(r"^[-+]?0x[0-9a-f]+$", na=False)

    out = pd.to_numeric(s_str.where(~hex_mask), errors="coerce")

    if hex_mask.any():
        def _hex_to_int(x: str) -> int:
            x = str(x).strip().lower()
            sign = -1 if x.startswith("-") else 1
            x = x.lstrip("+-")
            return sign * int(x, 16)

        out.loc[hex_mask] = s_low.loc[hex_mask].map(_hex_to_int)

    return out.astype(np.float64)


def coerce_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Convert the requested columns to float64, safely handling mixed numeric text.
    """
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = parse_mixed_numeric_series(df[c])
    return out

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 ── DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_splits(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV/XLSX, validate required columns, drop NaN rows, and
    **remove any rows whose `category` belongs to EXCLUDED_CLASSES**
    (empty by default — all classes are included).  Populate
    EXCLUDED_CLASSES at the top of this file if you ever need to drop
    a class before processing.
    """

    def _read(path: str) -> pd.DataFrame:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        p = p.resolve()

        if not p.exists():
            raise FileNotFoundError(
                f"\n\n{'='*60}\n"
                f"  FILE NOT FOUND:\n"
                f"    {p}\n\n"
                f"  Please fix the path in SECTION 0 at the top of this file.\n"
                f"{'='*60}\n"
            )

        if p.suffix.lower() == ".csv":
            df = pd.read_csv(str(p), low_memory=False)
        else:
            df = pd.read_excel(str(p))

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in {p.name}: {missing}\n"
                f"Expected columns include: {REQUIRED_COLS}"
            )

        log.info("Loaded  %-55s  →  %d rows", p.name, len(df))
        return df

    train, test = _read(train_path), _read(test_path)

    # ── EXCLUSION FILTER ──────────────────────────────────────────────────
    # Applied to BOTH splits so the LabelEncoder never sees excluded labels
    # and the node/IP mapping is also uncontaminated.
    if EXCLUDED_CLASSES:
        for split_name, df in [("train", train), ("test", test)]:
            mask_excl = df[TARGET_COL].astype(str).isin(EXCLUDED_CLASSES)
            n_dropped = mask_excl.sum()
            if n_dropped:
                log.info(
                    "Excluded %d '%s' rows from %s split  (classes: %s)",
                    n_dropped,
                    ", ".join(sorted(EXCLUDED_CLASSES)),
                    split_name,
                    sorted(df.loc[mask_excl, TARGET_COL].unique().tolist()),
                )
        train = train[~train[TARGET_COL].astype(str).isin(EXCLUDED_CLASSES)].reset_index(drop=True)
        test  = test[~test[TARGET_COL].astype(str).isin(EXCLUDED_CLASSES)].reset_index(drop=True)

    # ── NaN FILTER ────────────────────────────────────────────────────────
    key = EDGE_NUMERIC + EDGE_CATEGORICAL + NODE_AGG_COLS + ["saddr", "daddr", TARGET_COL]
    n_before = (len(train), len(test))
    train = train.dropna(subset=key).reset_index(drop=True)
    test  = test.dropna(subset=key).reset_index(drop=True)
    dropped = (n_before[0] - len(train), n_before[1] - len(test))
    if any(dropped):
        log.warning("NaN rows dropped — train: %d   test: %d", *dropped)

    return train, test

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 ── LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════

def encode_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, int]:
    """
    Fit LabelEncoder on training categories only.
    All classes present in training data are encoded (alphabetical order
    → integer IDs starting from 0).
    Test labels unseen in training → -1 (excluded from evaluation).
    """
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL].astype(str)).astype(np.int64)

    known = set(le.classes_)
    y_test = np.array(
        [le.transform([v])[0] if v in known else -1 for v in test_df[TARGET_COL].astype(str)],
        dtype=np.int64,
    )

    n_classes = len(le.classes_)
    log.info("Classes (%d): %s", n_classes, list(le.classes_))

    unseen = int((y_test == -1).sum())
    if unseen:
        log.warning("%d test edges have unseen category labels → masked out.", unseen)

    return y_train, y_test, le, n_classes

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 ── EDGE FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def _proto_ohe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """One-hot encode `proto` on training vocabulary. Unknown test protos → zeros."""
    vocab = sorted(train_df["proto"].astype(str).unique())

    def _enc(series: pd.Series) -> np.ndarray:
        cat = pd.Categorical(series.astype(str), categories=vocab)
        codes = cat.codes  # -1 for unseen
        mat = np.zeros((len(series), len(vocab)), dtype=np.float32)
        valid = codes >= 0
        rows = np.flatnonzero(valid)
        if len(rows):
            mat[rows, codes[valid]] = 1.0
        return mat

    return _enc(train_df["proto"]), _enc(test_df["proto"]), [f"proto_{p}" for p in vocab]


def build_edge_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Edge feature matrix = [StandardScaled numeric (10) | proto one-hot]
    StandardScaler fitted on TRAIN only, applied to both → no data leakage.
    """
    Xn_tr = coerce_numeric_frame(train_df, EDGE_NUMERIC).to_numpy(dtype=np.float32, copy=True)
    Xn_te = coerce_numeric_frame(test_df, EDGE_NUMERIC).to_numpy(dtype=np.float32, copy=True)

    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(Xn_tr)
    Xn_te = scaler.transform(Xn_te)

    ohe_tr, ohe_te, proto_cols = _proto_ohe(train_df, test_df)
    log.info("Proto OHE vocab: %s", proto_cols)

    X_tr = np.concatenate([Xn_tr, ohe_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([Xn_te, ohe_te], axis=1).astype(np.float32)

    log.info(
        "Edge feature dim: %d  (numeric=%d  proto_ohe=%d)",
        X_tr.shape[1], len(EDGE_NUMERIC), len(proto_cols)
    )
    return X_tr, X_te, scaler

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 ── NODE MAPPING
# ═══════════════════════════════════════════════════════════════════════════

def build_node_mapping(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, int]:
    """
    Integer index for every unique IP seen in train ∪ test.
    Using both splits ensures no unknown node IDs at test time.
    """
    all_ips = pd.concat([
        train_df["saddr"], train_df["daddr"],
        test_df["saddr"], test_df["daddr"],
    ]).astype(str).unique()

    ip2idx = {ip: i for i, ip in enumerate(sorted(all_ips))}
    log.info("Unique IP nodes: %d", len(ip2idx))
    return ip2idx

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 ── NODE FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def build_node_features(
    train_df: pd.DataFrame,
    ip2idx: Dict[str, int],
    n_nodes: int,
) -> torch.Tensor:
    """
    6 structural features per node — computed from TRAINING flows only.

    Features
    ────────
    [0] out_degree
    [1] in_degree
    [2] total_degree
    [3] mean_srate
    [4] mean_drate
    [5] frac_malicious
    """
    src = train_df["saddr"].astype(str).map(ip2idx)
    dst = train_df["daddr"].astype(str).map(ip2idx)
    valid = src.notna() & dst.notna()

    src_idx = src[valid].astype(np.int64).to_numpy()
    dst_idx = dst[valid].astype(np.int64).to_numpy()

    srate = parse_mixed_numeric_series(train_df.loc[valid, "srate"]).to_numpy(dtype=np.float64)
    drate = parse_mixed_numeric_series(train_df.loc[valid, "drate"]).to_numpy(dtype=np.float64)
    atk   = parse_mixed_numeric_series(train_df.loc[valid, "attack"]).to_numpy(dtype=np.float64)

    out_deg = np.bincount(src_idx, minlength=n_nodes).astype(np.float64)
    in_deg  = np.bincount(dst_idx, minlength=n_nodes).astype(np.float64)

    sum_srate    = np.bincount(src_idx, weights=srate, minlength=n_nodes).astype(np.float64)
    sum_drate    = np.bincount(dst_idx, weights=drate, minlength=n_nodes).astype(np.float64)
    sum_atk_out  = np.bincount(src_idx, weights=atk,   minlength=n_nodes).astype(np.float64)
    sum_atk_in   = np.bincount(dst_idx, weights=atk,   minlength=n_nodes).astype(np.float64)

    total = out_deg + in_deg

    def safe_div(num, den):
        return np.divide(num, den, where=den > 0, out=np.zeros_like(num, dtype=np.float64))

    X = np.stack([
        out_deg,
        in_deg,
        total,
        safe_div(sum_srate, out_deg),
        safe_div(sum_drate, in_deg),
        safe_div(sum_atk_out + sum_atk_in, total),
    ], axis=1).astype(np.float32)

    active = (total > 0)
    if active.sum() > 1:
        sc = StandardScaler()
        X[active] = sc.fit_transform(X[active])

    log.info("Node feature matrix: %s  (6 structural features)", X.shape)
    return torch.tensor(X, dtype=torch.float32)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 ── EDGE INDEX
# ═══════════════════════════════════════════════════════════════════════════

def build_edge_index(df: pd.DataFrame, ip2idx: Dict[str, int]) -> torch.Tensor:
    """Directed COO edge_index [2, E] from saddr → daddr."""
    src = df["saddr"].astype(str).map(ip2idx)
    dst = df["daddr"].astype(str).map(ip2idx)
    valid = src.notna() & dst.notna()
    skipped = int((~valid).sum())
    if skipped:
        log.warning("Skipped %d edges with unmapped IPs.", skipped)

    src_idx = src[valid].astype(np.int64).to_numpy()
    dst_idx = dst[valid].astype(np.int64).to_numpy()

    return torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 ── SELF-LOOPS
# ═══════════════════════════════════════════════════════════════════════════

def add_self_loops(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_label: torch.Tensor,
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Append one self-loop per node.
      edge_attr  → zero vector
      edge_label → -1
    """
    idx   = torch.arange(n_nodes, dtype=torch.long)
    sl_ei = torch.stack([idx, idx])
    sl_ea = torch.zeros(n_nodes, edge_attr.shape[1], dtype=torch.float32)
    sl_el = torch.full((n_nodes,), -1, dtype=torch.long)

    return (
        torch.cat([edge_index, sl_ei], dim=1),
        torch.cat([edge_attr,  sl_ea], dim=0),
        torch.cat([edge_label, sl_el], dim=0),
    )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 ── OUTPUT DIRECTORY HELPER
# ═══════════════════════════════════════════════════════════════════════════

def ensure_output_dir(save_dir: str) -> Path:
    """
    Create output directory robustly on Windows and POSIX.
    Returns an absolute Path guaranteed to exist.
    """
    out_dir = Path(save_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir.resolve()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13 ── MAIN BUILD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_graph(
    train_path: str = TRAIN_PATH,
    test_path: str  = TEST_PATH,
    save_dir: str   = SAVE_DIR,
    seed: int       = 42,
) -> Tuple[Data, Dict]:
    """
    Full pipeline: CSV files → PyG Data object + saved .pt / .json
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    log.info("━" * 60)
    log.info("STEP 1 / 7   Loading data splits  (excluded classes: %s)", sorted(EXCLUDED_CLASSES))
    train_df, test_df = load_splits(train_path, test_path)

    log.info("STEP 2 / 7   Encoding edge labels")
    y_train, y_test, le, n_classes = encode_labels(train_df, test_df)

    log.info("STEP 3 / 7   Building node index")
    ip2idx  = build_node_mapping(train_df, test_df)
    n_nodes = len(ip2idx)

    log.info("STEP 4 / 7   Building edge features")
    X_tr, X_te, _ = build_edge_features(train_df, test_df)

    log.info("STEP 5 / 7   Building node features")
    x_node = build_node_features(train_df, ip2idx, n_nodes)

    log.info("STEP 6 / 7   Building edge index")
    ei_tr = build_edge_index(train_df, ip2idx)
    ei_te = build_edge_index(test_df,  ip2idx)
    n_tr  = ei_tr.shape[1]
    n_te  = ei_te.shape[1]

    edge_index = torch.cat([ei_tr, ei_te], dim=1)
    edge_attr  = torch.tensor(np.concatenate([X_tr, X_te]), dtype=torch.float32)
    edge_label = torch.tensor(np.concatenate([y_train, y_test]), dtype=torch.long)

    log.info("STEP 7 / 7   Adding self-loops + building masks")
    edge_index, edge_attr, edge_label = add_self_loops(
        edge_index, edge_attr, edge_label, n_nodes
    )
    E = edge_index.shape[1]

    train_mask = torch.zeros(E, dtype=torch.bool)
    test_mask  = torch.zeros(E, dtype=torch.bool)

    train_mask[:n_tr] = True
    test_valid = edge_label[n_tr : n_tr + n_te] != -1
    test_mask[n_tr : n_tr + n_te] = test_valid

    data = Data(
        x=x_node,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        train_mask=train_mask,
        test_mask=test_mask,
        num_nodes=n_nodes,
    )
    data.num_classes = n_classes

    label_map   = {i: c for i, c in enumerate(le.classes_)}
    train_counts = {label_map[i]: int((y_train == i).sum()) for i in range(n_classes)}

    meta = {
        "num_nodes":              n_nodes,
        "num_train_edges":        n_tr,
        "num_test_edges":         n_te,
        "num_total_edges":        E,
        "num_classes":            n_classes,
        "node_feat_dim":          int(x_node.shape[1]),
        "edge_feat_dim":          int(edge_attr.shape[1]),
        "label_map":              label_map,
        "edge_numeric_features":  EDGE_NUMERIC,
        "excluded_classes":       sorted(EXCLUDED_CLASSES),
        "seed":                   seed,
        "train_class_distribution": train_counts,
    }

    out_dir    = ensure_output_dir(save_dir)
    graph_path = out_dir / "botnet_graph.pt"
    meta_path  = out_dir / "botnet_graph_meta.json"

    print(f"Saving graph to: {graph_path}")
    print(f"Saving meta  to: {meta_path}")
    print(f"Output dir exists? {out_dir.exists()}")

    with graph_path.open("wb") as f:
        torch.save(data, f)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("━" * 60)
    log.info("✅  GRAPH BUILT SUCCESSFULLY")
    log.info("    Nodes              : %d", n_nodes)
    log.info("    Train edges        : %d", n_tr)
    log.info("    Test  edges        : %d", n_te)
    log.info("    Total edges (+SL)  : %d", E)
    log.info("    Node feature dim   : %d", x_node.shape[1])
    log.info("    Edge feature dim   : %d", edge_attr.shape[1])
    log.info("    Classes            : %d   %s", n_classes, list(le.classes_))
    log.info("    Excluded classes   : %s",  sorted(EXCLUDED_CLASSES))
    log.info("    Saved → %s", str(out_dir))
    log.info("━" * 60)

    return data, meta

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14 ── SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def sanity_check(data: Data, meta: dict) -> None:
    """Lightweight assertions to catch construction bugs immediately."""
    E = data.edge_index.shape[1]
    assert data.edge_attr.shape[0]  == E,                "edge_attr row count mismatch"
    assert data.edge_label.shape[0] == E,                "edge_label length mismatch"
    assert data.train_mask.shape[0] == E,                "train_mask length mismatch"
    assert data.test_mask.shape[0]  == E,                "test_mask length mismatch"
    assert data.x.shape[0]          == meta["num_nodes"], "node count mismatch"
    assert not (data.train_mask & data.test_mask).any(), "train_mask and test_mask must be disjoint"
    assert data.edge_label[data.train_mask].min() >= 0,  "all training edges must have valid labels (>= 0)"
    assert data.test_mask.sum() > 0,                     "test_mask is empty — check label encoding"

    log.info(
        "✅  Sanity check passed   train=%d   test=%d   classes=%d   excluded=%s",
        data.train_mask.sum().item(),
        data.test_mask.sum().item(),
        data.num_classes,
        meta.get("excluded_classes", []),
    )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15 ── PRINT FULL REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_report(data: Data, meta: dict) -> None:
    sep = "═" * 58
    print(f"\n{sep}")
    print("  GRAPH REPORT — UNSW IoT Botnet  (10-Best Features, 5-Class)")
    print(sep)
    print(f"  Nodes  (unique IPs)        : {meta['num_nodes']:>12,}")
    print(f"  Train  edges (flows)       : {meta['num_train_edges']:>12,}")
    print(f"  Test   edges (flows)       : {meta['num_test_edges']:>12,}")
    print(f"  Total  edges (incl. SL)    : {meta['num_total_edges']:>12,}")
    print(f"  Node   feature dim         : {meta['node_feat_dim']:>12}")
    print(f"  Edge   feature dim         : {meta['edge_feat_dim']:>12}")
    print(f"  Number of classes          : {meta['num_classes']:>12}")
    if meta.get("excluded_classes"):
        print(f"  Excluded classes           : {', '.join(meta['excluded_classes'])}")

    print(f"\n  Label mapping (train class distribution):")
    max_count = max(meta["train_class_distribution"].values()) if meta["train_class_distribution"] else 1
    for cls_id, cls_name in sorted(meta["label_map"].items()):
        count = meta["train_class_distribution"].get(cls_name, "n/a")
        bar = ""
        if isinstance(count, int) and max_count > 0:
            bar = "█" * min(40, int(count / max_count * 40))
        print(f"    [{cls_id:>2}]  {cls_name:<28}  {count:>9,}  {bar}")

    print(f"\n  Edge numeric features ({len(meta['edge_numeric_features'])}):")
    for i, feat in enumerate(meta["edge_numeric_features"]):
        print(f"    {i+1:>2}. {feat}")

    print(f"\n  Output files saved to:")
    print(f"    {SAVE_DIR}")
    print(f"    ├── botnet_graph.pt         ← PyG Data object")
    print(f"    └── botnet_graph_meta.json  ← label map & stats")
    print(sep + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 16 ── RUN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    data, meta = build_graph(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        save_dir=SAVE_DIR,
        seed=42,
    )

    sanity_check(data, meta)
    print_report(data, meta)

    print("Ready-to-use tensors:")
    print(f"  data.x            {tuple(data.x.shape)}      ← node features")
    print(f"  data.edge_index   {tuple(data.edge_index.shape)}   ← directed COO edges")
    print(f"  data.edge_attr    {tuple(data.edge_attr.shape)}   ← edge features")
    print(f"  data.edge_label   {tuple(data.edge_label.shape)}     ← class ids (-1 = self-loop)")
    print(f"  data.train_mask   {tuple(data.train_mask.shape)}     ← {data.train_mask.sum()} training edges")
    print(f"  data.test_mask    {tuple(data.test_mask.shape)}     ← {data.test_mask.sum()} test edges")
    print(f"  data.num_classes  {data.num_classes}")
    print("\n  data is ready for GraphSAGE / GAT / GCN / GGNN training.")
    print("\n  Class mapping (5-class):")
    for cls_id, cls_name in sorted(meta["label_map"].items()):
        print(f"    {cls_id}  →  {cls_name}")