"""
ToN-IoT — Edge Classification Graph Builder
=============================================
Task  : Predict attack type for each network flow (edge)
Why   : ~98% of IPs are normal, but flows themselves are labeled
        (ddos=6.1M, scanning=7.1M, etc.) -> balanced edge labels

Strategy:
  1. Build a graph where edges = individual flows (sampled, not aggregated)
  2. Node features = IP behavior aggregated from all their flows
  3. Edge features = per-flow characteristics
  4. Train a GNN to produce node embeddings, then classify edges using:
     MLP(concat(h_src, h_edge_feat, h_dst))
"""

import pandas as pd
import numpy as np
import glob
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 20_000   # cap each class to this many edges (balance)
SEED              = 42
rng               = np.random.default_rng(SEED)

files = glob.glob("Network_dataset_*.csv")
if not files:
    raise FileNotFoundError("No Network_dataset_*.csv found.")
print(f"Found {len(files)} file(s)")

# ── Pass 1: fit encoders ──────────────────────────────────────────────────────
print("Pass 1: fitting encoders ...")
all_proto, all_conn, all_types = [], [], []
for f in files:
    for chunk in pd.read_csv(f, chunksize=100_000, low_memory=False):
        all_proto.extend(chunk["proto"].astype(str).values)
        all_conn.extend(chunk["conn_state"].astype(str).values)
        all_types.extend(chunk["type"].astype(str).values)

proto_enc = LabelEncoder().fit(all_proto)
conn_enc  = LabelEncoder().fit(all_conn)
type_enc  = LabelEncoder().fit(all_types)

print(f"  Attack classes : {list(type_enc.classes_)}")
print(f"  Class counts   :")
for cls, cnt in Counter(all_types).most_common():
    print(f"    {cls:>15} : {cnt:>10,}")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

# ── Pass 2: collect ALL rows, grouped by type, then sample ───────────────────
print("\nPass 2: collecting and sampling flows ...")

# We collect rows per class, then sample SAMPLES_PER_CLASS from each
class_rows = defaultdict(list)

for f in files:
    print(f"  Reading {f} ...")
    for chunk in pd.read_csv(f, chunksize=100_000, low_memory=False):
        for col in ["duration","src_bytes","dst_bytes","src_pkts","dst_pkts",
                    "src_port","dst_port"]:
            chunk[col] = to_num(chunk[col]).fillna(0)

        for attack_type, grp in chunk.groupby("type"):
            class_rows[str(attack_type)].append(grp)

# Subsample each class
sampled_frames = []
for cls, frames in class_rows.items():
    df_cls = pd.concat(frames, ignore_index=True)
    n      = min(len(df_cls), SAMPLES_PER_CLASS)
    df_cls = df_cls.sample(n=n, random_state=SEED)
    sampled_frames.append(df_cls)
    print(f"  {cls:>15} : {len(df_cls):>7,} rows sampled (total {len(pd.concat(frames)):,})")

df = pd.concat(sampled_frames, ignore_index=True).sample(frac=1, random_state=SEED)
print(f"\nTotal sampled flows : {len(df):,}")

# ── Build IP → node ID mapping ────────────────────────────────────────────────
ip_to_id  = {}
node_ctr  = 0
node_stats = defaultdict(lambda: defaultdict(float))

for _, row in df.iterrows():
    for ip in [str(row["src_ip"]), str(row["dst_ip"])]:
        if ip not in ip_to_id:
            ip_to_id[ip] = node_ctr
            node_ctr += 1

num_nodes = len(ip_to_id)
print(f"Unique IPs (nodes) : {num_nodes:,}")

# Node features from sampled flows
for _, row in df.iterrows():
    sid = ip_to_id[str(row["src_ip"])]
    did = ip_to_id[str(row["dst_ip"])]
    for nid in [sid, did]:
        node_stats[nid]["flows"]     += 1
        node_stats[nid]["src_bytes"] += float(row["src_bytes"])
        node_stats[nid]["dst_bytes"] += float(row["dst_bytes"])
        node_stats[nid]["src_pkts"]  += float(row["src_pkts"])
        node_stats[nid]["dst_pkts"]  += float(row["dst_pkts"])
        node_stats[nid]["duration"]  += float(row["duration"])

# ── Node feature matrix ───────────────────────────────────────────────────────
X = np.zeros((num_nodes, 8), dtype=np.float32)
for nid in range(num_nodes):
    s  = node_stats[nid]
    sb = s["src_bytes"]; db = s["dst_bytes"]
    sp = s["src_pkts"];  dp = s["dst_pkts"]
    X[nid] = [sb, db, sp, dp, s["flows"], s["duration"],
              sb/(db+1), sp/(dp+1)]

# ── Edge index, edge features, edge labels ────────────────────────────────────
src_ids, dst_ids, edge_feats, edge_labels = [], [], [], []

for _, row in df.iterrows():
    sid = ip_to_id[str(row["src_ip"])]
    did = ip_to_id[str(row["dst_ip"])]
    src_ids.append(sid)
    dst_ids.append(did)
    edge_feats.append([
        float(row["duration"]),
        float(row["src_bytes"]),
        float(row["dst_bytes"]),
        float(proto_enc.transform([str(row["proto"])])[0]),
        float(conn_enc.transform([str(row["conn_state"])])[0]),
        float(row["src_port"]),
        float(row["dst_port"]),
        float(row["src_pkts"]) + float(row["dst_pkts"]),
    ])
    edge_labels.append(int(type_enc.transform([str(row["type"])])[0]))

edge_index  = torch.tensor([src_ids, dst_ids], dtype=torch.long)
edge_attr   = torch.tensor(edge_feats, dtype=torch.float32)
edge_y      = torch.tensor(edge_labels, dtype=torch.long)

# ── Stratified edge split ─────────────────────────────────────────────────────
num_edges   = edge_index.shape[1]
e_train = torch.zeros(num_edges, dtype=torch.bool)
e_val   = torch.zeros(num_edges, dtype=torch.bool)
e_test  = torch.zeros(num_edges, dtype=torch.bool)

for cls_id in range(len(type_enc.classes_)):
    idx = np.where(edge_y.numpy() == cls_id)[0]
    rng.shuffle(idx)
    n = len(idx)
    if n < 3:
        e_train[idx] = True
        continue
    t = int(0.70 * n)
    v = int(0.15 * n)
    e_train[idx[:t]]     = True
    e_val[idx[t:t+v]]    = True
    e_test[idx[t+v:]]    = True

print(f"\nEdge split (stratified):")
print(f"  Train : {e_train.sum():,}")
print(f"  Val   : {e_val.sum():,}")
print(f"  Test  : {e_test.sum():,}")

# ── Build node-level dummy labels (not used in training) ─────────────────────
# We still need y at node level for Data() compatibility; use zeros
node_y = torch.zeros(num_nodes, dtype=torch.long)

# ── Save ──────────────────────────────────────────────────────────────────────
data = Data(
    x           = torch.tensor(X, dtype=torch.float32),
    edge_index  = edge_index,
    edge_attr   = edge_attr,
    y           = node_y,
    edge_y      = edge_y,
    edge_train_mask = e_train,
    edge_val_mask   = e_val,
    edge_test_mask  = e_test,
)

print(f"\n{data}")
torch.save(data, "toniot_edge_graph.pt")
torch.save({"classes": list(type_enc.classes_)}, "edge_label_meta.pt")
print("\nSaved → toniot_edge_graph.pt  &  edge_label_meta.pt")
print("Class mapping:")
for i, c in enumerate(type_enc.classes_):
    print(f"  {i} = {c}")
