# GNN IoT Intrusion Detection Benchmark

Benchmarking four GNN architectures — GAT, GCN, GGNN, and GraphSAGE —
against three training strategies for edge-level, multi-class network
intrusion classification, on two IoT intrusion-detection datasets:
**BoT-IoT** and **ToN-IoT**.

Every model is trained to classify individual network flows (edges in the
graph, not nodes) into their attack category, using a common
graph-construction pipeline, a shared evaluation protocol, and multi-seed
runs for reported results.

## Repository structure

```
.
├── graph-builders/    # scripts that turn raw flow CSVs into PyG graphs
├── bot-iot/            # 12 notebooks: 4 architectures x 3 strategies
├── ton-iot/             # 3 notebooks: 1 per strategy, runs all 4 architectures
└── ablation/            # structural ablations isolating one design choice each
```

Each subfolder has its own README with the specifics for that part of the
pipeline.

## The benchmark

**Datasets**
| | BoT-IoT | ToN-IoT |
|---|---|---|
| Graph | IPs as nodes, flows as edges | IPs as nodes, flows as edges |
| Classes | DDoS, DoS, Normal, Reconnaissance, Theft (5) | backdoor, ddos, dos, injection, mitm, normal, password, ransomware, scanning, xss (10) |
| Source | UNSW BoT-IoT, 5%, 10-best-features | UNSW ToN-IoT, Processed Network dataset |

See `graph-builders/README.md` for exact download links and how the graphs
are built.

**Architectures**: GAT, GCN, GGNN, GraphSAGE — all edge classifiers built on
top of node embeddings, sharing the same encoder shape, edge-MLP head, and
hyperparameter search space per architecture.

**Training strategies**: each architecture is trained three ways per
dataset —

- **weighted loss** — class-weighted cross-entropy to counter class
  imbalance
- **unweighted loss** — plain cross-entropy, no reweighting
- **GraphSMOTE** — synthetic oversampling of minority-class edges during
  training

**Evaluation**: every reported result is a multi-seed run (2 or 3 seeds
depending on the notebook — noted in each subfolder) reporting mean ± std
for overall accuracy/precision/recall/F1 and per-class F1/precision/recall.

## Why bot-iot and ton-iot are organized differently

`ton-iot/` has 3 notebooks — one per strategy, each running all 4
architectures back-to-back and writing one consolidated results JSON.

`bot-iot/` has 12 notebooks — one per (architecture, strategy) pair. Running
all 4 architectures in a single BoT-IoT notebook took long enough that a
single Colab session would time out before finishing, so each architecture
was split into its own notebook. Each one still runs the full multi-seed
evaluation and writes its own results JSON; see `bot-iot/README.md` for the
exact file list and output naming.

## Ablations

`ablation/` isolates individual architectural choices — e.g. does GAT's
learned attention actually outperform uniform aggregation, does
GraphSAGE's explicit self-pathway matter versus GCN's folded self-loop —
by swapping one component while holding everything else fixed to the
corresponding full-benchmark run. See `ablation/README.md` for details on
each one.

## Reproducing a result

1. Build (or download, if provided separately) the graph `.pt` file for
   your dataset — see `graph-builders/README.md`.
2. Open the notebook for the architecture/strategy/dataset combination you
   want in Colab.
3. Set `GRAPH_PT_PATH` in the Configuration cell to your graph file.
4. Run all cells. Each notebook is self-contained and saves its own results
   JSON at the end.

## Results format

Every notebook writes one JSON with this shape:

```json
{
  "model": "...",
  "dataset": "...",
  "strategy": "...",
  "seeds": [0, 1, ...],
  "n_seeds": 2,
  "per_seed_results": [ ... ],
  "overall": {
    "accuracy":  {"mean": ..., "std": ...},
    "precision": {"mean": ..., "std": ...},
    "recall":    {"mean": ..., "std": ...},
    "f1":        {"mean": ..., "std": ...}
  },
  "per_class": {
    "<class name>": {
      "f1":        {"mean": ..., "std": ...},
      "precision": {"mean": ..., "std": ...},
      "recall":    {"mean": ..., "std": ...}
    },
    "...": { ... }
  }
}
```

(`ton-iot/` notebooks nest this per architecture inside one JSON per
strategy; `bot-iot/` and `ablation/` notebooks each write one JSON per run —
see the respective READMEs.)
