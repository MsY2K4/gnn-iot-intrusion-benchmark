# Graph Builders

Scripts that convert the raw network-flow records from BoT-IoT and ToN-IoT
into PyTorch Geometric `Data` objects for edge-level classification. Each
script produces one `.pt` graph file (plus a metadata file) consumed by the
training notebooks in `bot-iot/` and `ton-iot/`.

In both graphs, **nodes are IP addresses** and **edges are individual network
flows**, so classification happens at the edge level (predict the flow's
attack category) rather than the node level.

## Scripts

| Script | Dataset | Output |
|---|---|---|
| `build_bot_iot_graph.py` | BoT-IoT | `graph_data/botnet_graph.pt`, `graph_data/botnet_graph_meta.json` |
| `build_ton_iot_graph.py` | ToN-IoT | `toniot_edge_graph.pt`, `edge_label_meta.pt` |

Both scripts only need `pandas`, `numpy`, `scikit-learn`, `torch`, and
`torch_geometric`. `build_bot_iot_graph.py` will attempt to install any
missing dependency automatically on first run.

## 1. Download the raw data

**BoT-IoT** — UNSW 5%, 10-best-features, pre-split Training/Testing CSVs:
https://unsw-my.sharepoint.com/personal/z5131399_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5131399%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FBot%2DIoT%5FDataset%2FDataset%2F5%25%2F10%2Dbest%20features%2F10%2Dbest%20Training%2DTesting%20split&viewid=604d81f1%2D64a9%2D4a09%2D8464%2D3c45ff9ba8fe&ga=1

Download both CSVs and place them next to `build_bot_iot_graph.py`, or update
`TRAIN_PATH` / `TEST_PATH` at the top of the script:

```
UNSW_2018_IoT_Botnet_Final_10_best_Training.csv
UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv
```

**ToN-IoT** — Processed Network dataset:
https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FTON%5FIoT%20datasets%2FProcessed%5Fdatasets%2FProcessed%5FNetwork%5Fdataset&viewid=f8d1dec5%2Dcd5f%2D42ae%2D8b06%2D2fece580c74a&ga=1

Download the CSV file(s) and place them next to `build_ton_iot_graph.py`. The
script picks up every file matching `Network_dataset_*.csv` in the working
directory, so multiple part-files are fine as-is.

> Both datasets are hosted by UNSW behind an authenticated OneDrive link — a
> free Microsoft/UNSW sign-in may be required to download.

## 2. Run the builders

```bash
pip install pandas numpy scikit-learn torch torch_geometric

python build_bot_iot_graph.py
python build_ton_iot_graph.py
```

Each script prints a build report (node/edge counts, feature dims, class
distribution) when it finishes, and writes its outputs to the working
directory (BoT-IoT additionally nests its outputs under `graph_data/`).

Copy the resulting `.pt` file(s) into your Colab environment (e.g. Google
Drive) and point `GRAPH_PT_PATH` in the training notebooks at that path.

## What each script does

### `build_bot_iot_graph.py`

- Loads the pre-split Training/Testing CSVs, validates required columns, and
  drops rows with missing values in key fields.
- Encodes the `category` target with a `LabelEncoder` fit on the training
  split only; any category seen only in test is masked out (label `-1`).
- Builds a directed edge for every flow (`saddr -> daddr`), with edge
  features = 10 standard-scaled numeric fields (fit on train only) + a
  one-hot encoding of `proto`.
- Builds 6 structural node features per IP (in/out/total degree, mean
  send/receive rate, fraction of malicious traffic), computed from the
  training split only.
- Appends one self-loop per node (`edge_label = -1`, zero edge features) so
  every node has at least one incoming edge for message passing.
- Runs a sanity check (mask disjointness, label validity, shape consistency)
  before saving.
- Saves `botnet_graph.pt` (the PyG `Data` object) and
  `botnet_graph_meta.json` (label map, feature dims, class distribution).

### `build_ton_iot_graph.py`

- Reads all matching `Network_dataset_*.csv` files in two passes: first to
  fit label/categorical encoders across the full data, second to collect and
  class-balance a sample of flows (default: up to 20,000 per attack type).
- Builds one node per unique IP, with 8 aggregate features computed from
  that IP's sampled flows (total bytes/packets sent and received, flow
  count, total duration, and two send/receive ratios).
- Builds one edge per sampled flow, with 8 edge features (duration, bytes,
  encoded protocol/connection state, ports, packet count).
- Performs a stratified 70/15/15 train/val/test split per class on the
  edges.
- Saves `toniot_edge_graph.pt` (the PyG `Data` object, with
  `edge_train_mask` / `edge_val_mask` / `edge_test_mask`) and
  `edge_label_meta.pt` (the class name list).

## Notes

- Both builders are deterministic given the same input files (`SEED = 42`
  throughout), so re-running produces the same graph.
- The BoT-IoT builder can exclude specific attack categories via the
  `EXCLUDED_CLASSES` set at the top of the script (empty by default — all
  classes are included).
- The ToN-IoT builder's `SAMPLES_PER_CLASS` constant controls the per-class
  cap used for balancing; lower it for a faster/smaller build, or remove the
  cap entirely by setting it above your largest class count.
