# Ablations

Structural ablations that isolate one architectural choice at a time,
holding everything else (encoder, depth, width, edge MLP, hyperparameters,
data split) fixed to the corresponding full-benchmark notebook. Each
notebook keeps **only** the ablated model — the original architecture is
not run here; its numbers already live in the `bot-iot/` / `ton-iot/`
results.

Each notebook runs a 2-seed evaluation (seeds `0, 1`) and saves one
consolidated JSON with macro and per-class metrics, in the same format as
the main benchmark notebooks.

## What's being tested

| Notebook | Base architecture | Ablated component | Question it answers |
|---|---|---|---|
| `bot_iot_weighted_loss_GAT_uniformattn_ablation.ipynb` | GAT (bot-iot, weighted loss) | Learned attention → uniform (mean) aggregation | Does GAT's learned attention actually help, or would unweighted neighbor averaging do just as well? |
| `bot_iot_weighted_loss_GraphSAGE_gcnstyle_ablation.ipynb` | GraphSAGE (bot-iot, weighted loss) | `SAGEConv` → `GCNConv` | Does GraphSAGE's explicit self/neighbor separation matter, or is GCN's symmetric-normalized self-loop folding equally good? |
| `ton_iot_unweighted_loss_GraphSAGE_gcnstyle_ablation.ipynb` | GraphSAGE (ton-iot, unweighted loss) | `SAGEConv` → `GCNConv` | Same question as above, replicated on ToN-IoT to check the effect isn't dataset-specific. |

## 1. GAT — Uniform Attention Ablation

`UniformAttentionConv` replaces `GATConv` with a `MessagePassing` layer that
keeps the exact same multi-head linear projection and concat/mean output
shape, but fixes the attention coefficients to uniform (i.e. plain per-head
mean aggregation over neighbors + self-loop) instead of the learned,
softmax-normalized attention GAT uses. There are no `att_src` / `att_dst`
parameters in this layer at all — attention isn't down-weighted, it's
removed.

`GATUniformAblation` is otherwise identical to `GATv1EdgeClassifier`: same
node encoder, layer count, head count, per-layer concat/mean rule, edge MLP,
and hyperparameters.

## 2. GraphSAGE — GCN-Style Aggregation Ablation

`GCNStyleAblation` swaps every `SAGEConv` layer for a `GCNConv` layer. The
difference being isolated: GraphSAGE keeps an explicit, separately-weighted
pathway for a node's own features alongside its aggregated neighbor
features, while GCN folds the self-loop into the same symmetric-normalized
average used for neighbors — no separate self-pathway.

Encoder, edge MLP, depth, width, and dropout are all unchanged from
`GraphSAGEEdgeClassifier`. This ablation is run on both datasets
(`bot-iot`/weighted-loss and `ton-iot`/unweighted-loss) to check whether any
effect generalizes across datasets or is specific to one.

## Running a notebook

Each notebook is self-contained, same as the main benchmark notebooks:

1. Open in Colab (or locally with a GPU).
2. Set `GRAPH_PT_PATH` in the Configuration cell to your graph file
   (`botnet_graph.pt` for bot-iot notebooks, `toniot_edge_graph.pt` for the
   ton-iot notebook — see `../graph-builders/`).
3. Run all cells. Only the ablated model is defined/trained — there's no
   fixed-seed run and no original-architecture comparison baked in.

Each notebook saves one JSON to `/content/results/` (or the working
directory if not on Colab) and offers a Colab download prompt:

| Notebook | Output file |
|---|---|
| GAT uniform-attention ablation | `bot_iot_weighted_loss_GAT_UniformAttnAblation_multiseed.json` |
| GraphSAGE GCN-style ablation (bot-iot) | `bot_iot_weighted_loss_GraphSAGE_GCNStyleAblation_multiseed.json` |
| GraphSAGE GCN-style ablation (ton-iot) | `ton_iot_unweighted_loss_GraphSAGE_GCNStyleAblation_multiseed.json` |

Each JSON contains per-seed results plus mean ± std for overall
(accuracy/precision/recall/F1) and per-class metrics — same schema as the
main benchmark results, so ablation numbers can be compared directly against
the corresponding original architecture's entry in `bot-iot/` / `ton-iot/`
results.

## Why 2 seeds instead of 3

The main benchmark notebooks evaluate over 3 seeds; these ablations use 2.
Ablations are meant to isolate a single design choice relative to an
already-established baseline, not to independently establish a new
result — 2 seeds is enough to see whether an effect is consistent while
keeping runtime down, especially for the bot-iot notebooks where compute
budget is already tight (each notebook is a single architecture run, split
out precisely because a full multi-architecture run exceeds a typical
Colab session).
