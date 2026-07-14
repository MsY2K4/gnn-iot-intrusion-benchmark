"""
analyze_results.py

Processes multi-seed GNN experiment results (Bot-IoT dataset) across three
class-imbalance handling strategies:
    - unweighted_loss
    - weighted_loss
    - GraphSMOTE

Each strategy x architecture combination lives in its own JSON file (e.g.
"bot_iot_GraphSMOTE_GAT_multiseed.json"), with "dataset", "strategy", "model",
"overall", and "per_class" fields at the TOP LEVEL of the file (no nested
"architectures" dict). This script auto-discovers all such files in the
input directory and produces the tables needed for a results section / paper:

    1. results_across_strategies.csv
       Overall (dataset-level, all-architecture) summary per strategy.

    2. test_macro_f1_all.csv
       Test Macro-F1 (mean +/- std) for every architecture x strategy x dataset.

    3. architecture_ranking.csv
       Architectures ranked by the BEST macro-F1 they achieve across any
       imbalance strategy.

    4. macro_f1_gain_vs_unweighted.csv
       Delta F1 = strategy_macro_f1 - unweighted_macro_f1, per architecture.

    5. weighted_vs_graphsmote.csv
       Direct head-to-head comparison of WeightedLoss vs GraphSMOTE macro-F1.

    6. per_class_f1.csv
       Per-class F1 (mean +/- std) for every architecture x strategy x class.

    7. summary_report.md
       Human-readable markdown version of all the above, ready to paste
       into a paper/report.

Input files: by default, ALL "*_multiseed.json" files found in --input-dir
are loaded (one file per strategy x architecture combination), e.g.:
    bot_iot_GraphSMOTE_GAT_multiseed.json
    bot_iot_GraphSMOTE_GCN_multiseed.json
    bot_iot_unweighted_loss_GAT_multiseed.json
    bot_iot_weighted_loss_GraphSAGE_multiseed.json
    ... etc.

Usage:
    python3 analyze_results.py --input-dir /path/to/jsons --output-dir /path/to/out
"""

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Friendly display names for strategies (keys must match the "strategy"
# field found inside each JSON file).
STRATEGY_DISPLAY = {
    "unweighted_loss": "Unweighted",
    "weighted_loss": "WeightedLoss",
    "GraphSMOTE": "GraphSMOTE",
}

# Glob pattern used to auto-discover input files when --files is not given.
DEFAULT_FILE_GLOB = "*_multiseed.json"

BASELINE_STRATEGY = "unweighted_loss"


# ---------------------------------------------------------------------------
# Loading & flattening
# ---------------------------------------------------------------------------

def load_json_files(paths):
    """Load each JSON file and return list of raw dicts."""
    payloads = []
    for p in paths:
        with open(p, "r") as f:
            payloads.append(json.load(f))
    return payloads


def mean_std_from_per_seed(values):
    """Fallback: compute mean/std from a list of raw per-seed values."""
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def flatten_overall(payloads):
    """
    Build a tidy long-format list of dicts, one row per (dataset, strategy,
    architecture), with overall metrics (mean/std) and n_seeds.

    Each payload here is a single-architecture file with "overall" and
    "per_seed_results" at the TOP LEVEL (no nested "architectures" dict).
    """
    rows = []
    for payload in payloads:
        dataset = payload.get("dataset", "unknown")
        strategy = payload.get("strategy", "unknown")
        arch_name = payload.get("model", "unknown")

        overall = payload.get("overall", {})
        n_seeds = payload.get("n_seeds", len(payload.get("seeds", [])))

        # Prefer precomputed overall stats; fall back to per_seed_results.
        def get_metric(metric_key):
            if metric_key in overall and "mean" in overall[metric_key]:
                return overall[metric_key]["mean"], overall[metric_key]["std"]
            vals = [
                r.get(metric_key)
                for r in payload.get("per_seed_results", [])
                if r.get(metric_key) is not None
            ]
            return mean_std_from_per_seed(vals)

        acc_mean, acc_std = get_metric("acc")
        f1_mean, f1_std = get_metric("macro_f1")
        prec_mean, prec_std = get_metric("precision")
        rec_mean, rec_std = get_metric("recall")

        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "strategy_display": STRATEGY_DISPLAY.get(strategy, strategy),
                "architecture": arch_name,
                "n_seeds": n_seeds,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "precision_mean": prec_mean,
                "precision_std": prec_std,
                "recall_mean": rec_mean,
                "recall_std": rec_std,
            }
        )
    return pd.DataFrame(rows)


def flatten_per_class(payloads):
    """
    Build a tidy long-format list of dicts, one row per
    (dataset, strategy, architecture, class), with per-class F1/precision/recall.

    Each payload here is a single-architecture file with "per_class" and
    "per_seed_results" at the TOP LEVEL (no nested "architectures" dict).
    """
    rows = []
    for payload in payloads:
        dataset = payload.get("dataset", "unknown")
        strategy = payload.get("strategy", "unknown")
        arch_name = payload.get("model", "unknown")

        per_class = payload.get("per_class", {})

        if per_class:
            for cls_name, cls_stats in per_class.items():
                def get_stat(stat_dict):
                    if isinstance(stat_dict, dict) and "mean" in stat_dict:
                        return stat_dict["mean"], stat_dict["std"]
                    return None, None

                f1_mean, f1_std = get_stat(cls_stats.get("f1", {}))
                prec_mean, prec_std = get_stat(cls_stats.get("precision", {}))
                rec_mean, rec_std = get_stat(cls_stats.get("recall", {}))

                rows.append(
                    {
                        "dataset": dataset,
                        "strategy": strategy,
                        "strategy_display": STRATEGY_DISPLAY.get(strategy, strategy),
                        "architecture": arch_name,
                        "class": cls_name,
                        "f1_mean": f1_mean,
                        "f1_std": f1_std,
                        "precision_mean": prec_mean,
                        "precision_std": prec_std,
                        "recall_mean": rec_mean,
                        "recall_std": rec_std,
                    }
                )
        else:
            # Fallback: derive per-class stats from per_seed_results
            per_seed = payload.get("per_seed_results", [])
            class_vals = defaultdict(lambda: defaultdict(list))
            for seed_result in per_seed:
                for cls_name, cls_stats in seed_result.get("per_class", {}).items():
                    for metric in ("f1", "precision", "recall"):
                        if metric in cls_stats:
                            class_vals[cls_name][metric].append(cls_stats[metric])

            for cls_name, metrics in class_vals.items():
                f1_mean, f1_std = mean_std_from_per_seed(metrics.get("f1", []))
                prec_mean, prec_std = mean_std_from_per_seed(metrics.get("precision", []))
                rec_mean, rec_std = mean_std_from_per_seed(metrics.get("recall", []))
                rows.append(
                    {
                        "dataset": dataset,
                        "strategy": strategy,
                        "strategy_display": STRATEGY_DISPLAY.get(strategy, strategy),
                        "architecture": arch_name,
                        "class": cls_name,
                        "f1_mean": f1_mean,
                        "f1_std": f1_std,
                        "precision_mean": prec_mean,
                        "precision_std": prec_std,
                        "recall_mean": rec_mean,
                        "recall_std": rec_std,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def fmt_mean_std(mean, std, decimals=4):
    if mean is None:
        return "n/a"
    if std is None:
        std = 0.0
    return f"{mean:.{decimals}f} \u00b1 {std:.{decimals}f}"


def build_table_results_across_strategies(df_overall):
    """
    Table 1: Results across dataset and all three imbalance strategies.
    One row per (dataset, strategy), aggregated across architectures
    (mean of architecture means, plus best architecture achieved).
    """
    records = []
    for (dataset, strategy), g in df_overall.groupby(["dataset", "strategy"]):
        best_row = g.loc[g["macro_f1_mean"].idxmax()]
        records.append(
            {
                "dataset": dataset,
                "strategy": STRATEGY_DISPLAY.get(strategy, strategy),
                "n_architectures": g["architecture"].nunique(),
                "avg_macro_f1_mean": g["macro_f1_mean"].mean(),
                "avg_macro_f1_std": g["macro_f1_std"].mean(),
                "avg_accuracy_mean": g["acc_mean"].mean(),
                "best_architecture": best_row["architecture"],
                "best_macro_f1_mean": best_row["macro_f1_mean"],
                "best_macro_f1_std": best_row["macro_f1_std"],
            }
        )
    out = pd.DataFrame(records).sort_values(["dataset", "strategy"])
    out["avg_macro_f1"] = out.apply(
        lambda r: fmt_mean_std(r["avg_macro_f1_mean"], r["avg_macro_f1_std"]), axis=1
    )
    out["best_macro_f1"] = out.apply(
        lambda r: fmt_mean_std(r["best_macro_f1_mean"], r["best_macro_f1_std"]), axis=1
    )
    return out


def build_table_test_macro_f1_all(df_overall):
    """
    Table 2: Test Macro-F1 scores for all architectures, datasets, strategies.
    Wide format: rows = architecture, columns = strategy, values = "mean +/- std".
    Also returns a long-format version.
    """
    long = df_overall[
        ["dataset", "architecture", "strategy", "strategy_display",
         "macro_f1_mean", "macro_f1_std", "n_seeds"]
    ].copy()
    long["macro_f1"] = long.apply(
        lambda r: fmt_mean_std(r["macro_f1_mean"], r["macro_f1_std"]), axis=1
    )

    wide = long.pivot_table(
        index=["dataset", "architecture"],
        columns="strategy_display",
        values="macro_f1",
        aggfunc="first",
    ).reset_index()

    return long.sort_values(["dataset", "architecture", "strategy"]), wide


def build_table_architecture_ranking(df_overall):
    """
    Table 3: Architecture ranking by BEST macro-F1 achieved across any strategy.
    """
    records = []
    for (dataset, arch), g in df_overall.groupby(["dataset", "architecture"]):
        best_row = g.loc[g["macro_f1_mean"].idxmax()]
        records.append(
            {
                "dataset": dataset,
                "architecture": arch,
                "best_strategy": STRATEGY_DISPLAY.get(
                    best_row["strategy"], best_row["strategy"]
                ),
                "best_macro_f1_mean": best_row["macro_f1_mean"],
                "best_macro_f1_std": best_row["macro_f1_std"],
            }
        )
    out = pd.DataFrame(records)
    out = out.sort_values(["dataset", "best_macro_f1_mean"], ascending=[True, False])
    out["rank"] = out.groupby("dataset")["best_macro_f1_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    out["best_macro_f1"] = out.apply(
        lambda r: fmt_mean_std(r["best_macro_f1_mean"], r["best_macro_f1_std"]), axis=1
    )
    cols = ["dataset", "rank", "architecture", "best_strategy", "best_macro_f1",
            "best_macro_f1_mean", "best_macro_f1_std"]
    return out[cols].reset_index(drop=True)


def build_table_gain_vs_unweighted(df_overall, baseline=BASELINE_STRATEGY):
    """
    Table 4: Macro-F1 gain relative to unweighted baseline.
    Delta F1 = strategy_macro_f1 - unweighted_macro_f1, per architecture/dataset.
    """
    records = []
    for (dataset, arch), g in df_overall.groupby(["dataset", "architecture"]):
        base_rows = g[g["strategy"] == baseline]
        if base_rows.empty:
            continue
        base_f1 = base_rows.iloc[0]["macro_f1_mean"]

        for _, row in g.iterrows():
            if row["strategy"] == baseline:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "strategy": STRATEGY_DISPLAY.get(row["strategy"], row["strategy"]),
                    "unweighted_macro_f1_mean": base_f1,
                    "strategy_macro_f1_mean": row["macro_f1_mean"],
                    "delta_f1": row["macro_f1_mean"] - base_f1,
                }
            )
    out = pd.DataFrame(records).sort_values(
        ["dataset", "architecture", "strategy"]
    ).reset_index(drop=True)
    out["delta_f1_pct"] = out["delta_f1"] * 100
    return out


def build_table_weighted_vs_graphsmote(df_overall):
    """
    Table 5: Direct comparison of WeightedLoss vs GraphSMOTE macro-F1.
    """
    records = []
    for (dataset, arch), g in df_overall.groupby(["dataset", "architecture"]):
        w_rows = g[g["strategy"] == "weighted_loss"]
        gs_rows = g[g["strategy"] == "GraphSMOTE"]
        if w_rows.empty or gs_rows.empty:
            continue
        w = w_rows.iloc[0]
        gs = gs_rows.iloc[0]
        records.append(
            {
                "dataset": dataset,
                "architecture": arch,
                "weighted_loss_macro_f1_mean": w["macro_f1_mean"],
                "weighted_loss_macro_f1_std": w["macro_f1_std"],
                "graphsmote_macro_f1_mean": gs["macro_f1_mean"],
                "graphsmote_macro_f1_std": gs["macro_f1_std"],
                "delta_f1_graphsmote_minus_weighted": (
                    gs["macro_f1_mean"] - w["macro_f1_mean"]
                ),
                "better_strategy": (
                    "GraphSMOTE"
                    if gs["macro_f1_mean"] > w["macro_f1_mean"]
                    else "WeightedLoss"
                    if w["macro_f1_mean"] > gs["macro_f1_mean"]
                    else "Tie"
                ),
            }
        )
    out = pd.DataFrame(records).sort_values(["dataset", "architecture"]).reset_index(drop=True)
    out["weighted_loss_macro_f1"] = out.apply(
        lambda r: fmt_mean_std(
            r["weighted_loss_macro_f1_mean"], r["weighted_loss_macro_f1_std"]
        ),
        axis=1,
    )
    out["graphsmote_macro_f1"] = out.apply(
        lambda r: fmt_mean_std(
            r["graphsmote_macro_f1_mean"], r["graphsmote_macro_f1_std"]
        ),
        axis=1,
    )
    return out


def build_table_per_class_f1(df_per_class):
    """
    Table 6: Per-class F1, long format and a pivoted wide format
    (rows = architecture+strategy, columns = class).
    """
    long = df_per_class.copy()
    long["f1"] = long.apply(lambda r: fmt_mean_std(r["f1_mean"], r["f1_std"]), axis=1)
    long = long.sort_values(["dataset", "architecture", "strategy", "class"])

    wide = long.pivot_table(
        index=["dataset", "strategy_display", "architecture"],
        columns="class",
        values="f1",
        aggfunc="first",
    ).reset_index()

    return long, wide


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def df_to_markdown(df, float_format="{:.4f}"):
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        # tabulate not installed -- minimal fallback
        return df.to_string(index=False)


def write_markdown_report(path, tables: dict, dataset_label="Bot-IoT"):
    lines = [f"# {dataset_label} Multi-Seed Results Summary\n"]
    for title, df in tables.items():
        lines.append(f"## {title}\n")
        lines.append(df_to_markdown(df))
        lines.append("\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing the three input JSON files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit list of JSON file paths (overrides --input-dir auto-discovery).",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_FILE_GLOB,
        help=f"Glob pattern (relative to --input-dir) used to auto-discover "
             f"input files when --files is not given. Default: {DEFAULT_FILE_GLOB!r}",
    )
    parser.add_argument(
        "--output-dir",
        default="./results_output",
        help="Directory to write output CSVs and markdown report.",
    )
    args = parser.parse_args()

    if args.files:
        paths = args.files
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Could not find input file(s): {missing}")
    else:
        paths = sorted(glob.glob(os.path.join(args.input_dir, args.glob)))
        # analyze_results.py itself lives alongside the JSONs sometimes; make
        # sure we don't accidentally pick up non-JSON matches.
        paths = [p for p in paths if p.lower().endswith(".json")]
        if not paths:
            raise FileNotFoundError(
                f"No input files found in {args.input_dir!r} matching "
                f"glob {args.glob!r}."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    payloads = load_json_files(paths)

    df_overall = flatten_overall(payloads)
    df_per_class = flatten_per_class(payloads)

    # --- Table 1 ---
    t1 = build_table_results_across_strategies(df_overall)

    # --- Table 2 ---
    t2_long, t2_wide = build_table_test_macro_f1_all(df_overall)

    # --- Table 3 ---
    t3 = build_table_architecture_ranking(df_overall)

    # --- Table 4 ---
    t4 = build_table_gain_vs_unweighted(df_overall)

    # --- Table 5 ---
    t5 = build_table_weighted_vs_graphsmote(df_overall)

    # --- Table 6 ---
    t6_long, t6_wide = build_table_per_class_f1(df_per_class)

    # Save CSVs
    t1.to_csv(os.path.join(args.output_dir, "1_results_across_strategies.csv"), index=False, encoding="utf-8")
    t2_long.to_csv(os.path.join(args.output_dir, "2_test_macro_f1_all_long.csv"), index=False, encoding="utf-8")
    t2_wide.to_csv(os.path.join(args.output_dir, "2_test_macro_f1_all_wide.csv"), index=False, encoding="utf-8")
    t3.to_csv(os.path.join(args.output_dir, "3_architecture_ranking.csv"), index=False, encoding="utf-8")
    t4.to_csv(os.path.join(args.output_dir, "4_macro_f1_gain_vs_unweighted.csv"), index=False, encoding="utf-8")
    t5.to_csv(os.path.join(args.output_dir, "5_weighted_vs_graphsmote.csv"), index=False, encoding="utf-8")
    t6_long.to_csv(os.path.join(args.output_dir, "6_per_class_f1_long.csv"), index=False, encoding="utf-8")
    t6_wide.to_csv(os.path.join(args.output_dir, "6_per_class_f1_wide.csv"), index=False, encoding="utf-8")

    # Also dump the raw flattened overall/per-class tables for reference
    df_overall.to_csv(os.path.join(args.output_dir, "0_raw_overall_flat.csv"), index=False, encoding="utf-8")
    df_per_class.to_csv(os.path.join(args.output_dir, "0_raw_per_class_flat.csv"), index=False, encoding="utf-8")

    # Markdown report with display-friendly subsets
    display_tables = {
        "1. Results Across Dataset & Imbalance Strategies": t1[
            ["dataset", "strategy", "n_architectures", "avg_macro_f1",
             "avg_accuracy_mean", "best_architecture", "best_macro_f1"]
        ],
        "2. Test Macro-F1 -- All Architectures / Datasets / Strategies (wide)": t2_wide,
        "3. Architecture Ranking by Best Macro-F1 (any strategy)": t3[
            ["dataset", "rank", "architecture", "best_strategy", "best_macro_f1"]
        ],
        "4. Macro-F1 Gain vs Unweighted Baseline (ΔF1 = strategy - unweighted)": t4[
            ["dataset", "architecture", "strategy", "delta_f1", "delta_f1_pct"]
        ],
        "5. WeightedLoss vs GraphSMOTE (Macro-F1)": t5[
            ["dataset", "architecture", "weighted_loss_macro_f1",
             "graphsmote_macro_f1", "delta_f1_graphsmote_minus_weighted",
             "better_strategy"]
        ],
        "6. Per-Class F1 (wide)": t6_wide,
    }
    write_markdown_report(
        os.path.join(args.output_dir, "summary_report.md"), display_tables
    )

    print(f"Done. Outputs written to: {os.path.abspath(args.output_dir)}")
    for fn in sorted(os.listdir(args.output_dir)):
        print(f"  - {fn}")


if __name__ == "__main__":
    main()