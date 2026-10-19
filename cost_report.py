"""
Reads results/raw_results.csv (produced by run_benchmark.py) and produces
the actual thesis deliverable: for each (scale, system), the cheapest
ef_search value that meets the target recall, translated into real cloud
cost -- plus a decision-framework table recommending a system per scale.

Region handling: throughput is a property of where the benchmark was RUN
(`measured_in`, from raw_results.csv), price is a property of where you'd
DEPLOY (`priced_for`, from config.yaml's cost.regions). Every measured run is
priced at every configured region, so you can both re-price US measurements
at Indian rates and, once a Mumbai run exists, compare measured-in-Mumbai
against measured-in-US directly. Both columns are kept in the output so
nobody mistakes a re-priced US measurement for an actual Indian run.

Usage:
    python cost_report.py
"""

import os

import pandas as pd
import yaml

from cost import cost_per_million_queries, monthly_cost_at_load, pick_operating_point


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_results(results_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(results_dir, "raw_results.csv"))
    if "region" not in df.columns:
        # Produced before run_benchmark.py tagged rows with a region.
        df.insert(0, "region", "unspecified")
    return df


def build_summary(df: pd.DataFrame, regions: dict, target_recall: float,
                  hours_per_month: int) -> pd.DataFrame:
    """One row per (measured_in, priced_for, scale, system). Shared by this
    report and app.py, so the app's live "what if my target recall were
    X?" view uses exactly the same operating-point logic as the thesis
    numbers -- not a reimplementation that could drift."""
    summary_rows = []
    for (measured_in, scale, system), group in df.groupby(["region", "scale", "system"]):
        sweep_results = group.to_dict("records")
        # The operating point depends only on the measurement, not on price,
        # so it's picked once and then priced at every region.
        operating_point = pick_operating_point(sweep_results, target_recall)
        for priced_for, region_cfg in regions.items():
            price = region_cfg["hourly_price_usd"]
            summary_rows.append(summarize(measured_in, priced_for, price, scale, system,
                                           group, operating_point, hours_per_month))
    return pd.DataFrame(summary_rows)


def main():
    config = load_config()
    results_dir = config["paths"]["results_dir"]
    target_recall = config["cost"]["target_recall"]
    regions = config["cost"]["regions"]
    hours_per_month = config["cost"]["hours_per_month"]

    df = load_raw_results(results_dir)
    summary = build_summary(df, regions, target_recall, hours_per_month)
    summary_path = os.path.join(results_dir, "cost_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved cost summary to {summary_path}\n")
    print(summary.to_string(index=False))

    # Decision framework: cheapest system that meets target recall, per
    # (deployment region, measurement region, scale).
    print(f"\n--- Decision framework (target recall >= {target_recall}) ---")
    met = summary[summary["met_target_recall"]]
    for (priced_for, measured_in), region_group in met.groupby(["priced_for", "measured_in"]):
        label = regions[priced_for].get("label", priced_for)
        print(f"\n  Deploying in {priced_for} ({label}), throughput measured in {measured_in}:")
        for scale, group in region_group.groupby("scale"):
            best = group.loc[group["cost_per_million_queries_usd"].idxmin()]
            print(f"    {scale}: recommend {best['system']} "
                  f"(${best['cost_per_million_queries_usd']:.4f} per 1M queries, "
                  f"{best['recall']:.3f} recall)")


def summarize(measured_in, priced_for, price, scale, system, group, operating_point, hours_per_month):
    base = {
        "measured_in": measured_in,
        "priced_for": priced_for,
        "hourly_price_usd": price,
        "scale": scale,
        "system": system,
        "n_passages": int(group["n_passages"].iloc[0]),
    }
    if operating_point is None:
        return {
            **base,
            "met_target_recall": False,
            "ef_search_used": None,
            "recall": group["recall"].max(),
            "qps": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "cost_per_million_queries_usd": None,
            "monthly_cost_at_100qps_usd": None,
            "index_memory_mb": group["index_memory_mb"].iloc[0],
            "build_time_s": group["build_time_s"].iloc[0],
        }

    cost_per_m = cost_per_million_queries(price, operating_point["qps"])
    monthly_cost = monthly_cost_at_load(price, target_qps=100,
                                         achieved_qps_per_instance=operating_point["qps"],
                                         hours_per_month=hours_per_month)
    return {
        **base,
        "met_target_recall": True,
        "ef_search_used": operating_point["ef_search"],
        "recall": operating_point["recall"],
        "qps": operating_point["qps"],
        "p50_latency_ms": operating_point["p50_latency_ms"],
        "p95_latency_ms": operating_point["p95_latency_ms"],
        "p99_latency_ms": operating_point["p99_latency_ms"],
        "cost_per_million_queries_usd": round(cost_per_m, 4),
        "monthly_cost_at_100qps_usd": round(monthly_cost, 2),
        "index_memory_mb": operating_point["index_memory_mb"],
        "build_time_s": operating_point["build_time_s"],
    }


if __name__ == "__main__":
    main()
