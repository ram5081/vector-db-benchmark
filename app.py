"""
The panel-facing demo: type in your data scale, target recall, and query
load, get back a recommended system and its real cost. Reads
results/cost_summary.csv, which cost_report.py produces from actual
benchmark runs -- this app never invents numbers, it only looks them up.

Usage:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
import yaml

from cost import monthly_cost_at_load


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


st.set_page_config(page_title="Vector DB Decision Tool", layout="centered")
st.title("Vector DB Decision Tool")
st.caption(
    "Cost-normalized recommendations from real benchmark runs across "
    "pgvector, Qdrant, and FAISS -- not vendor claims."
)

config = load_config()

try:
    summary = pd.read_csv("results/cost_summary.csv")
except FileNotFoundError:
    st.error(
        "No results yet. Run data_prep.py, then run_benchmark.py, then "
        "cost_report.py first -- this app only reports real measured "
        "numbers, it doesn't estimate."
    )
    st.stop()

if "priced_for" not in summary.columns:
    st.error(
        "results/cost_summary.csv predates per-region pricing. Re-run "
        "cost_report.py to regenerate it."
    )
    st.stop()

met = summary[summary["met_target_recall"]]
if met.empty:
    st.warning("No system in the current results met the configured target recall.")
    st.stop()

regions = config["cost"]["regions"]
deploy_regions = [r for r in regions if r in set(met["priced_for"])]
deploy_region = st.segmented_control(
    "Deployment region",
    deploy_regions,
    default=deploy_regions[0],
    required=True,
    format_func=lambda r: regions[r].get("label", r),
)

scales_available = sorted(met["scale"].unique())
scale_choice = st.selectbox("Your data scale", scales_available)

target_qps = st.number_input("Expected query load (queries/sec)", min_value=1, value=100)

subset = met[(met["scale"] == scale_choice) & (met["priced_for"] == deploy_region)].copy()
if subset.empty:
    st.warning(f"No results at scale {scale_choice} for {deploy_region}.")
    st.stop()

# Prefer throughput actually measured in the deployment region; otherwise
# fall back to another region's measurement, re-priced at this region's rate,
# and say so -- a re-priced number must never pass for a measured one.
measured_regions = sorted(subset["measured_in"].unique())
measured_in = deploy_region if deploy_region in measured_regions else measured_regions[0]
subset = subset[subset["measured_in"] == measured_in]
if measured_in != deploy_region:
    st.info(
        f"No benchmark run exists in {deploy_region} yet. Showing throughput "
        f"measured in {measured_in}, priced at {deploy_region} rates.",
        icon=":material/info:",
    )

hourly_price = float(subset["hourly_price_usd"].iloc[0])
subset["monthly_cost_usd"] = subset["qps"].apply(
    lambda qps: monthly_cost_at_load(
        hourly_price, target_qps, qps, config["cost"]["hours_per_month"],
    )
)
subset = subset.sort_values("monthly_cost_usd")

best = subset.iloc[0]
st.success(
    f"**Recommended: {best['system']}** -- "
    f"${best['monthly_cost_usd']:.2f}/month at {target_qps} QPS, "
    f"{best['recall']:.3f} recall, ef_search={int(best['ef_search_used'])}"
)

st.subheader("All systems at this scale")
st.dataframe(
    subset[["system", "recall", "qps", "monthly_cost_usd", "index_memory_mb", "build_time_s"]]
    .rename(columns={
        "qps": "qps_per_instance",
        "monthly_cost_usd": "monthly_cost_usd_at_target_load",
    })
    .set_index("system"),
    width="stretch",
)

st.caption(
    f"Instance: {config['cost']['instance_type']} "
    f"(${hourly_price}/hr, on-demand list price, pre-tax, {deploy_region}). "
    f"Throughput measured in: {measured_in}. "
    f"Target recall: {config['cost']['target_recall']}."
)
