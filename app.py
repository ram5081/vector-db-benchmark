"""
The panel-facing demo: pick a deployment region, data scale, target recall,
and query load, get back a recommended system, its real cost, and the
measurements behind it. Reads results/raw_results.csv (produced by
run_benchmark.py) and prices it with the same build_summary() that
cost_report.py uses -- this app never invents numbers, it only looks them
up and re-applies the thesis's own cost rules to them.

Usage:
    streamlit run app.py
"""

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from cost import instances_needed_for_load, monthly_cost_at_load
from cost_report import build_summary, load_raw_results

st.set_page_config(
    page_title="Vector DB decision tool",
    page_icon=":material/database:",
    layout="wide",
)

# Fixed system -> color slot, so a system keeps its color on every chart and
# under every filter. Slots are the first three of a palette validated for
# colorblind separation (all pairs, light and dark).
SYSTEM_ORDER = ["pgvector", "qdrant", "faiss"]
SYSTEM_LABELS = {"pgvector": "pgvector", "qdrant": "Qdrant", "faiss": "FAISS"}
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
INK = {
    "light": {"secondary": "#52514e", "muted": "#898781"},
    "dark": {"secondary": "#c3c2b7", "muted": "#898781"},
}
RECALL_TARGETS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]


@st.cache_data(ttl="10m")
def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_data(ttl="10m")
def load_raw(results_dir: str, mtime: float) -> pd.DataFrame:
    # mtime is only part of the cache key, so a re-run benchmark shows up
    # without waiting for the TTL.
    return load_raw_results(results_dir)


@st.cache_data(ttl="10m")
def summarize(raw: pd.DataFrame, regions: dict, target_recall: float, hours_per_month: int):
    return build_summary(raw, regions, target_recall, hours_per_month)


def label(system: str) -> str:
    return SYSTEM_LABELS.get(system, system)


def scale_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M passages"
    if n >= 1_000:
        return f"{n / 1_000:g}K passages"
    return f"{n} passages"


def usd(x: float) -> str:
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}"


def pick_measured_in(rows: pd.DataFrame, deploy_region: str) -> str:
    """Prefer throughput measured in the deployment region itself; else fall
    back to another region's measurement (re-priced at this region's rate)."""
    measured = sorted(rows["measured_in"].unique())
    return deploy_region if deploy_region in measured else measured[0]


def log_ticks(lo: float, hi: float) -> list:
    """1-2-5 ticks spanning [lo, hi], so a log axis gets a few readable
    labels instead of a dense run of minor gridlines."""
    ticks, decade = [], 10 ** int(np.floor(np.log10(max(lo, 1e-9))))
    while decade <= hi * 10:
        ticks += [m * decade for m in (1, 2, 5) if lo <= m * decade <= hi]
        decade *= 10
    return ticks or [lo, hi]


def color_scale(systems, mode: str) -> alt.Scale:
    domain = [s for s in SYSTEM_ORDER if s in systems] + sorted(set(systems) - set(SYSTEM_ORDER))
    slot = {s: i for i, s in enumerate(SYSTEM_ORDER + sorted(set(systems) - set(SYSTEM_ORDER)))}
    return alt.Scale(domain=[label(s) for s in domain],
                     range=[PALETTE[mode][slot[s] % 8] for s in domain])


# ---------------------------------------------------------------- data

config = load_config()
results_dir = config["paths"]["results_dir"]
raw_path = os.path.join(results_dir, "raw_results.csv")
regions = config["cost"]["regions"]
hours_per_month = config["cost"]["hours_per_month"]
mode = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"

st.title("Vector DB decision tool", icon=":material/database:")
st.caption(
    "Which self-hosted vector database is cheapest for your workload? "
    "Cost-normalized results from real benchmark runs of pgvector, Qdrant and "
    "FAISS on MS MARCO, not vendor claims."
)

if not os.path.exists(raw_path):
    st.error(
        "No results yet. Run data_prep.py, then run_benchmark.py first -- this "
        "app only reports real measured numbers, it doesn't estimate.",
        icon=":material/error:",
    )
    st.stop()

raw = load_raw(results_dir, os.path.getmtime(raw_path))

if raw["n_passages"].max() < config["dataset"]["scales"][0]["n_passages"]:
    st.warning(
        f"These are smoke-test numbers ({raw['n_passages'].max():,} passages, "
        f"{raw['system'].nunique()} system(s)), only proving the pipeline works "
        "end to end. They aren't thesis results yet.",
        title="Preliminary data",
        icon=":material/science:",
    )

# ---------------------------------------------------------------- controls

scales = (raw[["scale", "n_passages"]].drop_duplicates("scale")
          .sort_values("n_passages"))
scale_names = scales["scale"].tolist()
scale_sizes = dict(zip(scales["scale"], scales["n_passages"]))
default_recall = config["cost"]["target_recall"]
if default_recall not in RECALL_TARGETS:
    RECALL_TARGETS = sorted(RECALL_TARGETS + [default_recall])

with st.container(border=True):
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    c3, c4 = st.columns(2, vertical_alignment="bottom")
    with c1:
        deploy_region = st.segmented_control(
            "Deployment region", list(regions), default=list(regions)[0], required=True,
            format_func=lambda r: regions[r].get("label", r),
        )
    with c2:
        scale_choice = st.segmented_control(
            "Data scale", scale_names, default=scale_names[-1], required=True,
            format_func=lambda s: scale_label(scale_sizes[s]),
        )
    with c3:
        target_recall = st.select_slider(
            "Target recall@10", RECALL_TARGETS, value=default_recall,
            help="Each system is costed at the cheapest ef_search that reaches "
                 f"this recall. The thesis reports {default_recall}.",
        )
    with c4:
        target_qps = st.number_input("Query load (QPS)", min_value=1, value=100, step=50)

summary = summarize(raw, regions, target_recall, hours_per_month)
in_region = summary[summary["priced_for"] == deploy_region]
at_scale = in_region[in_region["scale"] == scale_choice]
measured_in = pick_measured_in(at_scale, deploy_region)
at_scale = at_scale[at_scale["measured_in"] == measured_in].copy()
hourly_price = float(at_scale["hourly_price_usd"].iloc[0])

met = at_scale[at_scale["met_target_recall"]].copy()
missed = at_scale[~at_scale["met_target_recall"]]
met["instances"] = met["qps"].apply(lambda q: instances_needed_for_load(target_qps, q))
met["monthly_cost"] = met["qps"].apply(
    lambda q: monthly_cost_at_load(hourly_price, target_qps, q, hours_per_month))
met = met.sort_values("cost_per_million_queries_usd")

if measured_in != deploy_region:
    st.caption(
        f":material/info: No benchmark has been run in {deploy_region} yet: showing "
        f"throughput measured in **{measured_in}**, priced at {deploy_region} rates."
    )

# ---------------------------------------------------------------- recommendation

if met.empty:
    st.warning(
        f"No system reached recall {target_recall} at any ef_search tested at this "
        "scale. Try a lower target -- this is a real finding, not an error.",
        icon=":material/warning:",
    )
else:
    best = met.iloc[0]
    st.subheader(f"Recommended: {label(best['system'])}", icon=":material/verified:")
    if len(met) > 1:
        runner_up = met.iloc[1]
        ratio = runner_up["cost_per_million_queries_usd"] / best["cost_per_million_queries_usd"]
        st.markdown(
            f"**{ratio:.1f}× cheaper** per query than the runner-up, "
            f"{label(runner_up['system'])}, at the same recall target."
        )

    with st.container(horizontal=True):
        st.metric("Monthly cost at your load", usd(best["monthly_cost"]), border=True,
                  help=f"Always-on instances needed for {target_qps} QPS × "
                       f"${hourly_price}/hr × {hours_per_month} h.")
        st.metric("Cost per 1M queries", usd(best["cost_per_million_queries_usd"]), border=True)
        st.metric("Recall@10", f"{best['recall']:.3f}", border=True,
                  help=f"At ef_search={int(best['ef_search_used'])}, the cheapest "
                       f"setting reaching {target_recall}.")
        st.metric("Throughput per instance", f"{best['qps']:,.0f} QPS", border=True)
        st.metric("Instances needed", f"{int(best['instances'])}", border=True,
                  help=f"{config['cost']['instance_type']} instances to sustain {target_qps} QPS.")

for _, row in missed.iterrows():
    st.caption(
        f":material/block: {label(row['system'])} never reached recall {target_recall} "
        f"at this scale (best: {row['recall']:.3f}), so it isn't costed."
    )

# ---------------------------------------------------------------- charts

ink = INK[mode]
colors = color_scale(raw["system"].unique(), mode)
legend = alt.Legend(title=None, orient="top")

left, right = st.columns(2)

with left, st.container(border=True):
    st.markdown("**Speed vs accuracy**")
    st.caption("Each point is one ef_search setting. Up and to the right is better; "
               "rings mark the setting each system is costed at.")
    sweep = raw[(raw["scale"] == scale_choice) & (raw["region"] == measured_in)].copy()
    sweep["System"] = sweep["system"].map(label)
    base = alt.Chart(sweep).encode(
        x=alt.X("recall:Q", title="Recall@10", scale=alt.Scale(zero=False, nice=False,
                domain=[min(sweep["recall"].min(), target_recall) - 0.01, 1.0])),
        y=alt.Y("qps:Q", title="Queries per second (log)", scale=alt.Scale(type="log"),
                axis=alt.Axis(format="~s", values=log_ticks(sweep["qps"].min(), sweep["qps"].max()))),
        color=alt.Color("System:N", scale=colors, legend=legend),
    )
    tooltip = [
        alt.Tooltip("System:N"),
        alt.Tooltip("ef_search:Q", title="ef_search"),
        alt.Tooltip("recall:Q", format=".3f", title="Recall@10"),
        alt.Tooltip("qps:Q", format=",.0f", title="QPS"),
        alt.Tooltip("p95_latency_ms:Q", format=".2f", title="p95 latency (ms)"),
    ]
    target_rule = alt.Chart(pd.DataFrame({"recall": [target_recall]})).mark_rule(
        color=ink["muted"], strokeWidth=1).encode(x="recall:Q")
    target_text = target_rule.mark_text(
        text=f"target {target_recall}", align="left", dx=4, baseline="top",
        color=ink["secondary"], fontSize=11).encode(y=alt.value(4))
    ops = met.assign(System=met["system"].map(label))
    rings = alt.Chart(ops).mark_point(size=260, filled=False, strokeWidth=2).encode(
        x="recall:Q", y="qps:Q", color=alt.Color("System:N", scale=colors, legend=None))
    starts = sweep.loc[sweep.groupby("system")["ef_search"].idxmin()]
    line_labels = alt.Chart(starts).mark_text(
        align="left", dx=6, dy=-10, color=ink["secondary"], fontSize=12,
    ).encode(x="recall:Q", y="qps:Q", text="System:N")
    chart = (
        target_rule + target_text + line_labels
        + base.mark_line(strokeWidth=2)
        + base.mark_point(size=64, filled=True, opacity=1)
        + rings
        + base.mark_point(size=500, opacity=0).encode(tooltip=tooltip)  # larger hover target
    ).properties(height=320)
    st.altair_chart(chart, alt="Recall versus throughput per system across ef_search settings")

with right, st.container(border=True):
    st.markdown(f"**Monthly cost at {target_qps:,} QPS**")
    st.caption(f"Always-on {config['cost']['instance_type']} fleet sized to your load, "
               f"{deploy_region} on-demand price. Lower is better.")
    if met.empty:
        st.caption("No system met the recall target, so there's nothing to cost.")
    else:
        bars = met.assign(System=met["system"].map(label))
        bars["cost_label"] = bars["monthly_cost"].map(usd)
        bar_base = alt.Chart(bars).encode(
            y=alt.Y("System:N", sort=alt.EncodingSortField("monthly_cost"), title=None),
            x=alt.X("monthly_cost:Q", title="USD per month", axis=alt.Axis(format="$,.0f")),
        )
        bar_chart = (
            bar_base.mark_bar(cornerRadiusEnd=4, size=20).encode(
                color=alt.Color("System:N", scale=colors, legend=None),
                tooltip=[alt.Tooltip("System:N"),
                         alt.Tooltip("monthly_cost:Q", format="$,.2f", title="Monthly cost"),
                         alt.Tooltip("instances:Q", title="Instances"),
                         alt.Tooltip("cost_per_million_queries_usd:Q", format="$.4f",
                                     title="Per 1M queries")])
            + bar_base.mark_text(align="left", dx=6, color=ink["secondary"], fontSize=12)
            .encode(text="cost_label:N")
        ).properties(height=60 + 56 * len(bars))
        st.altair_chart(bar_chart, alt=f"Monthly cost per system, lowest for {label(met.iloc[0]['system'])}")

left, right = st.columns(2)

# Cost per scale, each scale using the measurement chosen for this region.
per_scale = []
for scale_name, rows in in_region.groupby("scale"):
    m = pick_measured_in(rows, deploy_region)
    per_scale.append(rows[(rows["measured_in"] == m) & rows["met_target_recall"]])
per_scale = pd.concat(per_scale) if per_scale else in_region.head(0)

with left, st.container(border=True):
    st.markdown("**How cost grows with data size**")
    st.caption(f"Cost per 1M queries at recall ≥ {target_recall}, per scale tier.")
    if per_scale["scale"].nunique() < 2:
        st.caption(":material/hourglass_empty: Appears once more than one scale has been "
                   "benchmarked (50K / 250K / 1M).")
    else:
        growth = per_scale.assign(System=per_scale["system"].map(label))
        growth_base = alt.Chart(growth).encode(
            x=alt.X("n_passages:Q", title="Passages indexed (log)",
                    scale=alt.Scale(type="log", nice=False,
                                    domain=[growth["n_passages"].min(), growth["n_passages"].max() * 1.6]),
                    axis=alt.Axis(format="~s", values=sorted(growth["n_passages"].unique().tolist()))),
            y=alt.Y("cost_per_million_queries_usd:Q", title="USD per 1M queries",
                    axis=alt.Axis(format="$.2f" if growth["cost_per_million_queries_usd"].max() >= 0.1
                                  else "$.4f")),
            color=alt.Color("System:N", scale=colors, legend=legend),
        )
        ends = growth.loc[growth.groupby("system")["n_passages"].idxmax()]
        growth_chart = (
            growth_base.mark_line(strokeWidth=2)
            + growth_base.mark_point(size=64, filled=True, opacity=1).encode(tooltip=[
                alt.Tooltip("System:N"), alt.Tooltip("scale:N", title="Scale"),
                alt.Tooltip("cost_per_million_queries_usd:Q", format="$.4f", title="Per 1M queries"),
                alt.Tooltip("ef_search_used:Q", title="ef_search")])
            + alt.Chart(ends).mark_text(align="left", dx=8, color=ink["secondary"], fontSize=12)
            .encode(x="n_passages:Q", y="cost_per_million_queries_usd:Q",
                    text=alt.Text("System:N"))
        ).properties(height=280)
        st.altair_chart(growth_chart, alt="Cost per million queries as the corpus grows, per system")

with right, st.container(border=True):
    st.markdown("**Decision framework**")
    st.caption(f"Cheapest system per scale at recall ≥ {target_recall} and {target_qps:,} QPS, "
               f"{deploy_region} pricing.")
    decisions = []
    for scale_name in scale_names:
        rows = per_scale[per_scale["scale"] == scale_name].sort_values("cost_per_million_queries_usd")
        if rows.empty:
            decisions.append({"Scale": scale_label(scale_sizes[scale_name]),
                              "Recommended": "None met target", "Per 1M queries": None,
                              "Monthly cost": None, "Runner-up": None, "Advantage": None})
            continue
        top = rows.iloc[0]
        nxt = rows.iloc[1] if len(rows) > 1 else None
        decisions.append({
            "Scale": scale_label(scale_sizes[scale_name]),
            "Recommended": label(top["system"]),
            "Per 1M queries": top["cost_per_million_queries_usd"],
            "Monthly cost": monthly_cost_at_load(top["hourly_price_usd"], target_qps,
                                                 top["qps"], hours_per_month),
            "Runner-up": label(nxt["system"]) if nxt is not None else None,
            "Advantage": (nxt["cost_per_million_queries_usd"] / top["cost_per_million_queries_usd"]
                          if nxt is not None else None),
        })
    st.dataframe(
        pd.DataFrame(decisions),
        hide_index=True,
        column_config={
            "Per 1M queries": st.column_config.NumberColumn(format="$%.4f"),
            "Monthly cost": st.column_config.NumberColumn(format="$%.2f"),
            "Advantage": st.column_config.NumberColumn(
                format="%.1f×", help="How many times cheaper per query than the runner-up"),
        },
        alt="Recommended system for each data scale",
    )

# ---------------------------------------------------------------- detail table

with st.container(border=True):
    st.markdown(f"**All systems at {scale_label(scale_sizes[scale_choice])}**")
    detail = at_scale.copy()
    detail["System"] = detail["system"].map(label)
    detail = detail.merge(met[["system", "monthly_cost", "instances"]], on="system", how="left")
    st.dataframe(
        detail.sort_values("cost_per_million_queries_usd", na_position="last"),
        hide_index=True,
        column_order=["System", "met_target_recall", "ef_search_used", "recall", "qps",
                      "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                      "cost_per_million_queries_usd", "monthly_cost", "instances",
                      "index_memory_mb", "build_time_s"],
        column_config={
            "met_target_recall": st.column_config.CheckboxColumn("Meets target"),
            "ef_search_used": st.column_config.NumberColumn("ef_search"),
            "recall": st.column_config.NumberColumn("Recall@10", format="%.3f"),
            "qps": st.column_config.NumberColumn("QPS / instance", format="%,d"),
            "p50_latency_ms": st.column_config.NumberColumn("p50 (ms)", format="%.2f"),
            "p95_latency_ms": st.column_config.NumberColumn("p95 (ms)", format="%.2f"),
            "p99_latency_ms": st.column_config.NumberColumn("p99 (ms)", format="%.2f"),
            "cost_per_million_queries_usd": st.column_config.NumberColumn("Per 1M queries", format="$%.4f"),
            "monthly_cost": st.column_config.NumberColumn("Monthly at load", format="$%.2f"),
            "instances": st.column_config.NumberColumn("Instances"),
            "index_memory_mb": st.column_config.NumberColumn("Index memory (MB)", format="%.1f"),
            "build_time_s": st.column_config.NumberColumn("Build time (s)", format="%.2f"),
        },
        alt="Full measurements for every system at the selected scale",
    )

with st.expander("How these numbers are measured", icon=":material/rule:"):
    hnsw = config["hnsw"]
    st.markdown(
        f"""
- **Same index for everyone.** Every system builds HNSW with M={hnsw['M']} and
  ef_construction={hnsw['ef_construction']}, uses inner product on normalized
  {config['dataset']['embedding_dim']}-dim `{config['dataset']['embedding_model']}` embeddings,
  and is queried with the same {raw['n_queries'].max()} MS MARCO queries.
- **Recall is checked against exact answers**, a brute-force top-10 recomputed for each scale.
- **Cost is taken at the cheapest ef_search that reaches the recall target**, not the fastest
  setting, so every system is compared at the same accuracy.
- **Build time covers the full path** from raw vectors to a searchable index (for pgvector:
  table creation, bulk load and `CREATE INDEX`).
- **Hardware & price:** {config['cost']['instance_type']}, on-demand Linux list price in USD,
  pre-tax (${hourly_price}/hr in {deploy_region}). Throughput measured in: {measured_in}.
- **Limitations:** FAISS has no memory API, so its memory figure is an analytical estimate;
  each configuration is a single run; Indian billing (GST 18%) is not included.
"""
    )
