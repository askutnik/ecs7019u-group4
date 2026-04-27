from __future__ import annotations
import pandas as pd
import streamlit as st
from src.recommendations import get_cluster_recommendation
import os
import subprocess
import sys
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="UK Demographic Marketing Dashboard",
    layout="wide",
)

FEATURE_COLS = [
    "imd", "pct_econ_active", "pct_econ_inactive", "pct_retired",
    "pct_students", "pct_home_family", "pct_long_term_sick",
    "pct_unemployed", "pct_self_employed",
]

CLUSTER_COLOURS = {
    0: "#E63946",
    1: "#2A9D8F",
    2: "#E9C46A",
    3: "#4361EE",
}


@st.cache_data
def load_artifacts():
    usecols = ["pcds", "oa21", "lat", "long", "imd", "cluster"]
    dtype = {
        "pcds": "string", "oa21": "string", "cluster": "int16",
        "lat": "float32", "long": "float32", "imd": "float32",
    }
    lookup = pd.read_csv("processed/postcode_lookup.csv.gz", usecols=usecols, dtype=dtype)
    lookup["pcds"] = lookup["pcds"].astype(str).str.strip().str.upper()
    profiles_mean = pd.read_csv("processed/cluster_profiles_mean.csv")
    profiles_delta = pd.read_csv("processed/cluster_profiles_delta_vs_global.csv")
    sizes = pd.read_csv("processed/cluster_sizes.csv")
    for df in (profiles_mean, profiles_delta, sizes):
        df["cluster"] = df["cluster"].astype(int)
    return lookup, profiles_mean, profiles_delta, sizes


def normalise_postcode(pc: str) -> str:
    return str(pc).strip().upper()


def top_deltas(delta_row: pd.Series, n: int = 3):
    deltas = delta_row.drop(labels=["cluster"])
    deltas = deltas.reindex(deltas.abs().sort_values(ascending=False).index)
    return deltas.head(n)


def cluster_label(cid: int) -> str:
    try:
        r = get_cluster_recommendation(cid)
        return f"Cluster {cid} — {r.get('cluster_name', '')}"
    except Exception:
        return f"Cluster {cid}"


# ── Page header ───────────────────────────────────────────────────────────────
st.title("UK Demographic Marketing Dashboard")
st.caption("Enter a UK postcode to view its demographic cluster, summary profile and transparent marketing recommendations.")

lookup, profiles_mean, profiles_delta, sizes = load_artifacts()

CLUSTER_LABELS = {cid: cluster_label(cid) for cid in range(4)}

# ── Sidebar (outside tabs so it's always visible) ─────────────────────────────
with st.sidebar:
    st.header("Lookup")
    postcode = st.text_input("Postcode (e.g., E1 4PD)", value="")
    st.markdown("---")
    st.subheader("About")
    st.write(
        "Clusters are learned using K-means on public UK demographic indicators (Census + ONSPD). "
        "Recommendations are rule-based for interpretability."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Dashboard", "🗺️ Cluster Map"])



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not postcode:
        st.info("Enter a postcode in the sidebar to begin.")
        st.stop()

    pc = normalise_postcode(postcode)
    matches = lookup[lookup["pcds"] == pc]

    if matches.empty:
        st.error(
            f"Postcode **{pc}** not found in the lookup table. "
            "Check formatting (include a space if applicable, e.g. 'E1 4PD')."
        )
        st.stop()

    row = matches.iloc[0]
    cluster_id = int(row["cluster"])
    rec = get_cluster_recommendation(cluster_id)

    mean_row = profiles_mean[profiles_mean["cluster"] == cluster_id]
    delta_row = profiles_delta[profiles_delta["cluster"] == cluster_id]
    size_row = sizes[sizes["cluster"] == cluster_id]

    if mean_row.empty or delta_row.empty or size_row.empty:
        st.warning("Cluster profile tables are missing this cluster ID.")
        st.stop()

    mean_row = mean_row.iloc[0]
    delta_row = delta_row.iloc[0]
    oa_count = int(size_row.iloc[0]["oa_count"])

    colA, colB = st.columns([1.2, 1])

    with colA:
        st.subheader("📍 Postcode result")
        st.markdown(
            f"**{pc}** falls within Output Area `{row['oa21']}`, "
            f"assigned to **Cluster {cluster_id} — {rec.get('cluster_name', '')}**. "
            f"The area has an IMD score of **{float(row['imd']):.1f}**."
        )
        st.caption(
            "The IMD measure is included as an area-level index; all other indicators "
            "are percentages derived from Census 2021 economic activity categories."
        )
        st.markdown("---")

        st.subheader("🎯 Marketing recommendation")
        st.markdown(f"#### {rec['cluster_name']}")
        st.markdown(f"**Who to target:** {rec['target_profile']}")

        st.markdown("**Recommended channels**")
        channels = rec["recommended_channels"]
        if isinstance(channels, list):
            for ch in channels:
                st.markdown(f"- {ch}")
        else:
            st.markdown(channels)

        st.markdown("**Messaging strategy**")
        strategy = rec["messaging_strategy"]
        if isinstance(strategy, list):
            for point in strategy:
                st.markdown(f"- {point}")
        else:
            paragraphs = [p.strip() for p in str(strategy).split("\n") if p.strip()]
            for para in paragraphs:
                st.markdown(para)

        with st.expander("💡 Rationale"):
            st.markdown(rec["rationale"])

    with colB:
        st.subheader("Cluster context")
        st.markdown(f"This cluster contains **{oa_count:,}** Output Areas across the UK.")
        st.markdown("---")

        st.markdown("**Top distinguishing features (vs UK average):**")
        top = top_deltas(delta_row, n=5)
        top_df = (
            top.rename("delta_vs_global")
            .to_frame()
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        st.dataframe(top_df, use_container_width=True)

        st.subheader("Average profile (cluster means)")
        PCT_FEATURES = [
            "pct_econ_active", "pct_econ_inactive", "pct_retired",
            "pct_students", "pct_home_family", "pct_long_term_sick",
            "pct_unemployed", "pct_self_employed",
        ]
        profile_df = (
            mean_row[PCT_FEATURES]
            .rename("cluster_mean")
            .to_frame()
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        st.bar_chart(profile_df.set_index("feature"))

        imd_percentile = (mean_row["imd"] / 32844) * 100
        st.metric("Deprivation percentile", f"{imd_percentile:.1f}%")

    st.markdown("---")
    st.subheader("Clustering Model Comparison")

    try:
        kmeans = pd.read_csv("processed/k_sweep_results.csv")
        gmm = pd.read_csv("processed/gmm_k_sweep_results.csv")
        kmeans["model"] = "KMeans"
        gmm["model"] = "GMM"
        comparison = pd.concat([
            kmeans[["k", "silhouette", "model"]],
            gmm[["k", "silhouette", "model"]],
        ])
        st.dataframe(comparison, use_container_width=True)
        st.line_chart(comparison.pivot(index="k", columns="model", values="silhouette"))
    except FileNotFoundError:
        st.info("Run training scripts to generate model comparison data.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cluster Map
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Geographic cluster visualisation")
    st.caption("All four clusters plotted simultaneously. Each colour represents a distinct demographic segment.")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

    with ctrl_col1:
        map_mode = st.radio(
            "Display mode",
            options=["Scatter", "Density"],
            horizontal=True,
            help="Scatter shows individual postcodes; Density shows concentration hotspots.",
        )

    with ctrl_col2:
        visible_clusters = st.multiselect(
            "Show clusters",
            options=list(range(4)),
            default=list(range(4)),
            format_func=lambda cid: CLUSTER_LABELS[cid],
        )

    with ctrl_col3:
        sample_size = st.slider(
            "Sample size per cluster",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="Higher values are slower but more representative.",
        )

    if not visible_clusters:
        st.warning("Select at least one cluster to display.")
        st.stop()

    frames = []
    for cid in visible_clusters:
        subset = lookup[lookup["cluster"] == cid]
        sampled = subset.sample(n=min(sample_size, len(subset)), random_state=42).copy()
        sampled["cluster_label"] = CLUSTER_LABELS[cid]
        frames.append(sampled)

    map_df = pd.concat(frames, ignore_index=True)
    map_df["cluster_label"] = map_df["cluster_label"].astype(str)

    colour_map = {CLUSTER_LABELS[cid]: CLUSTER_COLOURS[cid] for cid in visible_clusters}

    if map_mode == "Scatter":
        fig = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="long",
            color="cluster_label",
            color_discrete_map=colour_map,
            hover_data={"lat": False, "long": False, "cluster_label": True, "imd": True},
            zoom=5,
            center={"lat": 53.5, "lon": -1.5},
            opacity=0.55,
            size_max=6,
        )
    else:
        fig = go.Figure()
        for cid in visible_clusters:
            subset = map_df[map_df["cluster"] == cid]
            fig.add_trace(go.Densitymapbox(
                lat=subset["lat"],
                lon=subset["long"],
                radius=8,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, CLUSTER_COLOURS[cid]]],
                showscale=False,
                name=CLUSTER_LABELS[cid],
                opacity=0.6,
            ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=5,
        mapbox_center={"lat": 53.5, "lon": -1.5},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        legend=dict(
            title="Demographic segment",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cluster sizes (Output Areas)**")
    size_cols = st.columns(4)
    for i, cid in enumerate(range(4)):
        size_val = sizes[sizes["cluster"] == cid]["oa_count"].values
        count_str = f"{int(size_val[0]):,}" if len(size_val) else "—"
        colour = CLUSTER_COLOURS[cid]
        size_cols[i].markdown(
            f"<div style='border-left:4px solid {colour}; padding-left:10px'>"
            f"<strong>{CLUSTER_LABELS[cid]}</strong><br/>{count_str} OAs"
            f"</div>",
            unsafe_allow_html=True,
        )
