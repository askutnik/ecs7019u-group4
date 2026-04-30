from __future__ import annotations
import pandas as pd
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
from fpdf import FPDF
import base64

# Ensure this import points to your corrected src/recommendations.py
from src.recommendations import get_ai_recommendation,get_ai_recommendation_hybrid
from src.recommendations_old import get_cluster_recommendation

st.set_page_config(
    page_title="UK Demographic Marketing Dashboard",
    layout="wide",
)

CLUSTER_COLOURS = {
    0: "#E63946",
    1: "#2A9D8F",
    2: "#E9C46A",
    3: "#4361EE",
}

@st.cache_data
def load_artifacts():
    # Keep your existing lookup/profiles/sizes logic, but ensure ALL are here:
    lookup = pd.read_csv("processed/postcode_lookup.csv.gz")
    profiles_mean = pd.read_csv("processed/cluster_profiles_mean.csv")
    profiles_delta = pd.read_csv("processed/cluster_profiles_delta_vs_global.csv")
    sizes = pd.read_csv("processed/cluster_sizes.csv")
    
    # Must include all three keys for the model comparison tab
    umaps = {
        "KMeans": pd.read_csv("processed/kmeans_umap.csv"),
        "GMM": pd.read_csv("processed/gmm_umap.csv"),
        "Hybrid": pd.read_csv("processed/hybrid_umap.csv")
    }
    return lookup, profiles_mean,profiles_delta, sizes, umaps
@st.cache_data
def load_kmeans_output():
    try:
        kmeans = pd.read_csv(
            "processed/kmeans_output.csv",
            usecols=["oa21", "confidence", "entropy", "is_ambiguous"],
        )
        return kmeans
    except FileNotFoundError:
        return None

@st.cache_data
def load_hybrid_output():
    try:
        return pd.read_csv("notebooks/processed/advanced_hybrid_output.csv")
    except FileNotFoundError:
        return None

def normalise_postcode(pc: str) -> str:
    return str(pc).strip().upper()
def top_deltas(delta_row: pd.Series, n: int = 3):
    deltas = delta_row.drop(labels=["cluster"])
    deltas = deltas.reindex(deltas.abs().sort_values(ascending=False).index)
    return deltas.head(n)

st.title("UK Demographic Marketing Dashboard")
lookup, profiles_mean,profiles_delta, sizes, umaps= load_artifacts()
hybrid_output = load_hybrid_output()
kmeans_output = load_kmeans_output()


def cluster_label(cid: int) -> str:
    return f"Cluster {cid}"




CLUSTER_LABELS = {cid: cluster_label(cid) for cid in range(4)}

with st.sidebar:
    st.header("Lookup")
    postcode = st.text_input("Postcode (e.g., E1 4PD)", value="")
    st.header("Product decsription")
    product_desc=st.text_area(
        "Describe your product or service",
        placeholder="e.g. A budgeting app for young professionals struggling with saving money",
        height=120
    )


tab1, tab2 = st.tabs([" Dashboard", "Cluster Map"])

with tab1:

    if not postcode:
        st.warning("Enter a postcode in the sidebar to begin.")
        st.stop()
    if not product_desc:
        st.warning("Please enter a product description to generate tailored marketing strategies.")
        st.stop()

    pc = normalise_postcode(postcode)
    matches = lookup[lookup["pcds"] == pc]

    st.markdown("### Campaign Context")
    st.info(
    f"**Product / Service Description**\n\n"
    f"{product_desc}"
)

    if matches.empty:
        st.error(
            f"Postcode *{pc}* not found in the lookup table. "
            "Check formatting (include a space if applicable, e.g. 'E1 4PD')."
        )
        st.stop()

    row = matches.iloc[0]
    cluster_id = int(row["cluster"])      # KMeans cluster from lookup
    oa_code = str(row["oa21"])

    # ── Resolve hybrid row ────────────────────────────────────────────────────
    hybrid_row = None
    if hybrid_output is not None:
        _hr = hybrid_output[hybrid_output["oa21"] == oa_code]
        if not _hr.empty:
            hybrid_row = _hr.iloc[0]

    # ── Probability columns ───────────────────────────────────────────────────
    PROB_COLS = ["prob_0", "prob_1", "prob_2", "prob_3"]
    has_probs = (
        hybrid_row is not None
        and all(c in hybrid_output.columns for c in PROB_COLS)
    )

    # ── KMeans cluster stats (for Strategy 1) ────────────────────────────────
    kmeans_stats = profiles_mean[profiles_mean["cluster"] == cluster_id].iloc[0].to_dict()

    # ── Hybrid blend data (for Strategy 2) ───────────────────────────────────
    if has_probs:
        raw_probs = [float(hybrid_row[c]) for c in PROB_COLS]
        # Keep only clusters with >5% weight to avoid noise
        cluster_probs = [(i, p) for i, p in enumerate(raw_probs) if p >= 0.02]
        # Renormalise so weights sum to 1
        total_w = sum(p for _, p in cluster_probs)
        cluster_probs = [(cid, p / total_w) for cid, p in cluster_probs]
        cluster_stats_map = {
            cid: profiles_mean[profiles_mean["cluster"] == cid].iloc[0].to_dict()
            for cid, _ in cluster_probs
        }
    else:
        cluster_probs = [(cluster_id, 1.0)]
        cluster_stats_map = {cluster_id: kmeans_stats}

    # ── Generate both AI strategies ───────────────────────────────────────────
    rec_kmeans = get_ai_recommendation(cluster_id, kmeans_stats,product_desc)
    rec_hybrid = get_ai_recommendation_hybrid(cluster_probs, cluster_stats_map,product_desc,CLUSTER_LABELS)

    # ── Terminal debug prints ─────────────────────────────────────────────────
    kmeans_label = get_cluster_recommendation(cluster_id).get("cluster_name", f"Cluster {cluster_id}")
    print(f"\n{'='*60}")
    print(f"POSTCODE: {pc}  |  OA: {oa_code}")
    print(f"{'='*60}")
    print(f"[KMeans] Predicted cluster: Cluster {cluster_id} — {kmeans_label}")

    print(f"\n[Hybrid] Cluster distribution:")
    if has_probs:
        for cid, prob in sorted(enumerate(raw_probs), key=lambda x: x[1], reverse=True):
            if prob < 0.01:
                continue
            cname = get_cluster_recommendation(cid).get("cluster_name", f"Cluster {cid}")
            print(f"  Cluster {cid} — {cname}: {prob * 100:.0f}%")
    else:
        print("  (no hybrid probabilities available)")
    print(f"{'='*60}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # AI Marketing Strategies — full width, two sub-tabs
    # ═════════════════════════════════════════════════════════════════════════
    st.subheader("Marketing Strategies")

    icon_map = {
        "Email": "📧", "LinkedIn": "💼", "TikTok": "🎵",
        "Instagram": "📸", "Facebook": "👥", "Google": "🔍",
        "SMS": "💬", "Mail": "📮", "Community": "🏘️",
        "University": "🏫", "Streaming": "📺", "Search": "🔍",
        "Newspaper": "📰", "Event": "🎪", "Podcast": "🎙️",
    }

    strat_tabs = st.tabs([f"📍 Strategy 1 — {kmeans_label}", "🔀 Strategy 2 — Hybrid Blend"])

    # ── Strategy 1: KMeans prediction ────────────────────────────────────────
    with strat_tabs[0]:
        st.caption(
            f"Based on the *KMeans* predicted cluster for this postcode: "
            f"*Cluster {cluster_id} — {kmeans_label}*."
        )
        st.metric("Cluster", kmeans_label)
        st.markdown(f"*Target Audience:* {rec_kmeans.get('target_profile', '—')}")

        #with st.expander("💡 Rationale", expanded=False):
        st.markdown(rec_kmeans.get("rationale", "—"))

        st.markdown("##### 📡 Recommended Channels")
        channels_k = rec_kmeans.get("recommended_channels", [])
        cols_k = st.columns(2)
        for i, ch in enumerate(channels_k):
            icon = next((v for k, v in icon_map.items() if k.lower() in ch.lower()), "📢")
            with cols_k[i % 2]:
                with st.container(border=True):
                    st.markdown(f"*{icon} {ch}*")

        st.markdown("##### ✉️ Messaging Strategy")
        for point in rec_kmeans.get("messaging_strategy", []):
            st.markdown(f"- {point}")

    # ── Strategy 2: Hybrid blend ──────────────────────────────────────────────
    with strat_tabs[1]:
        if has_probs:
            mix_str = ", ".join(
                f"*{p * 100:.0f}% Cluster {cid}*" for cid, p in cluster_probs
            )
            st.caption(f"Based on the *Hybrid* probabilistic blend: {mix_str}.")
        else:
            st.caption("Hybrid probabilities unavailable — falling back to KMeans cluster.")
        
        # Get dominant hybrid cluster
        top_cluster = max(cluster_probs, key=lambda x: x[1])[0]
        hybrid_label = get_cluster_recommendation(top_cluster).get(
    "cluster_name", f"Cluster {top_cluster}")

        st.metric("Blended Profile", hybrid_label,top_cluster)
        st.markdown(f"*Target Audience:* {rec_hybrid.get('target_profile', '—')}")

        #with st.expander(" Rationale", expanded=False):
        st.markdown(rec_hybrid.get("rationale", "—"))

        st.markdown("##### Recommended Channels")
        channels_h = rec_hybrid.get("recommended_channels", [])
        cols_h = st.columns(2)
        for i, ch in enumerate(channels_h):
            icon = next((v for k, v in icon_map.items() if k.lower() in ch.lower()), "📢")
            with cols_h[i % 2]:
                with st.container(border=True):
                    st.markdown(f"*{icon} {ch}*")

        st.markdown("#####  Messaging Strategy")
        for point in rec_hybrid.get("messaging_strategy", []):
            st.markdown(f"- {point}")

    # ── Key local insights ────────────────────────────────────────────────────
    st.divider()
    mean_row = profiles_mean[profiles_mean["cluster"] == cluster_id].iloc[0]
    delta_row = profiles_delta[profiles_delta["cluster"] == cluster_id].iloc[0]

    BUSINESS_LABELS = {
        "pct_retired": "Retiree Population",
        "pct_students": "Student Density",
        "pct_econ_active": "Working Professionals",
        "pct_long_term_sick": "Health-Stressed Households",
        "pct_home_family": "Families with Children",
        "pct_unemployed": "Job Seekers",
        "imd": "Economic Deprivation",
    }

    st.markdown("###  Key Local Insights")
    st.caption("How this specific area differs from the national average.")

    top_defining_features = top_deltas(delta_row, n=4)
    for feature, delta_val in top_defining_features.items():
        friendly = BUSINESS_LABELS.get(feature, feature.replace("pct_", "").replace("_", " ").title())
        direction = "Higher" if delta_val > 0 else "Lower"
        strength = "Significantly " if abs(delta_val) > 15 else ""
        st.markdown(f"*{friendly}* is *{strength}{direction}* than average.")
        bar_val = min(abs(delta_val) / 50, 1.0)
        st.progress(bar_val, text=f"{friendly}: {delta_val:+.1f}% variance")

# TAB 2 — Cluster Map
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
    st.divider()

    # UMAP Visualisation 
    # Shows cluster structure in 2D feature space. Well-separated clouds
    # of points confirm the clusters are genuinely distinct in the data.
    # Inside Tab 1 code:
    st.subheader(" Geographic & Demographic Landscape")
    st.caption("This map visualizes how different postcodes relate to each other. Areas closer together have similar demographic profiles.")

    # Display ONLY the Hybrid UMAP
    hybrid_umap = umaps["Hybrid"]
    fig = px.scatter(
        hybrid_umap, x="umap_x", y="umap_y", color="cluster",
        color_discrete_map=CLUSTER_COLOURS,
        title="UK Demographic Clusters (Hybrid GMM Projection)"
    )
    st.plotly_chart(fig, use_container_width=True)

