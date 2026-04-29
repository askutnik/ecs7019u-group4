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
from src.recommendations import get_ai_recommendation
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
        return pd.read_csv("processed/advanced_hybrid_output.csv")
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

tab1, tab2 = st.tabs([" Dashboard", "Cluster Map"])

with tab1:
    if not postcode:
        st.info("Enter a postcode in the sidebar to begin.")
        st.stop()

    pc = normalise_postcode(postcode)
    matches = lookup[lookup["pcds"] == pc]
    if matches.empty:
        st.error("Postcode not found.")
        st.stop()
    row = matches.iloc[0]
    colA, colB = st.columns([1.5, 1])
    with colA:
        st.subheader("Marketing Strategy")
        
        if hybrid_output is not None:
            oa_code = str(row['oa21'])
            hybrid_row = hybrid_output[hybrid_output['oa21'] == oa_code]
            
            if not hybrid_row.empty:
                cluster_a = int(hybrid_row.iloc[0]['final_cluster'])
                cluster_b = int(hybrid_row.iloc[0]['kmeans_cluster'])
                if cluster_a == cluster_b:
                    cluster_b = (cluster_a + 1) % 4

                stats_a = profiles_mean[profiles_mean['cluster'] == cluster_a].iloc[0].to_dict()
                stats_b = profiles_mean[profiles_mean['cluster'] == cluster_b].iloc[0].to_dict()

                rec_a = get_ai_recommendation(cluster_a, stats_a)
                rec_b = get_ai_recommendation(cluster_b, stats_b)

                # Persist for colB
                st.session_state['stats_a'] = stats_a
                st.session_state['cluster_a'] = cluster_a
                st.session_state['rec_a'] = rec_a

                # Polished Strategy View
                strat_tabs = st.tabs([" Primary Strategy", " Alternative Strategy"])
                
                with strat_tabs[0]:
                    st.metric(label="Primary Cluster", value=rec_a.get('cluster_name', f"Cluster {cluster_a}"))
                    st.markdown(f"**Target Audience:** :blue[{rec_a.get('target_profile')}]")
                    st.info(f"**Rationale:** {rec_a.get('rationale')}")
                    
                    st.markdown("###  Recommended Channels")

                    channels = rec_a.get('recommended_channels', [])

# Create a 2-column grid layout
                    cols = st.columns(2)
                    icon_map = {
    "Email": "📧", 
    "Social Media": "📱", 
    "Direct Mail": "📮", 
    "SMS": "💬", 
    "Display Ads": "🖼️",
    "University": "🏫",
    "On-campus": "📢",
    "Streaming": "📺",
    "Partnerships": "🤝"
}
                for i, ch in enumerate(channels):
                    with cols[i % 2]:
                        icon = next((icon_map[key] for key in icon_map if key in ch), "📢")
                        with st.container(border=True):
                            st.markdown(f"**{icon} {ch}**", help=f"High impact channel for {rec_a.get('cluster_name')}")

                with strat_tabs[1]:
                    st.metric(label="Alternative Cluster", value=rec_b.get('cluster_name', f"Cluster {cluster_b}"))
                    st.markdown(f"**Target Audience:** :orange[{rec_b.get('target_profile')}]")
                    st.caption(f"**Rationale:** {rec_b.get('rationale')}")
            else:
                st.warning("Hybrid model data not found for this OA.")
        else:
            st.error("Hybrid model output file missing.")
   

    def create_download_link(val, filename):
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download Marketing Strategy (PDF)</a>'

    # Inside Tab 1, under the recommendation section:
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(40, 10, f"Marketing Strategy: {pc}")
        
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Cluster: {rec['cluster_name']}")
        pdf.multi_cell(0, 10, f"Target Profile: {rec['target_profile']}")
        
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Recommended Channels:")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        for ch in rec["recommended_channels"]:
            pdf.cell(0, 10, f"- {ch}", ln=True)
            
        pdf_output = pdf.output(dest="S").encode("latin-1")
        html = create_download_link(pdf_output, f"Strategy_{pc}")
        st.markdown(html, unsafe_allow_html=True)

    # with colB:
    #     st.subheader("📊 Demographic Profile")
    #     if 'stats_a' in st.session_state:
    #         stats_a = st.session_state['stats_a']
    #         cluster_a = st.session_state['cluster_a']
    #         rec_a = st.session_state['rec_a']
            
    #         plot_data = {k: v for k, v in stats_a.items() if k != 'cluster'}
    #         df_plot = pd.DataFrame(list(plot_data.items()), columns=['Feature', 'Value'])
            
    #         # Polished Bar Chart
    #         fig = px.bar(
    #             df_plot, x='Value', y='Feature', orientation='h',
    #             template="plotly_white",
    #             title=f"Avg Demographics: {rec_a.get('cluster_name')}",
    #             color_discrete_sequence=[CLUSTER_COLOURS.get(cluster_a, "#4361EE")]
    #         )
    #         fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    #         st.plotly_chart(fig, use_container_width=True)
            
    #         with st.expander("View raw demographic stats"):
    #             st.dataframe(df_plot.style.format({"Value": "{:.2f}"}), use_container_width=True)
    #     else:
    #         st.info("Select a postcode to view cluster context.")

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
    st.divider()
    delta_row = profiles_delta[profiles_delta["cluster"] == cluster_id].iloc[0]
    top_defining_features = top_deltas(delta_row, n=4)

    # Define a mapping for business clarity
    BUSINESS_LABELS = {
        "pct_retired": "Retiree Population",
        "pct_students": "Student Density",
        "pct_econ_active": "Working Professionals",
        "pct_long_term_sick": "Health-Stressed Households",
        "pct_home_family": "Families with Children",
        "pct_unemployed": "Job Seekers",
        "imd": "Economic Deprivation"
    }

    st.write("###  Key Local Insights")
    st.caption("How this specific area differs from the national average.")

    top_defining_features = top_deltas(delta_row, n=4)

    for feature, delta_val in top_defining_features.items():
        # Get the friendly name or fall back to a titled version of the column
        friendly_name = BUSINESS_LABELS.get(feature, feature.replace("pct_", "").replace("_", " ").title())
        
        # Create a dynamic interpretation string
        direction = "Higher" if delta_val > 0 else "Lower"
        strength = "Significantly " if abs(delta_val) > 15 else ""
        
        # Display as a clean metric or descriptive text
        st.markdown(f"**{friendly_name}** is **{strength}{direction}** than average.")
        
        # Visualise with a progress bar (normalised to a 0-1 scale)
        bar_val = min(abs(delta_val) / 50, 1.0)
        st.progress(bar_val, text=f"{friendly_name}: {delta_val:+.1f}% variance")

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

