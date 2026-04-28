from __future__ import annotations
import pandas as pd
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json

# Ensure this import points to your corrected src/recommendations.py
from src.recommendations import get_ai_recommendation

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
    return lookup, profiles_mean,  sizes, umaps

@st.cache_data
def load_hybrid_output():
    try:
        return pd.read_csv("processed/advanced_hybrid_output.csv")
    except FileNotFoundError:
        return None

def normalise_postcode(pc: str) -> str:
    return str(pc).strip().upper()

st.title("UK Demographic Marketing Dashboard")
lookup, profiles_mean, sizes, umaps = load_artifacts()
hybrid_output = load_hybrid_output()

def cluster_label(cid: int) -> str:
    return f"Cluster {cid}"

CLUSTER_LABELS = {cid: cluster_label(cid) for cid in range(4)}

with st.sidebar:
    st.header("Lookup")
    postcode = st.text_input("Postcode (e.g., E1 4PD)", value="")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🗺️ Cluster Map", "📈 Model Comparison"])

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
                strat_tabs = st.tabs(["🎯 Primary Strategy", "🔄 Alternative Strategy"])
                
                with strat_tabs[0]:
                    st.metric(label="Primary Cluster", value=rec_a.get('cluster_name', f"Cluster {cluster_a}"))
                    st.markdown(f"**Target Audience:** :blue[{rec_a.get('target_profile')}]")
                    st.info(f"**Rationale:** {rec_a.get('rationale')}")
                    
                    st.markdown("### 📢 Recommended Channels")

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

    with colB:
        st.subheader("📊 Demographic Profile")
        if 'stats_a' in st.session_state:
            stats_a = st.session_state['stats_a']
            cluster_a = st.session_state['cluster_a']
            rec_a = st.session_state['rec_a']
            
            plot_data = {k: v for k, v in stats_a.items() if k != 'cluster'}
            df_plot = pd.DataFrame(list(plot_data.items()), columns=['Feature', 'Value'])
            
            # Polished Bar Chart
            fig = px.bar(
                df_plot, x='Value', y='Feature', orientation='h',
                template="plotly_white",
                title=f"Avg Demographics: {rec_a.get('cluster_name')}",
                color_discrete_sequence=[CLUSTER_COLOURS.get(cluster_a, "#4361EE")]
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View raw demographic stats"):
                st.dataframe(df_plot.style.format({"Value": "{:.2f}"}), use_container_width=True)
        else:
            st.info("Select a postcode to view cluster context.")
# ... (Include your existing tab2 and tab3 code here)
    
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


    # st.markdown("### UMAP Cluster Visualisation (Feature Space)")
    # st.caption(
    #     "UMAP projects 9 demographic features into 2D. "
    #     "Clearly separated point clouds confirm the clusters capture real structure "
    #     "in the data — not just artefacts of the algorithm. "
    #     "Select a model to inspect its cluster geometry."
    # )

    # umap_model = st.radio(
    #     "Select model",
    #     options=["KMeans", "GMM", "Hybrid GMM"],
    #     horizontal=True,
    # )

    # umap_file_map = {
    #     "KMeans": "processed/kmeans_umap.csv",
    #     "GMM": "processed/gmm_umap.csv",
    #     "Hybrid GMM": "processed/hybrid_umap.csv",
    # }

    # try:
    #     umap_df = pd.read_csv(umap_file_map[umap_model])
    #     umap_df["cluster"] = umap_df["cluster"].astype(str)

    #     colour_seq = [CLUSTER_COLOURS[i] for i in sorted(CLUSTER_COLOURS.keys())]

    #     umap_fig = px.scatter(
    #         umap_df,
    #         x="umap_x",
    #         y="umap_y",
    #         color="cluster",
    #         color_discrete_sequence=colour_seq,
    #         opacity=0.5,
    #         title=f"UMAP Projection — {umap_model}",
    #         labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2"},
    #     )
    #     umap_fig.update_traces(marker=dict(size=3))
    #     umap_fig.update_layout(height=500, legend_title="Cluster")
    #     st.plotly_chart(umap_fig, use_container_width=True)

    #     #  Confidence overlay for Hybrid/GMM
    #     # If confidence column exists, offer a second view coloured by
    #     # confidence rather than cluster — shows boundary uncertainty visually.
    #     if "confidence" in umap_df.columns:
    #         show_conf = st.checkbox("Colour by confidence score instead of cluster")
    #         if show_conf:
    #             conf_fig = px.scatter(
    #                 umap_df,
    #                 x="umap_x",
    #                 y="umap_y",
    #                 color="confidence",
    #                 color_continuous_scale="RdYlGn",
    #                 opacity=0.5,
    #                 title=f"UMAP — Confidence Score Overlay ({umap_model})",
    #                 labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2",
    #                         "confidence": "Confidence"},
    #             )
    #             conf_fig.update_traces(marker=dict(size=3))
    #             conf_fig.update_layout(height=500)
    #             st.plotly_chart(conf_fig, use_container_width=True)
    #             st.caption(
    #                 "Green = high confidence (clearly in cluster). "
    #                 "Red = low confidence (sits on a boundary). "
    #                 "Red points in the interior of a cluster suggest sub-structure the model hasn't captured."
    #             )

    # except FileNotFoundError:
    #     st.info(f"Run training scripts to generate UMAP data. Install umap-learn if not already: `pip install umap-learn`")

# TAB 3 — Model Comparison

with tab3:
    st.subheader("Clustering Model Comparison")

    try:
        kmeans_sweep = pd.read_csv("processed/k_sweep_results.csv")
        gmm_sweep = pd.read_csv("processed/gmm_sweep_results.csv")
        hybrid_sweep = pd.read_csv("processed/gmm_k_sweep_results.csv")

        kmeans_sweep["model"] = "KMeans"
        gmm_sweep["model"] = "GMM"
        hybrid_sweep["model"] = "Hybrid GMM"

        # Silhouette Comparison
        st.markdown("### Model Performance Matrix: Silhouette Score")
        st.caption("Higher = better cluster separation. KMeans wins at K=4 (0.241).")

        sil_table = kmeans_sweep[["k", "silhouette"]].rename(columns={"silhouette": "KMeans"})
        sil_table = sil_table.merge(
            gmm_sweep[["k", "silhouette"]].rename(columns={"silhouette": "GMM"}), on="k"
        )
        sil_table = sil_table.merge(
            hybrid_sweep[["k", "silhouette"]].rename(columns={"silhouette": "Hybrid GMM"}), on="k"
        )
        st.line_chart(sil_table.set_index("k"))
        st.dataframe(sil_table, use_container_width=True)

        st.divider()

        # Calinski-Harabasz Comparison
        # Higher is better. Measures ratio of between-cluster to within-cluster
        # dispersion. Complements silhouette with a different geometric measure.
        st.markdown("### Calinski-Harabasz Score")
        st.caption("Higher = better defined clusters. Does not require a distance sample — computed on full dataset.")

        ch_table = kmeans_sweep[["k", "calinski_harabasz"]].rename(columns={"calinski_harabasz": "KMeans"})
        ch_table = ch_table.merge(
            gmm_sweep[["k", "calinski_harabasz"]].rename(columns={"calinski_harabasz": "GMM"}), on="k"
        )
        ch_table = ch_table.merge(
            hybrid_sweep[["k", "calinski_harabasz"]].rename(columns={"calinski_harabasz": "Hybrid GMM"}), on="k"
        )
        st.line_chart(ch_table.set_index("k"))
        st.dataframe(ch_table, use_container_width=True)

        st.divider()

        #  Davies-Bouldin Comparison
        # Lower is better. Measures average similarity between each cluster
        # and its most similar neighbour. Penalises overlapping clusters.
        st.markdown("### Davies-Bouldin Score")
        st.caption("Lower = better. Penalises clusters that overlap with their nearest neighbour.")

        db_table = kmeans_sweep[["k", "davies_bouldin"]].rename(columns={"davies_bouldin": "KMeans"})
        db_table = db_table.merge(
            gmm_sweep[["k", "davies_bouldin"]].rename(columns={"davies_bouldin": "GMM"}), on="k"
        )
        db_table = db_table.merge(
            hybrid_sweep[["k", "davies_bouldin"]].rename(columns={"davies_bouldin": "Hybrid GMM"}), on="k"
        )
        st.line_chart(db_table.set_index("k"))
        st.dataframe(db_table, use_container_width=True)

        st.divider()

        # BIC Comparison (GMM models only) 
        st.markdown("### BIC Score (GMM models only)")
        st.caption(
            "Lower = better probabilistic fit penalised for model complexity. "
            "KMeans excluded — it has no likelihood function. "
            "Hybrid BIC is marginally higher than pure GMM at K=4, reflecting "
            "the trade-off between cluster separability and probabilistic fit."
        )
        chosen_k = 4
        gmm_bic_at_k = gmm_sweep[gmm_sweep["k"] == chosen_k]["bic"].values[0]
        hybrid_bic_at_k = hybrid_sweep[hybrid_sweep["k"] == chosen_k]["bic"].values[0]
        diff = hybrid_bic_at_k - gmm_bic_at_k

        bic_col1, bic_col2, bic_col3 = st.columns(3)
        bic_col1.metric("GMM BIC at K=4", f"{gmm_bic_at_k:,.0f}")
        bic_col2.metric("Hybrid BIC at K=4", f"{hybrid_bic_at_k:,.0f}",
                        delta=f"+{diff:,.0f}", delta_color="inverse")
        bic_col3.metric("Penalty", f"{(diff/gmm_bic_at_k)*100:.2f}%",
                        help="Hybrid BIC overhead vs pure GMM — marginal cost for better separability.")

        bic_table = gmm_sweep[["k", "bic"]].rename(columns={"bic": "GMM BIC"})
        bic_table = bic_table.merge(
            hybrid_sweep[["k", "bic"]].rename(columns={"bic": "Hybrid BIC"}), on="k"
        )
        st.line_chart(bic_table.set_index("k"))
        st.dataframe(bic_table, use_container_width=True)

        st.divider()

        # ── Log-Likelihood Comparison ─────────────────────────────────────────
        st.markdown("### Log-Likelihood (GMM models only)")
        st.caption("Higher (less negative) = better data fit. Shows how well each model explains the observed data.")
        ll_table = gmm_sweep[["k", "log_likelihood"]].rename(columns={"log_likelihood": "GMM"})
        ll_table = ll_table.merge(
            hybrid_sweep[["k", "log_likelihood"]].rename(columns={"log_likelihood": "Hybrid GMM"}), on="k"
        )
        st.line_chart(ll_table.set_index("k"))

    except FileNotFoundError as e:
        st.info(f"Run training scripts to generate model comparison data. Missing: {e.filename}")

    st.divider()

    # ── [ADDITION] Feature Importance Comparison ──────────────────────────────
    # Loads the Random Forest proxy importance CSVs from all three models.
    # Shows which demographic features drive each model's segmentation.
    st.markdown("### Feature Importance (Random Forest Proxy)")
    st.caption(
        "A Random Forest was trained to predict each model's cluster labels. "
        "Its feature importances reveal which Census variables drive segmentation. "
        "Consistent rankings across models = robust finding."
    )
    try:
        km_imp = pd.read_csv("processed/kmeans_feature_importance.csv")
        gmm_imp = pd.read_csv("processed/gmm_feature_importance.csv")
        hybrid_imp = pd.read_csv("processed/hybrid_feature_importance.csv")

        imp_merged = km_imp[["feature", "importance"]].rename(columns={"importance": "KMeans"})
        imp_merged = imp_merged.merge(
            gmm_imp[["feature", "importance"]].rename(columns={"importance": "GMM"}), on="feature"
        )
        imp_merged = imp_merged.merge(
            hybrid_imp[["feature", "importance"]].rename(columns={"importance": "Hybrid GMM"}), on="feature"
        )
        imp_merged = imp_merged.sort_values("KMeans", ascending=False)

        st.bar_chart(imp_merged.set_index("feature"))
        st.dataframe(imp_merged, use_container_width=True)

    except FileNotFoundError:
        st.info("Run all three training scripts to generate feature importance data.")

    st.divider()

    #  Noise Robustness Comparison
    # Shows ARI at each noise level for all three models side by side.
    # A robust model maintains ARI > 0.8 even at 10% noise.
    st.markdown("### Noise Robustness (Adjusted Rand Index)")
    st.caption(
        "Gaussian noise was injected at increasing levels to simulate census measurement error. "
        "ARI = 1.0 means cluster assignments are identical to the noise-free model. "
        "ARI near 0 means assignments become random. Higher is better."
    )
    try:
        km_noise = pd.read_csv("processed/kmeans_noise_robustness.csv")
        gmm_noise = pd.read_csv("processed/gmm_noise_robustness.csv")
        hybrid_noise = pd.read_csv("processed/hybrid_noise_robustness.csv")

        noise_merged = km_noise[["noise_level", "ari"]].rename(columns={"ari": "KMeans"})
        noise_merged = noise_merged.merge(
            gmm_noise[["noise_level", "ari"]].rename(columns={"ari": "GMM"}), on="noise_level"
        )
        noise_merged = noise_merged.merge(
            hybrid_noise[["noise_level", "ari"]].rename(columns={"ari": "Hybrid GMM"}), on="noise_level"
        )
        noise_merged["noise_level"] = (noise_merged["noise_level"] * 100).astype(int).astype(str) + "%"

        st.line_chart(noise_merged.set_index("noise_level"))
        st.dataframe(noise_merged, use_container_width=True)

    except FileNotFoundError:
        st.info("Run all three training scripts to generate noise robustness data.")

    st.divider()
    

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**K-Means (Baseline)**")
        st.plotly_chart(px.scatter(umaps["KMeans"], x="umap_x", y="umap_y", color="cluster"), use_container_width=True)
        st.caption("Rigid, spherical clusters.")

    with col2:
        st.markdown("**Standalone GMM**")
        st.plotly_chart(px.scatter(umaps["GMM"], x="umap_x", y="umap_y", color="cluster"), use_container_width=True)
        st.caption("Probabilistic but often noisy.")

    with col3:
        st.markdown("**Hybrid Ensemble (Proposed)**")
        st.plotly_chart(px.scatter(umaps["Hybrid"], x="umap_x", y="umap_y", color="cluster"), use_container_width=True)
        st.caption("Refined boundaries with K-Means seeding.")
        