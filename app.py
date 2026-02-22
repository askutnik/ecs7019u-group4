from __future__ import annotations
import pandas as pd
import streamlit as st
from src.recommendations import get_cluster_recommendation

st.set_page_config(
    page_title="UK Demographic Marketing Dashboard",
    layout="wide",
)

FEATURE_COLS = ["imd","pct_econ_active","pct_econ_inactive","pct_retired","pct_students","pct_home_family","pct_long_term_sick","pct_unemployed","pct_self_employed"]
@st.cache_data
def load_artifacts():
    usecols = ["pcds","oa21","lat","long","imd","cluster"]
    dtype = {"pcds":"string","oa21":"string","cluster":"int16","lat":"float32","long":"float32","imd":"float32"}
    lookup = pd.read_csv("processed/postcode_lookup.csv.gz",usecols=usecols,dtype=dtype)
    lookup["pcds"] = lookup["pcds"].astype(str).str.strip().str.upper()
    profiles_mean = pd.read_csv("processed/cluster_profiles_mean.csv")
    profiles_delta = pd.read_csv("processed/cluster_profiles_delta_vs_global.csv")
    sizes = pd.read_csv("processed/cluster_sizes.csv")
    #ensure int cluster for all tables
    for df in (profiles_mean,profiles_delta,sizes):
        df["cluster"] = df["cluster"].astype(int)
    return lookup,profiles_mean,profiles_delta,sizes

def normalise_postcode(pc:str)->str:
    return str(pc).strip().upper()

def top_deltas(delta_row:pd.Series,n:int=3):
    # pick top abs delta vals
    deltas = delta_row.drop(labels=["cluster"])
    # sort by abs val
    deltas = deltas.reindex(deltas.abs().sort_values(ascending=False).index)
    return deltas.head(n)

#UI
st.title("UK Demographic Marketing Dashboard")
st.caption("To start, please enter a UK postcode to view its demographic cluster, summary profile and transparent marketing recommendations.")

lookup,profiles_mean,profiles_delta,sizes = load_artifacts()

with st.sidebar:
    st.header("Lookup")
    postcode = st.text_input("Postcode (e.g., E1 4PD)",value="")
    st.markdown("---")
    st.subheader("About")
    st.write(
        "Clusters are learned using K-means on public UK demographic indicators (Census + ONSPD)."
        "Recommendations are rule-based for interpretability."
    )

if not postcode:
    st.info("Enter a postcode in the sidebar to begin.")
    st.stop()

pc = normalise_postcode(postcode)
matches = lookup[lookup["pcds"] == pc]

if matches.empty:
    st.error(
        f"Postcode '{pc}' not found in the lookup table. "
        "Check formatting (include a space if applicable, e.g. 'E1 4PD')."
    )
    st.stop()

row = matches.iloc[0]
cluster_id = int(row["cluster"])

rec = get_cluster_recommendation(cluster_id)

#pull cluster profile info
mean_row = profiles_mean[profiles_mean["cluster"]==cluster_id]
delta_row = profiles_delta[profiles_delta["cluster"]==cluster_id]
size_row = sizes[sizes["cluster"]==cluster_id]

if mean_row.empty or delta_row.empty or size_row.empty:
    st.warning("Cluster profile tables are missing this cluster ID.")
    st.stop()

mean_row = mean_row.iloc[0]
delta_row = delta_row.iloc[0]
oa_count = int(size_row.iloc[0]["oa_count"])

#layout
colA, colB = st.columns([1.2,1])
with colA:
    st.subheader("Postcode result")
    st.write({"postcode":pc,"oa21":str(row["oa21"]),"cluster":cluster_id,"cluster_name":rec.get("cluster_name", ""),"lat":float(row["lat"]),"long":float(row["long"]),"imd":float(row["imd"])})
    st.subheader("Marketing recommendation")
    st.markdown(f"**Cluster:** {rec['cluster_name']}")
    st.markdown(f"**Target profile:** {rec['target_profile']}")
    st.markdown("**Recommended channels:**")
    st.write(rec["recommended_channels"])
    st.markdown("**Messaging strategy:**")
    st.write(rec["messaging_strategy"])
    st.markdown("**Rationale:**")
    st.write(rec["rationale"])

with colB:
    st.subheader("Cluster context")
    st.write(f"Output Areas in this cluster: **{oa_count:,}**")
    #show top distinguishing features (delta vs global)
    st.markdown("**Top distinguishing features (vs UK average):**")
    top = top_deltas(delta_row,n=5)
    top_df = (top.rename("delta_vs_global").to_frame().reset_index().rename(columns={"index":"feature"}))
    st.dataframe(top_df,use_container_width=True)
    st.subheader("Average profile (cluster means)")
    PCT_FEATURES = ["pct_econ_active","pct_econ_inactive","pct_retired","pct_students","pct_home_family","pct_long_term_sick","pct_unemployed","pct_self_employed"]
    profile_df = (mean_row[PCT_FEATURES].rename("cluster_mean").to_frame().reset_index().rename(columns={"index":"feature"}))
    st.bar_chart(profile_df.set_index("feature"))
    imd_percentile = (mean_row["imd"]/32844)*100
    st.metric("Deprivation percentile",f"{imd_percentile:.1f}%")
st.markdown("---")
st.caption("The IMD measure is included as an area-level index, all other indicators are percentages derived from Census 2021 economic activity categories.")