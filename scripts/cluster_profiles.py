from __future__ import annotations
import argparse
import pandas as pd
from src.config import PROCESSED_DIR

FEATURE_COLS = ["imd","pct_econ_active","pct_econ_inactive","pct_retired","pct_students","pct_home_family","pct_long_term_sick","pct_unemployed","pct_self_employed",]

ID_COLS = ["oa21","pcds","lat","long"]

def main(in_csv:str)->None:
    in_path = PROCESSED_DIR/in_csv
    df = pd.read_csv(in_path)

    required = {"oa21","cluster",*FEATURE_COLS}
    missing = required-set(df.columns)
    if missing:
        raise ValueError(f"missing required cols in {in_path}:{sorted(missing)}")
    # build profiles at OA level
    df_oa = df.drop_duplicates(subset="oa21").reset_index(drop=True)

    # cluster sizes (OA counts)
    sizes = (df_oa.groupby("cluster")["oa21"].nunique().rename("oa_count").reset_index().sort_values("cluster"))

    means = df_oa.groupby("cluster")[FEATURE_COLS].mean().reset_index().sort_values("cluster")
    stds = df_oa.groupby("cluster")[FEATURE_COLS].std().reset_index().sort_values("cluster")

    # global mean
    global_mean = df_oa[FEATURE_COLS].mean()
    delta = means.copy()
    for c in FEATURE_COLS:
        delta[c] = delta[c]-float(global_mean[c])

    sizes_path = PROCESSED_DIR/"cluster_sizes.csv"
    means_path = PROCESSED_DIR/"cluster_profiles_mean.csv"
    stds_path = PROCESSED_DIR/"cluster_profiles_std.csv"
    delta_path = PROCESSED_DIR/"cluster_profiles_delta_vs_global.csv"

    sizes.to_csv(sizes_path,index=False)
    means.to_csv(means_path,index=False)
    stds.to_csv(stds_path,index=False)
    delta.to_csv(delta_path,index=False)

    print("cluster profiles summary:")
    print(f"input:{in_path}")
    print(f"OA rows used for profiling:{len(df_oa):,}")
    print(f"clusters found:{sorted(df_oa['cluster'].unique().tolist())}")
    print(f"wrote:{sizes_path}")
    print(f"wrote:{means_path}")
    print(f"wrote:{stds_path}")
    print(f"wrote:{delta_path}")
    print("\ncluster sizes (OA counts):")
    print(sizes.to_string(index=False))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",dest="in_csv",default="clustered_postcodes.csv",help="input CSV in processed- oa21 + cluster + features",)
    args = ap.parse_args()
    main(args.in_csv)