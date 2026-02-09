from __future__ import annotations
import argparse
import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR
from src.data_io import load_onspd,load_census
from src.features import add_econ_percentage

KEEP_COLS = ["oa21","pcds","lat","long","imd","total16plus","pct_econ_active","pct_econ_inactive","pct_retired","pct_students","pct_home_family","pct_long_term_sick","pct_unemployed","pct_self_employed",]

def main(onspd_csv:str,econ_csv:str,out_csv:str)->None:
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    onspd = load_onspd(RAW_DIR/onspd_csv)
    onspd = onspd[onspd["oa21"].astype(str).str.startswith("E")].copy() # england

    census = load_census(RAW_DIR/econ_csv)
    census = census[census["oa21"].notna()].copy()

    # ensure total col is numeric and not NaN
    total_col ="Total: All usual residents aged 16 years and over"
    census[total_col] = pd.to_numeric(census[total_col],errors="coerce")
    census = census[census[total_col].notna()].copy()

    merged = onspd.merge(census,on="oa21",how="left",validate="m:1")

    # join coverage
    joined_rate = merged[total_col].notna().mean()*100
    # add features
    feats = add_econ_percentage(merged)
    # keep model ready cols
    feats = feats[KEEP_COLS].copy()
    # deterministic sort
    feats = feats.sort_values(["oa21","pcds"]).reset_index(drop=True)
    # missing rep
    missing = feats.isna().mean().sort_values(ascending=False) * 100

    out_path = PROCESSED_DIR/out_csv
    feats.to_csv(out_path, index=False)

    print("build features:")
    print(f"ONSPD rows (England output areas):{len(onspd):,}")
    print(f"merged rows:{len(merged):,}")
    print(f"join coverage:{joined_rate:.2f}%")
    print(f"output written:{out_path}")
    print("\ntop missing columns (%):")
    print(missing.head(10).round(2).to_string())

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onspd",default="onspd.csv")
    ap.add_argument("--econ",default="econactivity.csv")
    ap.add_argument("--out",default="features_by_oa.csv")
    args = ap.parse_args()
    main(args.onspd,args.econ,args.out)
