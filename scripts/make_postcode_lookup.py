from __future__ import annotations
import pandas as pd
from src.config import PROCESSED_DIR

def main()->None:
    in_path = PROCESSED_DIR/"clustered_postcodes.csv"
    out_path = PROCESSED_DIR/"postcode_lookup.csv"

    usecols = ["pcds","oa21","lat","long","imd","cluster"]
    df = pd.read_csv(in_path,usecols=usecols)

    # normalise postcode for reliable lookup
    df["pcds"] = df["pcds"].astype(str).str.strip().str.upper()
    df.to_csv(out_path,index=False)

    print("postcode lookup table:")
    print(f"input:{in_path}")
    print(f"output:{out_path}")
    print(f"rows:{len(df):,}")

if __name__=="__main__":
    main()
