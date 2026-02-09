from pathlib import Path
import pandas as pd

def load_onspd(path:str|Path)->pd.DataFrame:
    usecols = ["pcd","pcds","lat","long","oa21","lsoa21","msoa21","imd"]
    df = pd.read_csv(path, usecols=usecols)
    df["pcds"] = df["pcds"].astype(str).str.strip().str.upper()
    df["oa21"] = df["oa21"].astype(str).str.strip()
    # ensure numeric
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["imd"] = pd.to_numeric(df["imd"], errors="coerce")
    return df

def load_census(path:str|Path)->pd.DataFrame:
    df = pd.read_csv(path, skiprows=7)
    df = df.rename(columns={"2021 output area": "oa21"})
    df["oa21"] = df["oa21"].astype(str).str.strip()
    return df
