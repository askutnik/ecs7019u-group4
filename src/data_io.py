from pathlib import Path
import pandas as pd

def load_onspd(path:str|Path)->pd.DataFrame:
    #load onspd
    usecols = [
        "pcd", "pcds",      #postcodes
        "lat", "long",      #coordinates
        "oa21", "lsoa21", "msoa21", #areas
        "imd"               #deprivation index
    ]
    df = pd.read_csv(path,usecols=usecols)
    #normalise postcode
    df["pcds"] = df["pcds"].str.strip().str.upper()
    return df

def load_census(path:str|Path)->pd.DataFrame:
    #load census economic activity data
    df = pd.read_csv(path,skiprows=7)
    #rename for easier joining
    df = df.rename(columns={
        "2021 output area": "oa21"
    })
    #normalize strings
    df["oa21"] = df["oa21"].astype(str).str.strip()
    return df
