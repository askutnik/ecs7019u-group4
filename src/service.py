from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any,Dict,Optional
import pandas as pd
from src.config import PROCESSED_DIR
from src.recommendations import get_cluster_recommendation

@dataclass
class PostcodeResult:
    postcode: str
    oa21: str
    lat: float
    long: float
    imd: float
    cluster: int
    recommendation: Dict[str, Any]


def _normalise_postcode(pc:str)->str:
    return str(pc).strip().upper()

def load_clustered_postcodes(csv_path:Optional[str|Path]=None)->pd.DataFrame:
    path = Path(csv_path) if csv_path else (PROCESSED_DIR/"postcode_lookup.csv")
    df = pd.read_csv(path)
    required = {"pcds","oa21","lat","long","imd","cluster"}
    missing = required-set(df.columns)
    if missing:
        raise ValueError(f"missing required cols in {path}: {sorted(missing)}")
    # normalise pcds to ensure consistent lookup
    df["pcds"] = df["pcds"].astype(str).str.strip().str.upper()
    return df


def lookup_postcode(postcode:str,df:pd.DataFrame)->Dict[str,Any]:
    pc = _normalise_postcode(postcode)
    matches = df[df["pcds"] == pc]
    if matches.empty:
        raise ValueError(
            f"Postcode '{pc}' not found in clustered table."
            f"Check formatting (e.g., 'E1 4PD') and ensure it exists in the dataset."
        )
    row = matches.iloc[0]
    cluster_id = int(row["cluster"])
    rec = get_cluster_recommendation(cluster_id)
    result = PostcodeResult(
        postcode=pc,
        oa21=str(row["oa21"]),
        lat=float(row["lat"]),
        long=float(row["long"]),
        imd=float(row["imd"]),
        cluster=cluster_id,
        recommendation=rec,
    )

    return asdict(result)
