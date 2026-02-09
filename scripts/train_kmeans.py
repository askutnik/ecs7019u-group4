from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import PROCESSED_DIR, MODELS_DIR

FEATURE_COLS = ["imd","pct_econ_active","pct_econ_inactive","pct_retired","pct_students","pct_home_family","pct_long_term_sick","pct_unemployed","pct_self_employed",]

def main(in_csv:str,k_min:int,k_max:int,chosen_k:int|None)->None:
    MODELS_DIR.mkdir(parents=True,exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR/in_csv)

    # cluster once per OA to avoid duplicates
    df_oa = df.drop_duplicates(subset="oa21").reset_index(drop=True)
    X = df_oa[FEATURE_COLS].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # silhouette sampling (key speed fix)
    rng = np.random.default_rng(42)
    sample_n = min(20000, X_scaled.shape[0])  # cap sample size
    sample_idx = rng.choice(X_scaled.shape[0],size=sample_n,replace=False)
    X_samp = X_scaled[sample_idx]

    # choose k with inertia and silhouette sweep
    ks = list(range(k_min,k_max+1))
    sils = []
    inertias = []

    for k in ks:
        km = KMeans(n_clusters=k,n_init=10,random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)

        labels_samp = labels[sample_idx]
        sils.append(silhouette_score(X_samp,labels_samp))

    results = pd.DataFrame({"k": ks,"silhouette":sils,"inertia":inertias})
    results_path = PROCESSED_DIR/"k_sweep_results.csv"
    results.to_csv(results_path,index=False)

    if chosen_k is None:
        best_row = results.sort_values("silhouette",ascending=False).iloc[0]
        chosen_k = int(best_row["k"])

    final = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
    df_oa["cluster"] = final.fit_predict(X_scaled)

    # map back to all postcodes
    df = df.merge(df_oa[["oa21","cluster"]],on="oa21",how="left",validate="m:1")

    clustered_path = PROCESSED_DIR / "clustered_postcodes.csv"
    df.to_csv(clustered_path, index=False)

    dump(imputer,MODELS_DIR/"imputer.joblib")
    dump(scaler,MODELS_DIR/"scaler.joblib")
    dump(final,MODELS_DIR/"kmeans.joblib")

    print("train kmeans summary:")
    print(f"input:{PROCESSED_DIR/in_csv}")
    print(f"k sweep written:{results_path}")
    print(f"chosen k:{chosen_k}")
    print(f"clustered output:{clustered_path}")
    print(f"models saved in:{MODELS_DIR}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",dest="in_csv",default="features_by_oa.csv")
    ap.add_argument("--kmin",type=int,default=2)
    ap.add_argument("--kmax",type=int,default=15)
    ap.add_argument("--k",type=int,default=None)
    args = ap.parse_args()

    main(args.in_csv,args.kmin,args.kmax,args.k)
