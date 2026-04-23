from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from src.config import PROCESSED_DIR, MODELS_DIR

FEATURE_COLS = [
    "imd","pct_econ_active","pct_econ_inactive","pct_retired",
    "pct_students","pct_home_family","pct_long_term_sick",
    "pct_unemployed","pct_self_employed"
]

def main(in_csv: str, k_min: int, k_max: int, chosen_k: int | None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / in_csv)

    # Same OA-level clustering
    df_oa = df.drop_duplicates(subset="oa21").reset_index(drop=True)
    X = df_oa[FEATURE_COLS].copy()

    # SAME preprocessing (important!)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # SAME sampling strategy
    rng = np.random.default_rng(42)
    sample_n = min(20000, X_scaled.shape[0])
    sample_idx = rng.choice(X_scaled.shape[0], size=sample_n, replace=False)
    X_samp = X_scaled[sample_idx]

    ks = list(range(k_min, k_max + 1))
    sils = []
    bics = []

    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)

        labels = gmm.predict(X_scaled)

        # silhouette (same as KMeans)
        labels_samp = labels[sample_idx]
        sils.append(silhouette_score(X_samp, labels_samp))

        # GMM-specific metric (VERY IMPORTANT for marks)
        bics.append(gmm.bic(X_scaled))

    results = pd.DataFrame({
        "k": ks,
        "silhouette": sils,
        "bic": bics
    })

    results_path = PROCESSED_DIR / "gmm_k_sweep_results.csv"
    results.to_csv(results_path, index=False)

    # choose best k
    if chosen_k is None:
        best_row = results.sort_values("silhouette", ascending=False).iloc[0]
        chosen_k = int(best_row["k"])

    # final model
    final = GaussianMixture(n_components=chosen_k, random_state=42)
    df_oa["cluster"] = final.fit_predict(X_scaled)

    # map back
    df = df.merge(df_oa[["oa21", "cluster"]], on="oa21", how="left")

    clustered_path = PROCESSED_DIR / "clustered_postcodes_gmm.csv"
    df.to_csv(clustered_path, index=False)

    dump(final, MODELS_DIR / "gmm.joblib")

    print("train GMM summary:")
    print(f"input: {PROCESSED_DIR/in_csv}")
    print(f"k sweep written: {results_path}")
    print(f"chosen k: {chosen_k}")
    print(f"clustered output: {clustered_path}")
    print(f"model saved: {MODELS_DIR/'gmm.joblib'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="features_by_oa.csv")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=15)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    main(args.in_csv, args.kmin, args.kmax, args.k)
