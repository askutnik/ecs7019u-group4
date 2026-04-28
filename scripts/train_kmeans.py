from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import src.config
from src.config import PROCESSED_DIR, MODELS_DIR

 
#Calinski-Harabasz & Davies-Bouldin
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
 
# Feature Importance via Random Forest proxy
from sklearn.ensemble import RandomForestClassifier
 
#Noise Injection / Robustness Testing
from sklearn.metrics import adjusted_rand_score
 
# Confidence Score via distances to centroids
from sklearn.metrics.pairwise import euclidean_distances

import umap
UMAP_AVAILABLE = True

FEATURE_COLS = ["imd","pct_econ_active","pct_econ_inactive","pct_retired","pct_students","pct_home_family","pct_long_term_sick","pct_unemployed","pct_self_employed",]

def main(in_csv:str,k_min:int,k_max:int,chosen_k:int|None)->None:
    MODELS_DIR.mkdir(parents=True,exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    print ("starting kmeans training...")

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

    # Calinski-Harabasz & Davies-Bouldin lists
    ch_scores = []
    db_scores = []

    for k in ks:
        km = KMeans(n_clusters=k,n_init=10,random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)

        labels_samp = labels[sample_idx]
        sils.append(silhouette_score(X_samp,labels_samp))
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))
 
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    results = pd.DataFrame({
        "k": ks,
        "silhouette": sils,
        "inertia": inertias,
        "calinski_harabasz": ch_scores,
        "davies_bouldin": db_scores,
    })
    results_path = PROCESSED_DIR/"k_sweep_results.csv"
    results.to_csv(results_path,index=False)

    if chosen_k is None:
        best_row = results.sort_values("silhouette",ascending=False).iloc[0]
        chosen_k = int(best_row["k"])

    final = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
    df_oa["cluster"] = final.fit_predict(X_scaled)

    # ── [ADDITION] Confidence Score ───────────────────────────────────────────
    # For KMeans, confidence = 1 - (distance to assigned centroid /
    #                                sum of distances to all centroids)
    # A score near 1.0 = clearly belongs to this cluster
    # A score near 0.0 = sits on the boundary between clusters
    dists = euclidean_distances(X_scaled, final.cluster_centers_)
    assigned_dist = dists[np.arange(len(dists)), df_oa["cluster"].values]
    total_dist = dists.sum(axis=1)
    df_oa["confidence"] = 1.0 - (assigned_dist / total_dist)
 
    # Flag ambiguous postcodes: bottom 15% confidence = boundary cases
    confidence_threshold = df_oa["confidence"].quantile(0.15)
    df_oa["is_ambiguous"] = df_oa["confidence"] < confidence_threshold
 
    print(f"Confidence score range: {df_oa['confidence'].min():.3f} – {df_oa['confidence'].max():.3f}")
    print(f"Ambiguous OAs (bottom 15%): {df_oa['is_ambiguous'].sum():,}")
 
    # ── [ADDITION] Feature Importance via Random Forest proxy ────────────────
    # KMeans has no built-in feature importance. We train a Random Forest
    # to predict cluster labels, then read its feature importances.
    # This answers: "Which features actually drive the segmentation?"
    print("Computing feature importance via Random Forest proxy...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, df_oa["cluster"].values)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False).reset_index()
    importances.columns = ["feature", "importance"]
    importances["model"] = "KMeans"
    importances_path = PROCESSED_DIR / "kmeans_feature_importance.csv"
    importances.to_csv(importances_path, index=False)
    print(f"Feature importance saved: {importances_path}")
    print(importances.to_string(index=False))
 
    # ── [ADDITION] Noise Injection / Robustness Testing ──────────────────────
    # We add Gaussian noise at increasing levels and measure how much
    # cluster assignments change using Adjusted Rand Index (ARI).
    # ARI = 1.0 means identical assignments; ARI near 0 = random.
    print("Running noise injection robustness test...")
    original_labels = final.predict(X_scaled)
    noise_levels = [0.01, 0.02, 0.05, 0.10]
    noise_results = []
    for noise_std in noise_levels:
        X_noisy = X_scaled + rng.normal(0, noise_std, X_scaled.shape)
        noisy_labels = final.predict(X_noisy)
        ari = adjusted_rand_score(original_labels, noisy_labels)
        noise_results.append({"noise_level": noise_std, "ari": ari, "model": "KMeans"})
        print(f"   Noise={noise_std*100:.0f}% | ARI: {ari:.4f}")
    noise_df = pd.DataFrame(noise_results)
    noise_path = PROCESSED_DIR / "kmeans_noise_robustness.csv"
    noise_df.to_csv(noise_path, index=False)
    print(f"Noise robustness saved: {noise_path}")
 
    # ── [ADDITION] UMAP Dimensionality Reduction ──────────────────────────────
    # UMAP projects 9 features down to 2D so we can visualise cluster
    # separation. A good clustering shows clearly separated clouds of points.
    # We sample 20k OAs for speed (UMAP is slow on 180k rows).
    if UMAP_AVAILABLE:
        print("Computing UMAP projection (this may take ~2 minutes)...")
        umap_sample_n = min(20000, X_scaled.shape[0])
        umap_idx = rng.choice(X_scaled.shape[0], size=umap_sample_n, replace=False)
        X_umap_sample = X_scaled[umap_idx]
        cluster_umap_sample = df_oa["cluster"].values[umap_idx]
 
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        embedding = reducer.fit_transform(X_umap_sample)
 
        umap_df = pd.DataFrame({
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
            "cluster": cluster_umap_sample,
            "model": "KMeans",
        })
        umap_path = PROCESSED_DIR / "kmeans_umap.csv"
        umap_df.to_csv(umap_path, index=False)
        print(f"UMAP projection saved: {umap_path}")
    else:
        print("Skipping UMAP — install umap-learn to enable.")

    # map back to all postcodes
    df = df.merge(
        df_oa[["oa21", "cluster", "confidence", "is_ambiguous"]],
        on="oa21", how="left", validate="m:1"
    )
    # df = df.merge(df_oa[["oa21","cluster"]],on="oa21",how="left",validate="m:1")

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
