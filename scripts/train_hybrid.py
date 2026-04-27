from __future__ import annotations
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, silhouette_score
from scipy.stats import entropy
from src.config import MODELS_DIR, PROCESSED_DIR

FEATURE_COLS = ["imd", "pct_econ_active", "pct_econ_inactive", "pct_retired", 
                "pct_students", "pct_home_family", "pct_long_term_sick", 
                "pct_unemployed", "pct_self_employed"]

def run_hybrid_sweep(X_scaled, ks=range(2, 11)):
    
    results = []
    print(f"Starting Hybrid Sweep (K={ks.start} to {ks.stop-1})...")
    
    for k in ks:
        # Standard GMM for the sweep to find the natural data structure
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(X_scaled)
        
        # 1. BIC (Calculated on the full dataset - very fast)
        bic_score = gmm.bic(X_scaled)
        
        # 2. Silhouette (Calculated on a sample for speed)
        idx = np.random.choice(len(X_scaled), 5000, replace=False)
        sil_score = silhouette_score(X_scaled[idx], labels[idx])
        
        results.append({
            "k": k, 
            "silhouette": sil_score, 
            "bic": bic_score,
            "model": "Hybrid_GMM"
        })
        print(f"   K={k} | Silhouette: {sil_score:.4f} | BIC: {bic_score:.2f}")
        
    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(PROCESSED_DIR / "gmm_k_sweep_results.csv", index=False)
    return sweep_df

def train_masters_ensemble():
    # load kmeans results and data
    df = pd.read_csv(PROCESSED_DIR / "features_by_oa.csv")
    imputer = joblib.load(MODELS_DIR / "imputer.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    kmeans = joblib.load(MODELS_DIR / "kmeans.joblib")
    
    X_scaled = scaler.transform(imputer.transform(df[FEATURE_COLS]))

    # sweep 
    run_hybrid_sweep(X_scaled)

    #train final GMM with kmeans initialization
    n_clusters = kmeans.n_clusters
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full', 
        means_init=kmeans.cluster_centers_, 
        max_iter=500,
        random_state=42
    )
    gmm_probs = gmm.fit_predict(X_scaled)
    all_probs = gmm.predict_proba(X_scaled)

    # hungarian aligned to kmeans
    km_labels = kmeans.predict(X_scaled)
    cm = confusion_matrix(km_labels, gmm_probs)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {gmm_idx: km_idx for km_idx, gmm_idx in zip(row_ind, col_ind)}
    aligned_gmm_labels = pd.Series(gmm_probs).map(mapping)

    # final outputs
    df['kmeans_cluster'] = km_labels
    df['final_cluster'] = aligned_gmm_labels
    df['confidence'] = all_probs.max(axis=1)
    df['entropy'] = entropy(all_probs.T)
    df['is_ambiguous'] = (df['confidence'] < 0.85) | (aligned_gmm_labels != km_labels)

    # Save Main Results
    df.to_csv(PROCESSED_DIR / "advanced_hybrid_output.csv", index=False)
    joblib.dump(gmm, MODELS_DIR / "advanced_gmm.joblib")
    
    # Cluster Means
    profiles_mean = df.groupby('final_cluster')[FEATURE_COLS + ['imd']].mean().reset_index()
    profiles_mean.rename(columns={'final_cluster': 'cluster'}, inplace=True)
    profiles_mean.to_csv(PROCESSED_DIR / "hybrid_cluster_profiles_mean.csv", index=False)

    # Cluster Sizes
    sizes = df['final_cluster'].value_counts().reset_index()
    sizes.columns = ['cluster', 'oa_count']
    sizes.to_csv(PROCESSED_DIR / "hybrid_cluster_sizes.csv", index=False)

    print(f" Ensemble Training Complete.")
    print(f" Results & Profile files saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    train_masters_ensemble()