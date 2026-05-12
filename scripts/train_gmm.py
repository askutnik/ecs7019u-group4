from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import adjusted_rand_score

from src.config import PROCESSED_DIR, MODELS_DIR
# feature columns 
FEATURE_COLS = [
    "imd","pct_econ_active","pct_econ_inactive","pct_retired",
    "pct_students","pct_home_family","pct_long_term_sick",
    "pct_unemployed","pct_self_employed"
]

def main(in_csv: str, k_min: int, k_max: int, chosen_k: int | None) -> None:
    print ("starting GMM training...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / in_csv)

    
    df_oa = df.drop_duplicates(subset="oa21").reset_index(drop=True)
    X = df_oa[FEATURE_COLS].copy()

    # preprocessing 
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # sampling 
    rng = np.random.default_rng(42)
    sample_n = min(20000, X_scaled.shape[0])
    sample_idx = rng.choice(X_scaled.shape[0], size=sample_n, replace=False)
    X_samp = X_scaled[sample_idx]

    ks = list(range(k_min, k_max + 1))
    sils = []
    bics = []
    log_likelihoods = []

    # Calinski-Harabasz & Davies-Bouldin lists
    ch_scores = []
    db_scores = []

    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)

        # silhouette scores
        labels_samp = labels[sample_idx]
        sils.append(silhouette_score(X_samp, labels_samp))

        # GMM-specific metrics
        bics.append(gmm.bic(X_scaled))
        log_likelihoods.append(gmm.score(X_scaled) * X_scaled.shape[0])

        # Calinski-Harabasz (higher = better separation) 
        ch_scores.append(calinski_harabasz_score(X_scaled, labels))

        #  Davies-Bouldin (lower = better separation) 
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    # Added ch_score and db_score columns to sweep results
    results = pd.DataFrame({
        "k": ks,
        "silhouette": sils,
        "log_likelihood": log_likelihoods,
        "bic": bics,
        "calinski_harabasz": ch_scores,
        "davies_bouldin": db_scores,
    })

    results_path = PROCESSED_DIR / "gmm_sweep_results.csv"
    results.to_csv(results_path, index=False)

    # choose best k
    if chosen_k is None:
        best_row = results.sort_values("silhouette", ascending=False).iloc[0]
        chosen_k = int(best_row["k"])

    # final model
    final = GaussianMixture(n_components=chosen_k, random_state=42)
    final.fit(X_scaled)
    df_oa["cluster"] = final.predict(X_scaled)

    # Confidence Score
    all_probs = final.predict_proba(X_scaled)
    df_oa["confidence"] = all_probs.max(axis=1)

    # flag confidence below 85% OR bottom 15th percentile
    confidence_threshold = max(0.85, df_oa["confidence"].quantile(0.15))
    df_oa["is_ambiguous"] = df_oa["confidence"] < confidence_threshold

    print(f"Confidence score range: {df_oa['confidence'].min():.3f} – {df_oa['confidence'].max():.3f}")
    print(f"Ambiguous OAs: {df_oa['is_ambiguous'].sum():,}")

    # Feature Importance via Random Forest proxy
    print("Computing feature importance via Random Forest proxy...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, df_oa["cluster"].values)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False).reset_index()
    importances.columns = ["feature", "importance"]
    importances["model"] = "GMM"
    importances_path = PROCESSED_DIR / "gmm_feature_importance.csv"
    importances.to_csv(importances_path, index=False)
    print(f"Feature importance saved: {importances_path}")
    print(importances.to_string(index=False))

    # Noise Injection / Robustness Testing
    
    print("Running noise injection robustness test...")
    original_labels = df_oa["cluster"].values
    noise_levels = [0.01, 0.02, 0.05, 0.10]
    noise_results = []
    for noise_std in noise_levels:
        X_noisy = X_scaled + rng.normal(0, noise_std, X_scaled.shape)
        noisy_labels = final.predict(X_noisy)
        ari = adjusted_rand_score(original_labels, noisy_labels)
        noise_results.append({"noise_level": noise_std, "ari": ari, "model": "GMM"})
        print(f"   Noise={noise_std*100:.0f}% | ARI: {ari:.4f}")
    noise_df = pd.DataFrame(noise_results)
    noise_path = PROCESSED_DIR / "gmm_noise_robustness.csv"
    noise_df.to_csv(noise_path, index=False)
    print(f"Noise robustness saved: {noise_path}")

    

  
    df = df.merge(
        df_oa[["oa21", "cluster", "confidence", "is_ambiguous"]],
        on="oa21", how="left"
    )

    clustered_path = PROCESSED_DIR / "clustered_postcodes_gmm.csv"
    df.to_csv(clustered_path, index=False)

    dump(final, MODELS_DIR / "gmm.joblib")
    # print results
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
