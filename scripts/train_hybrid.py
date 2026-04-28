# from __future__ import annotations
# import pandas as pd
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans
# from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import confusion_matrix, silhouette_score
# from scipy.stats import entropy

# # ── [ADDITION] Calinski-Harabasz & Davies-Bouldin ────────────────────────────
# from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# # ── [ADDITION] Feature Importance via Random Forest proxy ────────────────────
# from sklearn.ensemble import RandomForestClassifier

# # ── [ADDITION] Noise Injection / Robustness Testing ──────────────────────────
# from sklearn.metrics import adjusted_rand_score

# # ── [ADDITION] UMAP Dimensionality Reduction ─────────────────────────────────
# try:
#     import umap
#     UMAP_AVAILABLE = True
# except ImportError:
#     UMAP_AVAILABLE = False
#     print("WARNING: umap-learn not installed. Run: pip install umap-learn")

# from src.config import MODELS_DIR, PROCESSED_DIR

# FEATURE_COLS = ["imd", "pct_econ_active", "pct_econ_inactive", "pct_retired", 
#                 "pct_students", "pct_home_family", "pct_long_term_sick", 
#                 "pct_unemployed", "pct_self_employed"]

# def run_hybrid_sweep(X_scaled, ks=range(2, 10)):
#     results = []
#     print(f"Starting Hybrid Sweep (K={ks.start} to {ks.stop-1})...")

#     # ── [ADDITION] Calinski-Harabasz & Davies-Bouldin lists ──────────────────
#     for k in ks:
#         # Step 1: fit KMeans to get good starting centres
#         km = KMeans(n_clusters=k, n_init=10, random_state=42)
#         km.fit(X_scaled)

#         # Step 2: initialise GMM WITH those centres (this is what makes it hybrid)
#         gmm = GaussianMixture(
#             n_components=k,
#             covariance_type='full',
#             means_init=km.cluster_centers_,
#             max_iter=500,
#             random_state=42
#         )
#         labels = gmm.fit_predict(X_scaled)

#         bic_score = gmm.bic(X_scaled)
#         log_likelihood = gmm.score(X_scaled) * len(X_scaled)

#         idx = np.random.default_rng(42).choice(len(X_scaled), 5000, replace=False)
#         sil_score = silhouette_score(X_scaled[idx], labels[idx])

#         # ── [ADDITION] Calinski-Harabasz (higher = better) ───────────────────
#         ch_score = calinski_harabasz_score(X_scaled, labels)

#         # ── [ADDITION] Davies-Bouldin (lower = better) ───────────────────────
#         db_score = davies_bouldin_score(X_scaled, labels)

#         results.append({
#             "k": k,
#             "silhouette": sil_score,
#             "bic": bic_score,
#             "log_likelihood": log_likelihood,
#             "calinski_harabasz": ch_score,
#             "davies_bouldin": db_score,
#             "model": "Hybrid_GMM"
#         })
#         print(f"   K={k} | Silhouette: {sil_score:.4f} | BIC: {bic_score:.2f} | CH: {ch_score:.1f} | DB: {db_score:.4f}")

#     sweep_df = pd.DataFrame(results)
#     sweep_df.to_csv(PROCESSED_DIR / "gmm_k_sweep_results.csv", index=False)
#     return sweep_df

# def train_masters_ensemble():
#     df = pd.read_csv(PROCESSED_DIR / "features_by_oa.csv")
#     imputer = joblib.load(MODELS_DIR / "imputer.joblib")
#     scaler = joblib.load(MODELS_DIR / "scaler.joblib")
#     kmeans = joblib.load(MODELS_DIR / "kmeans.joblib")

#     # Deduplicate to OA level (same as train_gmm and train_kmeans)
#     df_oa = df.drop_duplicates(subset="oa21").reset_index(drop=True)
#     X_scaled = scaler.transform(imputer.transform(df_oa[FEATURE_COLS]))

#     run_hybrid_sweep(X_scaled)

#     # Train final GMM with KMeans initialisation
#     n_clusters = kmeans.n_clusters
#     gmm = GaussianMixture(
#         n_components=n_clusters,
#         covariance_type='full',
#         means_init=kmeans.cluster_centers_,
#         max_iter=500,
#         random_state=42
#     )
#     gmm_probs = gmm.fit_predict(X_scaled)
#     all_probs = gmm.predict_proba(X_scaled)

#     # Hungarian alignment to KMeans labels
#     km_labels = kmeans.predict(X_scaled)
#     cm = confusion_matrix(km_labels, gmm_probs)
#     row_ind, col_ind = linear_sum_assignment(-cm)
#     mapping = {gmm_idx: km_idx for km_idx, gmm_idx in zip(row_ind, col_ind)}
#     aligned_gmm_labels = pd.Series(gmm_probs).map(mapping)

#     # Final outputs
#     df_oa['kmeans_cluster'] = km_labels
#     df_oa['final_cluster'] = aligned_gmm_labels

#     # ── [ADDITION] Confidence Score ───────────────────────────────────────────
#     # GMM provides predict_proba natively — max probability = confidence.
#     # We also compute entropy: high entropy = uncertain = boundary postcode.
#     # is_ambiguous fires if EITHER confidence < 85% OR models disagree.
#     df_oa['confidence'] = all_probs.max(axis=1)
#     df_oa['entropy'] = entropy(all_probs.T)
#     df_oa['is_ambiguous'] = (df_oa['confidence'] < 0.85) | (aligned_gmm_labels != km_labels)

#     print(f"Confidence score range: {df_oa['confidence'].min():.3f} – {df_oa['confidence'].max():.3f}")
#     print(f"Ambiguous OAs (low confidence OR model disagreement): {df_oa['is_ambiguous'].sum():,}")

#     # ── [ADDITION] Feature Importance via Random Forest proxy ────────────────
#     # Train RF on hybrid final_cluster labels — compares to KMeans and GMM
#     # importance rankings to see if the hybrid changes which features matter.
#     print("Computing feature importance via Random Forest proxy...")
#     rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
#     rf.fit(X_scaled, aligned_gmm_labels.values)
#     importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
#     importances = importances.sort_values(ascending=False).reset_index()
#     importances.columns = ["feature", "importance"]
#     importances["model"] = "Hybrid_GMM"
#     importances_path = PROCESSED_DIR / "hybrid_feature_importance.csv"
#     importances.to_csv(importances_path, index=False)
#     print(f"Feature importance saved: {importances_path}")
#     print(importances.to_string(index=False))

#     # ── [ADDITION] Noise Injection / Robustness Testing ──────────────────────
#     # For the Hybrid we use the final trained GMM (with KMeans init) to predict
#     # on noisy data. ARI compares original vs noisy cluster assignments.
#     # This proves the hybrid architecture is robust to census measurement error.
#     print("Running noise injection robustness test...")
#     rng = np.random.default_rng(42)
#     original_labels = aligned_gmm_labels.values
#     noise_levels = [0.01, 0.02, 0.05, 0.10]
#     noise_results = []
#     for noise_std in noise_levels:
#         X_noisy = X_scaled + rng.normal(0, noise_std, X_scaled.shape)
#         noisy_probs = gmm.predict_proba(X_noisy)
#         noisy_raw = gmm.predict(X_noisy)
#         # Apply same Hungarian mapping so labels are comparable
#         noisy_aligned = pd.Series(noisy_raw).map(mapping).values
#         ari = adjusted_rand_score(original_labels, noisy_aligned)
#         noise_results.append({"noise_level": noise_std, "ari": ari, "model": "Hybrid_GMM"})
#         print(f"   Noise={noise_std*100:.0f}% | ARI: {ari:.4f}")
#     noise_df = pd.DataFrame(noise_results)
#     noise_path = PROCESSED_DIR / "hybrid_noise_robustness.csv"
#     noise_df.to_csv(noise_path, index=False)
#     print(f"Noise robustness saved: {noise_path}")

#     # ── [ADDITION] UMAP Dimensionality Reduction ──────────────────────────────
#     # Colour by final_cluster AND confidence so we can see boundary uncertainty
#     # visually. Points coloured by confidence show where the model is unsure.
#     if UMAP_AVAILABLE:
#         print("Computing UMAP projection (this may take ~2 minutes)...")
#         umap_sample_n = min(20000, X_scaled.shape[0])
#         umap_idx = rng.choice(X_scaled.shape[0], size=umap_sample_n, replace=False)
#         X_umap_sample = X_scaled[umap_idx]
#         cluster_umap_sample = aligned_gmm_labels.values[umap_idx]
#         confidence_umap_sample = df_oa["confidence"].values[umap_idx]
#         entropy_umap_sample = df_oa["entropy"].values[umap_idx]

#         reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
#         embedding = reducer.fit_transform(X_umap_sample)

#         umap_df = pd.DataFrame({
#             "umap_x": embedding[:, 0],
#             "umap_y": embedding[:, 1],
#             "cluster": cluster_umap_sample,
#             "confidence": confidence_umap_sample,
#             "entropy": entropy_umap_sample,
#             "model": "Hybrid_GMM",
#         })
#         umap_path = PROCESSED_DIR / "hybrid_umap.csv"
#         umap_df.to_csv(umap_path, index=False)
#         print(f"UMAP projection saved: {umap_path}")
#     else:
#         print("Skipping UMAP — install umap-learn to enable.")

#     # Merge OA-level results back to all postcodes
#     df = df.merge(
#         df_oa[["oa21", "kmeans_cluster", "final_cluster", "confidence", "entropy", "is_ambiguous"]],
#         on="oa21",
#         how="left"
#     )

#     df.to_csv(PROCESSED_DIR / "advanced_hybrid_output.csv", index=False)
#     joblib.dump(gmm, MODELS_DIR / "advanced_gmm.joblib")

#     profiles_mean = df_oa.groupby('final_cluster')[FEATURE_COLS].mean().reset_index()
#     profiles_mean.rename(columns={'final_cluster': 'cluster'}, inplace=True)
#     profiles_mean.to_csv(PROCESSED_DIR / "hybrid_cluster_profiles_mean.csv", index=False)

#     sizes = df_oa['final_cluster'].value_counts().reset_index()
#     sizes.columns = ['cluster', 'oa_count']
#     sizes.to_csv(PROCESSED_DIR / "hybrid_cluster_sizes.csv", index=False)

#     print(f"Ensemble Training Complete.")
#     print(f"Results saved to {PROCESSED_DIR}")

# if __name__ == "__main__":
#     train_masters_ensemble()

from __future__ import annotations
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, silhouette_score
from scipy.stats import entropy

# ── [ADDITION] Calinski-Harabasz & Davies-Bouldin ────────────────────────────
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# ── [ADDITION] Feature Importance via Random Forest proxy ────────────────────
from sklearn.ensemble import RandomForestClassifier

# ── [ADDITION] Noise Injection / Robustness Testing ──────────────────────────
from sklearn.metrics import adjusted_rand_score

# ── [ADDITION] UMAP Dimensionality Reduction ─────────────────────────────────
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. Run: pip install umap-learn")

from src.config import MODELS_DIR, PROCESSED_DIR

FEATURE_COLS = ["imd", "pct_econ_active", "pct_econ_inactive", "pct_retired", 
                "pct_students", "pct_home_family", "pct_long_term_sick", 
                "pct_unemployed", "pct_self_employed"]

def run_hybrid_sweep(X_scaled, ks=range(2, 10)):
    results = []
    print(f"Starting Hybrid Sweep (K={ks.start} to {ks.stop-1})...")

    # ── [ADDITION] Calinski-Harabasz & Davies-Bouldin lists ──────────────────
    for k in ks:
        # Step 1: fit KMeans to get good starting centres
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)

        # Step 2: initialise GMM WITH those centres (this is what makes it hybrid)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            means_init=km.cluster_centers_,
            max_iter=500,
            random_state=42
        )
        labels = gmm.fit_predict(X_scaled)

        bic_score = gmm.bic(X_scaled)
        log_likelihood = gmm.score(X_scaled) * len(X_scaled)

        idx = np.random.default_rng(42).choice(len(X_scaled), 5000, replace=False)
        sil_score = silhouette_score(X_scaled[idx], labels[idx])

        # ── [ADDITION] Calinski-Harabasz (higher = better) ───────────────────
        ch_score = calinski_harabasz_score(X_scaled, labels)

        # ── [ADDITION] Davies-Bouldin (lower = better) ───────────────────────
        db_score = davies_bouldin_score(X_scaled, labels)

        results.append({
            "k": k,
            "silhouette": sil_score,
            "bic": bic_score,
            "log_likelihood": log_likelihood,
            "calinski_harabasz": ch_score,
            "davies_bouldin": db_score,
            "model": "Hybrid_GMM"
        })
        print(f"   K={k} | Silhouette: {sil_score:.4f} | BIC: {bic_score:.2f} | CH: {ch_score:.1f} | DB: {db_score:.4f}")

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(PROCESSED_DIR / "gmm_k_sweep_results.csv", index=False)
    return sweep_df

def train_masters_ensemble():
    df = pd.read_csv(PROCESSED_DIR / "features_by_oa.csv")
    imputer = joblib.load(MODELS_DIR / "imputer.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    kmeans = joblib.load(MODELS_DIR / "kmeans.joblib")

    # Deduplicate to OA level (same as train_gmm and train_kmeans)
    df_oa = df.drop_duplicates(subset="oa21").reset_index(drop=True)
    X_scaled = scaler.transform(imputer.transform(df_oa[FEATURE_COLS]))

    run_hybrid_sweep(X_scaled)

    # Train final GMM with KMeans initialisation
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

    # Hungarian alignment to KMeans labels
    km_labels = kmeans.predict(X_scaled)
    cm = confusion_matrix(km_labels, gmm_probs)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {gmm_idx: km_idx for km_idx, gmm_idx in zip(row_ind, col_ind)}
    aligned_gmm_labels = pd.Series(gmm_probs).map(mapping)

    # Final outputs
    df_oa['kmeans_cluster'] = km_labels
    df_oa['final_cluster'] = aligned_gmm_labels

    # ── Align per-cluster probabilities to KMeans label order ────────────────
    # all_probs columns correspond to GMM component indices (0..K-1).
    # The Hungarian mapping remaps GMM indices -> KMeans indices, so we
    # reorder the probability columns to match the KMeans cluster numbering.
    # Result: prob_0 = P(KMeans cluster 0), prob_1 = P(KMeans cluster 1), etc.
    n_components = all_probs.shape[1]
    # Build inverse mapping: kmeans_label -> gmm_component_index
    inv_mapping = {km_idx: gmm_idx for gmm_idx, km_idx in mapping.items()}
    aligned_probs = np.zeros_like(all_probs)
    for km_label in range(n_components):
        gmm_col = inv_mapping[km_label]
        aligned_probs[:, km_label] = all_probs[:, gmm_col]

    for i in range(n_components):
        df_oa[f'prob_{i}'] = aligned_probs[:, i]

    # ── [ADDITION] Confidence Score ───────────────────────────────────────────
    # GMM provides predict_proba natively — max probability = confidence.
    # We also compute entropy: high entropy = uncertain = boundary postcode.
    # is_ambiguous fires if EITHER confidence < 85% OR models disagree.
    df_oa['confidence'] = aligned_probs.max(axis=1)
    df_oa['entropy'] = entropy(aligned_probs.T)
    df_oa['is_ambiguous'] = (df_oa['confidence'] < 0.85) | (aligned_gmm_labels != km_labels)

    print(f"Confidence score range: {df_oa['confidence'].min():.3f} – {df_oa['confidence'].max():.3f}")
    print(f"Ambiguous OAs (low confidence OR model disagreement): {df_oa['is_ambiguous'].sum():,}")

    # ── [ADDITION] Feature Importance via Random Forest proxy ────────────────
    # Train RF on hybrid final_cluster labels — compares to KMeans and GMM
    # importance rankings to see if the hybrid changes which features matter.
    print("Computing feature importance via Random Forest proxy...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, aligned_gmm_labels.values)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False).reset_index()
    importances.columns = ["feature", "importance"]
    importances["model"] = "Hybrid_GMM"
    importances_path = PROCESSED_DIR / "hybrid_feature_importance.csv"
    importances.to_csv(importances_path, index=False)
    print(f"Feature importance saved: {importances_path}")
    print(importances.to_string(index=False))

    # ── [ADDITION] Noise Injection / Robustness Testing ──────────────────────
    # For the Hybrid we use the final trained GMM (with KMeans init) to predict
    # on noisy data. ARI compares original vs noisy cluster assignments.
    # This proves the hybrid architecture is robust to census measurement error.
    print("Running noise injection robustness test...")
    rng = np.random.default_rng(42)
    original_labels = aligned_gmm_labels.values
    noise_levels = [0.01, 0.02, 0.05, 0.10]
    noise_results = []
    for noise_std in noise_levels:
        X_noisy = X_scaled + rng.normal(0, noise_std, X_scaled.shape)
        noisy_probs = gmm.predict_proba(X_noisy)
        noisy_raw = gmm.predict(X_noisy)
        # Apply same Hungarian mapping so labels are comparable
        noisy_aligned = pd.Series(noisy_raw).map(mapping).values
        ari = adjusted_rand_score(original_labels, noisy_aligned)
        noise_results.append({"noise_level": noise_std, "ari": ari, "model": "Hybrid_GMM"})
        print(f"   Noise={noise_std*100:.0f}% | ARI: {ari:.4f}")
    noise_df = pd.DataFrame(noise_results)
    noise_path = PROCESSED_DIR / "hybrid_noise_robustness.csv"
    noise_df.to_csv(noise_path, index=False)
    print(f"Noise robustness saved: {noise_path}")

    # ── [ADDITION] UMAP Dimensionality Reduction ──────────────────────────────
    # Colour by final_cluster AND confidence so we can see boundary uncertainty
    # visually. Points coloured by confidence show where the model is unsure.
    if UMAP_AVAILABLE:
        print("Computing UMAP projection (this may take ~2 minutes)...")
        umap_sample_n = min(20000, X_scaled.shape[0])
        umap_idx = rng.choice(X_scaled.shape[0], size=umap_sample_n, replace=False)
        X_umap_sample = X_scaled[umap_idx]
        cluster_umap_sample = aligned_gmm_labels.values[umap_idx]
        confidence_umap_sample = df_oa["confidence"].values[umap_idx]
        entropy_umap_sample = df_oa["entropy"].values[umap_idx]

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        embedding = reducer.fit_transform(X_umap_sample)

        umap_df = pd.DataFrame({
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
            "cluster": cluster_umap_sample,
            "confidence": confidence_umap_sample,
            "entropy": entropy_umap_sample,
            "model": "Hybrid_GMM",
        })
        umap_path = PROCESSED_DIR / "hybrid_umap.csv"
        umap_df.to_csv(umap_path, index=False)
        print(f"UMAP projection saved: {umap_path}")
    else:
        print("Skipping UMAP — install umap-learn to enable.")

    # Merge OA-level results back to all postcodes
    prob_cols = [f"prob_{i}" for i in range(n_components)]
    df = df.merge(
        df_oa[["oa21", "kmeans_cluster", "final_cluster", "confidence", "entropy", "is_ambiguous"] + prob_cols],
        on="oa21",
        how="left"
    )

    df.to_csv(PROCESSED_DIR / "advanced_hybrid_output.csv", index=False)
    joblib.dump(gmm, MODELS_DIR / "advanced_gmm.joblib")

    profiles_mean = df_oa.groupby('final_cluster')[FEATURE_COLS].mean().reset_index()
    profiles_mean.rename(columns={'final_cluster': 'cluster'}, inplace=True)
    profiles_mean.to_csv(PROCESSED_DIR / "hybrid_cluster_profiles_mean.csv", index=False)

    sizes = df_oa['final_cluster'].value_counts().reset_index()
    sizes.columns = ['cluster', 'oa_count']
    sizes.to_csv(PROCESSED_DIR / "hybrid_cluster_sizes.csv", index=False)

    print(f"Ensemble Training Complete.")
    print(f"Results saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    train_masters_ensemble()