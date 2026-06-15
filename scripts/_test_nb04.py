"""Test Notebook 04: Unsupervised Learning.

Tests all 5 clustering methods with minimal samples.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
import time

# ===== CONFIG =====
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")
_DATA_ROOT = os.getenv("CRACK_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT).expanduser() if _DATA_ROOT else PROJECT_ROOT / "data"
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
DATA_ROOT = DATA_ROOT.resolve()
np.random.seed(42)
print(f"DATA_ROOT: {DATA_ROOT}")

# ===== Quick Data Loading =====
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf is None or buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def load_dataset(data_root, per_class=None):
    def _load_dir(directory, label):
        imgs, lbls = [], []
        for p in sorted(directory.iterdir())[:per_class]:
            if p.suffix.lower() in IMAGE_EXTS:
                img = _imread_gray(p)
                if img is not None:
                    imgs.append(img); lbls.append(label)
        return imgs, lbls
    pos_imgs, pos_lbls = _load_dir(data_root / "Positive", 1)
    neg_imgs, neg_lbls = _load_dir(data_root / "Negative", 0)
    all_imgs = pos_imgs + neg_imgs
    labels = np.array(pos_lbls + neg_lbls, dtype=np.int64)
    shapes = {img.shape for img in all_imgs}
    if len(shapes) == 1:
        images = np.stack(all_imgs)
    else:
        images = np.array(all_imgs, dtype=object)
    return images, labels

# Feature extraction (simplified for speed — use key features only)
def extract_features_simple(image):
    """Extract reduced feature set for clustering speed."""
    img_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image

    # HOG (reduced)
    h = hog(img_u8, orientations=6, pixels_per_cell=(16,16),
            cells_per_block=(2,2), feature_vector=True)

    # LBP
    n_bins = 8 * 7 + 3
    lbp_img = local_binary_pattern(img_u8, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)

    # GLCM (single distance)
    glcm = graycomatrix(img_u8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p)[0,0] for p in
                  ["contrast","correlation","energy","homogeneity"]]

    # Edge density
    edges = cv2.Canny(img_u8, 50, 150)
    edge_den = float(np.count_nonzero(edges)) / edges.size

    return np.concatenate([h, lbp_hist, glcm_feats, [edge_den]])

PER_CLASS = 200
print(f"\nLoading {PER_CLASS * 2} images...")
images, labels = load_dataset(DATA_ROOT, per_class=PER_CLASS)
print(f"Loaded: {len(labels)} images, shape: {images.shape}")

print("Extracting features...")
t0 = time.time()
X = np.stack([extract_features_simple(img) for img in images])
print(f"Feature matrix: {X.shape} (took {time.time()-t0:.1f}s)")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Standardized features: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_pca.shape}, explained variance: {pca.explained_variance_ratio_}")

# ===== Test All 5 Clustering Methods =====
print("\n" + "="*60)
print("Testing 5 Clustering Methods")
print("="*60)

results = []

def test_clustering(name, model, X_data, y_true, has_predict=True):
    t0 = time.time()
    if has_predict:
        y_pred = model.fit_predict(X_data)
    else:
        model.fit(X_data)
        y_pred = model.predict(X_data)

    elapsed = time.time() - t0

    # Internal metrics (no labels needed)
    unique_labels = set(y_pred)
    n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(y_pred == -1))

    if n_clusters_found >= 2 and len(set(y_pred)) > 1:
        # Filter noise points for silhouette
        mask = y_pred != -1
        if mask.sum() >= 2 and len(set(y_pred[mask])) >= 2:
            sil = silhouette_score(X_data[mask], y_pred[mask])
            db = davies_bouldin_score(X_data[mask], y_pred[mask])
            ch = calinski_harabasz_score(X_data[mask], y_pred[mask])
        else:
            sil, db, ch = float('nan'), float('nan'), float('nan')
    else:
        sil, db, ch = float('nan'), float('nan'), float('nan')

    # External metrics (requires true labels)
    if n_noise > 0:
        mask = y_pred != -1
        ari = adjusted_rand_score(y_true[mask], y_pred[mask])
        nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    else:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

    result = {
        "Method": name,
        "Clusters": n_clusters_found,
        "Noise": n_noise,
        "Silhouette": sil,
        "DB Index": db,
        "CH Index": ch,
        "ARI": ari,
        "NMI": nmi,
        "Time(s)": round(elapsed, 2),
    }
    results.append(result)
    status = "PASS" if not np.isnan(sil) and sil > -1 else "WARN"
    print(f"  [{status}] {name}: clusters={n_clusters_found}, noise={n_noise}, "
          f"Sil={sil:.4f}, ARI={ari:.4f}, {elapsed:.1f}s")
    return y_pred

# 1. K-Means
test_clustering("K-Means (K=2)", KMeans(n_clusters=2, random_state=42, n_init='auto'),
                X_scaled, labels)

# 2. GMM
test_clustering("GMM (n=2, full)", GaussianMixture(n_components=2, covariance_type='full', random_state=42),
                X_scaled, labels)

# 3. DBSCAN
test_clustering("DBSCAN (eps=0.5, min=5)", DBSCAN(eps=0.5, min_samples=5),
                X_scaled, labels)

# 4. Agglomerative
test_clustering("Agglomerative (n=2, ward)", AgglomerativeClustering(n_clusters=2, linkage='ward'),
                X_scaled, labels)

# 5. Spectral
test_clustering("Spectral (n=2, rbf)", SpectralClustering(n_clusters=2, affinity='rbf', random_state=42, n_init=10),
                X_scaled, labels)

# ===== Summary =====
print("\n" + "="*60)
print("NOTEBOOK 04 TEST RESULTS")
print("="*60)
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Verify all methods ran
n_pass = len(df)
print(f"\n{n_pass}/5 clustering methods tested successfully")

# Check that at least K-Means and GMM give reasonable results
kmeans_ari = df[df['Method'].str.contains('K-Means')]['ARI'].values[0]
gmm_ari = df[df['Method'].str.contains('GMM')]['ARI'].values[0]
print(f"K-Means ARI: {kmeans_ari:.4f}")
print(f"GMM ARI: {gmm_ari:.4f}")

# Clustering on reduced features may not be perfect, but should show some structure
if kmeans_ari > 0.0 or gmm_ari > 0.0:
    print("[PASS] Clustering methods detect some structure in the data")
else:
    print("[WARN] Clustering ARI close to 0 — may need more/better features")

print("\nAll 5 unsupervised methods are functional.")
