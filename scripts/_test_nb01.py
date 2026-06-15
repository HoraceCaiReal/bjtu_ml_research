"""Test Notebook 01: Data Processing & Feature Engineering.

Runs all key functions with a small sample to verify correctness.
"""
import os, sys, warnings
# Fix Unicode output on Windows
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# ====== CONFIG (matching notebook 01 section 2.2) ======
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")
_DATA_ROOT = os.getenv("CRACK_DATA_ROOT")
if _DATA_ROOT:
    _data_root = Path(_DATA_ROOT).expanduser()
    DATA_ROOT = (_data_root if _data_root.is_absolute() else PROJECT_ROOT / _data_root).resolve()
else:
    DATA_ROOT = PROJECT_ROOT / "data"
POSITIVE_DIR = DATA_ROOT / "Positive"
NEGATIVE_DIR = DATA_ROOT / "Negative"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)

print(f"DATA_ROOT: {DATA_ROOT}")
print(f"DEVICE: {DEVICE}")

# ====== SECTION 3: Data Loading ======
print("\n" + "="*60)
print("SECTION 3: Data Loading")
print("="*60)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _imread_gray(path: Path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf is None or buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def load_dataset(data_root: Path, max_samples: int | None = None):
    def _load_dir(directory, label):
        imgs, lbls = [], []
        paths = sorted(directory.iterdir())
        for path in paths[:max_samples]:
            if path.suffix.lower() in IMAGE_EXTS:
                img = _imread_gray(path)
                if img is not None:
                    imgs.append(img)
                    lbls.append(label)
        return imgs, lbls
    pos_imgs, pos_lbls = _load_dir(data_root / "Positive", label=1)
    neg_imgs, neg_lbls = _load_dir(data_root / "Negative", label=0)
    all_imgs = pos_imgs + neg_imgs
    labels = np.array(pos_lbls + neg_lbls, dtype=np.int64)
    shapes = {img.shape for img in all_imgs}
    if len(shapes) == 1:
        images = np.stack(all_imgs)
    else:
        images = np.array(all_imgs, dtype=object)
    return images, labels

PER_DIR = 250  # max_samples per class directory
images, labels = load_dataset(DATA_ROOT, max_samples=PER_DIR)
n_total = len(labels)
n_pos = int(np.sum(labels == 1))
n_neg = int(np.sum(labels == 0))
print(f"Loaded: {n_total} images (Pos={n_pos}, Neg={n_neg})")
print(f"Image array shape: {images.shape}")
assert n_pos == n_neg == PER_DIR, f"Expected {PER_DIR} per class, got Pos={n_pos}, Neg={n_neg}"
print("✓ Data loading correct")

# ====== SECTION 4: Data Split Strategies ======
print("\n" + "="*60)
print("SECTION 4: Data Split Strategies")
print("="*60)

def split_dataset(images, labels, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    val_test_ratio = 1.0 - train_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=val_test_ratio, random_state=random_seed, stratify=labels)
    if val_ratio == 0.0:
        return X_train, None, X_temp, y_train, None, y_temp
    test_ratio_in_temp = (1.0 - train_ratio - val_ratio) / (1.0 - train_ratio)
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_in_temp, random_state=random_seed, stratify=y_temp)
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_in_temp, random_state=random_seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Test 70/30 holdout
X_tr, _, X_te, y_tr, _, y_te = split_dataset(images, labels, train_ratio=0.7, val_ratio=0.0)
print(f"70/30 split: train={len(y_tr)}, test={len(y_te)}")
print(f"Train class balance: Pos={int((y_tr==1).sum())}, Neg={int((y_tr==0).sum())}")
assert abs(len(y_tr) - 350) <= 2 and abs(len(y_te) - 150) <= 2, f"Unexpected split sizes: {len(y_tr)}/{len(y_te)}"
print("✓ Holdout split correct")

# Test stratified K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_sizes = []
for tr_idx, te_idx in skf.split(np.arange(len(labels)), labels):
    fold_sizes.append((len(tr_idx), len(te_idx)))
print(f"5-fold CV: fold sizes = {fold_sizes}")
assert len(fold_sizes) == 5
print("✓ K-fold CV correct")

# ====== SECTION 5: Preprocessing ======
print("\n" + "="*60)
print("SECTION 5: Preprocessing Methods")
print("="*60)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(image)

def apply_gaussian_filter(image, kernel_size=(5,5), sigma=1.0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

test_img = images[0]
print(f"Test image: shape={test_img.shape}, dtype={test_img.dtype}, range=[{test_img.min()},{test_img.max()}]")

# Test each preprocessing method
preproc_results = {}
for name, fn in [
    ("CLAHE", lambda img: apply_clahe(img)),
    ("Gaussian", lambda img: apply_gaussian_filter(img)),
    ("Median", lambda img: apply_median_filter(img)),
    ("CLAHE+Gaussian", lambda img: apply_gaussian_filter(apply_clahe(img))),
    ("CLAHE+Median", lambda img: apply_median_filter(apply_clahe(img))),
]:
    result = fn(test_img)
    preproc_results[name] = result
    assert result.shape == test_img.shape, f"{name}: shape mismatch {result.shape} vs {test_img.shape}"
    assert result.dtype == np.uint8, f"{name}: dtype mismatch {result.dtype}"
    print(f"  {name}: shape={result.shape}, range=[{result.min()},{result.max()}]")
print("✓ All preprocessing methods produce correct output")

# ====== SECTION 6: Feature Extraction ======
print("\n" + "="*60)
print("SECTION 6: Feature Extraction")
print("="*60)

def extract_hog_features(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
    return hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, feature_vector=True)

def extract_lbp_features(image, radius=1, n_points=8):
    n_bins = n_points * (n_points - 1) + 3  # uniform pattern bins
    lbp_image = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_glcm_features(image, distances=(1,3,5), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    img_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    props = []
    for d in distances:
        for a in angles:
            glcm = graycomatrix(img_u8, distances=[d], angles=[a],
                                levels=256, symmetric=True, normed=True)
            props.extend([
                graycoprops(glcm, "contrast")[0,0],
                graycoprops(glcm, "correlation")[0,0],
                graycoprops(glcm, "energy")[0,0],
                graycoprops(glcm, "homogeneity")[0,0],
            ])
    return np.array(props, dtype=np.float64)

def extract_edge_density(image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return float(np.count_nonzero(edges)) / edges.size

hog_feat = extract_hog_features(test_img)
lbp_feat = extract_lbp_features(test_img)
glcm_feat = extract_glcm_features(test_img)
edge_den = extract_edge_density(test_img)

print(f"HOG: {len(hog_feat)} dims, range=[{hog_feat.min():.4f}, {hog_feat.max():.4f}]")
print(f"LBP: {len(lbp_feat)} dims, sum={lbp_feat.sum():.4f} (should be ~1.0)")
print(f"GLCM: {len(glcm_feat)} dims (3 dist × 4 angles × 4 properties = 48)")
print(f"Edge Density: {edge_den:.4f}")

assert len(hog_feat) > 1000, "HOG features too few"
assert len(lbp_feat) == 59, f"LBP should be 59 uniform bins, got {len(lbp_feat)}"
assert len(glcm_feat) == 48, f"GLCM should be 48, got {len(glcm_feat)}"
assert 0 <= edge_den <= 1, f"Edge density out of range: {edge_den}"
print("✓ All feature dimensions correct")

# Full feature concatenation
def extract_all_features(image):
    return np.concatenate([
        extract_hog_features(image),
        extract_lbp_features(image),
        extract_glcm_features(image),
        np.array([extract_edge_density(image)]),
    ])

full_feat = extract_all_features(test_img)
expected_dim = len(hog_feat) + len(lbp_feat) + len(glcm_feat) + 1
assert len(full_feat) == expected_dim, f"Full feature dim mismatch: {len(full_feat)} vs {expected_dim}"
print(f"Full feature vector: {len(full_feat)} dims")
print("✓ Feature concatenation correct")

# ====== SECTION 7: Preprocessing & Feature Comparison ======
print("\n" + "="*60)
print("SECTION 7: Comparison Experiments")
print("="*60)

def _subsample_balanced(images, labels, max_samples, random_seed):
    rng = np.random.default_rng(random_seed)
    n_per_class = max_samples // 2
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    sp = rng.choice(pos_idx, min(n_per_class, len(pos_idx)), replace=False)
    sn = rng.choice(neg_idx, min(n_per_class, len(neg_idx)), replace=False)
    idx = np.concatenate([sp, sn])
    return images[idx], labels[idx]

# Test preprocessing comparison
def compare_preprocessing_pipelines(images, labels, max_samples=300, train_ratio=0.7, random_seed=42):
    pipelines = {
        "No preprocessing": None,
        "CLAHE": lambda img: apply_clahe(img),
        "Gaussian": lambda img: apply_gaussian_filter(img),
        "Median": lambda img: apply_median_filter(img),
        "CLAHE+Gaussian": lambda img: apply_gaussian_filter(apply_clahe(img)),
        "CLAHE+Median": lambda img: apply_median_filter(apply_clahe(img)),
    }
    images_sub, labels_sub = _subsample_balanced(images, labels, max_samples, random_seed)
    results = []
    for name, preproc_fn in pipelines.items():
        processed = images_sub if preproc_fn is None else np.array([preproc_fn(img) for img in images_sub])
        X_tr, _, X_te, y_tr, _, y_te = split_dataset(
            processed, labels_sub, train_ratio=train_ratio, val_ratio=0.0, random_seed=random_seed)
        X_tr_f = np.stack([extract_all_features(img) for img in X_tr])
        X_te_f = np.stack([extract_all_features(img) for img in X_te])
        model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=random_seed, n_jobs=-1)
        model.fit(X_tr_f, y_tr)
        y_pred = model.predict(X_te_f)
        results.append({
            "Pipeline": name,
            "Accuracy": accuracy_score(y_te, y_pred),
            "F1": f1_score(y_te, y_pred, zero_division=0),
        })
    return pd.DataFrame(results)

print("Running preprocessing comparison (6 pipelines × 300 samples)...")
df_prep = compare_preprocessing_pipelines(images, labels, max_samples=300)
print(df_prep.to_string(index=False))
best_prep = df_prep.loc[df_prep["F1"].idxmax()]
print(f"Best preprocessing: {best_prep['Pipeline']} (F1={best_prep['F1']:.4f})")
assert df_prep["F1"].max() > 0.5, "F1 too low - possible bug"
print("✓ Preprocessing comparison works")

# Test feature subset comparison
def extract_features_separate(image):
    return {
        "hog": extract_hog_features(image),
        "lbp": extract_lbp_features(image),
        "glcm": extract_glcm_features(image),
        "edge_density": np.array([extract_edge_density(image)]),
    }

def compare_feature_subsets(images, labels, max_samples=300, train_ratio=0.7, random_seed=42):
    feature_groups = {
        "Edge Only": ["edge_density"],
        "HOG Only": ["hog"],
        "LBP Only": ["lbp"],
        "GLCM Only": ["glcm"],
        "HOG+LBP+GLCM": ["hog", "lbp", "glcm"],
        "All Features": ["hog", "lbp", "glcm", "edge_density"],
    }
    images_sub, labels_sub = _subsample_balanced(images, labels, max_samples, random_seed)
    all_feats = [extract_features_separate(img) for img in images_sub]
    X_idx = np.arange(len(images_sub))[:, np.newaxis]
    idx_tr, _, idx_te, y_tr, _, y_te = split_dataset(
        X_idx, labels_sub, train_ratio=train_ratio, val_ratio=0.0, random_seed=random_seed)
    idx_tr = idx_tr.flatten().astype(int)
    idx_te = idx_te.flatten().astype(int)
    results = []
    for name, keys in feature_groups.items():
        X_tr_f = np.concatenate(
            [np.concatenate([all_feats[i][k] for k in keys]) for i in idx_tr]
        ).reshape(len(idx_tr), -1)
        X_te_f = np.concatenate(
            [np.concatenate([all_feats[i][k] for k in keys]) for i in idx_te]
        ).reshape(len(idx_te), -1)
        model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=random_seed, n_jobs=-1)
        model.fit(X_tr_f, y_tr)
        y_pred = model.predict(X_te_f)
        results.append({
            "Features": name,
            "Dim": X_tr_f.shape[1],
            "Accuracy": accuracy_score(y_te, y_pred),
            "F1": f1_score(y_te, y_pred, zero_division=0),
        })
    return pd.DataFrame(results)

print("\nRunning feature subset comparison (6 subsets × 300 samples)...")
df_feat = compare_feature_subsets(images, labels, max_samples=300)
print(df_feat.to_string(index=False))
best_feat = df_feat.loc[df_feat["F1"].idxmax()]
print(f"Best features: {best_feat['Features']} (F1={best_feat['F1']:.4f})")
assert df_feat["F1"].max() > 0.5, "F1 too low - possible bug"
print("✓ Feature comparison works")

# ====== SECTION 8: Split Strategy Comparison ======
print("\n" + "="*60)
print("SECTION 8: Split Strategy Comparison")
print("="*60)

def compare_split_strategies(images, labels, max_samples=300, random_seed=42):
    images_sub, labels_sub = _subsample_balanced(images, labels, max_samples, random_seed)
    print("  Extracting features...")
    X_all = np.stack([extract_all_features(img) for img in images_sub])
    y_all = labels_sub
    all_results = []
    for train_r in [0.5, 0.7, 0.9]:
        test_r = round(1.0 - train_r, 4)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=test_r, random_state=random_seed, stratify=y_all)
        model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=random_seed, n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        all_results.append({
            "Strategy": f"Holdout {train_r:.0%}/{test_r:.0%}",
            "Category": "Holdout",
            "F1": f1_score(y_te, y_pred, zero_division=0),
        })
    for k in [5, 10]:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
        fold_f1s = []
        for tr_idx, te_idx in kf.split(X_all, y_all):
            model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=random_seed, n_jobs=-1)
            model.fit(X_all[tr_idx], y_all[tr_idx])
            y_pred = model.predict(X_all[te_idx])
            fold_f1s.append(f1_score(y_all[te_idx], y_pred, zero_division=0))
        all_results.append({
            "Strategy": f"Stratified {k}-Fold CV",
            "Category": "K-Fold CV",
            "F1": np.mean(fold_f1s),
        })
    return pd.DataFrame(all_results)

print("Running split strategy comparison...")
df_split = compare_split_strategies(images, labels, max_samples=300)
print(df_split.to_string(index=False))
best_split = df_split.loc[df_split["F1"].idxmax()]
print(f"Best strategy: {best_split['Strategy']} (F1={best_split['F1']:.4f})")
assert df_split["F1"].max() > 0.5, "F1 too low"
print("✓ Split strategy comparison works")

# ====== SUMMARY ======
print("\n" + "="*60)
print("NOTEBOOK 01 — ALL TESTS PASSED ✓")
print("="*60)
print(f"  Data loading: {n_total} images ✓")
print(f"  Preprocessing: 5 methods ✓")
print(f"  Features: HOG(26244) + LBP(59) + GLCM(48) + Edge(1) = 26352 dims ✓")
print(f"  Best pipeline: {best_prep['Pipeline']} (F1={best_prep['F1']:.4f}) ✓")
print(f"  Best features: {best_feat['Features']} (F1={best_feat['F1']:.4f}) ✓")
print(f"  Best split: {best_split['Strategy']} (F1={best_split['F1']:.4f}) ✓")
