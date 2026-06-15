"""Phase 1.2: Quick smoke test — data loading, preprocessing, feature extraction."""
import os, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless testing
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import warnings
warnings.filterwarnings('ignore')

# --- Config ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")
_DATA_ROOT = os.getenv("CRACK_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT).expanduser() if _DATA_ROOT else PROJECT_ROOT / "data"
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
DATA_ROOT = DATA_ROOT.resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Project root: {PROJECT_ROOT}")
print(f"Data root: {DATA_ROOT}")
print(f"Device: {DEVICE}")

# --- Image loading (Chinese-path safe) ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf is None or buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

# --- Load small subset ---
MAX_SAMPLES = 200  # Very small for speed

print("\n=== 1. Data Loading ===")
pos_dir = DATA_ROOT / "Positive"
neg_dir = DATA_ROOT / "Negative"

pos_imgs, neg_imgs = [], []
for p in sorted(pos_dir.iterdir())[:MAX_SAMPLES//2]:
    if p.suffix.lower() in IMAGE_EXTS:
        img = imread_gray(p)
        if img is not None:
            pos_imgs.append(img)

for p in sorted(neg_dir.iterdir())[:MAX_SAMPLES//2]:
    if p.suffix.lower() in IMAGE_EXTS:
        img = imread_gray(p)
        if img is not None:
            neg_imgs.append(img)

all_imgs = pos_imgs + neg_imgs
labels = np.array([1]*len(pos_imgs) + [0]*len(neg_imgs), dtype=np.int64)

# Stack if uniform shapes, else use object array
shapes = {img.shape for img in all_imgs}
if len(shapes) == 1:
    images = np.stack(all_imgs)
else:
    images = np.array(all_imgs, dtype=object)

print(f"Loaded {len(images)} images (Pos={len(pos_imgs)}, Neg={len(neg_imgs)})")
print(f"Image shape: {images.shape}")

# --- Preprocessing ---
print("\n=== 2. Preprocessing ===")

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(img)

def apply_gaussian_filter(img, kernel_size=(5,5), sigma=1.0):
    return cv2.GaussianBlur(img, kernel_size, sigma)

def apply_median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

test_img = images[0] if images.dtype != object else images[0].astype(np.uint8)
print(f"Test image shape: {test_img.shape}, dtype: {test_img.dtype}")
clahe_result = apply_clahe(test_img)
gaussian_result = apply_gaussian_filter(test_img)
median_result = apply_median_filter(test_img)
print(f"CLAHE output shape: {clahe_result.shape}, dtype: {clahe_result.dtype}")
print(f"Gaussian output shape: {gaussian_result.shape}, dtype: {gaussian_result.dtype}")
print(f"Median output shape: {median_result.shape}, dtype: {median_result.dtype}")

# --- Feature Extraction ---
print("\n=== 3. Feature Extraction ===")

hog_feat = hog(test_img, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), feature_vector=True)
print(f"HOG features: {len(hog_feat)} dims")

lbp_img = local_binary_pattern(test_img, 8, 1, method="uniform")
n_bins = 8 * 7 + 3
lbp_hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
print(f"LBP features: {len(lbp_hist)} dims")

glcm = graycomatrix(test_img, distances=[1], angles=[0],
                    levels=256, symmetric=True, normed=True)
glcm_feats = [
    graycoprops(glcm, "contrast")[0,0],
    graycoprops(glcm, "correlation")[0,0],
    graycoprops(glcm, "energy")[0,0],
    graycoprops(glcm, "homogeneity")[0,0],
]
print(f"GLCM features: {len(glcm_feats)} dims (sample: contrast={glcm_feats[0]:.4f})")

edges = cv2.Canny(test_img, 50, 150)
edge_density = float(np.count_nonzero(edges)) / edges.size
print(f"Edge density: {edge_density:.4f}")

# --- Quick RF test ---
print("\n=== 4. Quick RF Classification Test ===")

# Build full feature vectors
def extract_all_features(img):
    h = hog(img, orientations=9, pixels_per_cell=(8,8),
            cells_per_block=(2,2), feature_vector=True)
    lbp = local_binary_pattern(img, 8, 1, method="uniform")
    l_hist, _ = np.histogram(lbp, bins=59, range=(0, 59), density=True)
    g = graycomatrix(img, distances=[1], angles=[0], levels=256,
                     symmetric=True, normed=True)
    g_feats = [graycoprops(g, p)[0,0] for p in ["contrast","correlation","energy","homogeneity"]]
    e = float(np.count_nonzero(cv2.Canny(img, 50, 150))) / img.size
    return np.concatenate([h, l_hist, g_feats, [e]])

print("Extracting features for all images...")
X = np.stack([extract_all_features(img if images.dtype != object else img.astype(np.uint8)) for img in images])
y = labels
print(f"Feature matrix: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Test set: {len(y_test)} samples")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# --- Verify outputs ---
print("\n=== 5. Verification ===")
checks = [
    ("Images loaded", len(images) == 200),
    ("Features extracted", X.shape[0] == 200 and X.shape[1] > 100),
    ("F1 > 0.5 (better than random)", f1 > 0.5),
    ("Accuracy > 0.5", acc > 0.5),
    ("GPU available", DEVICE.type == "cuda"),
]
all_pass = True
for name, result in checks:
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  [{status}] {name}")

if all_pass:
    print("\n*** ALL SMOKE TESTS PASSED ***")
else:
    print("\n*** SOME CHECKS FAILED ***")
    sys.exit(1)
