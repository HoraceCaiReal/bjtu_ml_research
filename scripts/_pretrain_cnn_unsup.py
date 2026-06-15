"""Pre-train CNN and unsupervised models only.
Traditional models are unchanged from the previous run.

Usage: python scripts/_pretrain_cnn_unsup.py
"""
import os, sys, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
from dotenv import load_dotenv
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score,
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import time

# ===== CONFIG =====
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CNN_DIR = OUTPUT_DIR / "models" / "cnn"
UNSUP_DIR = OUTPUT_DIR / "models" / "unsupervised"
SCALER_DIR = OUTPUT_DIR / "scalers"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [CNN_DIR, UNSUP_DIR, SCALER_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
_D = os.getenv("CRACK_DATA_ROOT")
DATA_ROOT = Path(_D).expanduser().resolve() if _D else (PROJECT_ROOT / "data").resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

N_CNN = 1000   # Per class for CNN
N_UNSUP = 1000 # Per class for unsupervised

print(f"DATA_ROOT: {DATA_ROOT}")
print(f"DEVICE: {DEVICE}")
print(f"CNN samples: {N_CNN * 2}, Unsup samples: {N_UNSUP * 2}")

# ===== COMMON =====
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf is not None and buf.size > 0 else None

def load_dataset(data_root, per_class=None):
    def _ld(d, lbl):
        imgs, lbls = [], []
        for p in sorted(d.iterdir())[:per_class]:
            if p.suffix.lower() in IMAGE_EXTS:
                img = _imread_gray(p)
                if img is not None: imgs.append(img); lbls.append(lbl)
        return imgs, lbls
    pi, pl = _ld(data_root / "Positive", 1)
    ni, nl = _ld(data_root / "Negative", 0)
    all_i = pi + ni
    labels = np.array(pl + nl, dtype=np.int64)
    shapes = {img.shape for img in all_i}
    images = np.stack(all_i) if len(shapes) == 1 else np.array(all_i, dtype=object)
    return images, labels

def apply_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
def apply_median_filter(img):
    return cv2.medianBlur(img, 5)
def default_preprocess(img):
    return apply_median_filter(apply_clahe(img))

# ===== CNN MODELS =====
class CrackCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.5):
        super().__init__()
        self.c1 = self._b(input_channels, 32)
        self.c2 = self._b(32, 64)
        self.c3 = self._b(64, 128)
        self.c4 = self._b(128, 256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout_rate)
        self.cls = nn.Linear(256, num_classes)
    def _b(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.MaxPool2d(2))
    def forward(self, x):
        for c in [self.c1, self.c2, self.c3, self.c4]: x = c(x)
        x = self.gap(x); x = x.view(x.size(0), -1)
        return self.cls(self.drop(x))

class FocalLoss(nn.Module):
    """Focal Loss. alpha=None for balanced data (no class weighting)."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        fw = (1 - pt) ** self.gamma
        if self.alpha is not None:
            at = torch.where(targets == 1,
                torch.tensor(self.alpha, device=inputs.device),
                torch.tensor(1.0 - self.alpha, device=inputs.device))
            fw = at * fw
        return (fw * ce).mean()

class LabelSmoothingCE(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, inputs, targets):
        n = inputs.size(-1)
        lp = F.log_softmax(inputs, dim=-1)
        s = torch.zeros_like(lp).fill_(self.epsilon / (n - 1))
        s.scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
        return (-s * lp).sum(dim=-1).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        t = F.one_hot(targets, num_classes=inputs.size(1)).float()
        inter = (p * t).sum(dim=0)
        card = (p + t).sum(dim=0)
        dice = (2.*inter + self.smooth) / (card + self.smooth)
        return 1 - dice.mean()

class CrackDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float().unsqueeze(1)
        self.labels = torch.from_numpy(labels).long()
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return self.images[i], self.labels[i]

# ===================================================================
# PART 1: CNN MODELS
# ===================================================================
print("\n" + "="*70)
print("PART 1: Training CNN Models")
print("="*70)

print(f"\nLoading {N_CNN * 2} images for CNN...")
cnn_images, cnn_labels = load_dataset(DATA_ROOT, per_class=N_CNN)
print(f"Loaded: {len(cnn_labels)} images ({cnn_labels.sum()} positive)")

print("Preprocessing (CLAHE+Median, resize to 128x128, normalize)...")
cnn_processed = []
for img in cnn_images:
    img = default_preprocess(img)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    cnn_processed.append(img)
cnn_processed = np.stack(cnn_processed)

# Three-way split: train(60%)/val(15%)/test(25%)
Xc_tr, Xc_tmp, yc_tr, yc_tmp = train_test_split(
    cnn_processed, cnn_labels, test_size=0.4, random_state=42, stratify=cnn_labels)
Xc_va, Xc_te, yc_va, yc_te = train_test_split(
    Xc_tmp, yc_tmp, test_size=0.625, random_state=42, stratify=yc_tmp)
print(f"CNN split: train={len(Xc_tr)} ({yc_tr.sum()} pos), "
      f"val={len(Xc_va)} ({yc_va.sum()} pos), test={len(Xc_te)} ({yc_te.sum()} pos)")

tr_ds = CrackDataset(Xc_tr, yc_tr)
va_ds = CrackDataset(Xc_va, yc_va)
te_ds = CrackDataset(Xc_te, yc_te)
tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=64)
te_ld = DataLoader(te_ds, batch_size=64)

CNN_EPOCHS = 15
BEST_CNN_CONFIG = {"dropout_rate": 0.5, "lr": 1e-3}

CNN_CONFIGS = [
    {"name": "cross_entropy", "loss_fn": lambda: nn.CrossEntropyLoss()},
    {"name": "focal_gamma2", "loss_fn": lambda: FocalLoss(alpha=None, gamma=2.0)},
    {"name": "focal_gamma3", "loss_fn": lambda: FocalLoss(alpha=None, gamma=3.0)},
    {"name": "focal_balanced", "loss_fn": lambda: FocalLoss(alpha=0.5, gamma=2.0)},
    {"name": "label_smoothing", "loss_fn": lambda: LabelSmoothingCE(epsilon=0.1)},
    {"name": "dice", "loss_fn": lambda: DiceLoss(smooth=1.0)},
]

cnn_results = []

for config in CNN_CONFIGS:
    print(f"\n{'─'*50}")
    print(f"Training CNN: {config['name']}")
    t0 = time.time()

    model = CrackCNN(
        num_classes=2, input_channels=1,
        dropout_rate=BEST_CNN_CONFIG["dropout_rate"]).to(DEVICE)
    criterion = config['loss_fn']()
    optimizer = Adam(model.parameters(), lr=BEST_CNN_CONFIG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val_f1 = 0.0; best_state = None
    history = {"train_loss": [], "train_acc": [], "val_f1": []}

    for epoch in range(CNN_EPOCHS):
        model.train()
        ep_loss = 0.0; correct = 0
        for d, tgt in tr_ld:
            d, tgt = d.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(d), tgt)
            loss.backward(); optimizer.step()
            ep_loss += loss.item()
            correct += model(d).argmax(1).eq(tgt).sum().item()

        history["train_loss"].append(ep_loss / len(tr_ld))
        history["train_acc"].append(correct / len(tr_ds))

        # Validation on val_loader (NOT test_loader)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for d, tgt in va_ld:
                d = d.to(DEVICE)
                preds.extend(model(d).argmax(1).cpu().numpy())
                trues.extend(tgt.numpy())
        val_f1 = f1_score(trues, preds, zero_division=0)
        history["val_f1"].append(val_f1)
        scheduler.step(1.0 - val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Final evaluation on held-out test set
    model.load_state_dict(best_state)
    model.eval()
    te_preds, te_trues = [], []
    with torch.no_grad():
        for d, tgt in te_ld:
            d = d.to(DEVICE)
            te_preds.extend(model(d).argmax(1).cpu().numpy())
            te_trues.extend(tgt.numpy())
    test_f1 = f1_score(te_trues, te_preds, zero_division=0)
    test_acc = accuracy_score(te_trues, te_preds)

    elapsed = time.time() - t0

    # Save model and history
    torch.save(best_state, CNN_DIR / f"crackcnn_{config['name']}_best.pth")
    with open(CNN_DIR / f"crackcnn_{config['name']}_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    cnn_results.append({
        "loss": config['name'],
        "val_f1": round(best_val_f1, 4),
        "test_f1": round(test_f1, 4),
        "test_acc": round(test_acc, 4),
        "train_time_s": round(elapsed, 1),
        "epochs": CNN_EPOCHS,
    })

    print(f"  Val F1={best_val_f1:.4f}, Test F1={test_f1:.4f}, "
          f"Test Acc={test_acc:.4f}, Time={elapsed:.1f}s")

# Save CNN results
df_cnn = pd.DataFrame(cnn_results).sort_values("test_f1", ascending=False)
df_cnn.to_csv(RESULTS_DIR / "cnn_comparison.csv", index=False, encoding='utf-8-sig')
print(f"\nCNN comparison:")
print(df_cnn.to_string(index=False))

# ===================================================================
# PART 2: UNSUPERVISED MODELS
# ===================================================================
print("\n" + "="*70)
print("PART 2: Unsupervised Models")
print("="*70)

print(f"\nLoading {N_UNSUP * 2} images...")
uns_images, uns_labels = load_dataset(DATA_ROOT, per_class=N_UNSUP)
print(f"Loaded: {len(uns_labels)} images")

# Feature extraction (reduced dimensionality for clustering speed)
def extract_features_reduced(img):
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    h = hog(img_u8, orientations=6, pixels_per_cell=(16,16),
            cells_per_block=(2,2), feature_vector=True)
    n_bins = 8 * 7 + 3
    lbp_img = local_binary_pattern(img_u8, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
    g = graycomatrix(img_u8, distances=[1], angles=[0],
                     levels=256, symmetric=True, normed=True)
    gf = [graycoprops(g, p)[0,0] for p in ("contrast","correlation","energy","homogeneity")]
    e = float(np.count_nonzero(cv2.Canny(img_u8, 50, 150))) / img_u8.size
    return np.concatenate([h, lbp_hist, gf, [e]]).astype(np.float64)

X_unsup = np.stack([extract_features_reduced(img) for img in uns_images])
print(f"Features: {X_unsup.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)
joblib.dump(scaler, SCALER_DIR / "unsupervised_scaler.joblib")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
np.savez(UNSUP_DIR / "pca_results.npz", X_pca=X_pca, labels=uns_labels)

def eval_cluster(name, y_pred, y_true, X_data):
    uniq = set(y_pred)
    n_cl = len(uniq) - (1 if -1 in uniq else 0)
    n_noise = int(np.sum(y_pred == -1))
    mask = y_pred != -1
    sil = (silhouette_score(X_data[mask], y_pred[mask])
           if mask.sum() >= 2 and len(set(y_pred[mask])) >= 2 else None)
    db = (davies_bouldin_score(X_data[mask], y_pred[mask])
          if mask.sum() >= 2 and len(set(y_pred[mask])) >= 2 else None)
    if n_noise > 0:
        ari = adjusted_rand_score(y_true[mask], y_pred[mask])
        nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    else:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
    return {"method": name, "n_clusters": n_cl, "n_noise": n_noise,
            "silhouette": round(sil, 4) if sil is not None else None,
            "ari": round(ari, 4), "nmi": round(nmi, 4)}

unsup_results = []

# 1. K-Means
print("\nK-Means (K=2)...")
km = KMeans(n_clusters=2, random_state=42, n_init='auto')
km_l = km.fit_predict(X_scaled)
joblib.dump(km, UNSUP_DIR / "kmeans_best.joblib")
unsup_results.append(eval_cluster("K-Means", km_l, uns_labels, X_scaled))
print(f"  ARI={unsup_results[-1]['ari']:.4f}, Silhouette={unsup_results[-1]['silhouette']}")

# 2. GMM
print("GMM (full covariance)...")
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_l = gmm.fit_predict(X_scaled)
joblib.dump(gmm, UNSUP_DIR / "gmm_best.joblib")
unsup_results.append(eval_cluster("GMM", gmm_l, uns_labels, X_scaled))
print(f"  ARI={unsup_results[-1]['ari']:.4f}")

# 3. DBSCAN (grid search)
print("DBSCAN (grid search eps×min_samples)...")
best_ari = -1; best_db = None; best_db_l = None
for eps in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    for ms in [3, 5, 10]:
        db = DBSCAN(eps=eps, min_samples=ms)
        db_l = db.fit_predict(X_scaled)
        nc = len(set(db_l)) - (1 if -1 in db_l else 0)
        if nc >= 2:
            mask = db_l != -1
            if mask.sum() > 10:
                ari = adjusted_rand_score(uns_labels[mask], db_l[mask])
                if ari > best_ari:
                    best_ari, best_db, best_db_l = ari, db, db_l
if best_db is not None:
    joblib.dump(best_db, UNSUP_DIR / "dbscan_best.joblib")
    unsup_results.append(eval_cluster(
        f"DBSCAN(eps={best_db.eps},ms={best_db.min_samples})",
        best_db_l, uns_labels, X_scaled))
    print(f"  Best: eps={best_db.eps}, ms={best_db.min_samples}, "
          f"ARI={unsup_results[-1]['ari']:.4f}")
else:
    unsup_results.append({"method": "DBSCAN", "error": "No valid clustering found"})
    print("  No valid DBSCAN clustering found.")

# 4. Agglomerative
print("Agglomerative (ward)...")
agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
agg_l = agg.fit_predict(X_scaled)
joblib.dump(agg, UNSUP_DIR / "agglomerative_best.joblib")
unsup_results.append(eval_cluster("Agglomerative(ward)", agg_l, uns_labels, X_scaled))
print(f"  ARI={unsup_results[-1]['ari']:.4f}")

# 5. Spectral
print("Spectral (rbf)...")
spec = SpectralClustering(n_clusters=2, affinity='rbf', random_state=42, n_init=10)
spec_l = spec.fit_predict(X_scaled)
joblib.dump(spec, UNSUP_DIR / "spectral_best.joblib")
unsup_results.append(eval_cluster("Spectral(rbf)", spec_l, uns_labels, X_scaled))
print(f"  ARI={unsup_results[-1]['ari']:.4f}")

# Save results
df_unsup = pd.DataFrame(unsup_results)
df_unsup.to_csv(RESULTS_DIR / "unsupervised_comparison.csv", index=False, encoding='utf-8-sig')
print(f"\nUnsupervised comparison:")
print(df_unsup.to_string(index=False))

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"\nCNN models ({len(cnn_results)}):")
for r in cnn_results:
    print(f"  {r['loss']}: Val F1={r['val_f1']:.4f}, Test F1={r['test_f1']:.4f}, "
          f"Test Acc={r['test_acc']:.4f}, Time={r['train_time_s']}s")

print(f"\nUnsupervised ({len(unsup_results)}):")
for r in unsup_results:
    if 'error' not in r:
        print(f"  {r['method']}: ARI={r['ari']:.4f}, NMI={r['nmi']:.4f}")
    else:
        print(f"  {r['method']}: {r['error']}")

print(f"\nAll models in: {OUTPUT_DIR}")
print("Ready for visualization system!")
