"""Fast pre-training with small samples for immediate model availability.

Saves all models to outputs/ for the visualization system.
Uses 300 samples per class for traditional, 200 for CNN, 200 for unsupervised.
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
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
TRAD_DIR = OUTPUT_DIR / "models" / "traditional"
CNN_DIR = OUTPUT_DIR / "models" / "cnn"
UNSUP_DIR = OUTPUT_DIR / "models" / "unsupervised"
SCALER_DIR = OUTPUT_DIR / "scalers"
RESULTS_DIR = OUTPUT_DIR / "results"
for d in [TRAD_DIR, CNN_DIR, UNSUP_DIR, SCALER_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
_D = os.getenv("CRACK_DATA_ROOT")
DATA_ROOT = Path(_D).expanduser().resolve() if _D else (PROJECT_ROOT / "data").resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42); torch.manual_seed(42)

N_TRAD = 300   # Per class for traditional
N_CNN = 200    # Per class for CNN
N_UNSUP = 200  # Per class for unsupervised

print(f"Fast pre-training: Trad={N_TRAD*2}, CNN={N_CNN*2}, Unsup={N_UNSUP*2}")
print(f"Device: {DEVICE}")

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

def extract_hog(img):
    return hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
def extract_lbp(img):
    n_bins = 8 * 7 + 3
    lbp_img = local_binary_pattern(img, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
    return hist
def extract_glcm(img):
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    props = []
    for d in (1,3,5):
        for a in (0, np.pi/4, np.pi/2, 3*np.pi/4):
            g = graycomatrix(img_u8, distances=[d], angles=[a], levels=256, symmetric=True, normed=True)
            props.extend([graycoprops(g, p)[0,0] for p in ("contrast","correlation","energy","homogeneity")])
    return np.array(props, dtype=np.float64)
def extract_edge(img):
    return float(np.count_nonzero(cv2.Canny(img, 50, 150))) / img.size
def extract_all_features(img):
    img = img.astype(np.uint8) if img.dtype != np.uint8 else img
    return np.concatenate([extract_hog(img), extract_lbp(img), extract_glcm(img),
                           np.array([extract_edge(img)], dtype=np.float64)]).astype(np.float64)

# ===== PART 1: TRADITIONAL =====
print("\n=== Part 1: Traditional Models ===")
images, labels = load_dataset(DATA_ROOT, per_class=N_TRAD)
print(f"Loaded {len(labels)} images")
processed = np.array([default_preprocess(img) for img in images])
X = np.stack([extract_all_features(img) for img in processed])
y = labels
print(f"Features: {X.shape}")

# Save scaler
scaler = StandardScaler().fit(X)
joblib.dump(scaler, SCALER_DIR / "standard_scaler.joblib")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train: {len(y_tr)}, Test: {len(y_te)}")

trad_models = {
    "decision_tree": (DecisionTreeClassifier(random_state=42),
        {'max_depth': [5, 10, 15, None], 'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 5]}),
    "svm": (Pipeline([('s', StandardScaler()), ('m', SVC(probability=True, random_state=42))]),
        {'m__kernel': ['linear', 'rbf'], 'm__C': [0.1, 1, 10]}),
    "naive_bayes": (Pipeline([('s', StandardScaler()), ('m', GaussianNB())]),
        {'m__var_smoothing': [1e-9, 1e-7, 1e-5]}),
    "random_forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
        {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}),
    "logistic_regression": (Pipeline([('s', StandardScaler()),
        ('m', LogisticRegression(random_state=42, max_iter=2000))]),
        {'m__C': [0.1, 1, 10], 'm__penalty': ['l1', 'l2'], 'm__solver': ['liblinear']}),
}

try:
    from xgboost import XGBClassifier
    trad_models["xgboost"] = (XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
        {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.3], 'subsample': [0.8, 1.0]})
except ImportError: pass

try:
    from lightgbm import LGBMClassifier
    trad_models["lightgbm"] = (LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        {'n_estimators': [50, 100], 'max_depth': [3, 6], 'num_leaves': [31, 63], 'learning_rate': [0.1, 0.3]})
except ImportError: pass

trad_results = []
for key, (base_model, param_grid) in trad_models.items():
    print(f"\n  {key}...", end=" ", flush=True)
    t0 = time.time()
    grid = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    y_pred = best.predict(X_te)
    try:
        auc = roc_auc_score(y_te, best.predict_proba(X_te)[:, 1])
    except Exception:
        auc = float('nan')
    f1 = f1_score(y_te, y_pred, zero_division=0)

    joblib.dump(best, TRAD_DIR / f"{key}_best.joblib")
    with open(TRAD_DIR / f"{key}_best_params.json", 'w') as f:
        json.dump(grid.best_params_, f, indent=2, default=str)

    elapsed = time.time() - t0
    trad_results.append({"model": key, "cv_f1": round(grid.best_score_,4),
                         "test_f1": round(f1,4), "test_auc": round(auc,4), "time_s": round(elapsed,1)})
    print(f"F1={f1:.4f}, AUC={auc:.4f}, {elapsed:.1f}s ✓")

df_trad = pd.DataFrame(trad_results).sort_values("test_f1", ascending=False)
df_trad.to_csv(RESULTS_DIR / "traditional_comparison.csv", index=False, encoding='utf-8-sig')
print(f"\nTraditional models saved. Best: {df_trad.iloc[0]['model']} (F1={df_trad.iloc[0]['test_f1']:.4f})")

# ===== PART 2: CNN =====
print("\n=== Part 2: CNN Models ===")

class CrackCNN(nn.Module):
    def __init__(self, nc=2, ic=1, dr=0.5):
        super().__init__()
        self.c1 = self._b(ic, 32); self.c2 = self._b(32, 64)
        self.c3 = self._b(64, 128); self.c4 = self._b(128, 256)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(dr)
        self.cls = nn.Linear(256, nc)
    def _b(self, i, o):
        return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.MaxPool2d(2))
    def forward(self, x):
        for c in [self.c1, self.c2, self.c3, self.c4]: x = c(x)
        x = self.gap(x); x = x.view(x.size(0), -1)
        return self.cls(self.drop(x))

class FocalLoss(nn.Module):
    """Focal Loss for balanced binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : float or None
        Positive-class weight. None = no class balancing (best for balanced data).
        0.25 = common default for highly imbalanced data.
        0.50 = neutral weighting.
    gamma : float
        Focusing parameter. 0 = standard CE; larger = more focus on hard examples.
    """
    def __init__(self, a=None, g=2.0): super().__init__(); self.a=a; self.g=g
    def forward(self, i, t):
        ce = F.cross_entropy(i, t, reduction='none')
        pt = torch.exp(-ce)
        fw = (1 - pt) ** self.g
        if self.a is not None:
            at = torch.where(t == 1,
                torch.tensor(self.a, device=i.device),
                torch.tensor(1.0 - self.a, device=i.device))
            fw = at * fw
        return (fw * ce).mean()

class LabelSmoothingCE(nn.Module):
    def __init__(self, e=0.1): super().__init__(); self.e=e
    def forward(self, i, t):
        n = i.size(-1); lp = F.log_softmax(i, dim=-1)
        s = torch.zeros_like(lp).fill_(self.e/(n-1))
        s.scatter_(1, t.unsqueeze(1), 1-self.e)
        return (-s*lp).sum(dim=-1).mean()

class DiceLoss(nn.Module):
    def __init__(self, sm=1.0): super().__init__(); self.sm=sm
    def forward(self, i, t):
        p = F.softmax(i, dim=1)[:,1]; tf = t.float()
        inter = (p*tf).sum()
        return 1-(2*inter+self.sm)/(p.sum()+tf.sum()+self.sm)

class CrackDS(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = torch.from_numpy(imgs).float().unsqueeze(1)
        self.lbls = torch.from_numpy(lbls).long()
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.imgs[i], self.lbls[i]

# Load CNN data
cnn_imgs, cnn_lbls = load_dataset(DATA_ROOT, per_class=N_CNN)
cnn_proc = []
for img in cnn_imgs:
    img = default_preprocess(img)
    img = cv2.resize(img, (128, 128))
    cnn_proc.append(img.astype(np.float32) / 255.0)
cnn_proc = np.stack(cnn_proc)
print(f"CNN data: {cnn_proc.shape}")

# 三路划分：train (60%) / val (15%) / test (25%)
Xc_tr, Xc_tmp, yc_tr, yc_tmp = train_test_split(
    cnn_proc, cnn_lbls, test_size=0.4, random_state=42, stratify=cnn_lbls)
Xc_va, Xc_te, yc_va, yc_te = train_test_split(
    Xc_tmp, yc_tmp, test_size=0.625, random_state=42, stratify=yc_tmp)
tr_ds = CrackDS(Xc_tr, yc_tr)
va_ds = CrackDS(Xc_va, yc_va)
te_ds = CrackDS(Xc_te, yc_te)
tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=32)
te_ld = DataLoader(te_ds, batch_size=32)
print(f"CNN Train: {len(Xc_tr)}, Val: {len(Xc_va)}, Test: {len(Xc_te)}")

CNN_EPOCHS = 5  # Fast training

loss_configs = [
    ("cross_entropy", nn.CrossEntropyLoss()),
    ("focal_gamma2", FocalLoss(None, 2.0)),
    ("focal_gamma3", FocalLoss(None, 3.0)),
    ("focal_balanced", FocalLoss(0.5, 2.0)),
    ("label_smoothing", LabelSmoothingCE(0.1)),
    ("dice", DiceLoss(1.0)),
]

cnn_results = []
for name, criterion in loss_configs:
    print(f"\n  CNN {name}...", end=" ", flush=True)
    t0 = time.time()
    model = CrackCNN(nc=2, ic=1, dr=0.5).to(DEVICE)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None; best_f1 = 0.0

    for ep in range(CNN_EPOCHS):
        model.train()
        for d, tgt in tr_ld:
            d, tgt = d.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(d), tgt)
            loss.backward(); opt.step()
        # Validation on independent val set (NOT test set)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for d, tgt in va_ld:
                d = d.to(DEVICE)
                preds.extend(model(d).argmax(1).cpu().numpy())
                trues.extend(tgt.numpy())
        val_f1 = f1_score(trues, preds, zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation on held-out test set
    model.eval()
    te_preds, te_trues = [], []
    with torch.no_grad():
        for d, tgt in te_ld:
            d = d.to(DEVICE)
            te_preds.extend(model(d).argmax(1).cpu().numpy())
            te_trues.extend(tgt.numpy())
    test_f1 = f1_score(te_trues, te_preds, zero_division=0)

    torch.save(best_state or model.state_dict(), CNN_DIR / f"crackcnn_{name}_best.pth")
    elapsed = time.time() - t0
    cnn_results.append({"loss": name, "val_f1": round(best_f1,4), "test_f1": round(test_f1,4), "time_s": round(elapsed,1)})
    print(f"Val F1={best_f1:.4f}, Test F1={test_f1:.4f}, {elapsed:.1f}s ✓")

df_cnn = pd.DataFrame(cnn_results).sort_values("val_f1", ascending=False)
df_cnn.to_csv(RESULTS_DIR / "cnn_comparison.csv", index=False, encoding='utf-8-sig')

with open(CNN_DIR / "crackcnn_best_config.json", 'w') as f:
    json.dump({"dropout_rate": 0.5, "lr": 1e-3, "input_size": 128, "num_classes": 2}, f)
print(f"\nCNN models saved. Best: {df_cnn.iloc[0]['loss']} (F1={df_cnn.iloc[0]['val_f1']:.4f})")

# ===== PART 3: UNSUPERVISED =====
print("\n=== Part 3: Unsupervised Models ===")

def extract_features_reduced(img):
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    h = hog(img_u8, orientations=6, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
    n_bins = 8*7+3
    lbp_img = local_binary_pattern(img_u8, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
    g = graycomatrix(img_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    gf = [graycoprops(g, p)[0,0] for p in ("contrast","correlation","energy","homogeneity")]
    e = float(np.count_nonzero(cv2.Canny(img_u8, 50, 150))) / img_u8.size
    return np.concatenate([h, lbp_hist, gf, [e]])

unsup_imgs, unsup_lbls = load_dataset(DATA_ROOT, per_class=N_UNSUP)
X_unsup = np.stack([extract_features_reduced(img) for img in unsup_imgs])
print(f"Unsup features: {X_unsup.shape}")

sc = StandardScaler(); X_s = sc.fit_transform(X_unsup)
joblib.dump(sc, SCALER_DIR / "unsupervised_scaler.joblib")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_s)
np.savez(UNSUP_DIR / "pca_results.npz", X_pca=X_pca, labels=unsup_lbls)

def eval_cluster(name, yp, yt, Xd):
    mask = yp != -1
    sil = silhouette_score(Xd[mask], yp[mask]) if mask.sum()>=2 and len(set(yp[mask]))>=2 else float('nan')
    ari = adjusted_rand_score(yt[mask] if mask.sum()<len(yt) else yt, yp[mask] if mask.sum()<len(yt) else yp)
    nmi = normalized_mutual_info_score(yt[mask] if mask.sum()<len(yt) else yt, yp[mask] if mask.sum()<len(yt) else yp)
    return {"method": name, "silhouette": round(sil,4) if not np.isnan(sil) else None,
            "ari": round(ari,4), "nmi": round(nmi,4)}

unsup_results = []

# KMeans
km = KMeans(n_clusters=2, random_state=42, n_init='auto')
km_l = km.fit_predict(X_s)
joblib.dump(km, UNSUP_DIR / "kmeans_best.joblib")
unsup_results.append(eval_cluster("K-Means", km_l, unsup_lbls, X_s))
print(f"  K-Means: ARI={unsup_results[-1]['ari']:.4f} ✓")

# GMM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_l = gmm.fit_predict(X_s)
joblib.dump(gmm, UNSUP_DIR / "gmm_best.joblib")
unsup_results.append(eval_cluster("GMM", gmm_l, unsup_lbls, X_s))
print(f"  GMM: ARI={unsup_results[-1]['ari']:.4f} ✓")

# DBSCAN (grid search)
best_ari = -1; best_db = None; best_db_l = None
for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for ms in [3, 5, 10]:
        db = DBSCAN(eps=eps, min_samples=ms)
        db_l = db.fit_predict(X_s)
        nc = len(set(db_l)) - (1 if -1 in db_l else 0)
        if nc >= 2:
            mask = db_l != -1
            if mask.sum() > 10:
                ari = adjusted_rand_score(unsup_lbls[mask], db_l[mask])
                if ari > best_ari: best_ari, best_db, best_db_l = ari, db, db_l
if best_db is not None:
    joblib.dump(best_db, UNSUP_DIR / "dbscan_best.joblib")
    unsup_results.append(eval_cluster(f"DBSCAN(eps={best_db.eps},ms={best_db.min_samples})",
                                       best_db_l, unsup_lbls, X_s))
    print(f"  DBSCAN: ARI={unsup_results[-1]['ari']:.4f} ✓")
else:
    print("  DBSCAN: No valid clustering found")

# Agglomerative
agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
agg_l = agg.fit_predict(X_s)
joblib.dump(agg, UNSUP_DIR / "agglomerative_best.joblib")
unsup_results.append(eval_cluster("Agglomerative", agg_l, unsup_lbls, X_s))
print(f"  Agglomerative: ARI={unsup_results[-1]['ari']:.4f} ✓")

# Spectral
spec = SpectralClustering(n_clusters=2, affinity='rbf', random_state=42, n_init=10)
spec_l = spec.fit_predict(X_s)
joblib.dump(spec, UNSUP_DIR / "spectral_best.joblib")
unsup_results.append(eval_cluster("Spectral", spec_l, unsup_lbls, X_s))
print(f"  Spectral: ARI={unsup_results[-1]['ari']:.4f} ✓")

df_unsup = pd.DataFrame(unsup_results)
df_unsup.to_csv(RESULTS_DIR / "unsupervised_comparison.csv", index=False, encoding='utf-8-sig')

# ===== DONE =====
print("\n" + "="*60)
print("FAST PRE-TRAINING COMPLETE!")
print("="*60)
print(f"\nTraditional: {len(trad_results)} models")
for r in trad_results:
    print(f"  {r['model']}: F1={r['test_f1']:.4f}")

print(f"\nCNN: {len(cnn_results)} loss configs")
for r in cnn_results:
    print(f"  {r['loss']}: F1={r['val_f1']:.4f}")

print(f"\nUnsupervised: {len(unsup_results)} methods")
for r in unsup_results:
    print(f"  {r['method']}: ARI={r['ari']:.4f}")

print(f"\nAll models saved to: {OUTPUT_DIR}")
print("Ready for Gradio visualization system!")
