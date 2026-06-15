"""Pre-train all models and save to outputs/ for the visualization system.

This script:
1. Trains all 7 traditional models with GridSearchCV on 4000 samples
2. Saves best models to outputs/models/traditional/
3. Trains CNN with 6 loss configs (5 epochs each for speed)
4. Saves CNN models to outputs/models/cnn/
5. Runs all 5 unsupervised methods and saves results
6. Saves comparison results to outputs/results/

Usage: python scripts/_pretrain_all.py
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
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
import time

# ===== CONFIG =====
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
TRAD_DIR = MODELS_DIR / "traditional"
CNN_DIR = MODELS_DIR / "cnn"
UNSUP_DIR = MODELS_DIR / "unsupervised"
SCALER_DIR = OUTPUT_DIR / "scalers"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [TRAD_DIR, CNN_DIR, UNSUP_DIR, SCALER_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")
_DATA_ROOT = os.getenv("CRACK_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT).expanduser() if _DATA_ROOT else PROJECT_ROOT / "data"
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
DATA_ROOT = DATA_ROOT.resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

TRAD_SAMPLES = 2000  # Per class for traditional models
CNN_SAMPLES = 1000   # Per class for CNN
UNSUP_SAMPLES = 1000 # Per class for unsupervised

print(f"DATA_ROOT: {DATA_ROOT}")
print(f"DEVICE: {DEVICE}")
print(f"Traditional samples: {TRAD_SAMPLES * 2}")
print(f"CNN samples: {CNN_SAMPLES * 2}")

# ===== COMMON DATA LOADING =====
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

# Preprocessing & features
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(img)

def apply_median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

def default_preprocess(img):
    return apply_median_filter(apply_clahe(img))

def extract_hog_features(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
    return hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, feature_vector=True)

def extract_lbp_features(img, radius=1, n_points=8):
    n_bins = n_points * (n_points - 1) + 3
    lbp_img = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_glcm_features(img, distances=(1,3,5), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    props = []
    for d in distances:
        for a in angles:
            glcm = graycomatrix(img_u8, distances=[d], angles=[a],
                                levels=256, symmetric=True, normed=True)
            props.extend([graycoprops(glcm, "contrast")[0,0],
                          graycoprops(glcm, "correlation")[0,0],
                          graycoprops(glcm, "energy")[0,0],
                          graycoprops(glcm, "homogeneity")[0,0]])
    return np.array(props, dtype=np.float64)

def extract_edge_density(img, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return float(np.count_nonzero(edges)) / edges.size

def extract_all_features(image):
    image = image.astype(np.uint8) if image.dtype != np.uint8 else image
    return np.concatenate([
        extract_hog_features(image),
        extract_lbp_features(image),
        extract_glcm_features(image),
        np.array([extract_edge_density(image)], dtype=np.float64),
    ]).astype(np.float64)

# ===================================================================
# PART 1: TRADITIONAL MODELS
# ===================================================================
print("\n" + "="*70)
print("PART 1: Training Traditional Models")
print("="*70)

print(f"\nLoading {TRAD_SAMPLES * 2} images...")
images, labels = load_dataset(DATA_ROOT, per_class=TRAD_SAMPLES)
print(f"Loaded: {len(labels)} images")

print("Preprocessing (CLAHE + Median)...")
processed = np.array([default_preprocess(img) for img in images])
print("Extracting features...")
X = np.stack([extract_all_features(img) for img in processed])
y = labels
print(f"Feature matrix: {X.shape}")

# Save scaler
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, SCALER_DIR / "standard_scaler.joblib")
print(f"Scaler saved to {SCALER_DIR / 'standard_scaler.joblib'}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# Define models to train
TRAD_MODELS = {
    "decision_tree": {
        "name": "决策树 (DT)",
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            'max_depth': [3, 5, 10, 15, 20, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10],
        },
        "pipeline": False,
    },
    "svm": {
        "name": "SVM",
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ]),
        "params": {
            'svc__kernel': ['linear', 'rbf'],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto'],
        },
        "pipeline": True,
    },
    "naive_bayes": {
        "name": "朴素贝叶斯 (NB)",
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('nb', GaussianNB())
        ]),
        "params": {'nb__var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]},
        "pipeline": True,
    },
    "random_forest": {
        "name": "随机森林 (RF)",
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
        },
        "pipeline": False,
    },
    "logistic_regression": {
        "name": "逻辑回归 (LR)",
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(random_state=42, max_iter=2000))
        ]),
        "params": {
            'logreg__C': [0.01, 0.1, 1, 10, 100],
            'logreg__penalty': ['l1', 'l2'],
            'logreg__solver': ['liblinear', 'lbfgs'],
        },
        "pipeline": True,
    },
}

# Add XGBoost and LightGBM if available
try:
    from xgboost import XGBClassifier
    TRAD_MODELS["xgboost"] = {
        "name": "XGBoost",
        "model": XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
        },
        "pipeline": False,
    }
    print("XGBoost available")
except ImportError:
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    TRAD_MODELS["lightgbm"] = {
        "name": "LightGBM",
        "model": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.01, 0.1, 0.3],
        },
        "pipeline": False,
    }
    print("LightGBM available")
except ImportError:
    print("LightGBM not available")

# Train all traditional models
trad_results = []

for key, config in TRAD_MODELS.items():
    print(f"\n{'─'*50}")
    print(f"Training: {config['name']}")
    t0 = time.time()

    grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='f1',
                        n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    try:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float('nan')

    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    save_path = TRAD_DIR / f"{key}_best.joblib"
    joblib.dump(best_model, save_path)

    # Save best params
    params_path = TRAD_DIR / f"{key}_best_params.json"
    with open(params_path, 'w') as f:
        json.dump(grid.best_params_, f, indent=2, default=str)

    trad_results.append({
        "model": config['name'],
        "key": key,
        "best_score_cv": round(grid.best_score_, 4),
        "test_f1": round(f1, 4),
        "test_accuracy": round(acc, 4),
        "test_auc": round(auc, 4),
        "train_time_s": round(elapsed, 1),
        "best_params": str(grid.best_params_),
    })

    print(f"  CV F1: {grid.best_score_:.4f}, Test F1: {f1:.4f}, "
          f"Test AUC: {auc:.4f}, Time: {elapsed:.1f}s")
    print(f"  Saved: {save_path}")

# Save traditional comparison
df_trad = pd.DataFrame(trad_results).sort_values("test_f1", ascending=False)
df_trad.to_csv(RESULTS_DIR / "traditional_comparison.csv", index=False, encoding='utf-8-sig')
print(f"\nTraditional comparison saved to {RESULTS_DIR / 'traditional_comparison.csv'}")
print(df_trad.to_string(index=False))

# ===================================================================
# PART 2: CNN MODELS
# ===================================================================
print("\n" + "="*70)
print("PART 2: Training CNN Models")
print("="*70)

# CrackCNN
class CrackCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.5):
        super().__init__()
        self.conv1 = self._make_block(input_channels, 32)
        self.conv2 = self._make_block(32, 64)
        self.conv3 = self._make_block(64, 128)
        self.conv4 = self._make_block(128, 256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x)
        x = self.conv3(x); x = self.conv4(x)
        x = self.gap(x); x = x.view(x.size(0), -1)
        x = self.dropout(x); x = self.classifier(x)
        return x

# Custom losses
class FocalLoss(nn.Module):
    """Focal Loss for balanced binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : float or None
        Positive-class weight. None = no class balancing (best for balanced data).
        0.25 = common default for highly imbalanced data (few positives).
        0.50 = neutral weighting (equal weight to both classes).
    gamma : float
        Focusing parameter. 0 = standard CE; larger = more focus on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = torch.where(
                targets == 1,
                torch.tensor(self.alpha, device=inputs.device),
                torch.tensor(1.0 - self.alpha, device=inputs.device),
            )
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce).mean()

class LabelSmoothingCE(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, inputs, targets):
        n = inputs.size(-1)
        logp = F.log_softmax(inputs, dim=-1)
        smooth = torch.zeros_like(logp).fill_(self.epsilon / (n - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
        return (-smooth * logp).sum(dim=-1).mean()

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

# Load CNN data
print(f"\nLoading {CNN_SAMPLES * 2} images for CNN...")
cnn_images, cnn_labels = load_dataset(DATA_ROOT, per_class=CNN_SAMPLES)
print(f"Loaded: {len(cnn_labels)} images")

# Resize and normalize
print("Preprocessing for CNN (resize 128x128, CLAHE+Median, normalize)...")
cnn_processed = []
for img in cnn_images:
    img = default_preprocess(img)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    cnn_processed.append(img)
cnn_processed = np.stack(cnn_processed)

# 三路划分：train (60%) / val (15%) / test (25%)
Xc_train, Xc_temp, yc_train, yc_temp = train_test_split(
    cnn_processed, cnn_labels, test_size=0.4, random_state=42, stratify=cnn_labels)
Xc_val, Xc_test, yc_val, yc_test = train_test_split(
    Xc_temp, yc_temp, test_size=0.625, random_state=42, stratify=yc_temp)
# Result: train=60%, val=15%, test=25%
print(f"CNN Train: {len(Xc_train)}, Val: {len(Xc_val)}, Test: {len(Xc_test)}")

train_ds = CrackDataset(Xc_train, yc_train)
val_ds = CrackDataset(Xc_val, yc_val)
test_ds = CrackDataset(Xc_test, yc_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# CNN training configs
CNN_CONFIGS = [
    {"name": "cross_entropy", "loss_fn": lambda: nn.CrossEntropyLoss()},
    {"name": "focal_gamma2", "loss_fn": lambda: FocalLoss(alpha=None, gamma=2.0)},
    {"name": "focal_gamma3", "loss_fn": lambda: FocalLoss(alpha=None, gamma=3.0)},
    {"name": "focal_balanced", "loss_fn": lambda: FocalLoss(alpha=0.5, gamma=2.0)},
    {"name": "label_smoothing", "loss_fn": lambda: LabelSmoothingCE(epsilon=0.1)},
    {"name": "dice", "loss_fn": lambda: DiceLoss(smooth=1.0)},
]

CNN_EPOCHS = 15  # Reduced for practical timing; increase for production
BEST_CNN_CONFIG = {"dropout_rate": 0.5, "lr": 1e-3}

cnn_results = []

for config in CNN_CONFIGS:
    print(f"\n{'─'*50}")
    print(f"Training CNN: {config['name']}")
    t0 = time.time()

    model = CrackCNN(num_classes=2, input_channels=1,
                     dropout_rate=BEST_CNN_CONFIG["dropout_rate"]).to(DEVICE)
    criterion = config['loss_fn']()
    optimizer = Adam(model.parameters(), lr=BEST_CNN_CONFIG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val_f1 = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_f1": []}

    for epoch in range(CNN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / len(train_ds)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)

        # Validation (on independent val set, NOT test set)
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        history["val_f1"].append(val_f1)
        scheduler.step(1.0 - val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{CNN_EPOCHS}: loss={avg_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_f1={val_f1:.4f}")

    elapsed = time.time() - t0

    # Final evaluation on held-out test set (never used during training)
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_preds.extend(output.argmax(dim=1).cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)

    # Save best model
    save_path = CNN_DIR / f"crackcnn_{config['name']}_best.pth"
    torch.save(best_state, save_path)

    # Save history
    hist_path = CNN_DIR / f"crackcnn_{config['name']}_history.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    cnn_results.append({
        "loss_fn": config['name'],
        "best_val_f1": round(best_val_f1, 4),
        "test_f1": round(test_f1, 4),
        "final_train_loss": round(history["train_loss"][-1], 4),
        "final_train_acc": round(history["train_acc"][-1], 4),
        "train_time_s": round(elapsed, 1),
        "epochs": CNN_EPOCHS,
    })

    print(f"  Best val F1: {best_val_f1:.4f}, Test F1: {test_f1:.4f}, Time: {elapsed:.1f}s")
    print(f"  Saved: {save_path}")

# Save CNN comparison and best config
df_cnn = pd.DataFrame(cnn_results).sort_values("best_val_f1", ascending=False)
df_cnn.to_csv(RESULTS_DIR / "cnn_comparison.csv", index=False, encoding='utf-8-sig')

cnn_config_json = {
    "dropout_rate": BEST_CNN_CONFIG["dropout_rate"],
    "learning_rate": BEST_CNN_CONFIG["lr"],
    "input_size": 128,
    "num_classes": 2,
    "input_channels": 1,
    "num_params": sum(p.numel() for p in CrackCNN().parameters()),
}
with open(CNN_DIR / "crackcnn_best_config.json", 'w') as f:
    json.dump(cnn_config_json, f, indent=2)

print(f"\nCNN comparison saved to {RESULTS_DIR / 'cnn_comparison.csv'}")
print(df_cnn.to_string(index=False))

# ===================================================================
# PART 3: UNSUPERVISED MODELS
# ===================================================================
print("\n" + "="*70)
print("PART 3: Unsupervised Clustering")
print("="*70)

print(f"\nLoading {UNSUP_SAMPLES * 2} images...")
unsup_images, unsup_labels = load_dataset(DATA_ROOT, per_class=UNSUP_SAMPLES)
print(f"Loaded: {len(unsup_labels)} images")

print("Extracting features (reduced dimension)...")
# Use reduced HOG for clustering speed
def extract_features_reduced(img):
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    # Reduced HOG
    h = hog(img_u8, orientations=6, pixels_per_cell=(16,16),
            cells_per_block=(2,2), feature_vector=True)
    # LBP
    n_bins = 8 * 7 + 3
    lbp_img = local_binary_pattern(img_u8, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp_img, bins=n_bins, range=(0, n_bins), density=True)
    # GLCM reduced
    glcm = graycomatrix(img_u8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p)[0,0] for p in
                  ["contrast","correlation","energy","homogeneity"]]
    # Edge density
    edges = cv2.Canny(img_u8, 50, 150)
    edge_den = float(np.count_nonzero(edges)) / edges.size
    return np.concatenate([h, lbp_hist, glcm_feats, [edge_den]])

X_unsup = np.stack([extract_features_reduced(img) for img in unsup_images])
print(f"Unsupervised features: {X_unsup.shape}")

# Standardize
scaler_unsup = StandardScaler()
X_unsup_scaled = scaler_unsup.fit_transform(X_unsup)
joblib.dump(scaler_unsup, SCALER_DIR / "unsupervised_scaler.joblib")

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_unsup_scaled)

def eval_clustering(name, y_pred, y_true, X_data):
    """Evaluate clustering results with internal and external metrics."""
    unique_labels = set(y_pred)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(y_pred == -1))

    # Filter noise for silhouette
    mask = y_pred != -1
    if mask.sum() >= 2 and len(set(y_pred[mask])) >= 2:
        sil = silhouette_score(X_data[mask], y_pred[mask])
        db_idx = davies_bouldin_score(X_data[mask], y_pred[mask])
        ch_idx = calinski_harabasz_score(X_data[mask], y_pred[mask])
    else:
        sil, db_idx, ch_idx = float('nan'), float('nan'), float('nan')

    # External metrics
    if n_noise > 0:
        mask2 = y_pred != -1
        ari = adjusted_rand_score(y_true[mask2], y_pred[mask2])
        nmi = normalized_mutual_info_score(y_true[mask2], y_pred[mask2])
    else:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

    return {
        "method": name,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": round(sil, 4) if not np.isnan(sil) else None,
        "db_index": round(db_idx, 4) if not np.isnan(db_idx) else None,
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
    }

unsup_results = []

# 1. K-Means
print("\nK-Means...")
km = KMeans(n_clusters=2, random_state=42, n_init='auto')
km_labels = km.fit_predict(X_unsup_scaled)
joblib.dump(km, UNSUP_DIR / "kmeans_best.joblib")
unsup_results.append(eval_clustering("K-Means", km_labels, unsup_labels, X_unsup_scaled))

# 2. GMM
print("GMM...")
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(X_unsup_scaled)
joblib.dump(gmm, UNSUP_DIR / "gmm_best.joblib")
unsup_results.append(eval_clustering("GMM", gmm_labels, unsup_labels, X_unsup_scaled))

# 3. DBSCAN (grid search for best eps)
print("DBSCAN (grid search for eps)...")
best_db_ari = -1
best_db = None
best_db_labels = None
for eps in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
    for ms in [3, 5, 10]:
        db = DBSCAN(eps=eps, min_samples=ms)
        db_labels = db.fit_predict(X_unsup_scaled)
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        if n_clusters >= 2:
            mask = db_labels != -1
            if mask.sum() > 10:
                ari = adjusted_rand_score(unsup_labels[mask], db_labels[mask])
                if ari > best_db_ari:
                    best_db_ari = ari
                    best_db = db
                    best_db_labels = db_labels
if best_db is not None:
    joblib.dump(best_db, UNSUP_DIR / "dbscan_best.joblib")
    unsup_results.append(eval_clustering(
        f"DBSCAN(eps={best_db.eps},ms={best_db.min_samples})",
        best_db_labels, unsup_labels, X_unsup_scaled))
else:
    unsup_results.append({"method": "DBSCAN", "error": "No valid clustering found"})

# 4. Agglomerative
print("Agglomerative...")
agg = AgglomerativeClustering(n_clusters=2, linkage='ward')
agg_labels = agg.fit_predict(X_unsup_scaled)
joblib.dump(agg, UNSUP_DIR / "agglomerative_best.joblib")
unsup_results.append(eval_clustering("Agglomerative(ward)", agg_labels, unsup_labels, X_unsup_scaled))

# 5. Spectral
print("Spectral...")
spec = SpectralClustering(n_clusters=2, affinity='rbf', random_state=42, n_init=10)
spec_labels = spec.fit_predict(X_unsup_scaled)
joblib.dump(spec, UNSUP_DIR / "spectral_best.joblib")
unsup_results.append(eval_clustering("Spectral(rbf)", spec_labels, unsup_labels, X_unsup_scaled))

# Save results
df_unsup = pd.DataFrame(unsup_results)
df_unsup.to_csv(RESULTS_DIR / "unsupervised_comparison.csv", index=False, encoding='utf-8-sig')
print(f"\nUnsupervised comparison:")
print(df_unsup.to_string(index=False))

# Save PCA data for visualization
np.savez(UNSUP_DIR / "pca_results.npz", X_pca=X_pca, labels=unsup_labels)

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*70)
print("PRE-TRAINING COMPLETE")
print("="*70)
print(f"\nModels saved to: {MODELS_DIR}")
print(f"Results saved to: {RESULTS_DIR}")
print(f"Scalers saved to: {SCALER_DIR}")

print(f"\nTraditional models ({len(trad_results)}):")
for r in trad_results:
    print(f"  {r['model']}: F1={r['test_f1']:.4f}, AUC={r['test_auc']:.4f}, "
          f"Time={r['train_time_s']}s")

print(f"\nCNN models ({len(cnn_results)}):")
for r in cnn_results:
    print(f"  {r['loss_fn']}: val_f1={r['best_val_f1']:.4f}, Time={r['train_time_s']}s")

print(f"\nUnsupervised methods ({len(unsup_results)}):")
for r in unsup_results:
    if 'error' not in r:
        print(f"  {r['method']}: ARI={r['ari']:.4f}, NMI={r['nmi']:.4f}, "
              f"Sil={r['silhouette']:.4f}")
    else:
        print(f"  {r['method']}: ERROR - {r['error']}")

print("\nReady for visualization system!")
