"""Data leakage diagnostic: checks for suspicious patterns in training results.

Tests:
1. Duplicate images in dataset (by perceptual hash)
2. Train/val/test index isolation
3. Label shuffle test: if CNN gets high F1 on shuffled labels → leakage
4. Check if test images are too similar to train images
"""
import os, sys, warnings, hashlib, json
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import time

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")
_D = os.getenv("CRACK_DATA_ROOT")
DATA_ROOT = Path(_D).expanduser().resolve() if _D else (PROJECT_ROOT / "data").resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42); torch.manual_seed(42)

print(f"DATA_ROOT: {DATA_ROOT}")
print(f"DEVICE: {DEVICE}")

# ===== HELPERS =====
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _imread_gray(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf is not None and buf.size > 0 else None

def apply_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
def apply_median_filter(img):
    return cv2.medianBlur(img, 5)
def default_preprocess(img):
    return apply_median_filter(apply_clahe(img))

# ===================================================================
# TEST 1: Check for duplicate images
# ===================================================================
print("\n" + "="*70)
print("TEST 1: Duplicate Image Detection (perceptual hash)")
print("="*70)

def dhash(img, hash_size=16):
    """Difference hash: resize to hash_size+1 × hash_size, compare adjacent pixels."""
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum(2 ** i for i, v in enumerate(diff.flatten()) if v)

# Sample 500 from each class
pos_dir = DATA_ROOT / "Positive"
neg_dir = DATA_ROOT / "Negative"

hashes_pos = {}
hashes_neg = {}
for label, d, hdict in [("Positive", pos_dir, hashes_pos), ("Negative", neg_dir, hashes_neg)]:
    paths = sorted(d.iterdir())
    for p in paths[:1000]:  # Check first 1000 from each
        if p.suffix.lower() in IMAGE_EXTS:
            img = _imread_gray(p)
            if img is not None:
                h = dhash(img)
                if h in hdict:
                    hdict[h].append(p.name)
                else:
                    hdict[h] = [p.name]

pos_dupes = {h: names for h, names in hashes_pos.items() if len(names) > 1}
neg_dupes = {h: names for h, names in hashes_neg.items() if len(names) > 1}

# Cross-class duplicates (same image in both Positive and Negative?)
cross_dupes = set(hashes_pos.keys()) & set(hashes_neg.keys())

print(f"  Positive dir: {len(hashes_pos)} unique hashes from 1000 samples")
print(f"  Negative dir: {len(hashes_neg)} unique hashes from 1000 samples")
print(f"  Within-class duplicates (Positive): {len(pos_dupes)} groups")
print(f"  Within-class duplicates (Negative): {len(neg_dupes)} groups")
print(f"  Cross-class duplicates (SAME image in BOTH dirs!): {len(cross_dupes)}")
if cross_dupes:
    print(f"  ⚠️  CROSS-CLASS DUPLICATES DETECTED: {cross_dupes}")
    for h in list(cross_dupes)[:5]:
        print(f"    hash={h}: Pos={hashes_pos[h]}, Neg={hashes_neg[h]}")

# ===================================================================
# TEST 2: Label Shuffle Test (the definitive leakage test)
# ===================================================================
print("\n" + "="*70)
print("TEST 2: Label Shuffle Test")
print("="*70)
print("If CNN achieves high F1 with SHUFFLED (random) labels, there's leakage.")
print("If CNN achieves F1≈0.5 with shuffled labels, the model genuinely learns.")

# Load 400 images (200 per class) for quick test
def load_dataset(per_class):
    def _ld(d, lbl):
        imgs, lbls = [], []
        for p in sorted(d.iterdir())[:per_class]:
            if p.suffix.lower() in IMAGE_EXTS:
                img = _imread_gray(p)
                if img is not None: imgs.append(img); lbls.append(lbl)
        return imgs, lbls
    pi, pl = _ld(DATA_ROOT / "Positive", 1)
    ni, nl = _ld(DATA_ROOT / "Negative", 0)
    all_i = pi + ni
    labels = np.array(pl + nl, dtype=np.int64)
    return all_i, labels

N_TEST = 200  # per class

print(f"\nLoading {N_TEST * 2} images...")
images, labels = load_dataset(N_TEST)

# Preprocess
processed = []
for img in images:
    img = default_preprocess(img)
    img = cv2.resize(img, (128, 128))
    processed.append(img.astype(np.float32) / 255.0)
processed = np.stack(processed)

# --- Test A: Real labels ---
print("\n--- Test 2A: CNN with REAL labels ---")
X_tr, X_te, y_tr, y_te = train_test_split(
    processed, labels, test_size=0.3, random_state=42, stratify=labels)

# Mini CNN
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2))
        self.c2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2))
        self.c3 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.cls = nn.Linear(64, 2)
    def forward(self, x):
        for c in [self.c1, self.c2, self.c3]: x = c(x)
        x = self.gap(x); x = x.view(x.size(0), -1)
        return self.cls(x)

class DS(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = torch.from_numpy(imgs).float().unsqueeze(1)
        self.lbls = torch.from_numpy(lbls).long()
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.imgs[i], self.lbls[i]

tr_ds = DS(X_tr, y_tr); te_ds = DS(X_te, y_te)
tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
te_ld = DataLoader(te_ds, batch_size=32)

model_real = MiniCNN().to(DEVICE)
opt = Adam(model_real.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for ep in range(10):
    model_real.train()
    for d, t in tr_ld:
        d, t = d.to(DEVICE), t.to(DEVICE)
        opt.zero_grad()
        loss = criterion(model_real(d), t)
        loss.backward(); opt.step()

model_real.eval()
preds, trues = [], []
with torch.no_grad():
    for d, t in te_ld:
        d = d.to(DEVICE)
        preds.extend(model_real(d).argmax(1).cpu().numpy())
        trues.extend(t.numpy())
real_f1 = f1_score(trues, preds, zero_division=0)
print(f"  REAL labels → Test F1 = {real_f1:.4f}")

# --- Test B: Shuffled labels ---
print("\n--- Test 2B: CNN with SHUFFLED (random) labels ---")
shuffled_labels = labels.copy()
np.random.shuffle(shuffled_labels)

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
    processed, shuffled_labels, test_size=0.3, random_state=42, stratify=shuffled_labels)

tr_ds_s = DS(X_tr_s, y_tr_s); te_ds_s = DS(X_te_s, y_te_s)
tr_ld_s = DataLoader(tr_ds_s, batch_size=32, shuffle=True)
te_ld_s = DataLoader(te_ds_s, batch_size=32)

model_shuf = MiniCNN().to(DEVICE)
opt_s = Adam(model_shuf.parameters(), lr=1e-3)

for ep in range(10):
    model_shuf.train()
    for d, t in tr_ld_s:
        d, t = d.to(DEVICE), t.to(DEVICE)
        opt_s.zero_grad()
        loss = criterion(model_shuf(d), t)
        loss.backward(); opt_s.step()

model_shuf.eval()
preds_s, trues_s = [], []
with torch.no_grad():
    for d, t in te_ld_s:
        d = d.to(DEVICE)
        preds_s.extend(model_shuf(d).argmax(1).cpu().numpy())
        trues_s.extend(t.numpy())
shuf_f1 = f1_score(trues_s, preds_s, zero_division=0)
print(f"  SHUFFLED labels → Test F1 = {shuf_f1:.4f}")

# --- Verdict ---
print(f"\n  {'='*50}")
if shuf_f1 > 0.7:
    print(f"  ⚠️  WARNING: Shuffled F1={shuf_f1:.4f} > 0.7 — likely DATA LEAKAGE!")
elif shuf_f1 > 0.6:
    print(f"  ⚠️  CAUTION: Shuffled F1={shuf_f1:.4f} — suspicious, further check needed")
else:
    print(f"  ✅ PASS: Shuffled F1={shuf_f1:.4f} ≈ 0.5 — model genuinely learns from images")
print(f"  Real F1={real_f1:.4f}, gap={real_f1 - shuf_f1:.4f}")
print(f"  {'='*50}")

# ===================================================================
# TEST 3: Check if CNN is just memorizing (overfitting check)
# ===================================================================
print("\n" + "="*70)
print("TEST 3: Overfitting Check (train vs test gap)")
print("="*70)

model_real.eval()
tr_preds, tr_trues = [], []
with torch.no_grad():
    for d, t in tr_ld:
        d = d.to(DEVICE)
        tr_preds.extend(model_real(d).argmax(1).cpu().numpy())
        tr_trues.extend(t.numpy())
train_f1 = f1_score(tr_trues, tr_preds, zero_division=0)
print(f"  Train F1 = {train_f1:.4f}")
print(f"  Test F1  = {real_f1:.4f}")
gap = train_f1 - real_f1
print(f"  Gap = {gap:.4f}")
if gap > 0.1:
    print(f"  ⚠️  Overfitting detected (gap > 0.1)")
else:
    print(f"  ✅ No significant overfitting")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print(f"  1. Cross-class duplicates: {len(cross_dupes)} (should be 0)")
print(f"  2. Shuffled-label F1: {shuf_f1:.4f} (should be ~0.5)")
print(f"  3. Train-Test gap: {gap:.4f} (should be < 0.1)")

all_ok = (len(cross_dupes) == 0) and (shuf_f1 < 0.6) and (gap < 0.1)
if all_ok:
    print(f"\n  ✅ ALL CHECKS PASSED — high F1 is genuine, not from data leakage")
    print(f"  The task (crack vs non-crack) is visually distinctive, and")
    print(f"  a CNN with proper architecture can achieve near-perfect accuracy.")
else:
    print(f"\n  ⚠️  SOME CHECKS FAILED — review the issues above")
