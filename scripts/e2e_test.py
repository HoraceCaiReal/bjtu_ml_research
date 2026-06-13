"""
裂纹图像识别系统 — 端到端测试脚本

运行方式:
    conda activate bjtu_ml
    python scripts/e2e_test.py

设计文档: docs/superpowers/specs/2026-06-13-e2e-test-design.md
"""

import io
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Windows UTF-8 输出：解决 GBK 编码不支持 Unicode 符号的问题
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 项目路径（必须在其他本地模块导入前设置，确保 src/ 可被找到）
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class StageRunner:
    """逐阶段运行测试，记录通过/失败，最后打印汇总报告。"""

    def __init__(self):
        self.results: list[dict] = []
        self.total_start = time.time()

    def run(self, name: str, fn):
        """执行 fn()，捕获异常，记录耗时和结果。"""
        print(f"\n{'─' * 50}")
        print(f"▶ {name}")
        print(f"{'─' * 50}")
        t0 = time.time()
        try:
            detail = fn()
            elapsed = time.time() - t0
            self.results.append(
                {
                    "name": name,
                    "passed": True,
                    "detail": detail or "",
                    "elapsed": elapsed,
                }
            )
            print(f"  ✅ PASS  {detail or ''}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            self.results.append(
                {
                    "name": name,
                    "passed": False,
                    "detail": str(e),
                    "elapsed": elapsed,
                }
            )
            print(f"  ❌ FAIL  {e}  ({elapsed:.1f}s)")
            print(tb)

    def report(self):
        total_elapsed = time.time() - self.total_start
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"\n{'=' * 50}")
        print("  裂纹图像识别系统 — 端到端测试报告")
        print(f"{'=' * 50}")
        for r in self.results:
            icon = "✅" if r["passed"] else "❌"
            print(f"  {icon} {r['name']:<20s} {r['detail']}")
        print(f"{'─' * 50}")
        print(f"  总计: {passed}/{total} PASS  |  耗时: {total_elapsed:.1f}s")
        print(f"{'=' * 50}")
        return passed == total


# ===========================================================================
# Stage 0: 环境检查
# ===========================================================================
def stage0_env() -> str:
    """验证 Python 版本和关键依赖可导入。"""
    assert sys.version_info >= (3, 10), (
        f"Python >= 3.10 required, got {sys.version_info}"
    )

    missing = []
    for mod in [
        "torch",
        "sklearn",
        "cv2",
        "xgboost",
        "lightgbm",
        "gradio",
        "skimage",
        "joblib",
        "dotenv",
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    assert not missing, f"缺失依赖: {missing}"

    import torch

    cuda = torch.cuda.is_available()
    torch_ver = torch.__version__.split("+")[0]
    return f"torch {torch_ver}, CUDA: {cuda}"


# ===========================================================================
# Stage 1: 数据路径验证
# ===========================================================================
def stage1_data_path() -> str:
    """验证 .env 中的数据路径存在，且 Positive/Negative 目录含图片。"""
    import cv2
    import numpy as np
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    data_root_str = os.getenv("CRACK_DATA_ROOT")
    assert data_root_str, ".env 中未找到 CRACK_DATA_ROOT"

    data_root = Path(data_root_str)
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    pos_dir = data_root / "Positive"
    neg_dir = data_root / "Negative"
    assert pos_dir.is_dir(), f"Positive 目录不存在: {pos_dir}"
    assert neg_dir.is_dir(), f"Negative 目录不存在: {neg_dir}"

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pos_count = sum(1 for p in pos_dir.iterdir() if p.suffix.lower() in image_exts)
    neg_count = sum(1 for p in neg_dir.iterdir() if p.suffix.lower() in image_exts)
    assert pos_count > 0, f"Positive 目录无图片: {pos_dir}"
    assert neg_count > 0, f"Negative 目录无图片: {neg_dir}"

    # 抽样加载 1 张图片验证可读
    sample = next(p for p in pos_dir.iterdir() if p.suffix.lower() in image_exts)
    buf = np.fromfile(str(sample), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"无法读取图片: {sample}"

    return f"{pos_count + neg_count} images, 抽样读取 OK ({img.shape})"


# ===========================================================================
# Stage 2: 特征提取验证
# ===========================================================================
def _extract_features(img):
    """提取 HOG+LBP+GLCM+edge_density 特征，与 gradio_app.py 保持一致。"""
    import cv2
    import numpy as np
    from skimage.feature import (
        graycomatrix,
        graycoprops,
        hog,
        local_binary_pattern,
    )

    # HOG（与 gradio_app.extract_hog_features 一致）
    h = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    # LBP（与 gradio_app.extract_lbp_features 一致: 59 bins）
    n_points, radius = 8, 1
    n_bins = n_points * (n_points - 1) + 3  # 59
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    # GLCM（与 gradio_app.extract_glcm_features 一致: 3×4×4=48 维）
    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    glcm_props = []
    for d in (1, 3, 5):
        for a in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            glcm = graycomatrix(
                img_u8,
                distances=[d],
                angles=[a],
                levels=256,
                symmetric=True,
                normed=True,
            )
            glcm_props.extend(
                [
                    graycoprops(glcm, "contrast")[0, 0],
                    graycoprops(glcm, "correlation")[0, 0],
                    graycoprops(glcm, "energy")[0, 0],
                    graycoprops(glcm, "homogeneity")[0, 0],
                ]
            )
    glcm_vec = np.array(glcm_props, dtype=np.float64)

    # Edge density（与 gradio_app.extract_edge_density 一致）
    edges = cv2.Canny(img, 50, 150)
    ed = np.array([float(np.count_nonzero(edges)) / edges.size])

    return np.concatenate([h, hist, glcm_vec, ed])


def _load_sample_image(data_root, label_dir="Positive"):
    """加载一张图片并应用 clahe+median 预处理。"""
    import cv2
    import numpy as np

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    sample = next(
        p for p in (data_root / label_dir).iterdir() if p.suffix.lower() in image_exts
    )
    buf = np.fromfile(str(sample), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    # clahe+median 预处理（与 gradio_app.prepare_data 一致）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 5)
    return img


def stage2_features() -> str:
    """对 1 张图片提取 HOG/LBP/GLCM/edge_density 特征，验证输出维度。"""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    data_root = Path(os.getenv("CRACK_DATA_ROOT"))
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    img = _load_sample_image(data_root)

    dims = {}

    # HOG
    import numpy as np
    from skimage.feature import hog

    h = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    dims["HOG"] = len(h)
    assert len(h) > 0, "HOG 特征为空"

    # LBP
    from skimage.feature import local_binary_pattern

    n_points, radius = 8, 1
    n_bins = n_points * (n_points - 1) + 3
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    dims["LBP"] = len(hist)
    assert len(hist) > 0, "LBP 特征为空"

    # GLCM
    from skimage.feature import graycomatrix, graycoprops

    img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
    glcm = graycomatrix(
        img_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    dims["GLCM"] = 48  # 3 distances × 4 angles × 4 properties
    assert graycoprops(glcm, "contrast").size > 0, "GLCM 特征为空"

    # Edge density
    import cv2

    edges = cv2.Canny(img, 50, 150)
    ed = float(np.count_nonzero(edges)) / edges.size
    dims["ED"] = round(ed, 4)
    assert 0 <= ed <= 1, f"edge_density 超出 [0,1]: {ed}"

    return f"HOG:{dims['HOG']} LBP:{dims['LBP']} GLCM:{dims['GLCM']} ED:{dims['ED']}"


# ===========================================================================
# Stage 3: 传统模型推理验证
# ===========================================================================
def stage3_trad_models() -> str:
    """加载 4 个传统模型，对同一张图提取特征后推理。"""
    # 安全说明: joblib 加载本项目自身训练产出的 sklearn 模型，来源可信。
    import joblib
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    data_root = Path(os.getenv("CRACK_DATA_ROOT"))
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    img = _load_sample_image(data_root)
    feature_vec = _extract_features(img).reshape(1, -1)

    model_dir = PROJECT_ROOT / "outputs" / "models" / "traditional"
    models_to_test = ["random_forest", "xgboost", "svm", "logistic_regression"]
    ok_count = 0

    for name in models_to_test:
        path = model_dir / f"{name}_best.joblib"
        assert path.exists(), f"模型文件不存在: {path}"
        model = joblib.load(path)
        pred = model.predict(feature_vec)
        assert pred[0] in (0, 1), f"{name} 输出非法: {pred[0]}"
        ok_count += 1

    return f"{ok_count}/{len(models_to_test)} models OK"


# ===========================================================================
# Stage 4: CNN 推理验证
# ===========================================================================
def stage4_cnn() -> str:
    """加载 CrackCNN 权重，对 3 张图片推理，验证输出概率。"""
    import json

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from dotenv import load_dotenv

    # -- CrackCNN 类定义（与 notebook 03 / gradio_app.py 保持一致）--
    class CrackCNN(nn.Module):
        def __init__(self, num_classes=2, input_channels=1, dropout_rate=0.5):
            super().__init__()
            self.block1 = self._make_block(input_channels, 32)
            self.block2 = self._make_block(32, 64)
            self.block3 = self._make_block(64, 128)
            self.block4 = self._make_block(128, 256)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(256, num_classes)

        @staticmethod
        def _make_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.classifier(x)
            return x

    # -- 加载配置和权重 --
    cnn_dir = PROJECT_ROOT / "outputs" / "models" / "cnn"
    config_path = cnn_dir / "crackcnn_best_config.json"
    weights_path = cnn_dir / "crackcnn_cross_entropy_best.pth"
    assert config_path.exists(), f"配置文件不存在: {config_path}"
    assert weights_path.exists(), f"权重文件不存在: {weights_path}"

    with open(config_path) as f:
        config = json.load(f)

    input_size = config.get("input_size", 128)
    dropout = config.get("dropout_rate", 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrackCNN(num_classes=2, input_channels=1, dropout_rate=dropout).to(device)
    # 安全说明: 加载本项目自身训练产出的 PyTorch 权重，来源可信。
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    # 兼容旧版权重 key (c1/c2/c3/c4/cls → block1/block2/block3/block4/classifier)
    _LEGACY_KEY_MAP = {
        "c1": "block1",
        "c2": "block2",
        "c3": "block3",
        "c4": "block4",
        "cls": "classifier",
    }
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in _LEGACY_KEY_MAP.items():
            if k.startswith(f"{old_prefix}."):
                new_key = f"{new_prefix}.{k[len(old_prefix) + 1 :]}"
                break
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # -- 加载 3 张图片推理 --
    load_dotenv(PROJECT_ROOT / ".env")
    data_root = Path(os.getenv("CRACK_DATA_ROOT"))
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []
    for d in [data_root / "Positive", data_root / "Negative"]:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in image_exts:
                samples.append(p)
            if len(samples) >= 3:
                break
        if len(samples) >= 3:
            break

    probs = []
    with torch.no_grad():
        for sp in samples[:3]:
            buf = np.fromfile(str(sp), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (input_size, input_size))
            # CLAHE + median 预处理（与训练时一致）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            img = cv2.medianBlur(img, 5)
            tensor = (
                torch.tensor(img, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            out = model(tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            probs.append(round(prob, 4))

    for p in probs:
        assert 0 <= p <= 1, f"概率超出 [0,1]: {p}"

    return f"probs={probs}"


# ===========================================================================
# Stage 5: Gradio 模块检查
# ===========================================================================
def stage5_gradio_module() -> str:
    """验证 gradio_app 可导入、界面可创建、run_pipeline 可调用。"""
    import importlib

    gradio_app = importlib.import_module("src.gradio_app")

    assert hasattr(gradio_app, "create_interface"), "缺少 create_interface"
    assert hasattr(gradio_app, "run_pipeline"), "缺少 run_pipeline"
    assert callable(gradio_app.create_interface), "create_interface 不可调用"
    assert callable(gradio_app.run_pipeline), "run_pipeline 不可调用"

    # 尝试创建界面（不启动服务器）
    app = gradio_app.create_interface()
    assert app is not None, "create_interface 返回 None"

    return "interface created, run_pipeline callable"


# ===========================================================================
# Stage 6: 方法组合实际运行
# ===========================================================================


class _MockProgress:
    """替代 gr.Progress，供 run_pipeline 在 Gradio 外调用时使用。"""

    def __call__(self, progress=None, desc=None):
        pass


def _build_pipeline_args(**overrides):
    """构建 run_pipeline 所需的完整参数列表，匹配 Gradio UI 的默认值。

    参数顺序严格对齐 run_pipeline 签名。
    """
    # 默认值（对齐 Gradio UI 初始状态）
    defaults = {
        "split_method": "holdout",
        "split_ratio": 0.7,
        "use_stratify": True,
        "preprocessing": "clahe+median",
        "features": ["hog", "lbp", "glcm", "edge_density"],
        "max_samples": 200,
        "model_name": "random_forest",
        # 传统模型参数
        "dt_max_depth": 15,
        "dt_min_samples_split": 5,
        "svm_C": 1.0,
        "nb_var_smoothing": 1e-9,
        "rf_n_estimators": 100,
        "rf_max_depth": 20,
        "rf_min_samples_split": 5,
        "lr_C": 1.0,
        "xgb_n_estimators": 100,
        "xgb_max_depth": 6,
        "xgb_subsample": 0.8,
        "lgbm_n_estimators": 100,
        "lgbm_max_depth": 6,
        "lgbm_num_leaves": 31,
        # CNN 参数
        "cnn_dropout": 0.5,
        "cnn_batch_size": 64,
        "cnn_epochs": 30,
        "cnn_early_stopping": 10,
        "cnn_input_size": 128,
        "cnn_weight_decay": 1e-4,
        # 无监督参数
        "unsup_n_clusters": 2,
        "unsup_eps": 0.5,
        "unsup_min_samples": 5,
        # Step 4: 传统模型 loss
        "dt_criterion": "gini",
        "svm_kernel": "rbf",
        "svm_gamma": "scale",
        "rf_criterion": "gini",
        "lr_penalty": "l2",
        "lr_solver": "lbfgs",
        "lr_l1_ratio": 0.5,
        "xgb_objective": "binary:logistic",
        "xgb_learning_rate": 0.1,
        "lgbm_objective": "binary",
        "lgbm_learning_rate": 0.1,
        # Step 4: CNN loss
        "cnn_loss_fn": "cross_entropy",
        "cnn_focal_alpha": "None",
        "cnn_focal_gamma": 2.0,
        "cnn_label_smoothing_epsilon": 0.1,
        "cnn_optimizer": "adam",
        "cnn_learning_rate": 0.001,
        # Step 4: 无监督 loss
        "kmeans_algorithm": "lloyd",
        "gmm_covariance_type": "full",
        "agg_linkage": "ward",
        "spec_affinity": "rbf",
        # Step 5
        "optimization_strategy": "manual",
        "cv_folds_opt": 3,
        "n_iter": 30,
        "validation_method": "holdout",
        "scoring_metric": "f1",
        "unsup_val_method": "internal_external",
        "random_seed": 42,
    }
    defaults.update(overrides)
    d = defaults
    return (
        d["split_method"],
        d["split_ratio"],
        d["use_stratify"],
        d["preprocessing"],
        d["features"],
        d["max_samples"],
        d["model_name"],
        d["dt_max_depth"],
        d["dt_min_samples_split"],
        d["svm_C"],
        d["nb_var_smoothing"],
        d["rf_n_estimators"],
        d["rf_max_depth"],
        d["rf_min_samples_split"],
        d["lr_C"],
        d["xgb_n_estimators"],
        d["xgb_max_depth"],
        d["xgb_subsample"],
        d["lgbm_n_estimators"],
        d["lgbm_max_depth"],
        d["lgbm_num_leaves"],
        d["cnn_dropout"],
        d["cnn_batch_size"],
        d["cnn_epochs"],
        d["cnn_early_stopping"],
        d["cnn_input_size"],
        d["cnn_weight_decay"],
        d["unsup_n_clusters"],
        d["unsup_eps"],
        d["unsup_min_samples"],
        d["dt_criterion"],
        d["svm_kernel"],
        d["svm_gamma"],
        d["rf_criterion"],
        d["lr_penalty"],
        d["lr_solver"],
        d["lr_l1_ratio"],
        d["xgb_objective"],
        d["xgb_learning_rate"],
        d["lgbm_objective"],
        d["lgbm_learning_rate"],
        d["cnn_loss_fn"],
        d["cnn_focal_alpha"],
        d["cnn_focal_gamma"],
        d["cnn_label_smoothing_epsilon"],
        d["cnn_optimizer"],
        d["cnn_learning_rate"],
        d["kmeans_algorithm"],
        d["gmm_covariance_type"],
        d["agg_linkage"],
        d["spec_affinity"],
        d["optimization_strategy"],
        d["cv_folds_opt"],
        d["n_iter"],
        d["validation_method"],
        d["scoring_metric"],
        d["unsup_val_method"],
        d["random_seed"],
    )


def stage6_combos() -> str:
    """调用 run_pipeline 跑 3 个代表性组合，验证完整链路。"""
    import importlib

    gradio_app = importlib.import_module("src.gradio_app")
    run_pipeline = gradio_app.run_pipeline
    mock_progress = _MockProgress()

    combos = [
        {
            "label": "RF",
            "args": _build_pipeline_args(
                model_name="random_forest",
                features=["hog", "lbp"],
                optimization_strategy="manual",
            ),
        },
        {
            "label": "CNN",
            "args": _build_pipeline_args(
                model_name="cnn",
                cnn_epochs=2,
                cnn_batch_size=32,
                cnn_loss_fn="cross_entropy",
                optimization_strategy="manual",
            ),
        },
        {
            "label": "KM",
            "args": _build_pipeline_args(
                model_name="kmeans",
                features=["hog", "lbp", "glcm"],
                unsup_n_clusters=2,
                optimization_strategy="manual",
            ),
        },
    ]

    ok_count = 0
    details = []

    for combo in combos:
        label = combo["label"]
        print(f"  ▶ 组合 {label}...")
        try:
            result = run_pipeline(*combo["args"], progress=mock_progress)
            status_text = result[0]  # 第一个返回值是 status markdown

            # 检查是否有错误
            if "❌" in status_text:
                raise RuntimeError(f"run_pipeline 返回错误: {status_text[:200]}")

            details.append(f"{label}: OK")
            ok_count += 1
            print(f"    ✅ {label} 完成")
        except Exception as e:
            details.append(f"{label}: FAIL ({e})")
            print(f"    ❌ {label}: {e}")
            traceback.print_exc()

    assert ok_count == len(combos), f"仅 {ok_count}/{len(combos)} 组合通过"
    return f"{ok_count}/{len(combos)} combos OK ({', '.join(details)})"


# ===========================================================================
# 主入口
# ===========================================================================
def main():
    runner = StageRunner()
    runner.run("Stage 0: 环境检查", stage0_env)
    runner.run("Stage 1: 数据路径验证", stage1_data_path)
    runner.run("Stage 2: 特征提取验证", stage2_features)
    runner.run("Stage 3: 传统模型推理", stage3_trad_models)
    runner.run("Stage 4: CNN 推理", stage4_cnn)
    runner.run("Stage 5: Gradio 模块检查", stage5_gradio_module)
    runner.run("Stage 6: 方法组合运行", stage6_combos)

    all_pass = runner.report()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
