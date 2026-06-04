"""
基础冒烟测试
验证项目核心模块能正常导入，模型能正常实例化。
运行方式：pytest tests/ -v
"""

import torch


def test_import_config():
    """验证 src.config 模块可正常导入。"""
    from src.config import DATA_ROOT, DEVICE, PROJECT_ROOT

    assert DATA_ROOT is not None
    assert DEVICE is not None
    assert PROJECT_ROOT.exists()


def test_import_plot_config():
    """验证 src.plot_config 模块可正常导入。"""
    from src.plot_config import set_chinese_font

    assert callable(set_chinese_font)


def test_import_data_utils():
    """验证 src.data_utils 模块可正常导入（函数签名存在）。"""
    from src import data_utils

    assert hasattr(data_utils, "apply_clahe")
    assert hasattr(data_utils, "extract_hog_features")
    assert hasattr(data_utils, "load_dataset")
    assert hasattr(data_utils, "split_dataset")


def test_import_traditional_models():
    """验证传统机器学习模型可正常导入和实例化。"""
    from src.models.traditional import (
        DecisionTreeClassifier,
        KNNClassifier,
        SVMClassifier,
    )

    dt = DecisionTreeClassifier()
    svm = SVMClassifier()
    knn = KNNClassifier()
    assert dt.model_type == "decision_tree"
    assert svm.model_type == "svm"
    assert knn.model_type == "knn"


def test_import_unsupervised_models():
    """验证无监督学习模型可正常导入和实例化。"""
    from src.models.unsupervised import (
        DBSCANClusterer,
        GMMClusterer,
        KMeansClusterer,
        UnsupervisedPipeline,
    )

    km = KMeansClusterer(n_clusters=2)
    gmm = GMMClusterer(n_components=2)
    db = DBSCANClusterer(eps=0.5)
    pipeline = UnsupervisedPipeline()
    assert km.n_clusters == 2
    assert gmm.n_components == 2
    assert db.eps == 0.5
    assert "kmeans" in pipeline.METHODS


def test_crack_cnn_instantiation():
    """验证 CrackCNN 模型可正常实例化并前向传播。"""
    from src.models.cnn import CrackCNN, get_cnn_model

    model = CrackCNN(num_classes=2, input_channels=1)
    assert model.num_classes == 2

    # 验证前向传播（使用随机输入）
    dummy_input = torch.randn(2, 1, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 2)

    # 验证 get_cnn_model 辅助函数
    model2 = get_cnn_model(num_classes=2, input_channels=1)
    assert isinstance(model2, CrackCNN)


def test_crack_cnn_param_count():
    """验证 CrackCNN 参数量在设计约束范围内（占位实现会很小，正式实现后应满足 500K-2M）。"""
    from src.models.cnn import CrackCNN

    model = CrackCNN(num_classes=2, input_channels=1)
    total_params = sum(p.numel() for p in model.parameters())
    # 占位实现参数量很小，正式实现后请调整此断言为：
    # assert 500_000 <= total_params <= 2_000_000
    assert total_params > 0
    print(f"CrackCNN 参数量: {total_params:,}")


def test_import_metrics():
    """验证评价指标模块可正常导入。"""
    from src.evaluation.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    assert callable(accuracy_score)
    assert callable(f1_score)
    assert callable(confusion_matrix)
    assert callable(classification_report)


def test_import_losses():
    """验证损失函数模块可正常导入。"""
    from src.training.losses import get_cross_entropy_loss

    loss_fn = get_cross_entropy_loss()
    assert loss_fn is not None
