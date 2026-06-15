"""Phase 1.1: Environment verification script."""
import sys
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')

# Test all critical imports
imports_ok = []
imports_fail = []

modules = {
    'numpy': 'np',
    'pandas': 'pd',
    'sklearn': 'sklearn',
    'cv2': 'cv2 (opencv)',
    'torch': 'torch',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'skimage': 'skimage',
    'matplotlib': 'matplotlib',
    'seaborn': 'sns',
    'dotenv': 'dotenv',
    'PIL': 'pillow',
    'scipy': 'scipy',
}

for mod, name in modules.items():
    try:
        __import__(mod)
        imports_ok.append(name)
    except Exception as e:
        imports_fail.append(f'{name}: {e}')

print(f'\nSuccessful imports ({len(imports_ok)}): {imports_ok}')
if imports_fail:
    print(f'FAILED imports ({len(imports_fail)}): {imports_fail}')
else:
    print('All imports successful!')

# Check PyTorch CUDA
import torch
print(f'\nPyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Running on CPU')

# Check key package versions
import sklearn
print(f'scikit-learn version: {sklearn.__version__}')
import cv2
print(f'OpenCV version: {cv2.__version__}')
try:
    import xgboost
    print(f'XGBoost version: {xgboost.__version__}')
except Exception:
    print('XGBoost not available')
try:
    import lightgbm
    print(f'LightGBM version: {lightgbm.__version__}')
except Exception:
    print('LightGBM not available')
