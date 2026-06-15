"""Phase 1.1: Data path verification script."""
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
print(f'Project root: {PROJECT_ROOT}')
load_dotenv(PROJECT_ROOT / '.env')

DATA_ROOT = os.getenv('CRACK_DATA_ROOT')
print(f'CRACK_DATA_ROOT from .env: {DATA_ROOT}')

data_root = Path(DATA_ROOT).expanduser()
if not data_root.is_absolute():
    data_root = PROJECT_ROOT / data_root
data_root = data_root.resolve()
print(f'Resolved data root: {data_root}')
print(f'Data root exists: {data_root.exists()}')

pos_dir = data_root / 'Positive'
neg_dir = data_root / 'Negative'
print(f'Positive dir exists: {pos_dir.exists()}')
print(f'Negative dir exists: {neg_dir.exists()}')

if pos_dir.exists():
    pos_count = len(list(pos_dir.glob('*.jpg')))
    print(f'Positive images: {pos_count}')
if neg_dir.exists():
    neg_count = len(list(neg_dir.glob('*.jpg')))
    print(f'Negative images: {neg_count}')

real_test = data_root / 'real_test'
print(f'real_test dir exists: {real_test.exists()}')
if real_test.exists():
    real_imgs = list(real_test.glob('*.jpg')) + list(real_test.glob('*.png'))
    print(f'real_test images: {len(real_imgs)}')
    for img in real_imgs[:5]:
        print(f'  - {img.name}')
