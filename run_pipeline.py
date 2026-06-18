import sys, numpy as np
sys.path.insert(0, '.')

from agent.core.tee_logger import TeeLogger
from agent.data.pipeline import DataPipeline

logger = TeeLogger(master_log_dir='master_log')
logger.info('=== Phase 1: Full Pipeline Run ===')

pipe = DataPipeline(splits_dir='data/splits')

# Step 1: Load & expand
logger.info('Loading and expanding CSV ...')
df = pipe.load_and_expand('data/raw/alpha_synuclein.csv')
logger.info(f'Expanded shape: {df.shape}')
print()
print('--- Sample rows ---')
print(df.head(8).to_string(index=False))
print()
print('--- Class distribution ---')
print(df['label_str'].value_counts().sort_index().to_string())
print()
print('--- Concentration values ---')
print(sorted(df['concentration'].unique().tolist()))

# Step 2: Build features
logger.info('Engineering features (2-mer + 3-mer) ...')
X, y = pipe.build_features(df)
logger.info(f'Feature matrix: X={X.shape}  y={y.shape}  dtype={X.dtype}')

# Step 3: Stratified split
logger.info('Stratified split 70/15/15 ...')
pipe.stratified_split(X, y, train=0.70, val=0.15, test=0.15)

# Step 4: Save
logger.info('Saving splits ...')
pipe.save_splits()

# Step 5: Seal
logger.info('Sealing mathematical wall ...')
pipe.seal_test_set()

# Step 6: Verify
logger.info('Verifying wall ...')
ok = pipe.verify_wall()

print()
print('=' * 42)
print(f'Total samples      : {len(X)}')
print(f'Feature dimensions : {X.shape[1]}')
unique, counts = np.unique(y, return_counts=True)
label_names = {0: 'No', 1: 'Low', 2: 'Medium', 3: 'High'}
for u, c in zip(unique.tolist(), counts.tolist()):
    print(f'  Class {u} ({label_names[u]:<6}) : {c:3d} samples')
print(f'Wall intact        : {ok}')
print('=' * 42)
logger.info('=== Phase 1 COMPLETE ===')
