import sys, textwrap
sys.path.insert(0, '.')

print('--- Test 1: System prompt ---')
from agent.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_SHORT
print(f'  SYSTEM_PROMPT      : {len(SYSTEM_PROMPT):,} chars')
print(f'  SYSTEM_PROMPT_SHORT: {len(SYSTEM_PROMPT_SHORT):,} chars')
assert 'val_f1_macro' in SYSTEM_PROMPT
assert 'class_weights' in SYSTEM_PROMPT
assert 'NEVER load test.pkl' in SYSTEM_PROMPT
assert 'TIER 1' in SYSTEM_PROMPT
assert 'TIER 10' in SYSTEM_PROMPT
print('  All required content present.')

print()
print('--- Test 2: ArxivTool import ---')
from agent.tools.arxiv_tool import ArxivTool
t = ArxivTool()
print(f'  ArxivTool name: {t.name}')

print()
print('--- Test 3: Pipeline.reduce_features ---')
import numpy as np
from agent.data.pipeline import DataPipeline
pipe = DataPipeline(splits_dir='data/splits')
rng = np.random.default_rng(42)
X_train = rng.random((276, 8427), dtype='float64').astype('float32')
X_val   = rng.random((60, 8427), dtype='float64').astype('float32')
X_test  = rng.random((60, 8427), dtype='float64').astype('float32')
y_train = np.array([0]*193 + [1]*10 + [2]*37 + [3]*36)
rng.shuffle(y_train)

Xtr, Xv, Xte = pipe.reduce_features(X_train, X_val, X_test, y_train, k_best=500)
print(f'  reduce_features: 8427 -> {Xtr.shape[1]} features')
assert Xtr.shape[1] == 500
assert Xv.shape[1]  == 500
assert Xte.shape[1] == 500

from pathlib import Path
assert Path('data/processed/scaler.pkl').exists()
assert Path('data/processed/selector.pkl').exists()
print('  Artifacts saved: scaler.pkl, selector.pkl, variance_threshold.pkl')

print()
print('--- Test 4: Pipeline.get_class_weights ---')
cw = pipe.get_class_weights(y_train)
print(f'  Weights: {cw}')
assert len(cw) == 4
assert Path('data/processed/class_weights.pkl').exists()

print()
print('--- Test 5: AuditTool ---')
from agent.tools.audit_tool import AuditTool
audit = AuditTool()

clean = 'X_train, y_train = d["X"], d["y"]'
r1 = audit.forward(clean)
assert r1 == 'PASS', f'Expected PASS, got {r1}'
print(f'  Clean code: {r1}')

dirty = 'X_test, y_test = d["X"], d["y"]'
r2 = audit.forward(dirty)
assert r2.startswith('FAIL'), f'Expected FAIL, got {r2}'
print(f'  Dirty code: {r2[:60]}...')

print()
print('ALL PHASE 3 SMOKE TESTS PASSED')
