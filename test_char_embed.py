import numpy as np
from agent.tools.char_embedding_template import build_and_train, encode_sequence, AA_VOCAB

# Test vocab
print(f"Vocab size: {len(AA_VOCAB)} AA tokens (+PAD=0 => 22 total)")
print(f"encode_sequence(MKQLEDKVEELLSK) => {encode_sequence('MKQLEDKVEELLSK')}")

# Test model build + predict with tiny dataset
rng = np.random.default_rng(0)
X_tr = rng.standard_normal((40, 189)).astype(np.float32)
y_tr = np.array([0,1,2,3]*10)
X_v  = rng.standard_normal((12, 189)).astype(np.float32)
y_v  = np.array([0,1,2,3,0,1,2,3,0,1,2,3])
cw   = {0: 0.32, 1: 5.75, 2: 2.76, 3: 3.14}

print("Building and training RoPEAttentionPeptide (proxy token mode)...")
model = build_and_train(X_tr, y_tr, X_v, y_v, cw)
preds = model.predict(X_v)
print(f"Predictions : {preds}  shape={preds.shape}")
print(f"Unique classes: {sorted(set(preds.tolist()))}")
print("Full smoke test PASSED")
