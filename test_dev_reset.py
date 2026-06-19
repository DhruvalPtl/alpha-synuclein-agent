"""test_dev_reset.py — verify dev_reset() works and wall survives."""
import sys, os
sys.path.insert(0, '.')
from pathlib import Path
from agent.tools.dev_reset import dev_reset

# ── 1. Record wall hash BEFORE reset ──────────────────────────────────────────
hash_file = Path('data/splits/split_hash.sha256')
hash_before = hash_file.read_text().strip().split()[0] if hash_file.exists() else None
print(f"Hash before: {hash_before}")

# ── 2. Run reset ───────────────────────────────────────────────────────────────
result = dev_reset('.')

# ── 3. Check experiments/ is empty ────────────────────────────────────────────
exp_items = [p for p in Path('experiments').iterdir() if p.name != '.gitkeep']
print(f"\nexperiments/ non-.gitkeep items: {len(exp_items)}")
assert len(exp_items) == 0, f"Expected 0 items, got: {exp_items}"
print("PASS  experiments/ empty")

# ── 4. Check sessions/ is empty ───────────────────────────────────────────────
ses_items = [p for p in Path('sessions').iterdir() if p.name != '.gitkeep']
print(f"sessions/ non-.gitkeep items:    {len(ses_items)}")
assert len(ses_items) == 0, f"Expected 0, got: {ses_items}"
print("PASS  sessions/ empty")

# ── 5. Check leaderboard.json was recreated as empty ─────────────────────────
import json
lb = json.loads(Path('master_log/leaderboard.json').read_text())
assert lb['total_runs'] == 0
print(f"PASS  leaderboard.json reset (total_runs={lb['total_runs']})")

# ── 6. Wall hash unchanged ────────────────────────────────────────────────────
assert result['wall_intact'], "Wall integrity FAILED!"
hash_after = result['wall_hash']
if hash_before:
    assert hash_before == hash_after, (
        f"Hash changed! before={hash_before[:16]} after={hash_after[:16]}"
    )
print(f"PASS  Wall intact — hash unchanged: {hash_after[:32]}...")

# ── 7. Protected dirs untouched ───────────────────────────────────────────────
for protected in ['data/raw', 'data/processed', 'data/splits', 'agent', '.env']:
    p = Path(protected)
    assert p.exists(), f"Protected path was deleted: {protected}"
print("PASS  All protected paths intact")

print()
print("ALL DEV_RESET TESTS PASSED")
