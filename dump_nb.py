import json

nb = json.load(open("notebooks/run_agent.ipynb", encoding="utf-8"))

FULL_CELLS = [1, 5]   # bootstrap + launch
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    ctype = cell["cell_type"]
    if i in FULL_CELLS:
        print(f"\n{'='*70}")
        print(f"CELL {i} [{ctype}] — FULL ({len(src)} chars):")
        print(src)
