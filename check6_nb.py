import json
from pathlib import Path

nb = json.loads(Path("notebooks/run_agent.ipynb").read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"Cell count: {len(cells)}")
print()
for i, cell in enumerate(cells):
    src = cell["source"]
    comments = [l.strip() for l in src if l.strip().startswith("#")]
    code_lines = [l.rstrip() for l in src if l.strip() and not l.strip().startswith("#")]
    first_comment = comments[0] if comments else "(no comment)"
    first_code    = code_lines[0] if code_lines else "(empty or comments only)"
    cell_id = cell.get("id", "?")
    print(f"Cell {i+1}  id={cell_id}")
    print(f"  Banner  : {first_comment[:80]}")
    print(f"  1st code: {first_code[:100]}")
    print()
