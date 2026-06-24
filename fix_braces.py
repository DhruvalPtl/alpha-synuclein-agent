import re
from pathlib import Path

path = Path("agent/tools/harness_template.py")
text = path.read_text(encoding="utf-8")

start_marker = "# ── Visualization ──────────────────────────────────────────────"

if start_marker in text:
    start_idx = text.find(start_marker)
    end_idx = text.find("'''", start_idx)
    
    viz_block = text[start_idx:end_idx]
    
    replacements = {
        "{val_f1:.4f}": "{{val_f1:.4f}}",
        "{_e}": "{{_e}}",
        "{_f1s[-1]:.4f}": "{{_f1s[-1]:.4f}}",
        "{_plots_dir}": "{{_plots_dir}}",
        "{_fams}": "{{_fams}}"
    }
    
    new_viz_block = viz_block
    for k, v in replacements.items():
        new_viz_block = new_viz_block.replace(k, v)
        
    text = text[:start_idx] + new_viz_block + text[end_idx:]
    path.write_text(text, encoding="utf-8")
    print("Fixed braces in harness_template.py")
else:
    print("Marker not found")
