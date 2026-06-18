"""
agent/tools/audit_tool.py
────────────────────────────────────────────────────────────────────────────
Anti-cheat scanner — MUST run before any experiment executes.

Scans all generated code files for forbidden patterns that would leak
test-set information into training or validation metrics.

Forbidden patterns
------------------
1.  Loading test.pkl  or  test split file
2.  Accessing data/splits/test* paths
3.  Variable names starting with "test_" used in metric calls
4.  sklearn / torch metric functions called on "test*" variables
5.  Hardcoded ground-truth arrays that match test-set size (60 samples)

Usage
-----
    from agent.tools.audit_tool import AuditTool
    audit = AuditTool()
    result = audit.forward(code_block="... python code ...")
    # result: "PASS" or "FAIL: <reason>"
"""

import re
from typing import List, Tuple

try:
    from smolagents import Tool
    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False
    # Fallback base class so the module is importable even without smolagents
    class Tool:  # type: ignore[no-redef]
        pass

from agent.core.tee_logger import TeeLogger


# ── Forbidden pattern catalogue ───────────────────────────────────────────────

_FORBIDDEN: List[Tuple[str, str, bool]] = [
    # (regex_pattern, human_readable_reason, is_regex)

    # Direct file access
    (r"test\.pkl",
     "Loading 'test.pkl' directly", True),

    (r"['\"]data[\\/]splits[\\/]test",
     "Accessing data/splits/test* path", True),

    (r"load_splits\s*\(.*\)\s*(?:#.*)?$",
     "Calling load_splits() — only allowed in eval_final.py", True),

    # Variable-based leakage
    (r"\bX_test\b",
     "Variable 'X_test' found — test features must never appear in experiment code", True),

    (r"\by_test\b",
     "Variable 'y_test' found — test labels must never appear in experiment code", True),

    # Metric computation on test-named variables
    (
        r"(?:accuracy_score|f1_score|classification_report|confusion_matrix"
        r"|precision_score|recall_score|roc_auc_score)\s*\("
        r"\s*(?:y_test|test_labels|test_targets|labels_test)",
        "Metric function called on test-set labels", True,
    ),

    # Torch eval on test loader
    (r"\btest_loader\b",
     "Variable 'test_loader' found — test DataLoader forbidden", True),

    (r"\btest_dataset\b",
     "Variable 'test_dataset' found — test Dataset forbidden", True),

    # Hardcoded suspiciously-sized label arrays (exact test size = 60)
    (
        r"\[\s*(?:\d+\s*,\s*){59}\d+\s*\]",
        "Hardcoded array of exactly 60 elements — may be test labels", True,
    ),

    # Direct pickle load of any split file
    (r"pickle\.load.*test",
     "pickle.load on a 'test*' variable or path", True),

    # numpy load of test file
    (r"np\.load.*test(?:\.pkl|\.npy|\.npz)",
     "np.load of test file", True),
]


class AuditTool(Tool if _SMOLAGENTS_AVAILABLE else object):  # type: ignore[misc]
    """
    Smolagents Tool that scans generated ML code for test-set leakage.

    Call before executing any experiment to ensure the mathematical wall
    is respected.
    """

    name        = "audit_code"
    description = (
        "Anti-cheat scanner. Checks Python code for any forbidden patterns "
        "that would leak test-set information. "
        "Call this BEFORE run_experiment. "
        "Input: a string containing the full Python code to audit. "
        "Returns 'PASS' or 'FAIL: <reason>'."
    )
    inputs = {
        "code_block": {
            "type": "string",
            "description": (
                "The complete Python code to audit. "
                "Can be the concatenation of model.py + train.py + eval.py."
            ),
        }
    }
    output_type = "string"

    def __init__(self) -> None:
        if _SMOLAGENTS_AVAILABLE:
            super().__init__()
        self.logger = TeeLogger()

    def forward(self, code_block: str) -> str:  # noqa: D102
        """
        Scan *code_block* for all forbidden patterns.

        Returns
        -------
        "PASS"               — no violations found
        "FAIL: <reason>"     — first violation found (scanning continues for log)
        """
        failures: List[str] = []
        lines = code_block.splitlines()

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Skip comment lines (they can contain the words "test" legitimately)
            if stripped.startswith("#"):
                continue

            for pattern, reason, is_regex in _FORBIDDEN:
                try:
                    if is_regex:
                        match = re.search(pattern, line, re.IGNORECASE)
                    else:
                        match = pattern.lower() in line.lower()

                    if match:
                        msg = f"Line {i}: {reason}  |  >> {line.strip()[:120]}"
                        failures.append(msg)
                        self.logger.warning(f"[AUDIT FAIL] {msg}")
                        break   # one failure per line is enough
                except re.error:
                    pass

        if not failures:
            self.logger.info("[AUDIT] PASS — no forbidden patterns detected.")
            return "PASS"

        summary = f"FAIL: {len(failures)} violation(s) found:\n"
        summary += "\n".join(f"  • {f}" for f in failures)
        self.logger.error(f"[AUDIT] {summary}")
        return summary

    # ── Convenience: scan a directory of .py files ──────────────────────────────
    def audit_directory(self, directory: str) -> str:
        """Scan all .py files in *directory* and return consolidated result."""
        from pathlib import Path
        all_code = []
        for py_file in sorted(Path(directory).glob("*.py")):
            all_code.append(f"# === {py_file.name} ===")
            all_code.append(py_file.read_text(encoding="utf-8", errors="replace"))
        return self.forward("\n".join(all_code))
