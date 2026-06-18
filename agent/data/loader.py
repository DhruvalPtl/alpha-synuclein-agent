"""
agent/data/loader.py
────────────────────────────────────────────────────────────────────────────
Raw-format CSV loader for the alpha-synuclein dataset.

The CSV has a non-standard layout:
  Row 0 : title line
  Row 1 : RFU legend
  Row 2 : "X is K-acetylated" note
  Rows 3-9: multi-line column header (split across lines by Excel-style wrapping)
  Row 10+: data rows — Sr No. | Peptide sequence | 6× label columns

Concentration columns (mg/ml): 0.1, 0.25, 0.5, 1, 2, 4

This module exposes a single function:
    load_raw_csv(path) → pd.DataFrame
        columns: sequence, concentration, label_int, label_str
"""

import re
import io
import pandas as pd
from pathlib import Path
from typing import List

# Concentration column values in the same order as the CSV (mg/ml)
CONCENTRATIONS: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

# Label mapping
LABEL_MAP = {"no": 0, "low": 1, "medium": 2, "high": 3}


def load_raw_csv(path: str) -> pd.DataFrame:
    """
    Parse the alpha-synuclein CSV (with its 10-row preamble) into a clean
    long-format DataFrame ready for feature engineering.

    Parameters
    ----------
    path : str — path to alpha_synuclein.csv

    Returns
    -------
    pd.DataFrame with columns:
        sequence      str    peptide amino-acid sequence
        concentration float  µg/ml value (0.1 … 4.0 mg/ml expressed as float)
        label_int     int    {0, 1, 2, 3}
        label_str     str    {"No", "Low", "Medium", "High"}
    """
    # ── Read the file as raw lines to side-step the messy header ─────────────
    raw = Path(path).read_bytes().decode("utf-8-sig", errors="replace")
    lines = raw.splitlines()

    # ── Find the first actual data row ────────────────────────────────────────
    # Data rows start with a number followed by a dot (e.g. "1.", "10.")
    data_lines = [l for l in lines if re.match(r"^\d+[.,]", l.strip())]

    if not data_lines:
        raise ValueError(
            f"No data rows found in {path}. "
            "Expected rows starting with 'Sr No.' like '1.,SEQUENCE,...'"
        )

    # ── Parse each data row ───────────────────────────────────────────────────
    records = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        # parts[0] = Sr No., parts[1] = sequence, parts[2..7] = 6 labels
        if len(parts) < 8:
            continue                  # skip malformed rows
        sequence = parts[1].strip().upper()
        if not sequence:
            continue

        for i, conc in enumerate(CONCENTRATIONS):
            raw_label = parts[2 + i].strip().capitalize() if (2 + i) < len(parts) else "No"
            raw_label = raw_label if raw_label else "No"
            label_int = LABEL_MAP.get(raw_label.lower(), 0)
            label_str = ["No", "Low", "Medium", "High"][label_int]

            records.append({
                "sequence":      sequence,
                "concentration": conc,
                "label_int":     label_int,
                "label_str":     label_str,
            })

    df = pd.DataFrame(records)
    return df.reset_index(drop=True)
