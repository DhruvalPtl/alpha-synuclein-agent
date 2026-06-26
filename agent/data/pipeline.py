"""
agent/data/pipeline.py
────────────────────────────────────────────────────────────────────────────
DataPipeline — single entry-point for every data operation.

This V2 version does NOT pre-extract features. It passes the raw sequence
and concentration values as a pandas DataFrame directly to the models,
allowing them to handle variable-length sequences natively (e.g. using ESM,
LSTMs, or dynamic k-mer extraction).

Dataset layout (CSV on disk)
─────────────────────────────
• 65 rows   → one peptide per row
• 6+ columns → concentration columns
• One column named 'sequence'
• Cell values  = integer label  {0=No, 1=Low, 2=Medium, 3=High}

After load_and_expand():
• ~390 rows
• Columns: sequence, concentration, label_int, label_str
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

from sklearn.model_selection import StratifiedShuffleSplit
from agent.data.loader import load_raw_csv

# ── Label vocabulary ──────────────────────────────────────────────────────────
LABEL_MAP: Dict[int, str] = {
    0: "No",
    1: "Low",
    2: "Medium",
    3: "High",
}
INV_LABEL_MAP: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}

# ── Default paths ─────────────────────────────────────────────────────────────
_SPLITS_DIR     = Path("data/splits")
_HASH_FILE      = _SPLITS_DIR / "split_hash.sha256"
_TRAIN_PATH     = _SPLITS_DIR / "train.pkl"
_VAL_PATH       = _SPLITS_DIR / "val.pkl"
_TEST_PATH      = _SPLITS_DIR / "test.pkl"


class DataPipeline:
    def __init__(self, splits_dir: str = str(_SPLITS_DIR), random_state: int = 42) -> None:
        self.splits_dir   = Path(splits_dir)
        self.random_state = random_state

        # Split DataFrames
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val  : Optional[pd.DataFrame] = None
        self.df_test : Optional[pd.DataFrame] = None

        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def load_and_expand(self, csv_path: str) -> pd.DataFrame:
        """Read the wide-format CSV and melt into a long-format DataFrame."""
        try:
            df_out = load_raw_csv(csv_path)
            return df_out
        except Exception:
            pass

        df_raw = pd.read_csv(csv_path)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        seq_col = self._find_sequence_column(df_raw)
        conc_cols = [c for c in df_raw.columns if c != seq_col]
        
        df_long = df_raw[[seq_col] + conc_cols].melt(
            id_vars=seq_col, var_name="concentration_str", value_name="raw_value"
        )
        df_long = df_long.rename(columns={seq_col: "sequence"})
        df_long = df_long.dropna(subset=["raw_value"])

        df_long["concentration"] = df_long["concentration_str"].apply(self._parse_concentration)
        df_long["label_int"] = self._resolve_labels(df_long["raw_value"])
        df_long["label_str"] = df_long["label_int"].map(LABEL_MAP)

        return df_long[["sequence", "concentration", "label_int", "label_str"]].reset_index(drop=True)

    def stratified_split(self, df: pd.DataFrame, train: float = 0.70, val: float = 0.15, test: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train / val / test split on the dataframe."""
        assert abs(train + val + test - 1.0) < 1e-6

        X = df.index.values
        y = df["label_int"].values

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=self.random_state)
        train_val_idx, test_idx = next(sss_test.split(X, y))

        df_test = df.iloc[test_idx].copy()
        df_tv = df.iloc[train_val_idx].copy()
        y_tv = y[train_val_idx]

        val_fraction_of_tv = val / (train + val)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction_of_tv, random_state=self.random_state)
        train_idx, val_idx = next(sss_val.split(df_tv.index.values, y_tv))

        df_train = df_tv.iloc[train_idx].copy()
        df_val = df_tv.iloc[val_idx].copy()

        self.df_train, self.df_val, self.df_test = df_train, df_val, df_test
        return df_train, df_val, df_test

    def get_class_weights(self, df_train: pd.DataFrame) -> Dict[int, float]:
        """Compute balanced class weights for imbalanced training."""
        from sklearn.utils.class_weight import compute_class_weight
        y_train = df_train["label_int"].values
        classes = np.unique(y_train)
        raw_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        
        weight_dict = {int(c): float(w) for c, w in zip(classes, raw_w)}
        
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(processed_dir / "class_weights.pkl", "wb") as fh:
            pickle.dump(weight_dict, fh)
            
        return weight_dict

    def seal_test_set(self) -> str:
        """Compute SHA-256 of the test split pickle and write it to disk."""
        if self.df_test is None:
            raise RuntimeError("No test split available.")

        with open(_TEST_PATH, "wb") as fh:
            pickle.dump(self.df_test, fh)

        digest = self._sha256(_TEST_PATH)
        _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HASH_FILE.write_text(f"{digest}  {_TEST_PATH.name}\\n", encoding="utf-8")
        return digest

    def verify_wall(self) -> bool:
        if not _HASH_FILE.exists() or not _TEST_PATH.exists():
            return False
            
        sealed_digest = _HASH_FILE.read_text(encoding="utf-8").strip().split()[0]
        current_digest = self._sha256(_TEST_PATH)
        return current_digest == sealed_digest

    def save_splits(self) -> None:
        if self.df_train is None:
            raise RuntimeError("No splits to save.")
            
        with open(_TRAIN_PATH, "wb") as f:
            pickle.dump(self.df_train, f)
        with open(_VAL_PATH, "wb") as f:
            pickle.dump(self.df_val, f)
        with open(_TEST_PATH, "wb") as f:
            pickle.dump(self.df_test, f)

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with open(_TRAIN_PATH, "rb") as f:
            self.df_train = pickle.load(f)
        with open(_VAL_PATH, "rb") as f:
            self.df_val = pickle.load(f)
        with open(_TEST_PATH, "rb") as f:
            self.df_test = pickle.load(f)
            
        return self.df_train, self.df_val, self.df_test

    @staticmethod
    def _sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _find_sequence_column(df: pd.DataFrame) -> str:
        for candidate in ["sequence", "Sequence", "peptide", "Peptide"]:
            if candidate in df.columns:
                return candidate
        for col in df.columns:
            try:
                pd.to_numeric(df[col])
            except (ValueError, TypeError):
                return col
        return "sequence"

    @staticmethod
    def _parse_concentration(raw: str) -> float:
        import re
        s = re.sub(r"[µu]m.*", "", str(raw).strip().lower())
        s = re.sub(r"[^0-9.\-eE+]", "", s)
        try:
            return float(s)
        except ValueError:
            return 0.0

    @staticmethod
    def _resolve_labels(series: pd.Series) -> pd.Series:
        if pd.api.types.is_integer_dtype(series) or (pd.api.types.is_float_dtype(series) and series.dropna().isin([0.0, 1.0, 2.0, 3.0]).all()):
            return series.fillna(0).astype(int)
        sample = series.dropna().iloc[0] if len(series.dropna()) > 0 else 0
        if isinstance(sample, str):
            return series.map(lambda v: INV_LABEL_MAP.get(str(v).strip().capitalize(), 0))
        return pd.cut(series.fillna(series.median()), bins=4, labels=[0, 1, 2, 3]).astype(int)
