"""
agent/data/pipeline.py
────────────────────────────────────────────────────────────────────────────
DataPipeline — single entry-point for every data operation.

Dataset layout (CSV on disk)
─────────────────────────────
• 65 rows   → one peptide per row
• 6+ columns → concentration columns (headers = numeric µM values, e.g. 0.5,1,2,5,10,20)
• One column named 'sequence' (or first non-numeric column)
• Cell values  = integer label  {0=No, 1=Low, 2=Medium, 3=High}

After load_and_expand():
• ~390 rows (65 peptides × 6 concentrations)
• Columns: sequence, concentration, label_int, label_str

Feature vector (per row)
─────────────────────────
  aa_composition     20  dims   (normalised)
  physicochemical     6  dims   (charge, hydrophob, MW, arom, pI, instab)
  2-mer frequencies 400  dims   (normalised)
  3-mer frequencies 8000 dims   (normalised)
  concentration       1  dim    (log10-scaled to [0,1])
  ─────────────────────────────
  TOTAL            8427  dims

Mathematical Wall
─────────────────
call pipe.seal_test_set()  →  SHA-256 hash of test.pkl written to
                               data/splits/split_hash.sha256
call pipe.verify_wall()    →  True / False (recomputes and compares)
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

from sklearn.model_selection import StratifiedShuffleSplit

from agent.data import features as F

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
_META_PATH      = _SPLITS_DIR / "split_meta.pkl"


class DataPipeline:
    """
    Stateful pipeline: load → engineer features → split → seal.

    Example
    -------
        pipe = DataPipeline()
        df   = pipe.load_and_expand("data/raw/alpha_synuclein.csv")
        X, y = pipe.build_features(df)
        pipe.save_splits()
        pipe.seal_test_set()
        ok = pipe.verify_wall()
    """

    def __init__(
        self,
        splits_dir: str = str(_SPLITS_DIR),
        random_state: int = 42,
    ) -> None:
        self.splits_dir   = Path(splits_dir)
        self.random_state = random_state

        # Split tensors — populated by stratified_split()
        self.X_train: Optional[np.ndarray] = None
        self.X_val  : Optional[np.ndarray] = None
        self.X_test : Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val  : Optional[np.ndarray] = None
        self.y_test : Optional[np.ndarray] = None

        # Concentration range — fitted during build_features()
        self._conc_log_min: float = -2.0
        self._conc_log_max: float =  2.0

        # 2-mer / 3-mer key order (fixed after first build_features call)
        self._2mer_keys = F._2MER_KEYS
        self._3mer_keys = F._3MER_KEYS

        self.splits_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & expand ──────────────────────────────────────────────────────

    def load_and_expand(self, csv_path: str) -> pd.DataFrame:
        """
        Read the wide-format CSV and melt into a long-format DataFrame.

        Expected CSV format
        -------------------
        One row per peptide.  Columns:
          • 'sequence'  (or first string column)  — peptide amino-acid sequence
          • All other columns                      — concentration values (str/float)
            with cell values = int label {0,1,2,3}  **OR**  a float measurement
            that will be binned into 4 quantile classes.

        Returns
        -------
        pd.DataFrame with columns:
            sequence      str   peptide sequence
            concentration float numeric µM value of the column header
            label_int     int   {0,1,2,3}
            label_str     str   {"No","Low","Medium","High"}
        """
        df_raw = pd.read_csv(csv_path)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        # ── Identify sequence column ──────────────────────────────────────────
        seq_col = self._find_sequence_column(df_raw)

        # ── Identify concentration columns ───────────────────────────────────
        conc_cols = [c for c in df_raw.columns if c != seq_col]
        if not conc_cols:
            raise ValueError(
                f"No concentration columns found.  Columns present: {list(df_raw.columns)}"
            )

        # ── Melt to long format ───────────────────────────────────────────────
        df_long = df_raw[[seq_col] + conc_cols].melt(
            id_vars=seq_col,
            var_name="concentration_str",
            value_name="raw_value",
        )
        df_long = df_long.rename(columns={seq_col: "sequence"})
        df_long = df_long.dropna(subset=["raw_value"])

        # ── Parse concentration header → float µM ────────────────────────────
        df_long["concentration"] = df_long["concentration_str"].apply(
            self._parse_concentration
        )

        # ── Determine label_int ──────────────────────────────────────────────
        df_long["label_int"] = self._resolve_labels(df_long["raw_value"])
        df_long["label_str"] = df_long["label_int"].map(LABEL_MAP)

        # ── Drop helper columns; reset index ─────────────────────────────────
        df_out = (
            df_long[["sequence", "concentration", "label_int", "label_str"]]
            .reset_index(drop=True)
        )

        # ── Fit concentration range for log-encoding ─────────────────────────
        valid_conc = df_out["concentration"].replace(0, np.nan).dropna()
        if len(valid_conc) > 0:
            import math
            log_vals          = np.log10(valid_conc.values.astype(float))
            self._conc_log_min = float(log_vals.min())
            self._conc_log_max = float(log_vals.max())
            # Avoid zero-division if all concentrations are identical
            if abs(self._conc_log_max - self._conc_log_min) < 1e-9:
                self._conc_log_min -= 1.0
                self._conc_log_max += 1.0

        return df_out

    # ── 2. Build features ─────────────────────────────────────────────────────

    def build_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer the full feature matrix X and label vector y.

        Feature layout (per row)
        ─────────────────────────
          aa_composition     [20]   normalised composition
          physicochemical     [6]   charge/hydrophob/MW/arom/pI/instab
          2-mer frequencies [400]   normalised 2-mer counts
          3-mer frequencies [8000]  normalised 3-mer counts
          concentration       [1]   log10-scaled to [0,1]
          ────────────────────────────────────────────────
          TOTAL             [8427]

        Parameters
        ----------
        df : pd.DataFrame  — output of load_and_expand()

        Returns
        -------
        X : np.ndarray  shape (N, 8427)
        y : np.ndarray  shape (N,)  dtype int
        """
        rows = []
        for _, row in df.iterrows():
            seq  = str(row["sequence"])
            conc = float(row["concentration"])

            aa_comp  = F.amino_acid_composition(seq)          # (20,)
            physchem = F.physicochemical_features(seq)         # (6,)

            mer2_dict = F.kmer_frequencies(seq, k=2)
            mer3_dict = F.kmer_frequencies(seq, k=3)
            mer2 = np.array([mer2_dict[k] for k in self._2mer_keys])   # (400,)
            mer3 = np.array([mer3_dict[k] for k in self._3mer_keys])   # (8000,)

            conc_enc = np.array([
                F.encode_concentration(conc, self._conc_log_min, self._conc_log_max)
            ])  # (1,)

            feat = np.concatenate([aa_comp, physchem, mer2, mer3, conc_enc])
            rows.append(feat)

        X = np.vstack(rows).astype(np.float32)
        y = df["label_int"].values.astype(np.int64)
        return X, y

    # ── 3. Stratified split ───────────────────────────────────────────────────

    def stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train: float = 0.70,
        val:   float = 0.15,
        test:  float = 0.15,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Stratified train / val / test split.

        Parameters
        ----------
        X, y   : full feature matrix and labels
        train  : fraction for training   (default 0.70)
        val    : fraction for validation  (default 0.15)
        test   : fraction for test        (default 0.15)

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        assert abs(train + val + test - 1.0) < 1e-6, \
            f"Fractions must sum to 1.0; got {train + val + test:.4f}"

        # ── Step 1: carve out the sealed test set ─────────────────────────────
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test,
            random_state=self.random_state,
        )
        train_val_idx, test_idx = next(sss_test.split(X, y))

        X_test  = X[test_idx]
        y_test  = y[test_idx]
        X_tv    = X[train_val_idx]
        y_tv    = y[train_val_idx]

        # ── Step 2: split train/val from the remaining set ────────────────────
        val_fraction_of_tv = val / (train + val)
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction_of_tv,
            random_state=self.random_state,
        )
        train_idx, val_idx = next(sss_val.split(X_tv, y_tv))

        X_train = X_tv[train_idx]
        y_train = y_tv[train_idx]
        X_val   = X_tv[val_idx]
        y_val   = y_tv[val_idx]

        # ── Store for later save / seal ───────────────────────────────────────
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── 4. Seal the mathematical wall ────────────────────────────────────────

    def seal_test_set(self, test_path: Optional[str] = None) -> str:
        """
        Compute SHA-256 of the test split pickle and write it to disk.

        The test set must have been created by stratified_split() or
        loaded via load_splits() before calling this method.

        Parameters
        ----------
        test_path : str, optional
            Override the default test.pkl path.

        Returns
        -------
        str — the hex SHA-256 digest
        """
        if self.X_test is None:
            raise RuntimeError(
                "No test split available.  Call stratified_split() or "
                "load_splits() first."
            )

        path = Path(test_path) if test_path else _TEST_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        # Persist test.pkl
        with open(path, "wb") as fh:
            pickle.dump({"X_test": self.X_test, "y_test": self.y_test}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)

        digest = self._sha256(path)

        # Write hash file
        _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HASH_FILE.write_text(f"{digest}  {path.name}\n", encoding="utf-8")

        print("═" * 60)
        print("  ███ MATHEMATICAL WALL SEALED ███")
        print("═" * 60)
        print(f"  HASH : {digest}")
        print(f"  FILE : {path.resolve()}")
        print(f"  LOCK : {_HASH_FILE.resolve()}")
        print("═" * 60)
        print("  Test set is now LOCKED.")
        print("  Never load outside eval_final.py")
        print("═" * 60)

        return digest

    # ── 5. Verify the wall ───────────────────────────────────────────────────

    def verify_wall(self, test_path: Optional[str] = None) -> bool:
        """
        Recompute the SHA-256 of test.pkl and compare with the sealed hash.

        Returns
        -------
        bool — True if hashes match (wall is intact)
        """
        path = Path(test_path) if test_path else _TEST_PATH

        if not _HASH_FILE.exists():
            print("[verify_wall] ✗  Hash file not found. Call seal_test_set() first.")
            return False
        if not path.exists():
            print(f"[verify_wall] ✗  Test pickle not found at {path}")
            return False

        sealed_line   = _HASH_FILE.read_text(encoding="utf-8").strip()
        sealed_digest = sealed_line.split()[0]
        current_digest = self._sha256(path)

        if current_digest == sealed_digest:
            print("═" * 60)
            print("  ✓  WALL INTACT — hashes match")
            print(f"  HASH : {current_digest}")
            print("═" * 60)
            return True
        else:
            print("═" * 60)
            print("  ✗  WALL BREACH DETECTED — hashes differ!")
            print(f"  SEALED  : {sealed_digest}")
            print(f"  CURRENT : {current_digest}")
            print("═" * 60)
            return False

    # ── 6. Save splits ────────────────────────────────────────────────────────

    def save_splits(self) -> None:
        """
        Persist all six split arrays and pipeline metadata to disk.

        Files written
        -------------
        data/splits/train.pkl
        data/splits/val.pkl
        data/splits/test.pkl
        data/splits/split_meta.pkl   (concentration range, kmer key order)
        """
        if self.X_train is None:
            raise RuntimeError(
                "No splits to save. Run stratified_split() first."
            )
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        with open(_TRAIN_PATH, "wb") as f:
            pickle.dump({"X": self.X_train, "y": self.y_train}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(_VAL_PATH, "wb") as f:
            pickle.dump({"X": self.X_val, "y": self.y_val}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(_TEST_PATH, "wb") as f:
            pickle.dump({"X": self.X_test, "y": self.y_test}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            "conc_log_min": self._conc_log_min,
            "conc_log_max": self._conc_log_max,
            "2mer_keys":    self._2mer_keys,
            "3mer_keys":    self._3mer_keys,
            "random_state": self.random_state,
        }
        with open(_META_PATH, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"[DataPipeline] Splits saved to {self.splits_dir.resolve()}\n"
            f"  train : X={self.X_train.shape}  y={self.y_train.shape}\n"
            f"  val   : X={self.X_val.shape}  y={self.y_val.shape}\n"
            f"  test  : X={self.X_test.shape}  y={self.y_test.shape}"
        )

    # ── 7. Load splits ────────────────────────────────────────────────────────

    def load_splits(
        self,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Restore split arrays from disk and repopulate self.X_* / self.y_*.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        for path, label in [(_TRAIN_PATH, "train"), (_VAL_PATH, "val"),
                             (_TEST_PATH, "test")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Split file not found: {path}.  Run save_splits() first."
                )

        with open(_TRAIN_PATH, "rb") as f:
            d = pickle.load(f);  self.X_train, self.y_train = d["X"], d["y"]
        with open(_VAL_PATH, "rb") as f:
            d = pickle.load(f);  self.X_val,   self.y_val   = d["X"], d["y"]
        with open(_TEST_PATH, "rb") as f:
            d = pickle.load(f);  self.X_test,  self.y_test  = d["X"], d["y"]

        if _META_PATH.exists():
            with open(_META_PATH, "rb") as f:
                meta = pickle.load(f)
            self._conc_log_min = meta["conc_log_min"]
            self._conc_log_max = meta["conc_log_max"]
            self._2mer_keys    = meta["2mer_keys"]
            self._3mer_keys    = meta["3mer_keys"]

        print(
            f"[DataPipeline] Splits loaded from {self.splits_dir.resolve()}\n"
            f"  train : X={self.X_train.shape}  y={self.y_train.shape}\n"
            f"  val   : X={self.X_val.shape}  y={self.y_val.shape}\n"
            f"  test  : X={self.X_test.shape}  y={self.y_test.shape}"
        )

        return (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _find_sequence_column(df: pd.DataFrame) -> str:
        """Return the name of the sequence column."""
        # Check common explicit names first
        for candidate in ["sequence", "Sequence", "peptide", "Peptide",
                          "seq", "Seq", "aa_sequence"]:
            if candidate in df.columns:
                return candidate
        # Fallback: first column that contains only strings (non-numeric)
        for col in df.columns:
            try:
                pd.to_numeric(df[col])
            except (ValueError, TypeError):
                return col
        raise ValueError(
            "Could not identify the sequence column. "
            "Please ensure one column is named 'sequence'."
        )

    @staticmethod
    def _parse_concentration(raw: str) -> float:
        """
        Parse a concentration column header to a float µM value.
        Handles formats like '0.5', '1uM', '2µM', '5 uM', '10µM', etc.
        """
        import re
        s = str(raw).strip().lower()
        s = re.sub(r"[µu]m.*", "", s)   # strip 'uM', 'µM', 'um'
        s = re.sub(r"[^0-9.\-eE+]", "", s)
        try:
            return float(s)
        except ValueError:
            return 0.0

    @staticmethod
    def _resolve_labels(series: pd.Series) -> pd.Series:
        """
        Convert raw cell values to integer labels {0, 1, 2, 3}.

        Handles:
        • Values already in {0, 1, 2, 3}  → pass through
        • String labels {'No','Low','Medium','High'}  → map to int
        • Continuous floats  → quantile-based binning into 4 classes
        """
        sample = series.dropna().iloc[0] if len(series.dropna()) > 0 else 0

        # ── Already integer labels ────────────────────────────────────────────
        if pd.api.types.is_integer_dtype(series) or (
            pd.api.types.is_float_dtype(series)
            and series.dropna().isin([0.0, 1.0, 2.0, 3.0]).all()
        ):
            return series.fillna(0).astype(int)

        # ── String labels ─────────────────────────────────────────────────────
        if isinstance(sample, str):
            return series.map(
                lambda v: INV_LABEL_MAP.get(str(v).strip().capitalize(), 0)
            )

        # ── Continuous values → quantile bins ─────────────────────────────────
        return pd.cut(
            series.fillna(series.median()),
            bins=4,
            labels=[0, 1, 2, 3],
        ).astype(int)
