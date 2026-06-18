"""
agent/data/features.py
────────────────────────────────────────────────────────────────────────────
All feature engineering functions for peptide sequences.
Zero external biology libraries — every property computed from first principles.

Functions
---------
amino_acid_composition(sequence)       → np.ndarray [20]   normalised counts
physicochemical_features(sequence)     → np.ndarray [6]    charge, hydrophob,
                                                            MW, aromaticity,
                                                            pI, instability
kmer_frequencies(sequence, k)          → dict[str, float]  normalised k-mer freq
encode_concentration(value,            → float [0, 1]      log10-scaled
                     log_min, log_max)
"""

import math
import itertools
from typing import Dict, List
import numpy as np

# ── Canonical amino-acid alphabet (sorted alphabetically) ────────────────────
AA_LIST: List[str] = list("ACDEFGHIKLMNPQRSTVWY")   # 20 standard amino acids
AA_SET  = set(AA_LIST)

# ── Kyte–Doolittle hydrophobicity scale ──────────────────────────────────────
KD_SCALE: Dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# ── Residue monoisotopic masses (Da) ─────────────────────────────────────────
RESIDUE_MW: Dict[str, float] = {
    "A":  89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G":  75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
_WATER_MW = 18.02   # added once per chain

# ── Aromatic residues ─────────────────────────────────────────────────────────
AROMATIC = {"F", "W", "Y", "H"}

# ── pKa values for isoelectric point (Lehninger / ExPASy convention) ─────────
# Each key maps to (pKa, sign_at_low_pH)
#   sign_at_low_pH = +1  → positive at very low pH (basic group)
#   sign_at_low_pH = -1  → neutral at very low pH, negative at high pH (acid group)
_IONIZABLE: List[tuple] = [
    # (residue_or_terminus, pKa, charge_when_fully_protonated)
    ("Nterm",  8.00,  +1),   # α-amino group
    ("Cterm",  3.10,  -1),   # α-carboxyl group
    ("D",      3.86,  -1),
    ("E",      4.25,  -1),
    ("H",      6.05,  +1),
    ("C",      8.33,  -1),
    ("Y",     10.07,  -1),
    ("K",     10.54,  +1),
    ("R",     12.48,  +1),
]

# ── Dipeptide Instability Weight Values (DIWV) ────────────────────────────────
# Source: Guruprasad et al. (1990) Protein Engineering 4(2):155-161
# Used by ProtParam / ExPASy.  Full 20×20 = 400-entry table.
DIWV: Dict[str, float] = {
    "AA": 1.0,  "AC": 44.94, "AD": -7.49, "AE": -7.49, "AF": 1.0,
    "AG": 1.0,  "AH": -7.49, "AI": 1.0,  "AK": 1.0,  "AL": 1.0,
    "AM": 1.0,  "AN": 23.51, "AP": 20.26, "AQ": 0.0,  "AR": 1.0,
    "AS": 1.0,  "AT": 1.0,  "AV": 1.0,  "AW": 1.0,  "AY": 1.0,

    "CA": -1.33,"CC": 1.0,  "CD": 20.26, "CE": 33.60, "CF": 1.0,
    "CG": 1.0,  "CH": 33.60,"CI": 1.0,  "CK": 1.0,  "CL": 20.26,
    "CM": 33.60,"CN": 1.0,  "CP": 20.26, "CQ": -6.54,"CR": 1.0,
    "CS": 1.0,  "CT": 33.60,"CV": -6.54,"CW": 24.68, "CY": 1.0,

    "DA": 1.0,  "DC": 1.0,  "DD": 1.0,  "DE": 1.0,  "DF": 1.0,
    "DG": 1.0,  "DH": 1.0,  "DI": 1.0,  "DK": -7.49,"DL": 1.0,
    "DM": 1.0,  "DN": 1.0,  "DP": 1.0,  "DQ": 1.0,  "DR": -6.54,
    "DS": 1.0,  "DT": 1.0,  "DV": 1.0,  "DW": 1.0,  "DY": 1.0,

    "EA": 1.0,  "EC": 44.94,"ED": 1.0,  "EE": -6.54,"EF": 1.0,
    "EG": 1.0,  "EH": -6.54,"EI": 1.0,  "EK": 1.0,  "EL": 1.0,
    "EM": 1.0,  "EN": 1.0,  "EP": 20.26,"EQ": 1.0,  "ER": 1.0,
    "ES": 1.0,  "ET": 1.0,  "EV": 1.0,  "EW": -14.03,"EY": 1.0,

    "FA": 1.0,  "FC": 1.0,  "FD": 13.34,"FE": 1.0,  "FF": 1.0,
    "FG": 1.0,  "FH": 1.0,  "FI": 1.0,  "FK": -14.03,"FL": 1.0,
    "FM": 1.0,  "FN": 1.0,  "FP": 20.26,"FQ": 1.0,  "FR": 1.0,
    "FS": 1.0,  "FT": 1.0,  "FV": 1.0,  "FW": 1.0,  "FY": 33.60,

    "GA": 1.0,  "GC": 1.0,  "GD": 1.0,  "GE": -6.54,"GF": 1.0,
    "GG": 13.34,"GH": 1.0,  "GI": -7.49,"GK": -7.49,"GL": 1.0,
    "GM": 1.0,  "GN": -7.49,"GP": 1.0,  "GQ": 1.0,  "GR": 1.0,
    "GS": 1.0,  "GT": -7.49,"GV": -7.49,"GW": 13.34, "GY": -7.49,

    "HA": 1.0,  "HC": 1.0,  "HD": 1.0,  "HE": 1.0,  "HF": -7.49,
    "HG": 1.0,  "HH": 1.0,  "HI": 1.0,  "HK": 1.0,  "HL": 1.0,
    "HM": 1.0,  "HN": 24.68,"HP": 1.0,  "HQ": 1.0,  "HR": 1.0,
    "HS": 1.0,  "HT": -7.49,"HV": 1.0,  "HW": -1.88,"HY": 44.94,

    "IA": 1.0,  "IC": 1.0,  "ID": 1.0,  "IE": 44.94,"IF": 1.0,
    "IG": 1.0,  "IH": -7.49,"II": 1.0,  "IK": -7.49,"IL": 20.26,
    "IM": 1.0,  "IN": 1.0,  "IP": -1.88,"IQ": 1.0,  "IR": 1.0,
    "IS": 1.0,  "IT": 1.0,  "IV": -7.49,"IW": 1.0,  "IY": 1.0,

    "KA": 1.0,  "KC": 1.0,  "KD": 1.0,  "KE": 1.0,  "KF": 1.0,
    "KG": -7.49,"KH": 1.0,  "KI": -7.49,"KK": 1.0,  "KL": -7.49,
    "KM": 33.60,"KN": 1.0,  "KP": -6.54,"KQ": 24.68,"KR": 33.60,
    "KS": 1.0,  "KT": 1.0,  "KV": -7.49,"KW": 1.0,  "KY": 1.0,

    "LA": 1.0,  "LC": 1.0,  "LD": 1.0,  "LE": 1.0,  "LF": 1.0,
    "LG": 1.0,  "LH": 1.0,  "LI": 1.0,  "LK": -7.49,"LL": 1.0,
    "LM": 1.0,  "LN": 1.0,  "LP": 20.26,"LQ": -6.54,"LR": 1.0,
    "LS": 1.0,  "LT": 1.0,  "LV": 1.0,  "LW": 24.68,"LY": 1.0,

    "MA": 1.0,  "MC": 1.0,  "MD": 1.0,  "ME": -6.54,"MF": 1.0,
    "MG": 1.0,  "MH": 1.0,  "MI": 1.0,  "MK": 33.60,"ML": 1.0,
    "MM": -1.88,"MN": 1.0,  "MP": 44.94,"MQ": -6.54,"MR": -6.54,
    "MS": 44.94,"MT": -1.88,"MV": 1.0,  "MW": 1.0,  "MY": 24.68,

    "NA": 1.0,  "NC": -6.54,"ND": 1.0,  "NE": 1.0,  "NF": -14.03,
    "NG": -14.03,"NH": 1.0,  "NI": 44.94,"NK": 24.68,"NL": 1.0,
    "NM": 1.0,  "NN": 1.0,  "NP": -1.88,"NQ": -6.54,"NR": 1.0,
    "NS": 1.0,  "NT": -7.49,"NV": 1.0,  "NW": -9.37,"NY": 1.0,

    "PA": 20.26,"PC": -6.54,"PD": -6.54,"PE": 18.38,"PF": 20.26,
    "PG": 1.0,  "PH": 1.0,  "PI": 1.0,  "PK": 1.0,  "PL": 1.0,
    "PM": -6.54,"PN": 1.0,  "PP": 20.26,"PQ": 20.26,"PR": -6.54,
    "PS": 20.26,"PT": 1.0,  "PV": 20.26,"PW": -1.88,"PY": 1.0,

    "QA": 1.0,  "QC": -6.54,"QD": 1.0,  "QE": -6.54,"QF": -6.54,
    "QG": 1.0,  "QH": 1.0,  "QI": 1.0,  "QK": 1.0,  "QL": 1.0,
    "QM": 1.0,  "QN": 1.0,  "QP": 20.26,"QQ": 20.26,"QR": 1.0,
    "QS": 44.94,"QT": 1.0,  "QV": -6.54,"QW": 1.0,  "QY": -6.54,

    "RA": 1.0,  "RC": 1.0,  "RD": 1.0,  "RE": 1.0,  "RF": 1.0,
    "RG": -6.54,"RH": 20.26,"RI": 1.0,  "RK": 1.0,  "RL": 1.0,
    "RM": 1.0,  "RN": 1.0,  "RP": 20.26,"RQ": 20.26,"RR": 58.28,
    "RS": 44.94,"RT": 1.0,  "RV": 1.0,  "RW": 58.28,"RY": -6.54,

    "SA": 1.0,  "SC": 33.60,"SD": 1.0,  "SE": 20.26,"SF": 1.0,
    "SG": 1.0,  "SH": 1.0,  "SI": 1.0,  "SK": -7.49,"SL": 1.0,
    "SM": 1.0,  "SN": 1.0,  "SP": 44.94,"SQ": 20.26,"SR": 20.26,
    "SS": 20.26,"ST": 1.0,  "SV": 1.0,  "SW": 1.0,  "SY": 1.0,

    "TA": 1.0,  "TC": 1.0,  "TD": 1.0,  "TE": 20.26,"TF": 13.34,
    "TG": -7.49,"TH": 1.0,  "TI": 1.0,  "TK": 1.0,  "TL": 1.0,
    "TM": 1.0,  "TN": -14.03,"TP": 1.0, "TQ": -6.54,"TR": 1.0,
    "TS": 1.0,  "TT": 1.0,  "TV": 1.0,  "TW": -14.03,"TY": 1.0,

    "VA": 1.0,  "VC": 1.0,  "VD": -14.03,"VE": 1.0, "VF": 1.0,
    "VG": -7.49,"VH": 1.0,  "VI": 1.0,  "VK": -1.88,"VL": 1.0,
    "VM": 1.0,  "VN": 1.0,  "VP": 20.26,"VQ": 1.0,  "VR": 1.0,
    "VS": 1.0,  "VT": -7.49,"VV": 1.0,  "VW": 1.0,  "VY": -6.54,

    "WA": -14.03,"WC": 1.0, "WD": 1.0,  "WE": 1.0,  "WF": 1.0,
    "WG": -9.37,"WH": 24.68,"WI": 1.0,  "WK": 1.0,  "WL": 13.34,
    "WM": 24.68,"WN": 13.34,"WP": 1.0,  "WQ": 1.0,  "WR": 1.0,
    "WS": 1.0,  "WT": -14.03,"WV": -7.49,"WW": 1.0, "WY": 1.0,

    "YA": 24.68,"YC": 1.0,  "YD": 24.68,"YE": -6.54,"YF": 1.0,
    "YG": -7.49,"YH": 13.34,"YI": 1.0,  "YK": 1.0,  "YL": 1.0,
    "YM": 44.94,"YN": 1.0,  "YP": 13.34,"YQ": 1.0,  "YR": -15.91,
    "YS": 1.0,  "YT": -7.49,"YV": 1.0,  "YW": -9.37,"YY": 13.34,
}

# ── Pre-build sorted kmer key lists (ensures consistent feature order) ────────
_2MER_KEYS: List[str] = sorted(
    "".join(p) for p in itertools.product(AA_LIST, repeat=2)
)
_3MER_KEYS: List[str] = sorted(
    "".join(p) for p in itertools.product(AA_LIST, repeat=3)
)


# ═══════════════════════════════════════════════════════════════════════════════
# Public functions
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(sequence: str) -> str:
    """Upper-case and strip non-standard residues."""
    return "".join(aa for aa in sequence.upper() if aa in AA_SET)


def amino_acid_composition(sequence: str) -> np.ndarray:
    """
    Compute normalised amino-acid composition.

    Parameters
    ----------
    sequence : str
        Raw peptide sequence (any case; non-standard residues ignored).

    Returns
    -------
    np.ndarray, shape (20,)
        Fraction of each canonical amino acid, ordered by AA_LIST.
        Sums to 1.0 for a valid sequence; returns zeros for empty input.
    """
    seq = _clean(sequence)
    counts = np.zeros(20, dtype=np.float64)
    if not seq:
        return counts
    for aa in seq:
        counts[AA_LIST.index(aa)] += 1.0
    return counts / len(seq)


def physicochemical_features(sequence: str) -> np.ndarray:
    """
    Compute six physicochemical descriptors without Biopython.

    Returns
    -------
    np.ndarray, shape (6,)
        [charge, hydrophobicity, molecular_weight,
         aromaticity, isoelectric_point, instability_index]

    Notes
    -----
    charge          : net charge at pH 7.4 (Henderson-Hasselbalch)
    hydrophobicity  : mean Kyte–Doolittle score  (range ≈ −4.5 … +4.5)
    molecular_weight: sum of residue masses + water  (Da)
    aromaticity     : fraction of {F, W, Y, H} residues
    isoelectric_point: pH at which net charge = 0 (binary search, Δ=0.001)
    instability_index: Guruprasad 1990 dipeptide index
                       >40 → unstable, ≤40 → stable
    """
    seq = _clean(sequence)
    n   = len(seq)

    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # ── Residue counts for ionisable groups ────────────────────────────────
    counts: Dict[str, int] = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1

    # ── 1. Charge at pH 7.4 ────────────────────────────────────────────────
    charge = _charge_at_ph(counts, ph=7.4)

    # ── 2. Hydrophobicity ──────────────────────────────────────────────────
    hydrophobicity = sum(KD_SCALE.get(aa, 0.0) for aa in seq) / n

    # ── 3. Molecular weight ────────────────────────────────────────────────
    mw = sum(RESIDUE_MW.get(aa, 0.0) for aa in seq) + _WATER_MW

    # ── 4. Aromaticity ─────────────────────────────────────────────────────
    aromaticity = sum(1 for aa in seq if aa in AROMATIC) / n

    # ── 5. Isoelectric point (binary search) ──────────────────────────────
    pi = _isoelectric_point(counts)

    # ── 6. Instability index ───────────────────────────────────────────────
    instability = _instability_index(seq)

    return np.array(
        [charge, hydrophobicity, mw, aromaticity, pi, instability],
        dtype=np.float64,
    )


def kmer_frequencies(sequence: str, k: int = 2) -> Dict[str, float]:
    """
    Compute normalised k-mer frequency dictionary.

    All possible k-mers over the 20-aa alphabet are present (zero-padded).
    This guarantees a consistent, fixed-width feature vector regardless
    of which k-mers appear in a given sequence.

    Parameters
    ----------
    sequence : str
    k        : int, typically 2 or 3

    Returns
    -------
    dict[str, float]
        Keys ordered alphabetically; values = count / (L - k + 1) or 0.
    """
    if k == 2:
        key_list = _2MER_KEYS
    elif k == 3:
        key_list = _3MER_KEYS
    else:
        key_list = sorted(
            "".join(p) for p in itertools.product(AA_LIST, repeat=k)
        )

    freq: Dict[str, float] = {km: 0.0 for km in key_list}
    seq  = _clean(sequence)
    total = len(seq) - k + 1

    if total <= 0:
        return freq

    for i in range(total):
        km = seq[i : i + k]
        if km in freq:
            freq[km] += 1.0

    for km in freq:
        freq[km] /= total

    return freq


def encode_concentration(
    value: float,
    log_min: float = -2.0,
    log_max: float =  2.0,
) -> float:
    """
    Encode a concentration value as log10-scaled and min-max normalised to [0, 1].

    Parameters
    ----------
    value   : float — raw concentration (e.g. µM)
    log_min : float — log10 of the minimum expected concentration (default −2 → 0.01 µM)
    log_max : float — log10 of the maximum expected concentration (default +2 → 100 µM)

    Returns
    -------
    float in [0, 1]
    """
    if value <= 0.0:
        return 0.0
    log_val    = math.log10(value)
    normalised = (log_val - log_min) / (log_max - log_min)
    return float(max(0.0, min(1.0, normalised)))


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _charge_at_ph(counts: Dict[str, int], ph: float) -> float:
    """Net charge of a peptide at a given pH via Henderson-Hasselbalch."""
    charge = 0.0
    for (group, pka, sign) in _IONIZABLE:
        if group == "Nterm":
            n = 1
        elif group == "Cterm":
            n = 1
        else:
            n = counts.get(group, 0)

        if n == 0:
            continue

        if sign == +1:
            # Basic group: positively charged when pH < pKa
            charge += n * (1.0 / (1.0 + 10 ** (ph - pka)))
        else:
            # Acidic group: negatively charged when pH > pKa
            charge -= n * (1.0 / (1.0 + 10 ** (pka - ph)))

    return charge


def _isoelectric_point(counts: Dict[str, int]) -> float:
    """Binary search for the pH at which net charge ≈ 0."""
    lo, hi = 0.0, 14.0
    for _ in range(1000):          # converges in < 15 iterations for pH res 0.001
        mid = (lo + hi) / 2.0
        c   = _charge_at_ph(counts, mid)
        if abs(c) < 1e-6:
            return mid
        if c > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _instability_index(seq: str) -> float:
    """
    Guruprasad et al. (1990) instability index.
    II = (10 / L) * Σ DIWV[aa_i][aa_{i+1}]
    Values > 40 indicate an unstable protein.
    """
    if len(seq) < 2:
        return 0.0
    total = sum(DIWV.get(seq[i] + seq[i + 1], 1.0) for i in range(len(seq) - 1))
    return (10.0 / len(seq)) * total
