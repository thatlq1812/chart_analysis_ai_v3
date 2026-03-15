"""
Shared Metric Functions

All benchmark suites use these functions for consistent, reproducible scoring.

Metrics implemented:
    anls(pred, target)                    - Answer Normalized Levenshtein Similarity
    exact_match(pred, target)             - Exact string match (case-insensitive)
    contains_match(pred, target)          - Target appears in prediction
    numeric_accuracy(pred, target, tol)   - Numeric relative error within tolerance
    cer(pred, target)                     - Character Error Rate
    text_overlap(preds, gts)              - Set-based precision/recall/F1
    table_value_recall(table_str, values) - How many GT values appear in VLM table

References:
    ANLS: Biten et al. 2019 (DocVQA) - https://arxiv.org/abs/1907.00490
    ChartQA metric: Masry et al. 2022 - https://arxiv.org/abs/2203.10244
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ANLS (Answer Normalized Levenshtein Similarity)
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j] + (c1 != c2), curr[j] + 1, prev[j + 1] + 1))
        prev = curr
    return prev[-1]


def anls(pred: str, target: str, threshold: float = 0.5) -> float:
    """
    Answer Normalized Levenshtein Similarity.

    Score = 1 - NED(pred, target),  where NED = edit_distance / max(len)
    If NED > threshold (default 0.5), score is clipped to 0.

    Used in DocVQA, ChartQA, and other VQA benchmarks.

    Args:
        pred: Predicted answer string
        target: Ground truth answer string
        threshold: NED threshold above which score = 0

    Returns:
        Float in [0.0, 1.0]
    """
    pred = _normalize(pred)
    target = _normalize(target)
    if not target:
        return 1.0 if not pred else 0.0
    max_len = max(len(pred), len(target))
    if max_len == 0:
        return 1.0
    ned = _levenshtein(pred, target) / max_len
    return 0.0 if ned > threshold else 1.0 - ned


def _normalize(text: str) -> str:
    """Lowercase, strip, normalize unicode, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text.lower().strip())
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(pred: str, target: str) -> bool:
    """Case-insensitive exact match after normalization."""
    return _normalize(pred) == _normalize(target)


# ---------------------------------------------------------------------------
# Contains Match
# ---------------------------------------------------------------------------

def contains_match(pred: str, target: str) -> bool:
    """True if normalized target appears anywhere in normalized prediction."""
    return _normalize(target) in _normalize(pred)


# ---------------------------------------------------------------------------
# Numeric Accuracy
# ---------------------------------------------------------------------------

def _parse_number(text: str) -> Optional[float]:
    """Extract a float from a string, handling %, commas, K/M/B suffixes."""
    text = text.strip().replace(",", "").replace("%", "")
    # Handle K/M/B suffixes
    multipliers = {"k": 1e3, "m": 1e6, "b": 1e9}
    if text and text[-1].lower() in multipliers:
        try:
            return float(text[:-1]) * multipliers[text[-1].lower()]
        except ValueError:
            pass
    try:
        return float(text)
    except ValueError:
        return None


def numeric_accuracy(pred: str, target: str, tolerance: float = 0.05) -> bool:
    """
    Return True if numeric values in pred and target match within relative tolerance.

    Tolerance of 0.05 means within 5% relative error (ChartQA standard).
    """
    p = _parse_number(pred)
    t = _parse_number(target)
    if p is None or t is None:
        return False
    if t == 0:
        return abs(p) < 1e-9
    return abs(p - t) / abs(t) <= tolerance


# ---------------------------------------------------------------------------
# Character Error Rate
# ---------------------------------------------------------------------------

def cer(pred: str, target: str) -> float:
    """
    Character Error Rate: edit_distance(pred, target) / len(target).

    CER = 0.0 is perfect. CER > 1.0 means more insertions than GT chars.
    """
    pred = _normalize(pred)
    target = _normalize(target)
    if not target:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, target) / len(target)


# ---------------------------------------------------------------------------
# Text Set Overlap (OCR precision/recall/F1)
# ---------------------------------------------------------------------------

def text_overlap(
    preds: List[str],
    gts: List[str],
    match_fn: str = "anls",
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 between predicted text set and GT text set.

    Each GT text is matched to best prediction (greedy, no duplicate matching).

    Args:
        preds: Predicted text strings from OCR/VLM
        gts: Ground truth text strings
        match_fn: "exact" | "anls" | "contains"
        threshold: ANLS score threshold for a match to count

    Returns:
        Dict with keys: precision, recall, f1, matches
    """
    if not gts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matches": 0}
    if not preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": 0}

    gts_norm = [_normalize(g) for g in gts]
    preds_norm = [_normalize(p) for p in preds]

    matched_preds = set()
    matched_count = 0

    for g in gts_norm:
        best_score = 0.0
        best_idx = -1
        for idx, p in enumerate(preds_norm):
            if idx in matched_preds:
                continue
            if match_fn == "exact":
                score = 1.0 if g == p else 0.0
            elif match_fn == "contains":
                score = 1.0 if g in p or p in g else 0.0
            else:
                score = anls(p, g)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score >= threshold and best_idx >= 0:
            matched_count += 1
            matched_preds.add(best_idx)

    precision = len(matched_preds) / len(preds) if preds else 0.0
    recall = matched_count / len(gts)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matches": matched_count,
    }


# ---------------------------------------------------------------------------
# VLM Table Value Recall
# ---------------------------------------------------------------------------

def table_value_recall(
    table_str: str,
    gt_values: List[str],
    tolerance: float = 0.05,
) -> Dict[str, float]:
    """
    Check how many GT values (tick labels, data labels) appear in VLM table output.

    Handles both numeric tolerance match and ANLS string match.

    Args:
        table_str: Linearized table string from DePlot/MatCha output
        gt_values: List of expected values (tick labels, titles, etc.)
        tolerance: Numeric match tolerance

    Returns:
        Dict with: recall, numeric_recall, text_recall, total_gt
    """
    if not gt_values:
        return {"recall": 1.0, "numeric_recall": 1.0, "text_recall": 1.0, "total_gt": 0}

    found_any = 0
    found_numeric = 0
    found_text = 0
    numeric_gt = 0
    text_gt = 0

    # Extract all tokens from table string
    table_tokens = re.findall(r"[\w.,%-]+", table_str.lower())
    table_text = " ".join(table_tokens)

    for val in gt_values:
        val_norm = _normalize(val)
        num = _parse_number(val)
        if num is not None:
            numeric_gt += 1
            # Try to find numeric match in table tokens
            found = False
            for tok in table_tokens:
                tok_num = _parse_number(tok)
                if tok_num is not None and (
                    abs(num) < 1e-9 and abs(tok_num) < 1e-9
                    or abs(num) > 1e-9 and abs(tok_num - num) / abs(num) <= tolerance
                ):
                    found = True
                    break
            if found:
                found_numeric += 1
                found_any += 1
        else:
            text_gt += 1
            # ANLS-based text match
            score = max(
                (anls(tok, val_norm) for tok in table_tokens), default=0.0
            )
            if score >= 0.7 or val_norm in table_text:
                found_text += 1
                found_any += 1

    total_gt = len(gt_values)
    recall = found_any / total_gt
    num_recall = found_numeric / numeric_gt if numeric_gt > 0 else 1.0
    txt_recall = found_text / text_gt if text_gt > 0 else 1.0

    return {
        "recall": round(recall, 4),
        "numeric_recall": round(num_recall, 4),
        "text_recall": round(txt_recall, 4),
        "total_gt": total_gt,
    }
