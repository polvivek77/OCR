"""Accuracy metrics for OCR evaluation.

- Character accuracy: SequenceMatcher ratio on normalized text
- Word accuracy: SequenceMatcher ratio on token lists
- Number accuracy: precision / recall over numeric tokens (integers & decimals)
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip().lower()


def char_accuracy(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    return round(SequenceMatcher(None, _norm(ref), _norm(hyp)).ratio(), 4)


def word_accuracy(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    ref_tokens = _norm(ref).split()
    hyp_tokens = _norm(hyp).split()
    if not ref_tokens:
        return 0.0
    return round(SequenceMatcher(None, ref_tokens, hyp_tokens).ratio(), 4)


def _numbers(s: str) -> list[str]:
    return [m.group().replace(",", "") for m in NUM_RE.finditer(s or "")]


def number_metrics(ref: str, hyp: str) -> dict:
    ref_nums = _numbers(ref)
    hyp_nums = _numbers(hyp)
    ref_set = set(ref_nums)
    hyp_set = set(hyp_nums)
    tp = len(ref_set & hyp_set)
    precision = tp / len(hyp_set) if hyp_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    missed = sorted(ref_set - hyp_set)
    extra = sorted(hyp_set - ref_set)
    return {
        "ref_count": len(ref_set),
        "hyp_count": len(hyp_set),
        "matched": tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "missed": missed[:20],
        "extra": extra[:20],
    }


def evaluate(ref: str, hyp: str) -> dict:
    return {
        "char_acc": char_accuracy(ref, hyp),
        "word_acc": word_accuracy(ref, hyp),
        "numbers": number_metrics(ref, hyp),
    }
