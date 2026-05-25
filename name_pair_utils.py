"""
name_pair_utils.py
------------------
Lightweight utilities for name similarity feature computation.
Used by training_pair_generator.py.
"""

import re
import unicodedata
from typing import List, Tuple, Set

import jellyfish
from metaphone import doublemetaphone


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Lowercase, strip accents, remove punctuation except spaces."""
    name = str(name).lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def tokenize(name: str) -> List[str]:
    return [t for t in normalize(name).split() if t]


def initials(name: str) -> str:
    return "".join(t[0] for t in tokenize(name))


# ── Phonetics ─────────────────────────────────────────────────────────────────

def metaphones(name: str) -> Set[str]:
    """Return set of non-empty double-metaphone codes for all tokens."""
    codes = set()
    for token in tokenize(name):
        p1, p2 = doublemetaphone(token)
        if p1:
            codes.add(p1)
        if p2:
            codes.add(p2)
    return codes


def soundex_tokens(name: str) -> List[str]:
    return [jellyfish.soundex(t) for t in tokenize(name)]


# ── Similarity features ────────────────────────────────────────────────────────

def shared_tokens(a: str, b: str) -> int:
    return len(set(tokenize(a)) & set(tokenize(b)))


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta and not tb:
        return 1.0
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def char_ngrams(s: str, n: int = 3) -> Set[str]:
    s = normalize(s).replace(" ", "")
    return set(s[i:i+n] for i in range(len(s) - n + 1))


def char_ngram_similarity(a: str, b: str, n: int = 3) -> float:
    ga, gb = char_ngrams(a, n), char_ngrams(b, n)
    if not ga and not gb:
        return 1.0
    union = ga | gb
    return len(ga & gb) / len(union) if union else 0.0


def shared_phonetics(a: str, b: str) -> int:
    return len(metaphones(a) & metaphones(b))


def phonetic_similarity(a: str, b: str) -> float:
    ma, mb = metaphones(a), metaphones(b)
    if not ma and not mb:
        return 1.0
    union = ma | mb
    return len(ma & mb) / len(union) if union else 0.0


def edit_distance(a: str, b: str) -> int:
    return jellyfish.levenshtein_distance(normalize(a), normalize(b))


def compute_features(a: str, b: str) -> dict:
    return {
        "shared_tokens":        shared_tokens(a, b),
        "shared_phonetics":     shared_phonetics(a, b),
        "token_jaccard":        round(token_jaccard(a, b), 4),
        "char_ngram_similarity": round(char_ngram_similarity(a, b), 4),
        "phonetic_similarity":  round(phonetic_similarity(a, b), 4),
        "edit_distance":        edit_distance(a, b),
    }


# ── Difficulty heuristic ───────────────────────────────────────────────────────

def pair_difficulty(features: dict, label: int) -> str:
    """
    Classify pair difficulty from precomputed features.
    Positives: hard = high edit distance / low jaccard
    Negatives: hard = high similarity despite different identity
    """
    tj  = features["token_jaccard"]
    ps  = features["phonetic_similarity"]
    ed  = features["edit_distance"]
    cns = features["char_ngram_similarity"]

    if label == 1:
        # Positive pairs
        if ed <= 2 and tj >= 0.8:
            return "easy"
        elif ed <= 5 or tj >= 0.5:
            return "medium"
        else:
            return "hard"
    else:
        # Negative pairs — hard = looks very similar
        score = (tj * 0.4) + (ps * 0.3) + (cns * 0.3)
        if score >= 0.65:
            return "hard"
        elif score >= 0.35:
            return "medium"
        else:
            return "easy"
