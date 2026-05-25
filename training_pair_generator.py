"""
training_pair_generator.py
--------------------------
Generates training pairs (positive / light-negative / hard-negative)
from an existing noisy-name CSV for supervised name-matching models.

Usage
-----
python training_pair_generator.py \
    --input  your_dataset.csv \
    --output training_pairs.csv \
    --pairs  500000 \
    --pos_ratio   0.40 \
    --light_ratio 0.30 \
    --hard_ratio  0.30 \
    --seed 42 \
    --chunk_size 50000

The script never fabricates names — every name in every pair
comes from your own dataset.
"""

import argparse
import hashlib
import itertools
import random
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from name_pair_utils import (
    char_ngrams,
    compute_features,
    compute_features_precomputed,
    initials,
    metaphones,
    normalize,
    pair_difficulty,
    tokenize,
)

# ── Index structures ───────────────────────────────────────────────────────────

class NameIndex:
    """
    Inverted indexes for fast approximate-neighbor lookup.
    Supports: char-3gram, token, metaphone-code, initials.
    Each slot stores a list of (latent_id, noisy_name, row_index).
    """

    def __init__(self):
        self.ngram_idx:    defaultdict = defaultdict(list)
        self.token_idx:    defaultdict = defaultdict(list)
        self.phonetic_idx: defaultdict = defaultdict(list)
        self.initials_idx: defaultdict = defaultdict(list)
        self.surname_idx:  defaultdict = defaultdict(list)

    def add(self, latent_id, name: str, row_idx: int, 
            tokens: list = None, ngrams: set = None, 
            phoneme_set: set = None, init: str = None):
        """Add entry to index. Optionally accepts pre-computed features to avoid recomputation."""
        entry = (latent_id, name, row_idx)

        # Use pre-computed or compute on demand
        if ngrams is None:
            ngrams = char_ngrams(name, 3)
        for gram in ngrams:
            self.ngram_idx[gram].append(entry)

        if tokens is None:
            tokens = tokenize(name)
        for tok in tokens:
            self.token_idx[tok].append(entry)

        if phoneme_set is None:
            phoneme_set = metaphones(name)
        for code in phoneme_set:
            self.phonetic_idx[code].append(entry)

        if init is None:
            init = initials(name)
        if init:
            self.initials_idx[init].append(entry)

        if tokens:
            self.surname_idx[tokens[-1]].append(entry)

    def candidates(
        self,
        name: str,
        query_latent_id,
        top_k: int = 50,
        tokens: list = None,
        ngrams: set = None,
        phoneme_set: set = None,
        init: str = None,
    ) -> List[Tuple]:
        """
        Return up to `top_k` candidate (latent_id, name, row_idx) tuples
        from DIFFERENT identities, ranked by neighbourhood hit count.
        Optionally accepts pre-computed features to avoid recomputation.
        """
        hit_count: Dict[int, int] = defaultdict(int)  # row_idx → hits

        # Use pre-computed or compute on demand
        if ngrams is None:
            ngrams = char_ngrams(name, 3)
        for gram in ngrams:
            for entry in self.ngram_idx.get(gram, []):
                if entry[0] != query_latent_id:
                    hit_count[entry[2]] += 2        # weight ngrams higher

        if tokens is None:
            tokens = tokenize(name)
        for tok in tokens:
            for entry in self.token_idx.get(tok, []):
                if entry[0] != query_latent_id:
                    hit_count[entry[2]] += 3        # token overlap is strong signal

        if phoneme_set is None:
            phoneme_set = metaphones(name)
        for code in phoneme_set:
            for entry in self.phonetic_idx.get(code, []):
                if entry[0] != query_latent_id:
                    hit_count[entry[2]] += 2

        if init is None:
            init = initials(name)
        if init:
            for entry in self.initials_idx.get(init, []):
                if entry[0] != query_latent_id:
                    hit_count[entry[2]] += 4        # initials collision = hard-negative gold

        if tokens:
            for entry in self.surname_idx.get(tokens[-1], []):
                if entry[0] != query_latent_id:
                    hit_count[entry[2]] += 3

        # Sort by hit count descending, take top_k
        ranked = sorted(hit_count.items(), key=lambda x: -x[1])[:top_k]
        return ranked   # [(row_idx, hit_count), ...]


# ── Pair builders ──────────────────────────────────────────────────────────────

def make_pair_id() -> str:
    return str(uuid.uuid4())[:16]


def build_positive_pair(
    q_latent_id,
    q_name: str,
    c_name: str,
    c_latent_id,
    notes: str = "",
    q_features: dict = None,
    c_features: dict = None,
) -> Optional[dict]:
    if normalize(q_name) == normalize(c_name):
        return None                         # skip identical strings
    
    # Use pre-computed features if available, otherwise compute
    if q_features is None or c_features is None:
        feats = compute_features(q_name, c_name)
    else:
        feats = compute_features_precomputed(
            q_features["tokens"], c_features["tokens"],
            q_features["ngrams3"], c_features["ngrams3"],
            q_features["metaphone_set"], c_features["metaphone_set"],
            q_features["norm_name"], c_features["norm_name"]
        )
    
    diff  = pair_difficulty(feats, label=1)
    return {
        "pair_id":             make_pair_id(),
        "query_name":          q_name,
        "candidate_name":      c_name,
        "query_latent_id":     q_latent_id,
        "candidate_latent_id": c_latent_id,
        "label":               1,
        "pair_type":           "positive",
        "difficulty":          diff,
        **feats,
        "notes":               notes,
    }


def build_negative_pair(
    q_latent_id,
    q_name: str,
    c_latent_id,
    c_name: str,
    pair_type: str,
    notes: str = "",
    q_features: dict = None,
    c_features: dict = None,
) -> Optional[dict]:
    if q_latent_id == c_latent_id:
        return None
    
    # Use pre-computed features if available, otherwise compute
    if q_features is None or c_features is None:
        feats = compute_features(q_name, c_name)
    else:
        feats = compute_features_precomputed(
            q_features["tokens"], c_features["tokens"],
            q_features["ngrams3"], c_features["ngrams3"],
            q_features["metaphone_set"], c_features["metaphone_set"],
            q_features["norm_name"], c_features["norm_name"]
        )
    
    diff  = pair_difficulty(feats, label=0)

    # Reject pairs that are too dissimilar to be useful negatives
    if feats["token_jaccard"] < 0.05 and feats["phonetic_similarity"] < 0.1 \
            and feats["char_ngram_similarity"] < 0.15:
        return None

    return {
        "pair_id":             make_pair_id(),
        "query_name":          q_name,
        "candidate_name":      c_name,
        "query_latent_id":     q_latent_id,
        "candidate_latent_id": c_latent_id,
        "label":               0,
        "pair_type":           pair_type,
        "difficulty":          diff,
        **feats,
        "notes":               notes,
    }


# ── Positive generation ────────────────────────────────────────────────────────

def generate_positives(
    identity_groups: Dict,
    target: int,
    rng: random.Random,
) -> List[dict]:
    """
    For each identity, pair variants together.
    50% single-noise positives, 50% combined-noise positives.
    """
    pairs = []
    latent_ids = list(identity_groups.keys())
    rng.shuffle(latent_ids)

    single_target   = target // 2
    combined_target = target - single_target

    single_pairs, combined_pairs = [], []

    for lid in latent_ids:
        rows = identity_groups[lid]
        if len(rows) < 2:
            continue

        single_rows   = [r for r in rows if not r.get("is_combined_noise", False)]
        combined_rows = [r for r in rows if r.get("is_combined_noise",  False)]

        # ---- single-noise positives ----
        if len(single_rows) >= 2:
            for a, b in itertools.combinations(single_rows, 2):
                p = build_positive_pair(
                    lid, a["noisy_name"], b["noisy_name"], lid,
                    notes="single_noise_pair",
                    q_features=a, c_features=b
                )
                if p:
                    single_pairs.append(p)

        # clean ↔ noisy always included
        clean_rows = [r for r in rows if r.get("mode") in ("clean", None, "")]
        noisy_rows = [r for r in rows if r.get("mode") not in ("clean", None, "")]
        for cr in clean_rows:
            for nr in noisy_rows:
                p = build_positive_pair(
                    lid, cr["noisy_name"], nr["noisy_name"], lid,
                    notes="clean_vs_noisy",
                    q_features=cr, c_features=nr
                )
                if p:
                    single_pairs.append(p)

        # ---- combined-noise positives ----
        for cr in combined_rows:
            for nr in rows:
                if cr is nr:
                    continue
                p = build_positive_pair(
                    lid, cr["noisy_name"], nr["noisy_name"], lid,
                    notes="combined_noise_pair",
                    q_features=cr, c_features=nr
                )
                if p:
                    combined_pairs.append(p)

        if len(single_pairs) >= single_target * 2 \
                and len(combined_pairs) >= combined_target * 2:
            break

    rng.shuffle(single_pairs)
    rng.shuffle(combined_pairs)
    pairs = single_pairs[:single_target] + combined_pairs[:combined_target]
    rng.shuffle(pairs)
    return pairs


# ── Negative generation ────────────────────────────────────────────────────────

def _hit_score_to_type(hit_count: int) -> Tuple[str, str]:
    """Map neighbourhood hit score to (pair_type, difficulty_hint)."""
    if hit_count >= 12:
        return "hard_negative", "hard"
    elif hit_count >= 6:
        return "hard_negative", "medium"
    elif hit_count >= 3:
        return "light_negative", "medium"
    else:
        return "light_negative", "easy"


def _mine_negatives_worker(
    chunk_rows: List[dict],
    index: NameIndex,
    rows_cache: List[dict],
    top_k: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Worker function for parallel negative mining.
    Processes a chunk of rows and returns (light_pairs, hard_pairs).
    """
    light_pairs: List[dict] = []
    hard_pairs: List[dict] = []
    seen_pairs = set()
    rng = random.Random(seed)

    for row in chunk_rows:
        q_name = row["noisy_name"]
        q_lid = row["latent_id"]

        ranked = index.candidates(
            q_name, q_lid, top_k=top_k,
            tokens=row.get("tokens"),
            ngrams=row.get("ngrams3"),
            phoneme_set=row.get("metaphone_set"),
            init=row.get("initials")
        )

        for row_idx, hit_cnt in ranked:
            c_row = rows_cache[row_idx]
            c_name = c_row["noisy_name"]
            c_lid = c_row["latent_id"]

            if c_lid == q_lid:
                continue

            pair_key = tuple(sorted([f"{q_lid}|{normalize(q_name)}", f"{c_lid}|{normalize(c_name)}"]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            ptype, _ = _hit_score_to_type(hit_cnt)

            p = build_negative_pair(
                q_lid, q_name, c_lid, c_name, ptype,
                notes=f"neighbourhood_hits={hit_cnt}",
                q_features=row,
                c_features=c_row
            )
            if p is None:
                continue

            if ptype == "hard_negative":
                hard_pairs.append(p)
            else:
                light_pairs.append(p)

    return light_pairs, hard_pairs


def generate_negatives(
    df: pd.DataFrame,
    index: NameIndex,
    light_target: int,
    hard_target: int,
    rng: random.Random,
    top_k: int = 40,
    sample_frac: float = 0.3,
    num_workers: int = 4,
) -> List[dict]:
    """
    Mine hard & light negatives using the NameIndex with multiprocessing.
    Hard negatives come from top-ranked (high-hit) neighbours.
    Light negatives come from mid-ranked neighbours.
    
    sample_frac: fraction of rows to sample for negative mining (default 0.3 = 30%).
    top_k: max neighbourhood candidates per query (default 40).
    num_workers: number of CPU cores to use for parallel mining (default 4).
    """
    # Cache entire dataframe as dict list to avoid expensive .iloc calls
    rows_cache = df.to_dict("records")

    # Sample a fraction of rows to speed up mining
    sample_size = max(1, int(len(df) * sample_frac))
    sampled_df = df.sample(n=sample_size, random_state=rng.randint(0, 9999))
    rows_list = sampled_df.to_dict("records")

    # Split rows into chunks for parallel processing
    chunk_size = max(1, len(rows_list) // num_workers)
    chunks = [rows_list[i:i + chunk_size] for i in range(0, len(rows_list), chunk_size)]

    print(f"   Distributing {len(rows_list):,} samples across {len(chunks)} workers…")

    # Mine negatives in parallel
    light_pairs_all: List[dict] = []
    hard_pairs_all: List[dict] = []
    seen_pairs_global = set()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _mine_negatives_worker,
                chunk,
                index,
                rows_cache,
                top_k,
                rng.randint(0, 9999999),
            )
            for chunk in chunks
        ]

        for future in tqdm(futures, desc="Mining negatives", leave=False):
            light_pairs, hard_pairs = future.result()
            
            # Deduplicate across chunks
            for pair in light_pairs:
                pair_key = tuple(sorted([
                    f"{pair['query_latent_id']}|{pair['query_name']}",
                    f"{pair['candidate_latent_id']}|{pair['candidate_name']}"
                ]))
                if pair_key not in seen_pairs_global:
                    light_pairs_all.append(pair)
                    seen_pairs_global.add(pair_key)

            for pair in hard_pairs:
                pair_key = tuple(sorted([
                    f"{pair['query_latent_id']}|{pair['query_name']}",
                    f"{pair['candidate_latent_id']}|{pair['candidate_name']}"
                ]))
                if pair_key not in seen_pairs_global:
                    hard_pairs_all.append(pair)
                    seen_pairs_global.add(pair_key)

            # Early exit if we have enough pairs
            if len(light_pairs_all) >= light_target and len(hard_pairs_all) >= hard_target:
                break

    # Trim to targets
    light_pairs_all = light_pairs_all[:light_target]
    hard_pairs_all = hard_pairs_all[:hard_target]

    return light_pairs_all + hard_pairs_all


# ── Index builder ──────────────────────────────────────────────────────────────

def build_index(df: pd.DataFrame) -> NameIndex:
    idx = NameIndex()
    rows_list = df.to_dict("records")
    for i, row in tqdm(enumerate(rows_list), total=len(df), desc="Building index"):
        idx.add(
            row["latent_id"], 
            row["noisy_name"], 
            i,
            tokens=row.get("tokens"),
            ngrams=row.get("ngrams3"),
            phoneme_set=row.get("metaphone_set"),
            init=row.get("initials")
        )
    return idx


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"\n📂  Loading {args.input} …")
    df = pd.read_csv(args.input, low_memory=False)

    # Validate required columns
    required = {"latent_id", "noisy_name"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"❌  Missing columns: {missing}")

    df = df.dropna(subset=["noisy_name", "latent_id"]).reset_index(drop=True)
    df["noisy_name"] = df["noisy_name"].astype(str)
    df["latent_id"]  = df["latent_id"].astype(str)

    # Fill optional columns
    for col in ("is_combined_noise", "mode", "clean_name"):
        if col not in df.columns:
            df[col] = None

    print(f"   Rows: {len(df):,}   |   Unique identities: {df['latent_id'].nunique():,}")

    print("\n⚡ Precomputing normalized features...")
    df["norm_name"] = df["noisy_name"].map(normalize)
    df["tokens"] = df["noisy_name"].map(tokenize)
    df["token_set"] = df["tokens"].map(set)
    df["ngrams3"] = df["noisy_name"].map(lambda x: char_ngrams(x, 3))
    df["metaphone_set"] = df["noisy_name"].map(metaphones)
    df["initials"] = df["noisy_name"].map(initials)
    print("   ✓ Feature precomputation complete")

    # Target pair counts
    total  = args.pairs
    n_pos  = int(total * args.pos_ratio)
    n_lneg = int(total * args.light_ratio)
    n_hneg = total - n_pos - n_lneg

    print(f"\n🎯  Target pairs  →  pos={n_pos:,}  light_neg={n_lneg:,}  hard_neg={n_hneg:,}")

    # ── Build identity groups ──
    print("\n🔧  Grouping by latent_id …")
    identity_groups: Dict[str, List[dict]] = defaultdict(list)
    rows_list = df.to_dict("records")
    for row in rows_list:
        identity_groups[row["latent_id"]].append(row)

    # ── Positives ──
    print("\n✅  Generating positive pairs …")
    pos_pairs = generate_positives(identity_groups, n_pos, rng)
    print(f"   Generated {len(pos_pairs):,} positive pairs")

    # ── Build index ──
    print("\n🔍  Building neighbourhood index …")
    idx = build_index(df)

    # ── Negatives ──
    print("\n❌  Mining negative pairs …")
    neg_pairs = generate_negatives(df, idx, n_lneg, n_hneg, rng,
                                    top_k=args.top_k, sample_frac=args.neg_sample_frac,
                                    num_workers=args.num_workers)
    print(f"   Generated {len(neg_pairs):,} negative pairs")

    # ── Combine & save ──
    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)

    output_df = pd.DataFrame(all_pairs, columns=[
        "pair_id", "query_name", "candidate_name",
        "query_latent_id", "candidate_latent_id",
        "label", "pair_type", "difficulty",
        "shared_tokens", "shared_phonetics",
        "token_jaccard", "char_ngram_similarity",
        "phonetic_similarity", "edit_distance",
        "notes",
    ])

    # ── Stats ──
    print("\n📊  Pair distribution:")
    print(output_df.groupby(["pair_type", "difficulty"]).size().to_string())
    print(f"\n   Total pairs written: {len(output_df):,}")

    output_df.to_csv(args.output, index=False)
    print(f"\n💾  Saved → {args.output}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Training pair generator for name matching")
    p.add_argument("--input",       required=True,    help="Path to your noisy-name CSV")
    p.add_argument("--output",      default="training_pairs.csv")
    p.add_argument("--pairs",       type=int, default=500_000,
                   help="Total pairs to generate (default 500k)")
    p.add_argument("--pos_ratio",   type=float, default=0.40)
    p.add_argument("--light_ratio", type=float, default=0.30)
    p.add_argument("--hard_ratio",  type=float, default=0.30,
                   help="Informational; hard = 1 - pos - light")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--chunk_size",  type=int,   default=50_000,
                   help="Reserved for future chunked writes")
    p.add_argument("--neg_sample_frac", type=float, default=0.3,
                   help="Fraction of rows to sample for negative mining (default 0.3 = 30%%)")
    p.add_argument("--top_k",       type=int,   default=40,
                   help="Max neighbourhood candidates per query (default 40)")
    p.add_argument("--num_workers", type=int,   default=4,
                   help="Number of CPU cores for parallel negative mining (default 4)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
