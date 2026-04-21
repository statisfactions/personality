#!/usr/bin/env python3
"""Deduplicate contrast pairs via sentence-transformer embeddings.

Embeds each pair's `situation` text, computes pairwise cosine similarity,
and removes near-duplicates above a threshold. Within-facet and against-anchor
(existing training + holdout) dedup are both applied.

Usage:
    python scripts/dedup_pairs.py --input instruments/contrast_pairs_stratified.json
    python scripts/dedup_pairs.py --threshold 0.85 --dry-run
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXISTING_TRAINING = Path("instruments/contrast_pairs.json")
EXISTING_HOLDOUT = Path("instruments/contrast_pairs_holdout.json")


def load_anchor_situations(trait):
    anchors = []
    for path in [EXISTING_TRAINING, EXISTING_HOLDOUT]:
        if not path.exists():
            continue
        with open(path) as f:
            cp = json.load(f)
        for p in cp.get("traits", {}).get(trait, {}).get("pairs", []):
            anchors.append({"situation": p["situation"], "source": str(path.name)})
    return anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="instruments/contrast_pairs_stratified.json")
    parser.add_argument("--output", default=None,
                        help="Output file (default: overwrite input)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for near-dup (default 0.85)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write; just report")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    with open(in_path) as f:
        data = json.load(f)

    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    report_rows = []
    total_in, total_out = 0, 0
    for trait, tdata in data["traits"].items():
        anchors = load_anchor_situations(trait)
        new_pairs = tdata["pairs"]
        if not new_pairs:
            continue

        # Embed anchors + new pairs together
        anchor_texts = [a["situation"] for a in anchors]
        new_texts = [p["situation"] for p in new_pairs]
        if anchor_texts:
            a_emb = model.encode(anchor_texts, convert_to_numpy=True, normalize_embeddings=True)
        else:
            a_emb = np.zeros((0, model.get_sentence_embedding_dimension()))
        n_emb = model.encode(new_texts, convert_to_numpy=True, normalize_embeddings=True)

        # Similarity of each new pair against anchors + earlier new pairs
        keep = []
        drop_reasons = []
        kept_emb_rows = []  # embeddings of already-kept new pairs
        for i, p in enumerate(new_pairs):
            # Check against anchors
            drop = None
            if a_emb.shape[0] > 0:
                sims_a = a_emb @ n_emb[i]
                j = int(np.argmax(sims_a))
                if sims_a[j] > args.threshold:
                    drop = (f"cos={sims_a[j]:.3f} vs anchor", anchors[j]["situation"])
            # Check against already-kept new pairs (within this trait)
            if drop is None and kept_emb_rows:
                kept_mat = np.vstack(kept_emb_rows)
                sims_k = kept_mat @ n_emb[i]
                j = int(np.argmax(sims_k))
                if sims_k[j] > args.threshold:
                    drop = (f"cos={sims_k[j]:.3f} vs within-batch",
                            new_pairs[keep[j]]["situation"])
            if drop is None:
                keep.append(i)
                kept_emb_rows.append(n_emb[i])
            else:
                drop_reasons.append((i, p["situation"], drop))

        kept = [new_pairs[i] for i in keep]
        print(f"\n{trait}: {len(new_pairs)} -> {len(kept)} (dropped {len(new_pairs) - len(kept)})")
        for idx, sit, (reason, match) in drop_reasons[:5]:
            print(f"  drop: \"{sit[:70]}...\"  [{reason}]")
            print(f"        matched: \"{match[:70]}...\"")

        # Per-facet counts
        by_facet = defaultdict(lambda: [0, 0])
        for p in new_pairs:
            by_facet[p.get("facet", "?")][0] += 1
        for p in kept:
            by_facet[p.get("facet", "?")][1] += 1
        facet_str = ", ".join(f"{f}:{b}/{a}" for f, (a, b) in sorted(by_facet.items()))
        print(f"  by facet (kept/generated): {facet_str}")

        total_in += len(new_pairs)
        total_out += len(kept)
        report_rows.append((trait, len(new_pairs), len(kept)))

        tdata["pairs"] = kept

    print(f"\n=== Totals ===\n  in: {total_in}  kept: {total_out}  dropped: {total_in - total_out}")

    if args.dry_run:
        print("\n(--dry-run, not writing)")
        return

    data.setdefault("generation", {})["dedup"] = {
        "embedding_model": EMBEDDING_MODEL,
        "threshold": args.threshold,
        "pre_dedup_counts": {r[0]: r[1] for r in report_rows},
        "post_dedup_counts": {r[0]: r[2] for r in report_rows},
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
