from collections import defaultdict
import random


def build_pairs(dataset, max_pairs=50, seed=42):
    """
    Build (faithful, hallucinated) pairs using:
    source_id grouping
    """

    random.seed(seed)

    # -----------------------------
    # Group by (source_id)
    # -----------------------------
    groups = defaultdict(lambda: {"faithful": [], "halluc": []})

    for s in dataset:
        key = s.source_id

        if s.gold_label == 1:
            groups[key]["faithful"].append(s)
        else:
            groups[key]["halluc"].append(s)

    pairs = []

    # -----------------------------
    # Build aligned pairs
    # -----------------------------
    
    for key, group in groups.items():
        f_list = group["faithful"]
        h_list = group["halluc"]

        if len(f_list) > 0 and len(h_list) > 0:
            for f in f_list:
                h = random.choice(h_list)
                pairs.append((f, h))


    # -----------------------------
    # Shuffle + trim
    # -----------------------------
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]

    if len(pairs) == 0:
        raise ValueError("No valid pairs constructed — check grouping logic")
    if len(pairs) < max_pairs:
        print("[PairSelector] Warning: low aligned pairs, adding random fallback")

    faithful = [s for s in dataset if s.gold_label == 1]
    halluc   = [s for s in dataset if s.gold_label == 0]

    random.shuffle(faithful)
    random.shuffle(halluc)

    for f, h in zip(faithful, halluc):
        pairs.append((f, h))
        if len(pairs) >= max_pairs:
            break

    print(f"[PairSelector] Built {len(pairs)} aligned pairs")

    return [(i, f, h) for i, (f, h) in enumerate(pairs)]