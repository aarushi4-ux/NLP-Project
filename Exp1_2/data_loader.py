"""
RAGTruth Data Loader — verified against actual file schema
===========================================================
response.jsonl columns:
    id, source_id, model, temperature, labels, split, quality, response

    labels: list of {start, end, text, meta, label_type, implicit_true, due_to_null}
    label_type values: 'Evident Conflict', 'Baseless Info', etc.

source.jsonl columns:
    source_id, task_type, source, source_info, prompt
"""

import json
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RAGSample:
    sample_id: str
    question: str
    context: str
    response: str
    label: int                    # 1 = hallucinated, 0 = faithful
    halluc_type: Optional[str]    # 'conflict' | 'baseless' | None
    halluc_spans: List[str] = field(default_factory=list)
    task_type: str = ""
    model: str = ""
    split: str = ""


def load_ragtruth(response_path: str, source_path: str,
                  split: str = None, quality: str = "good") -> List[RAGSample]:
    resp = pd.read_json(response_path, lines=True)
    src  = pd.read_json(source_path,   lines=True)

    if split is not None and "split" in resp.columns:
        resp = resp[resp["split"] == split]
    if quality is not None and "quality" in resp.columns:
        resp = resp[resp["quality"] == quality]

    merged = resp.merge(src, on="source_id")
    print(f"  Loaded {len(merged)} rows  (split={split}, quality={quality})")

    samples: List[RAGSample] = []
    for _, row in merged.iterrows():
        task     = str(row.get("task_type", ""))
        context  = str(row.get("source", ""))
        prompt   = str(row.get("prompt", ""))
        response = str(row.get("response", ""))

        # extract question from prompt
        question = ""
        if prompt:
            for line in prompt.split("\n"):
                if "question" in line.lower():
                    question = line.strip()
                    break
        if not question:
            question = prompt[:200]

        # labels
        raw_labels = row.get("labels", [])
        if not isinstance(raw_labels, list):
            try:
                raw_labels = json.loads(str(raw_labels))
            except Exception:
                raw_labels = []

        label = 0
        halluc_type = None
        spans = []

        if raw_labels and len(raw_labels) > 0:
            label = 1
            for l in raw_labels:
                if not isinstance(l, dict):
                    continue
                lt = l.get("label_type", "").lower()
                if "baseless" in lt:
                    halluc_type = "baseless"
                elif "conflict" in lt:
                    halluc_type = "conflict"
                else:
                    halluc_type = lt
                spans.append(l.get("text", ""))

        samples.append(RAGSample(
            sample_id   = str(row.get("id", len(samples))),
            question    = question,
            context     = context,
            response    = response,
            label       = label,
            halluc_type = halluc_type,
            halluc_spans= spans,
            task_type   = task,
            model       = str(row.get("model", "")),
            split       = str(row.get("split", "")),
        ))

    return samples


def load_ragtruth_official_splits(response_path: str, source_path: str) -> Tuple[List[RAGSample], List[RAGSample]]:
    """Use the dataset's own train/test split column."""
    train = load_ragtruth(response_path, source_path, split="train")
    test  = load_ragtruth(response_path, source_path, split="test")
    return train, test


def split_dataset(samples, train_ratio=0.7, val_ratio=0.1, seed=42):
    import random
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (
        [samples[i] for i in idx[:n_train]],
        [samples[i] for i in idx[n_train:n_train + n_val]],
        [samples[i] for i in idx[n_train + n_val:]],
    )
