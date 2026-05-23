import pandas as pd
from dataclasses import dataclass
from transformers import GPT2Tokenizer
import torch
import sys
sys.path.insert(0, "/Users/medihse/Documents/NLP_Project/FINAL/run8")

@dataclass
class Sample:
    example_id: str
    source_id: str
    prompt: str
    gold_label: int
    input_ids: torch.Tensor  # shape: (1, seq_len)

def load_dataset(csv_path: str, pt_path: str, tokenizer, max_length: int = 128, device: str = "cpu") -> list:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["retrieved_context"]).reset_index(drop=True)

    # Load .pt and extract test split info
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    test_ids  = set(str(i) for i in data["test_sample_ids"])
    labels_pt = data["test_labels"]  # tensor, indexed by position in test_sample_ids

    # Build a quick id→label lookup from the .pt
    id_to_label = {
        str(sid): int(labels_pt[i])
        for i, sid in enumerate(data["test_sample_ids"])
    }

    samples = []
    skipped = 0
    for _, row in df.iterrows():
        example_id = str(row["example_id"])

        # Only keep examples that are in the test split
        if example_id not in test_ids:
            skipped += 1
            continue

        prompt_val = row.get("prompt", "")
        if pd.notna(prompt_val) and str(prompt_val).strip():
            text = str(prompt_val).strip()
        else:
            text = str(row["retrieved_context"]).strip()

        label = id_to_label[example_id]  # use .pt label, not CSV label

        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(device)

        samples.append(Sample(
            example_id=example_id,
            source_id="all",
            prompt=text,
            gold_label=label,
            input_ids=input_ids,
        ))

    print(f"[Loader] Loaded {len(samples)} samples (skipped {skipped} non-test rows)")
    print(f"         Faithful: {sum(1 for s in samples if s.gold_label == 1)}, "
          f"Hallucinated: {sum(1 for s in samples if s.gold_label == 0)}")
    return samples