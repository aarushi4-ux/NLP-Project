import json
from transformers import AutoTokenizer

MODEL_NAME = "gpt2-medium"

# load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def get_onset_token(offsets, spans):
    if not spans:
        return None

    span_start = min(s for s, _ in spans)

    best_i = None
    best_dist = float("inf")

    for i, (s, e) in enumerate(offsets):
        # skip invalid
        if s == 0 and e == 0:
            continue

        # Case 1: exact overlap → return immediately
        if s <= span_start < e:
            return i

        # Case 2: find closest token to the RIGHT
        if s >= span_start:
            dist = s - span_start
            if dist < best_dist:
                best_dist = dist
                best_i = i

    return best_i


#convert character-level spans to token-level labels
def get_token_labels(text, spans, tokenizer):
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    
    offsets = enc["offset_mapping"]
    labels = [0] * len(offsets)
    
    for i, (tok_start, tok_end) in enumerate(offsets):
        for span in spans:
            span_start, span_end = span
            if not (tok_end < span_start or tok_start > span_end):
                labels[i] = 1
                break
    
    return enc["input_ids"], offsets, labels


def char_to_token_spans(text, spans, tokenizer):
    """
    Convert character-level spans to token-level spans.
    spans: list of (char_start, char_end)
    returns: list of (token_start, token_end)
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoding["offset_mapping"]
    token_spans = []

    for char_start, char_end in spans:
        token_start = None
        token_end = None

        for i, (start, end) in enumerate(offsets):
            if start <= char_start < end:
                token_start = i
            if start < char_end <= end:          # FIX: was < end, missed exact boundary
                token_end = i
                break

        # FIX: fallback is now OUTSIDE the loop, runs only when exact match failed
        if token_start is None:
            for i, (start, end) in enumerate(offsets):
                if start >= char_start:
                    token_start = i
                    break

        if token_end is None:
            for i in reversed(range(len(offsets))):
                start, end = offsets[i]
                if end <= char_end:              # FIX: was < char_end, off-by-one
                    token_end = i
                    break

        if token_start is not None and token_end is not None:
            token_spans.append((token_start, token_end))

    return token_spans
    

def extract_spans(example):
    """
    Extract (start, end) spans from RAGTruth labels.
    Returns a list of tuples.
    """
    spans = []
    for label in example.get("labels", []):
        spans.append((label["start"], label["end"]))

    return spans


def load_ragtruth(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def main():
    data_path = "../RAGTruth-main/dataset/response.jsonl"
    data = load_ragtruth(data_path)

    processed = []

    for ex in data:
        text = ex["response"]
        spans = extract_spans(ex)                         # FIX: single call, no duplicate

        token_spans = char_to_token_spans(text, spans, tokenizer)   # FIX: result is now used

        input_ids, offsets, token_labels = get_token_labels(
            text, spans, tokenizer
        )

        t_list = []
        for span in spans:
            t_span = get_onset_token(offsets, [span])
            t_list.append(t_span)

        processed.append({
            "input_ids": input_ids,
            "offsets": offsets,
            "labels": token_labels,
            "spans": spans,
            "token_spans": token_spans,
            "t": t_list,
            "seq_len": len(input_ids) 
        })

    with open("e4_processed.json", "w") as f:
        json.dump(processed, f)
        
if __name__ == "__main__":
    main()