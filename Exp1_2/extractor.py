"""
Hidden State Extractor — Track B
Extracts per-token hidden states from every layer of GPT-2-medium.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class HiddenStateBundle:
    """All representations needed for Track B metrics."""
    input_ids: torch.Tensor           # (seq_len,)
    tokens: List[str]
    hidden_states: torch.Tensor       # (n_layers+1, seq_len, hidden_dim)
    attentions: torch.Tensor          # (n_layers, n_heads, seq_len, seq_len)
    logits: torch.Tensor              # (seq_len, vocab_size)
    log_probs: torch.Tensor           # (seq_len, vocab_size)
    n_layers: int
    hidden_dim: int
    attn_outputs: torch.Tensor = None   # (n_layers, seq_len, hidden_dim)
    ffn_outputs:  torch.Tensor = None   # (n_layers, seq_len, hidden_dim)


class HiddenStateExtractor:
    def __init__(self, model_name: str = "gpt2-medium", device: str = None):
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Extractor] Loading {model_name} on {self.device} ...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_attentions=True,
        ).to(self.device).eval()
        self.n_layers = self.model.config.n_layer          # 24 for gpt2-medium
        self.hidden_dim = self.model.config.n_embd         # 1024 for gpt2-medium
        print(f"[Extractor] Ready — {self.n_layers} layers, hidden_dim={self.hidden_dim}")

        #ADDED (for marinus):
        self._attn_out = {}
        self._ffn_out  = {}

        for l, block in enumerate(self.model.transformer.h):
            block.attn.register_forward_hook(
                lambda m, inp, out, l=l:
                    self._attn_out.__setitem__(l, out[0].detach().cpu())
            )
            block.mlp.register_forward_hook(
                lambda m, inp, out, l=l:
                    self._ffn_out.__setitem__(l, out.detach().cpu())
            )

    @torch.no_grad()
    def extract(self, prompt: str, max_length: int = 512) -> HiddenStateBundle:
        """
        Forward-pass `prompt` through GPT-2-medium and return all hidden states.
        """
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(self.device)          # (1, seq_len)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

        # hidden_states: tuple of (n_layers+1) tensors each (1, seq_len, hidden_dim)
        hidden_states = torch.stack(
            [h[0] for h in outputs.hidden_states], dim=0
        )  # (n_layers+1, seq_len, hidden_dim)

        # attentions: tuple of (n_layers) tensors each (1, n_heads, seq_len, seq_len)
        attentions = torch.stack(
            [a[0] for a in outputs.attentions], dim=0
        )  # (n_layers, n_heads, seq_len, seq_len)

        logits = outputs.logits[0]           # (seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        #ADDED (for marinus):
        attn_out_tensor = torch.stack(
            [self._attn_out[l] for l in range(self.n_layers)], dim=0
        )
        ffn_out_tensor = torch.stack(
            [self._ffn_out[l] for l in range(self.n_layers)], dim=0
        )
        attn_out_tensor = attn_out_tensor.squeeze(1)
        ffn_out_tensor = ffn_out_tensor.squeeze(1)


        # Validate both shapes here, where the variables actually exist
        assert attn_out_tensor.shape == (self.n_layers, input_ids.shape[1], self.hidden_dim), \
            f"Unexpected attn_out shape: {attn_out_tensor.shape}"
        assert ffn_out_tensor.shape == (self.n_layers, input_ids.shape[1], self.hidden_dim), \
            f"Unexpected ffn_out shape: {ffn_out_tensor.shape}"
            
        self._attn_out.clear()
        self._ffn_out.clear()

        return HiddenStateBundle(
            input_ids=input_ids[0].cpu(),
            tokens=tokens,
            hidden_states=hidden_states.cpu(),
            attentions=attentions.cpu(),
            logits=logits.cpu(),
            log_probs=log_probs.cpu(),
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            attn_outputs=attn_out_tensor, #ADDED
            ffn_outputs=ffn_out_tensor, #ADDED
        )

    def build_prompt(self, sample) -> str:
        """Concatenate context + question + response as a single string."""
        parts = []
        if sample.context:
            parts.append(f"Context: {sample.context}")
        if sample.question:
            parts.append(f"Question: {sample.question}")
        if sample.response:
            parts.append(f"Answer: {sample.response}")
        return "\n".join(parts)

    def response_token_range(self, tokens: List[str], response: str) -> Tuple[int, int]:
        """
        Return (start, end) index of the response tokens inside the full sequence.
        Approximate: find where 'Answer:' prefix starts.
        """
        prefix = "Answer:"
        # convert tokens back to string and search
        full_str = self.tokenizer.convert_tokens_to_string(tokens)
        ans_pos = full_str.find("Answer:")
        if ans_pos == -1:
            ans_pos = full_str.find(response[:20])
        if ans_pos == -1:
            return (len(tokens) // 2, len(tokens))

        # count chars to find token index
        char_count = 0
        start_idx = 0
        for i, tok in enumerate(tokens):
            tok_str = self.tokenizer.convert_tokens_to_string([tok])
            if char_count >= ans_pos:
                start_idx = i
                break
            char_count += len(tok_str)

        return (start_idx + 1, len(tokens))   # +1 to skip "Answer:" token itself
