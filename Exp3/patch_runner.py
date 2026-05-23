import torch
import time
import os
from scipy.stats import wilcoxon
from collections import defaultdict
import numpy as np


def save_activations(pairs, model, device, out_dir="activations"):
    '''
    Resulting shape per file:
    (num_layers=24, seq_len, hidden_dim=1024)   # for gpt2-medium
    '''
    os.makedirs(out_dir, exist_ok=True)

    for i, f_sample, h_sample in pairs:
        f_ids = f_sample.input_ids.to(device)
        h_ids = h_sample.input_ids.to(device)

        f_attn, f_ffn, f_res = extract_components(model, f_ids)
        h_attn, h_ffn, h_res = extract_components(model, h_ids)

        torch.save(torch.stack(f_res), f"{out_dir}/pair{i}_faithful_res.pt")
        torch.save(torch.stack(h_res), f"{out_dir}/pair{i}_halluc_res.pt")

        # Stack layers → shape: (num_layers, seq_len, hidden_dim)
        torch.save(torch.stack(f_attn), f"{out_dir}/pair{i}_faithful_attn.pt")
        torch.save(torch.stack(f_ffn),  f"{out_dir}/pair{i}_faithful_ffn.pt")
        torch.save(torch.stack(h_attn), f"{out_dir}/pair{i}_halluc_attn.pt")
        torch.save(torch.stack(h_ffn),  f"{out_dir}/pair{i}_halluc_ffn.pt")


def forward_with_patch(model, input_ids, patch_dict):
    handles = []

    def make_hook(layer_idx, component, patch_dict):
        def hook(module, inp, out):
            if (layer_idx, component) not in patch_dict:
                return out  # not a patched layer, pass through
            if component == "attn" and isinstance(out, tuple):
                return (patch_dict[(layer_idx, component)],) + out[1:]
            return patch_dict[(layer_idx, component)]
        return hook

    for layer_idx, layer in enumerate(model.transformer.h):
        handles.append(
            layer.attn.register_forward_hook(
                make_hook(layer_idx, "attn", patch_dict)
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                make_hook(layer_idx, "ffn", patch_dict)
            )
        )

    with torch.no_grad():
        outputs = model(input_ids)

    for h in handles:
        h.remove()

    return outputs.logits

    
def extract_components(model, input_ids):
    attn_outputs = []
    ffn_outputs = []
    residual_stream = []  # ADD

    def attn_hook(module, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        attn_outputs.append(tensor.detach())

    def ffn_hook(module, inp, out):
        ffn_outputs.append(out.detach())

    def residual_hook(module, inp, out):  # ADD
        tensor = out[0] if isinstance(out, tuple) else out
        residual_stream.append(tensor.detach())

    handles = []

    for layer in model.transformer.h:
        handles.append(layer.attn.register_forward_hook(attn_hook))
        handles.append(layer.mlp.register_forward_hook(ffn_hook))
        handles.append(layer.register_forward_hook(residual_hook))  # ADD

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return attn_outputs, ffn_outputs, residual_stream  # ADD


def compute_cie(base_logits, patched_logits, mode="mean"):
    base_prob = torch.softmax(base_logits, dim=-1)
    patched_prob = torch.softmax(patched_logits, dim=-1)

    diff = torch.abs(base_prob - patched_prob).sum(dim=-1)  # (batch, seq)

    if mode == "mean":
        return diff.mean().item()
    elif mode == "max":
        return diff.max().item()
    elif mode == "last":
        return diff[:, -1].mean().item()

# ----Main Patch Runer----------------------------
def run_patching_experiment(model, pairs, device, target_layers=None):

    total_pairs = len(pairs)
    n_layers = len(model.transformer.h)
    total_steps = total_pairs * 2 * n_layers  # 2 directions

    step = 0
    start_time = time.time()

    results = []

    with torch.no_grad():
        for i, f_sample, h_sample in pairs:

            f_ids = f_sample.input_ids.to(device)
            h_ids = h_sample.input_ids.to(device)

            min_len = min(f_ids.shape[1], h_ids.shape[1])
            f_ids = f_ids[:, :min_len]
            h_ids = h_ids[:, :min_len]

            # Extract components
            f_attn, f_ffn, f_res = extract_components(model, f_ids)
            h_attn, h_ffn, h_res = extract_components(model, h_ids)

            # Base logits
            f_logits = model(f_ids).logits
            h_logits = model(h_ids).logits

            n_layers = len(f_attn)
            layers_to_run = target_layers if target_layers else range(n_layers)

            # ---- TWO DIRECTIONS ----
            directions = [
                ("f_to_h", f_ids, h_ids, f_attn, f_ffn, h_attn, h_ffn, h_logits),
                ("h_to_f", h_ids, f_ids, h_attn, h_ffn, f_attn, f_ffn, f_logits),
            ]

            for direction, src_ids, tgt_ids, src_attn, src_ffn, tgt_attn, tgt_ffn, tgt_logits in directions:

                for layer in layers_to_run:

                    assert src_attn[layer].shape == tgt_attn[layer].shape, f"Attn mismatch at layer {layer}"
                    assert src_ffn[layer].shape == tgt_ffn[layer].shape, f"FFN mismatch at layer {layer}"

                    # ---- ATTENTION PATCH ----
                    patch_dict = {
                        (layer, "attn"): src_attn[layer]
                    }

                    patched_logits = forward_with_patch(model, tgt_ids, patch_dict)
                    cie_attn = compute_cie(tgt_logits, patched_logits)

                    # ---- FFN PATCH ----
                    patch_dict = {
                        (layer, "ffn"): src_ffn[layer]
                    }

                    patched_logits = forward_with_patch(model, tgt_ids, patch_dict)
                    cie_ffn = compute_cie(tgt_logits, patched_logits)

                    results.append({
                        "pair_idx": i,
                        "pair_id": f"{f_sample.source_id}_{h_sample.source_id}",
                        "direction": direction,
                        "layer": layer,
                        "component": "attn",
                        "cie": cie_attn,
                        "n_layers": n_layers
                    })

                    results.append({
                        "pair_idx": i,
                        "pair_id": f"{f_sample.source_id}_{h_sample.source_id}",
                        "direction": direction,
                        "layer": layer,
                        "component": "ffn",
                        "cie": cie_ffn,
                        "n_layers": n_layers
                    })

                    step += 1

                    if step % 10 == 0:  # update every 10 steps to reduce spam
                        elapsed = time.time() - start_time
                        avg_time = elapsed / step
                        remaining = avg_time * (total_steps - step)

                        print(
                            f"\r[Exp3] {step}/{total_steps} steps | "
                            f"Elapsed: {elapsed/60:.1f} min | "
                            f"ETA: {remaining/60:.1f} min",
                            end="",
                            flush=True
                        )
            
            # Group CIE scores by (layer, component, direction)
            cie_by_group = defaultdict(list)
            for r in results:
                key = (r["layer"], r["component"], r["direction"])
                cie_by_group[key].append(r["cie"])

            # For each layer/component, test f_to_h vs h_to_f CIE distributions
            p_values = {}
            for layer in range(n_layers):
                for comp in ["attn", "ffn"]:
                    f2h = [r["cie"] for r in results if r["layer"]==layer and r["component"]==comp and r["direction"]=="f_to_h"]
                    h2f = [r["cie"] for r in results if r["layer"]==layer and r["component"]==comp and r["direction"]=="h_to_f"]
                    if len(f2h) > 10 and len(h2f) > 10:
                        min_len = min(len(f2h), len(h2f))
                        stat, p = wilcoxon(f2h[:min_len], h2f[:min_len])
                        p_values[(layer, comp)] = p

            # Add p_value to each result row
            for r in results:
                r["p_value"] = p_values.get((r["layer"], r["component"]), float("nan"))

        return results