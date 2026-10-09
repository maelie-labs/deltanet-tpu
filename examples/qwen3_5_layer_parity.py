# Copyright 2026 maelie-labs.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity verification for Qwen3.5 linear attention layer between PyTorch and JAX.

Loads weights from Qwen3.5 (e.g. Qwen3.5-2B or Qwen3.5-4B) layer 0 and compares:
1. Prefill recurrence: deltanet_tpu.chunk_gdn vs PyTorch reference
2. Single-token decode step: deltanet_tpu.gdn_decode_step vs PyTorch reference
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from deltanet_tpu import chunk_gdn, gdn_decode_step


def load_layer_weights(model_id_or_path: str, layer_idx: int = 0):
    """Load layer weights and config from local path or Hugging Face hub."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from safetensors import safe_open

    # Resolve local path if given a model repo name
    actual_path = model_id_or_path
    if not os.path.exists(actual_path):
        models_dir = os.environ.get("MODELS_DIR", "")
        if models_dir and os.path.exists(os.path.join(models_dir, model_id_or_path)):
            actual_path = os.path.join(models_dir, model_id_or_path)

    cfg = Qwen3_5Config.from_pretrained(actual_path)
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    layer = Qwen3_5GatedDeltaNet(text_cfg, layer_idx=layer_idx)

    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
    state_dict = {}

    # Check for safetensors index file (multi-shard models such as Qwen3.5-4B)
    index_file = os.path.join(actual_path, "model.safetensors.index.json") if os.path.isdir(actual_path) else None
    if index_file and os.path.exists(index_file):
        with open(index_file) as f:
            idx_data = json.load(f)
        needed_shards = set()
        for k, shard in idx_data.get("weight_map", {}).items():
            if k.startswith(prefix):
                needed_shards.add(os.path.join(actual_path, shard))
        for s_file in needed_shards:
            with safe_open(s_file, framework="pt") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        sub_k = k[len(prefix):]
                        state_dict[sub_k] = f.get_tensor(k)
    else:
        st_file = None
        if os.path.isdir(actual_path):
            candidates = [
                os.path.join(actual_path, "model.safetensors-00001-of-00001.safetensors"),
                os.path.join(actual_path, "model.safetensors"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    st_file = c
                    break
        if st_file is None and not os.path.isdir(actual_path):
            from huggingface_hub import hf_hub_download
            try:
                st_file = hf_hub_download(repo_id=actual_path, filename="model.safetensors")
            except Exception:
                st_file = hf_hub_download(
                    repo_id=actual_path, filename="model.safetensors-00001-of-00001.safetensors"
                )
        if st_file is not None and os.path.exists(st_file):
            with safe_open(st_file, framework="pt") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        sub_k = k[len(prefix):]
                        state_dict[sub_k] = f.get_tensor(k)

    expected_keys = {
        "in_proj_qkv.weight",
        "in_proj_a.weight",
        "in_proj_b.weight",
        "conv1d.weight",
        "A_log",
        "dt_bias",
    }
    missing_keys = expected_keys - set(state_dict.keys())
    assert not missing_keys, f"Missing expected layer weights from {actual_path}: {missing_keys}"

    layer.load_state_dict(state_dict, strict=False)
    layer = layer.float().eval()
    return layer, text_cfg


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 layer parity test")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="Path or Hugging Face repo ID for Qwen3.5 weights (e.g. 'Qwen/Qwen3.5-2B' or 'Qwen/Qwen3.5-4B'). "
             "If the MODELS_DIR environment variable is set, models will also be searched in $MODELS_DIR/<model>.",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for prefill")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    clean_model_name = args.model
    if "Qwen3.5-4B" in clean_model_name:
        clean_model_name = "Qwen/Qwen3.5-4B"
    elif "Qwen3.5-2B" in clean_model_name:
        clean_model_name = "Qwen/Qwen3.5-2B"

    print(f"Loading Qwen3.5 layer from {clean_model_name}...")
    layer, cfg = load_layer_weights(args.model, layer_idx=0)
    print(f"Loaded layer 0 (Hk={cfg.linear_num_key_heads}, Hv={cfg.linear_num_value_heads}, K={cfg.linear_key_head_dim}, V={cfg.linear_value_head_dim})")

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    B, L = 1, args.seq_len
    torch.manual_seed(42)
    x = torch.randn(B, L, cfg.hidden_size, dtype=torch.float32)

    # 1. Project inputs with layer weights
    mixed_qkv = layer.in_proj_qkv(x).transpose(1, 2)
    pad = layer.conv_kernel_size - 1
    padded = F.pad(mixed_qkv, (pad, 0))
    conv_out = F.conv1d(padded, layer.conv1d.weight, groups=layer.conv_dim)
    mixed_qkv = F.silu(conv_out)[:, :, :mixed_qkv.shape[-1]].transpose(1, 2)

    q, k, v = torch.split(mixed_qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    q = q.reshape(B, L, layer.num_k_heads, layer.head_k_dim)
    k = k.reshape(B, L, layer.num_k_heads, layer.head_k_dim)
    v = v.reshape(B, L, layer.num_v_heads, layer.head_v_dim)

    b = layer.in_proj_b(x)
    a = layer.in_proj_a(x)
    beta = b.sigmoid()
    g = -layer.A_log.exp() * F.softplus(a + layer.dt_bias)

    repeat_ratio = layer.num_v_heads // layer.num_k_heads
    q_pt = q.repeat_interleave(repeat_ratio, dim=2) if repeat_ratio > 1 else q
    k_pt = k.repeat_interleave(repeat_ratio, dim=2) if repeat_ratio > 1 else k

    # 2. Run PyTorch reference prefill
    pt_out, pt_state = torch_chunk_gated_delta_rule(
        q_pt, k_pt, v, g=g, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
    )

    # 3. Run JAX deltanet_tpu prefill (passes unrepeated q and k directly)
    q_j = jnp.array(q.detach().numpy())
    k_j = jnp.array(k.detach().numpy())
    v_j = jnp.array(v.detach().numpy())
    g_j = jnp.array(g.detach().numpy())
    beta_j = jnp.array(beta.detach().numpy())

    jax_out, jax_state = chunk_gdn(
        q_j, k_j, v_j, g_j, beta_j,
        output_final_state=True,
        is_inference=True,
        precision="fp32",
        qk_l2norm=True,
    )

    pt_out_np = pt_out.detach().numpy()
    jax_out_np = np.array(jax_out)
    rel_l2_prefill_out = np.linalg.norm(pt_out_np - jax_out_np) / np.linalg.norm(pt_out_np)

    pt_state_np = pt_state.detach().numpy()
    jax_state_np = np.array(jax_state)
    rel_l2_prefill_state = np.linalg.norm(pt_state_np - jax_state_np) / np.linalg.norm(pt_state_np)

    print(f"Prefill (L={L}) Output Rel L2 Error:      {rel_l2_prefill_out:.4e}")
    print(f"Prefill (L={L}) Final State Rel L2 Error: {rel_l2_prefill_state:.4e}")

    # 4. Decode step on subsequent token
    x_tok = torch.randn(B, 1, cfg.hidden_size, dtype=torch.float32)
    b_tok = layer.in_proj_b(x_tok)
    a_tok = layer.in_proj_a(x_tok)
    beta_tok = b_tok.sigmoid()
    g_tok = -layer.A_log.exp() * F.softplus(a_tok + layer.dt_bias)
    q_tok = torch.randn(B, 1, layer.num_k_heads, layer.head_k_dim, dtype=torch.float32)
    k_tok = torch.randn(B, 1, layer.num_k_heads, layer.head_k_dim, dtype=torch.float32)
    v_tok = torch.randn(B, 1, layer.num_v_heads, layer.head_v_dim, dtype=torch.float32)

    q_tok_pt = q_tok.repeat_interleave(repeat_ratio, dim=2) if repeat_ratio > 1 else q_tok
    k_tok_pt = k_tok.repeat_interleave(repeat_ratio, dim=2) if repeat_ratio > 1 else k_tok

    pt_dec_out, pt_dec_state = torch_recurrent_gated_delta_rule(
        q_tok_pt, k_tok_pt, v_tok, g=g_tok, beta=beta_tok, initial_state=pt_state, output_final_state=True, use_qk_l2norm_in_kernel=True
    )

    jax_dec_out, jax_dec_state = gdn_decode_step(
        jnp.array(q_tok.detach().numpy()),
        jnp.array(k_tok.detach().numpy()),
        jnp.array(v_tok.detach().numpy()),
        jnp.array(g_tok.detach().numpy()),
        jnp.array(beta_tok.detach().numpy()),
        jax_state,
        qk_l2norm=True,
    )

    pt_dec_out_np = pt_dec_out.detach().numpy()
    jax_dec_out_np = np.array(jax_dec_out)
    rel_l2_dec_out = np.linalg.norm(pt_dec_out_np - jax_dec_out_np) / np.linalg.norm(pt_dec_out_np)

    pt_dec_state_np = pt_dec_state.detach().numpy()
    jax_dec_state_np = np.array(jax_dec_state)
    rel_l2_dec_state = np.linalg.norm(pt_dec_state_np - jax_dec_state_np) / np.linalg.norm(pt_dec_state_np)

    print(f"Decode Step Output Rel L2 Error:          {rel_l2_dec_out:.4e}")
    print(f"Decode Step Final State Rel L2 Error:     {rel_l2_dec_state:.4e}")

    assert rel_l2_prefill_out < 1e-4, f"Prefill output diff too high: {rel_l2_prefill_out}"
    assert rel_l2_prefill_state < 1e-4, f"Prefill state diff too high: {rel_l2_prefill_state}"
    assert rel_l2_dec_out < 1e-4, f"Decode output diff too high: {rel_l2_dec_out}"
    assert rel_l2_dec_state < 1e-4, f"Decode state diff too high: {rel_l2_dec_state}"
    print("SUCCESS: JAX deltanet_tpu and PyTorch Qwen3.5 linear attention match to within tolerance.")

    import subprocess
    try:
        tree_hash = subprocess.check_output(["git", "rev-parse", "HEAD:src"], text=True).strip()
    except Exception:
        tree_hash = "unknown"

    import transformers
    results = {
        "model": clean_model_name,
        "layer_idx": 0,
        "device": str(jax.devices()[0]),
        "backend": jax.default_backend(),
        "tree_hash": tree_hash,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "jax_version": jax.__version__,
        "precision": "fp32",
        "shapes": {
            "Hk": int(cfg.linear_num_key_heads),
            "Hv": int(cfg.linear_num_value_heads),
            "K": int(cfg.linear_key_head_dim),
            "V": int(cfg.linear_value_head_dim),
        },
        "prefill": {
            "seq_len": L,
            "output_rel_l2": float(rel_l2_prefill_out),
            "final_state_rel_l2": float(rel_l2_prefill_state),
        },
        "decode": {
            "steps": 1,
            "output_rel_l2": float(rel_l2_dec_out),
            "final_state_rel_l2": float(rel_l2_dec_state),
        },
    }
    out_dir = "benchmarks/results"
    if os.path.exists(out_dir):
        if args.output:
            out_path = args.output
        elif "4B" in clean_model_name:
            out_path = os.path.join(out_dir, "qwen3_5_4b_parity_results.json")
        else:
            out_path = os.path.join(out_dir, "qwen3_5_parity_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved parity results to {out_path}")


if __name__ == "__main__":
    main()
