#!/usr/bin/env python3
"""Benchmark public model delta-rule layer shapes on Google Cloud TPU.

Evaluates single-layer latency and throughput across:
1. GDN-2 (12 key / 24 value heads): Hk=12, Hv=24, K=128, V=128 (grouped heads G=2, channel decay)
2. Qwen3.5-2B GDN: Hk=16, Hv=16, K=128, V=128 (scalar decay, shared gate beta)
3. Qwen3.5-4B / Qwen3-Next GDN: Hk=16, Hv=32, K=128, V=128 (scalar decay, shared gate beta, G=2)
4. Kimi Linear KDA: Hk=32, Hv=32, K=128, V=128 (channel decay, shared gate beta)
5. GDN-2 pure-JAX reference (fallback_scan): Hk=12, Hv=24, K=128, V=128
"""

import json
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import jax
import jax.numpy as jnp

from deltanet_tpu import (
    chunk_delta_rule,
    chunk_gdn,
    chunk_gdn2,
    chunk_kda,
    gdn_decode_step,
    gdn2_decode_step,
    kda_decode_step,
)


def bench_shape(name, variant_type, Hk, Hv, K, V, L=8192, iters=20, warmup=5):
    print(f"\n=== Benchmarking {name} ({variant_type}): Hk={Hk}, Hv={Hv}, K={K}, V={V}, L={L} ===", flush=True)
    key = jax.random.PRNGKey(42)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (1, L, Hk, K), dtype=jnp.float32)
    k = jax.random.normal(k1, (1, L, Hk, K), dtype=jnp.float32)
    v = jax.random.normal(k2, (1, L, Hv, V), dtype=jnp.float32)
    b = jax.nn.sigmoid(jax.random.normal(k4, (1, L, Hk, K), dtype=jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k5, (1, L, Hv, V), dtype=jnp.float32) * 0.39)

    if variant_type == "gdn":
        gate_h = Hv if (name.startswith("Qwen3.5-4B") or name.startswith("Qwen3-Next")) else Hk
        g = -jnp.exp(jax.random.normal(k3, (1, L, gate_h, 1), dtype=jnp.float32) * 0.39 - 1.0)
        b = jax.nn.sigmoid(jax.random.normal(k4, (1, L, gate_h, 1), dtype=jnp.float32) * 0.39)
        def fwd_fn(q, k, v, g, b):
            return chunk_gdn(q, k, v, g, b, precision="bf16")
        def inf_fn(q, k, v, g, b):
            return chunk_gdn(q, k, v, g, b, precision="bf16", is_inference=True)
        def loss_fn(q, k, v, g, b):
            out = chunk_gdn(q, k, v, g, b, precision="bf16")
            return jnp.sum(out)
        fwd_jit = jax.jit(fwd_fn)
        inf_jit = jax.jit(inf_fn)
        fwdbwd_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4)))
        call_fwd = lambda: fwd_jit(q, k, v, g, b)
        call_inf = lambda: inf_jit(q, k, v, g, b)
        call_fwdbwd = lambda: fwdbwd_jit(q, k, v, g, b)
        step_fn = jax.jit(gdn_decode_step)
        g_dec = g[:, 0]
        beta_dec = b[:, 0]
    elif variant_type == "kda":
        g = -jnp.exp(jax.random.normal(k3, (1, L, Hk, K), dtype=jnp.float32) * 0.39 - 1.0)
        b = jax.nn.sigmoid(jax.random.normal(k4, (1, L, Hk, 1), dtype=jnp.float32) * 0.39)
        def fwd_fn(q, k, v, g, b):
            return chunk_kda(q, k, v, g, b, precision="bf16")
        def inf_fn(q, k, v, g, b):
            return chunk_kda(q, k, v, g, b, precision="bf16", is_inference=True)
        def loss_fn(q, k, v, g, b):
            out = chunk_kda(q, k, v, g, b, precision="bf16")
            return jnp.sum(out)
        fwd_jit = jax.jit(fwd_fn)
        inf_jit = jax.jit(inf_fn)
        fwdbwd_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4)))
        call_fwd = lambda: fwd_jit(q, k, v, g, b)
        call_inf = lambda: inf_jit(q, k, v, g, b)
        call_fwdbwd = lambda: fwdbwd_jit(q, k, v, g, b)
        step_fn = jax.jit(kda_decode_step)
        g_dec = g[:, 0]
        beta_dec = b[:, 0]
    elif variant_type == "fallback_scan":
        g = -jnp.exp(jax.random.normal(k3, (1, L, Hk, K), dtype=jnp.float32) * 0.39 - 1.0)
        def fwd_fn(q, k, v, g, b, w):
            return chunk_gdn2(q, k, v, g, b, w, precision="bf16", fallback_scan=True)
        def inf_fn(q, k, v, g, b, w):
            return chunk_gdn2(q, k, v, g, b, w, precision="bf16", fallback_scan=True, is_inference=True)
        def loss_fn(q, k, v, g, b, w):
            out = chunk_gdn2(q, k, v, g, b, w, precision="bf16", fallback_scan=True)
            return jnp.sum(out)
        fwd_jit = jax.jit(fwd_fn)
        inf_jit = jax.jit(inf_fn)
        fwdbwd_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5)))
        call_fwd = lambda: fwd_jit(q, k, v, g, b, w)
        call_inf = lambda: inf_jit(q, k, v, g, b, w)
        call_fwdbwd = lambda: fwdbwd_jit(q, k, v, g, b, w)
        step_fn = jax.jit(gdn2_decode_step)
        g_dec = g[:, 0]
        beta_dec = b[:, 0]
    else:  # gdn2
        g = -jnp.exp(jax.random.normal(k3, (1, L, Hk, K), dtype=jnp.float32) * 0.39 - 1.0)
        def fwd_fn(q, k, v, g, b, w):
            return chunk_gdn2(q, k, v, g, b, w, precision="bf16")
        def inf_fn(q, k, v, g, b, w):
            return chunk_gdn2(q, k, v, g, b, w, precision="bf16", is_inference=True)
        def loss_fn(q, k, v, g, b, w):
            out = chunk_gdn2(q, k, v, g, b, w, precision="bf16")
            return jnp.sum(out)
        fwd_jit = jax.jit(fwd_fn)
        inf_jit = jax.jit(inf_fn)
        fwdbwd_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5)))
        call_fwd = lambda: fwd_jit(q, k, v, g, b, w)
        call_inf = lambda: inf_jit(q, k, v, g, b, w)
        call_fwdbwd = lambda: fwdbwd_jit(q, k, v, g, b, w)
        step_fn = jax.jit(gdn2_decode_step)
        g_dec = g[:, 0]
        beta_dec = b[:, 0]

    # 1. Training Forward
    for _ in range(warmup):
        res = call_fwd()
        res.block_until_ready()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = call_fwd()
        res.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000.0)
    fwd_ms = float(jnp.median(jnp.array(times)))

    # 2. Training Forward + Backward
    for _ in range(warmup):
        out_l, grads = call_fwdbwd()
        jax.block_until_ready((out_l, grads))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out_l, grads = call_fwdbwd()
        jax.block_until_ready((out_l, grads))
        times.append((time.perf_counter() - t0) * 1000.0)
    fwdbwd_ms = float(jnp.median(jnp.array(times)))

    # 3. Inference Prefill
    for _ in range(warmup):
        res = call_inf()
        res.block_until_ready()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = call_inf()
        res.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000.0)
    prefill_ms = float(jnp.median(jnp.array(times)))
    prefill_tok_s = L / (prefill_ms / 1000.0)

    # 4. Inference Decode (B=1 and B=8)
    if variant_type == "fallback_scan":
        decode_results = None
    else:
        decode_results = {}
        for B_dec in [1, 8]:
            q_d = jax.random.normal(k0, (B_dec, Hk, K), dtype=jnp.float32)
            k_d = jax.random.normal(k1, (B_dec, Hk, K), dtype=jnp.float32)
            v_d = jax.random.normal(k2, (B_dec, Hv, V), dtype=jnp.float32)
            gate_h = g_dec.shape[1]
            beta_h = beta_dec.shape[1]
            g_d = jnp.broadcast_to(g_dec, (B_dec, gate_h, g_dec.shape[-1]))
            b_d = jnp.broadcast_to(beta_dec, (B_dec, beta_h, beta_dec.shape[-1]))
            w_d = jnp.broadcast_to(w[:, 0], (B_dec, Hv, V))
            s_d = jnp.zeros((B_dec, Hv, K, V), dtype=jnp.float32)

            if variant_type in ["gdn", "kda"]:
                for _ in range(warmup):
                    o, s_d = step_fn(q_d, k_d, v_d, g_d, b_d, s_d)
                    o.block_until_ready()
                # Pipelined decode: 100 steps dispatched, synchronized at end
                t0 = time.perf_counter()
                for _ in range(100):
                    o, s_d = step_fn(q_d, k_d, v_d, g_d, b_d, s_d)
                o.block_until_ready()
                t1 = time.perf_counter()
                pipelined_ms = ((t1 - t0) / 100.0) * 1000.0

                # Step-by-step synchronous decode: synchronized at each step
                t0 = time.perf_counter()
                for _ in range(20):
                    o, s_d = step_fn(q_d, k_d, v_d, g_d, b_d, s_d)
                    o.block_until_ready()
                t1 = time.perf_counter()
                sync_ms = ((t1 - t0) / 20.0) * 1000.0
            else:
                for _ in range(warmup):
                    o, s_d = step_fn(q_d, k_d, v_d, g_d, b_d, w_d, s_d)
                    o.block_until_ready()
                t0 = time.perf_counter()
                for _ in range(100):
                    o, s_d = step_fn(q_d, k_d, v_d, g_d, b_d, w_d, s_d)
                o.block_until_ready()
                t1 = time.perf_counter()
                pipelined_ms = ((t1 - t0) / 100.0) * 1000.0

                t0 = time.perf_counter()
                for _ in range(20):
                    o, s_d = step_fn(q_d, k_d, v_d, g_d, b_d, w_d, s_d)
                    o.block_until_ready()
                t1 = time.perf_counter()
                sync_ms = ((t1 - t0) / 20.0) * 1000.0

            tok_s = (B_dec * 1000.0) / pipelined_ms
            decode_results[f"b{B_dec}"] = {
                "pipelined_step_ms": pipelined_ms,
                "step_by_step_ms": sync_ms,
                "tokens_per_second": tok_s,
            }

    print(f"Forward: {fwd_ms:.2f} ms | Forward+Backward: {fwdbwd_ms:.2f} ms", flush=True)
    print(f"Prefill: {prefill_ms:.2f} ms ({prefill_tok_s:.0f} tok/s)", flush=True)
    if decode_results is not None:
        print(f"Decode B=1: {decode_results['b1']['pipelined_step_ms']:.4f} ms pipelined ({decode_results['b1']['step_by_step_ms']:.4f} ms sync) [{decode_results['b1']['tokens_per_second']:.0f} tok/s]", flush=True)
        print(f"Decode B=8: {decode_results['b8']['pipelined_step_ms']:.4f} ms pipelined ({decode_results['b8']['step_by_step_ms']:.4f} ms sync) [{decode_results['b8']['tokens_per_second']:.0f} tok/s]", flush=True)

    return {
        "model_name": name,
        "variant": variant_type,
        "Hk": Hk,
        "Hv": Hv,
        "K": K,
        "V": V,
        "seq_len": L,
        "forward_ms": fwd_ms,
        "forward_backward_ms": fwdbwd_ms,
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_second": prefill_tok_s,
        "decode": decode_results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark public shapes")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args, _ = parser.parse_known_args()

    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"JAX devices: {jax.devices()}", flush=True)

    shapes = [
        ("GDN-2 (12 key / 24 value heads)", "gdn2", 12, 24, 128, 128),
        ("Qwen3.5-2B GDN", "gdn", 16, 16, 128, 128),
        ("Qwen3.5-4B / Qwen3-Next GDN", "gdn", 16, 32, 128, 128),
        ("Kimi Linear KDA", "kda", 32, 32, 128, 128),
        ("chunked JAX fallback (fallback_scan)", "fallback_scan", 12, 24, 128, 128),
    ]

    results = {}
    for name, vtype, Hk, Hv, K, V in shapes:
        res = bench_shape(name, vtype, Hk, Hv, K, V)
        results[name.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")] = res

    dev = jax.devices()[0]
    dev_str = str(dev)
    if "v6e" in dev_str.lower() or "trillium" in dev_str.lower() or "v6e" in dev_kind.lower() or "v6 lite" in dev_kind.lower():
        chip = "TPU v6e (Trillium, 1 chip)"
    elif "v5e" in dev_str.lower() or "v5 lite" in dev_kind.lower() or "v5e" in dev_kind.lower():
        chip = "TPU v5e (1 chip)"
    else:
        chip = f"{jax.default_backend()} ({dev_str})"

    import subprocess
    tree_hash = os.environ.get("DELTANET_SRC_TREE_HASH")
    if not tree_hash:
        try:
            tree_hash = subprocess.check_output(["git", "rev-parse", "HEAD:src"], text=True).strip()
        except Exception:
            tree_hash = "unknown"

    try:
        import libtpu
        libtpu_ver = getattr(libtpu, "__version__", getattr(libtpu, "version", "unknown"))
    except Exception:
        libtpu_ver = "unknown"

    output = {
        "metadata": {
            "chip": chip,
            "jax_version": jax.__version__,
            "libtpu_version": libtpu_ver,
            "tree_hash": tree_hash,
            "batch_size": 1,
            "sequence_length": 8192,
            "input_dtype": "float32",
            "precision": "bf16",
        },
        "shapes": results,
    }

    out_dir = os.path.join(repo_root, "benchmarks", "results")
    if args.output:
        out_path = os.path.abspath(args.output)
    elif os.environ.get("OUTPUT_JSON"):
        out_path = os.path.abspath(os.environ["OUTPUT_JSON"])
    elif "v5e" in chip:
        out_path = os.path.join(out_dir, "public_shapes_v5e.json")
    else:
        out_path = os.path.join(out_dir, "public_shapes_benchmark.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved public layer shapes benchmark to {out_path}", flush=True)


if __name__ == "__main__":
    main()
