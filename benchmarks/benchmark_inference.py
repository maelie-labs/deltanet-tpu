#!/usr/bin/env python3
"""Benchmark prefill and decode inference on Google Cloud TPU."""

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

from deltanet_tpu import chunk_gdn2, gdn2_decode_step


def benchmark_prefill(batch=1, seq_len=8192, chunk_size=128, iters=20, warmup=5):
    Hk, Hv, K, V = 12, 24, 128, 128
    key = jax.random.PRNGKey(0)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (batch, seq_len, Hk, K), dtype=jnp.float32)
    k = jax.random.normal(k1, (batch, seq_len, Hk, K), dtype=jnp.float32)
    v = jax.random.normal(k2, (batch, seq_len, Hv, V), dtype=jnp.float32)
    g = -jnp.exp(jax.random.normal(k3, (batch, seq_len, Hk, K), dtype=jnp.float32) - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (batch, seq_len, Hk, K), dtype=jnp.float32))
    w = jax.nn.sigmoid(jax.random.normal(k5, (batch, seq_len, Hv, V), dtype=jnp.float32))

    @jax.jit
    def prefill_fn(q, k, v, g, b, w):
        return chunk_gdn2(
            q, k, v, g, b, w,
            is_inference=True,
            output_final_state=False,
            chunk_size=chunk_size,
            precision="bf16",
        )

    for _ in range(warmup):
        res = prefill_fn(q, k, v, g, b, w)
        res.block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        res = prefill_fn(q, k, v, g, b, w)
        res.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    median_ms = float(jnp.median(jnp.array(times)))
    tok_per_sec = (batch * seq_len) / (median_ms / 1000.0)
    return {
        "batch": batch,
        "seq_len": seq_len,
        "median_ms": median_ms,
        "tokens_per_second": tok_per_sec,
    }


def benchmark_decode(batch=1, num_steps=100, warmup=10):
    Hk, Hv, K, V = 12, 24, 128, 128
    key = jax.random.PRNGKey(1)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (batch, Hk, K), dtype=jnp.float32)
    k = jax.random.normal(k1, (batch, Hk, K), dtype=jnp.float32)
    v = jax.random.normal(k2, (batch, Hv, V), dtype=jnp.float32)
    g = -jnp.ones((batch, Hk, K), dtype=jnp.float32)
    b = jnp.ones((batch, Hk, K), dtype=jnp.float32) * 0.5
    w = jnp.ones((batch, Hv, V), dtype=jnp.float32) * 0.5
    state = jnp.zeros((batch, Hv, K, V), dtype=jnp.float32)

    step_fn = jax.jit(gdn2_decode_step)

    # Warmup
    for _ in range(warmup):
        o, state = step_fn(q, k, v, g, b, w, state)
        o.block_until_ready()
        state.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(num_steps):
        o, state = step_fn(q, k, v, g, b, w, state)
    o.block_until_ready()
    state.block_until_ready()
    t1 = time.perf_counter()

    total_time_s = t1 - t0
    step_time_ms = (total_time_s / num_steps) * 1000.0
    tok_per_sec = (batch * num_steps) / total_time_s
    return {
        "batch": batch,
        "step_time_ms": step_time_ms,
        "tokens_per_second": tok_per_sec,
    }


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    results = {}

    print("--- Prefill Benchmarks (B=1, L=8,192) ---")
    p1 = benchmark_prefill(batch=1, seq_len=8192)
    print(f"B=1, L=8192: {p1['median_ms']:.2f} ms ({p1['tokens_per_second']:.0f} tok/s)")

    print("--- Prefill Benchmarks (B=2, L=8,192) ---")
    p2 = benchmark_prefill(batch=2, seq_len=8192)
    print(f"B=2, L=8192: {p2['median_ms']:.2f} ms ({p2['tokens_per_second']:.0f} tok/s)")

    print("--- Decode Benchmarks (B=1) ---")
    d1 = benchmark_decode(batch=1, num_steps=100)
    print(f"B=1 decode: {d1['step_time_ms']:.4f} ms/step ({d1['tokens_per_second']:.0f} tok/s)")

    print("--- Decode Benchmarks (B=8) ---")
    d8 = benchmark_decode(batch=8, num_steps=100)
    print(f"B=8 decode: {d8['step_time_ms']:.4f} ms/step ({d8['tokens_per_second']:.0f} tok/s)")

    results["prefill_b1_8k"] = p1
    results["prefill_b2_8k"] = p2
    results["decode_b1"] = d1
    results["decode_b8"] = d8

    dev_str = str(jax.devices()[0])
    if "v6e" in dev_str.lower() or "trillium" in dev_str.lower():
        chip = "TPU v6e (Trillium, 1 chip)"
    elif "v5e" in dev_str.lower():
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
            "input_dtype": "float32",
            "precision": "bf16",
        },
        "benchmarks": results,
    }

    out_dir = os.path.join(repo_root, "benchmarks", "results")
    out_path = os.path.join(out_dir, "inference_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved inference benchmark results to {out_path}")


if __name__ == "__main__":
    main()
