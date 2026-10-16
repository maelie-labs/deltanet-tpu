# deltanet-tpu

Pallas TPU kernels for GDN-2 (Gated DeltaNet-2), Gated DeltaNet (Qwen3.5 / Qwen3-Next), Kimi Delta Attention (Kimi Linear) and other delta-rule linear attention layers in JAX. The kernel computes the recurrence (token-mixing) core of a linear-attention layer. Projections, short convolution, output gating and norms come from your model code. Chunked forward and backward passes support training. Forward-only prefill and a single-token decode step support inference, with per-document resets for packed sequences.

```python
import jax
import jax.numpy as jnp
from deltanet_tpu import chunk_gdn2

B, L, Hk, Hv, K, V = 1, 2048, 12, 24, 128, 128
key = jax.random.PRNGKey(0)
k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

q = jax.random.normal(k0, (B, L, Hk, K), dtype=jnp.bfloat16)
k = jax.random.normal(k1, (B, L, Hk, K), dtype=jnp.bfloat16)
v = jax.random.normal(k2, (B, L, Hv, V), dtype=jnp.bfloat16)
# Initialize gate values in (0, 1)
b = jax.nn.sigmoid(jax.random.normal(k3, (B, L, Hk, K), dtype=jnp.float32) * 0.39)
w = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hv, V), dtype=jnp.float32) * 0.39)
# Nonpositive log-decay (mean around -1.0)
g = -jnp.exp(jax.random.normal(k5, (B, L, Hk, K), dtype=jnp.float32) * 0.39 - 1.0)
segment_ids = (jnp.arange(L) >= L // 2).astype(jnp.int32)[None]

out, final_state = chunk_gdn2(
    q, k, v, g, b, w,
    segment_ids=segment_ids,
    output_final_state=True,
    chunk_size=128,
    precision="bf16",
    qk_l2norm=True,
)
```

## Installation

```bash
pip install "deltanet-tpu[tpu]"
```

For local CPU testing: `pip install "deltanet-tpu[dev]"`.

Requires Python `>=3.12`, `jax>=0.11.2,<0.12`, and a TPU (0.3.0 tested on TPU v6e on Google Cloud; earlier versions also on TPU v5e). For local testing without TPU hardware, non-TPU backends run interpret mode (slow CPU emulation executing kernel loops in Python). Run tests with `pytest` on TPU hardware, or `PALLAS_INTERPRET=1 pytest` on CPU.

## Inputs

- `q`, `k`: Raw query and key tensors of shape `(B, L, Hk, K)`. By default, `qk_l2norm=False` to match fla; pass `qk_l2norm=True` for in-kernel L2 normalization as in Qwen3.5 GDN. Raw unnormalized q and k require `qk_l2norm=True` or prior normalization to avoid state overflow.
- `v`: Value tensor of shape `(B, L, Hv, V)`.
- `g`: Log-decay tensor of shape `(B, L, Hk, K)`, `(B, L, Hk, 1)`, `(B, L, Hk)`, or `(B, L, Hv)`. Values must satisfy `g <= 0`; positive values are automatically clamped to `0.0` (with zero gradient on clamped values). Example: `g = -jnp.exp(A_log) * jax.nn.softplus(a + dt_bias)`.
- `b`, `w`, `beta`: Erase, write, or shared gate tensors. In `chunk_gdn`, `beta` has per-head shape `(B, L, Hk)`, `(B, L, Hv)`, `(B, L, Hk, 1)`, or `(B, L, Hv, 1)`. In `chunk_gdn2`, `b` has shape `(B, L, Hk, K)` and `w` has shape `(B, L, Hv, V)`.
- Data types: Inputs keep their dtype in HBM and are upcast to float32 inside the kernels. Setting `precision="bf16"` uses bfloat16 matrix multiplication operands and keeps intermediate chunk matrices in bfloat16, with float32 accumulation and float32 carried state. Output matches the dtype of `v`. Gradients are returned in each input's dtype. With bfloat16 inputs, the scaled (or normalized) q and k are stored in bfloat16 and gradients of broadcast or per-head gates are rounded to bfloat16 before they are summed; pass float32 inputs when gradient accuracy matters more than memory.
- Grouped heads: Value head `j` corresponds to key head `j // G`, where `G = Hv // Hk`. If `g` or `beta` has `Hv` heads, each value head reads the `q` and `k` of its key head.
- Packed documents: `segment_ids` of shape `(B, L)` resets hidden state where the ID changes; this is the preferred, jit-friendly document boundary representation. `cu_seqlens` (1D array or list) is also supported and applies document boundaries to every batch row.
- State: Carried recurrent state tensor has shape `(B, Hv, K, V)` in float32. When a packed sequence is split across calls or chips, pass `initial_state_segment_id` (the segment ID of the token that produced `initial_state`) to `chunk_gdn2`; if token 0 belongs to a different document, the initial state is reset there. In decode, `state_dtype="bfloat16"` allows storing the recurrent state in HBM as bfloat16 between steps (halves the state's bytes), while internal updates execute in float32. Note: bf16 state validated for channel-wise GDN-2 only; scalar-decay decode not validated.
- Prologue fusion: Passing `fuse_prologue=True` fuses decay exponentiation, gate sigmoids, and q/k L2-normalization directly into the Pallas VMEM prepass kernel. Under `fuse_prologue=True`, g, b, and w are pre-activation inputs: g is the raw log decay magnitude (`g_act = -exp(min(g, 5.0))`), b is the erase gate logit (`b_act = sigmoid(b)`), and w is the write gate logit (`w_act = sigmoid(w)`). Requires `qk_l2norm=True`. In backward, exact analytical adjoints are evaluated directly in VMEM instead of materializing activated intermediate tensors.
- Long sequences: `recompute_states=True` makes the backward pass recompute chunk states instead of saving them (slower, less residual memory). For forward-only prefill (`is_inference=True`), `streaming_prefill=True` processes the sequence in blocks of `prefill_block_size` tokens and passes the state between them, so intermediate buffers are bounded by the block size.
- Optional guards and precision (all off by default): `stability_guard=True` clamps the erase gate to `b >= 0` and the log-decay to `g <= -1e-6`, and caps at `stability_tau` the Frobenius norm of the state returned by a forward-only call or a decode step (and of the state passed between streaming-prefill blocks); the training path is not capped. `mask_lambda_by_doc=True` restricts the within-chunk cumulative decay to same-document pairs, which avoids precision loss in a document that follows a strongly decayed one. `state_fp32_matmul=True` runs the state recurrence at full float32 precision under `precision="bf16"` (slower). `truncate_bwd_dT_bf16=True` restores the pre-0.3.0 bfloat16 rounding in the backward pass.
- Training: Use `jax.value_and_grad` on your loss function. Checkpoint tags (`gdn_W`, `gdn_U`, `gdn_M`, `gdn_qe`, `gdn_ke`, `gdn_lend`, `gdn_T`) allow remat (rematerialization: trading compute for memory by recomputing forward operations during backward); JAX checkpoint policies match these names from inside the custom VJP. `is_inference=True` is forward-only and raises ValueError if differentiated.
- Decode: `gdn2_decode_step` and variant decode steps process single tokens of shape `(B, Hk, K)` and `(B, Hv, V)`, with optional `reset=` boolean per batch row and optional `state_dtype=` (`"float32"` or `"bfloat16"`, defaulting to `"float32"`). Note: bf16 state validated for channel-wise GDN-2 only; scalar-decay decode not validated.

## Supported Variants

| Variant | Function | Decode Step | Decay | Gates |
| :--- | :--- | :--- | :--- | :--- |
| GDN-2 (arXiv:2605.22791) | `chunk_gdn2` | `gdn2_decode_step` | Channel `(B, L, Hk, K)` | Erase `b`, Write `w` |
| Gated DeltaNet (arXiv:2412.06464) | `chunk_gdn` | `gdn_decode_step` | Scalar `(B, L, Hk, 1)` | Shared `beta` |
| Kimi Delta Attention (arXiv:2510.26692) | `chunk_kda` | `kda_decode_step` | Channel `(B, L, Hk, K)` | Shared `beta` |
| Gated Linear Attention | `chunk_gla` | `gla_decode_step` | Channel or scalar | Zero `b = 0`, Unit `w = 1` |
| DeltaNet (arXiv:2102.11174) | `chunk_delta_rule` | `delta_rule_decode_step` | None (`g = 0`) | Shared `beta` |

Scalar decay in `chunk_gdn` is supported by broadcasting; no separate fast path yet.

### Comparison with Flash Linear Attention (fla)

| fla function (PyTorch/Triton) | deltanet-tpu function (JAX/Pallas) | Parameters and Differences |
| :--- | :--- | :--- |
| `chunk_gated_delta_rule` | `chunk_gdn` | Scalar decay g, shared gate beta; `use_qk_l2norm_in_kernel` maps to `qk_l2norm` (default False). |
| `chunk_kda` | `chunk_kda` | Channel-wise decay g, shared gate beta; fused KDA gate (`A_log`, `dt_bias`) not supported. |
| `chunk_gla` | `chunk_gla` | Channel or scalar decay g, b = 0, w = 1. Default `qk_l2norm=False` matches fla. |
| `chunk_delta_rule` | `chunk_delta_rule` | No decay (g = 0), shared gate beta. Default `qk_l2norm=False` matches fla. |
| N/A | `chunk_gdn2` | Decoupled erase gate b and write gate w. |

Differences from fla: fla always returns `(output, final_state)` while deltanet-tpu returns a tuple only when `output_final_state=True`; fla `cu_seqlens` expects `B=1`, while here it applies to every batch row. All functions accept arbitrary sequence lengths with internal padding. Argument layouts follow `(B, L, H, K)` convention.

## Multi-Chip Sharding

The forward pass composes directly with `jax.shard_map` over batch and head axes across TPU chips (tested on a 1-device mesh; multi-chip configurations are experimental; running `shard_map` on fake CPU devices in interpret mode is experimental and may fault due to CPU interpret limits):

```python
from functools import partial
import jax
from jax.sharding import AxisType, PartitionSpec as P
from deltanet_tpu import chunk_gdn2

mesh = jax.make_mesh((1, len(jax.devices())), ("dp", "tp"), axis_types=(AxisType.Auto, AxisType.Auto))

@partial(jax.shard_map, mesh=mesh, in_specs=(P("dp", None, "tp", None),) * 6, out_specs=P("dp", None, "tp", None), check_vma=False)
def sharded_gdn2(q, k, v, g, b, w):
    return chunk_gdn2(q, k, v, g, b, w, chunk_size=128, precision="bf16")
```

## Inference

For inference without backward pass, use `is_inference=True` during prefill to bypass saving intermediate recurrence tensors, and use `*_decode_step` for autoregressive token generation. `*_decode_step` is implemented in vectorized pure JAX:

```python
from deltanet_tpu import chunk_gdn2, gdn2_decode_step

# 1. Prefill prompt tokens
prompt_out, state = chunk_gdn2(
    q, k, v, g, b, w, segment_ids=segment_ids,
    is_inference=True, output_final_state=True,
)

# 2. Decode next token (from model projections of incoming token)
# Token shapes: (B, Hk, K) for q/k/g/b, (B, Hv, V) for v/w
token_out, state = gdn2_decode_step(
    q_next, k_next, v_next, g_next, b_next, w_next, state,
)
```

## The Recurrence

Per head, with state S (K x V, float32) where `exp(g_t)` scales the K rows of S:
`S = exp(g_t) * S`
`S = S + k_t * (w_t * v_t - S^T (b_t * k_t))^T`
`o_t = S^T (scale * q_t)`
When `qk_l2norm=True`, query and key are normalized as `k_hat = k / sqrt(||k||^2 + 1e-6)` and `q_hat = scale * q / sqrt(||q||^2 + 1e-6)` (`qk_l2norm=False` skips normalization). S resets to zero where `segment_ids` changes (`initial_state` applies to the first segment only). GDN, KDA and DeltaNet set `b = w = beta`; GLA sets `b = 0, w = 1`.

When `Hv > Hk` (grouped value heads) and `g` and `b` have `Hk` heads, key-side operations execute once per key head and broadcast across value head groups `G = Hv // Hk`. When `g` or `b` has `Hv` heads, the kernel runs once per value head (see Limitations).

## Correctness

Kernel outputs and gradients are verified against a token-serial reference (`reference.py`):
- Parity: relative L2 error within 1e-5 in fp32 and 1e-2 in bf16 across lengths L in {128, 256, 512, 8192} on TPU.
- Fused prologue parity: `fuse_prologue=True` outputs, final states, and input gradients (dq, dk, dv, dg, db, dw) match reference execution within 1e-5 relative L2 error in fp32 (1e-2 in bf16) across aligned and unaligned lengths (L in {128, 300, 384}), with document resets (`segment_ids`), initial states, and custom query scale (see `tests/test_parity.py`).
- Real-model parity: `examples/qwen3_5_layer_parity.py` loads Qwen3.5 linear attention layer 0 weights and verifies the recurrence core at sequence length 128 and one decode step against Hugging Face `transformers` 5.19.0 in fp32 on CPU. Relative L2 errors are 3.6e-7 (Qwen3.5-2B prefill), 4.2e-7 (Qwen3.5-4B prefill), 1.7e-7 (Qwen3.5-2B decode), and 2.2e-7 (Qwen3.5-4B decode). Results are recorded in `benchmarks/results/qwen3_5_parity_results.json` and `benchmarks/results/qwen3_5_4b_parity_results.json`.
- Arbitrary lengths: dynamic padding verified for sequence lengths L in {1, 127, 1000, 8191}.
- Causality: inputs at future positions produce zero change in past outputs and gradients.
- Document boundaries: intra-chunk document resets and single-token documents match reference outputs.
- Numerical stability: log-decay values g <= -150 maintain finite outputs and gradients without NaNs or Infs; positive decay g > 0 is clamped to 0.0.
- State continuity: passing recurrent state between split chunks matches single-call execution.
- Sharding: numerical outputs match within 1e-5 under `jax.shard_map` across batch and head partitions on a 1-device mesh.
- Inference: split prefill chunks match single-call execution; N decode steps match chunked execution with document resets.
- Long-sequence options: `recompute_states=True` gives the same outputs (relative L2 < 1e-6) and gradients as the default path; streaming prefill matches single-pass prefill, including document resets at block boundaries, 1D segment IDs and padding; a document starting at token 0 with `initial_state_segment_id` gives the same result as no initial state.
- Hardware test suite: 380 tests pass on one TPU v6e chip (0.3.0; 0 failed, 0 skipped). Raw log: `benchmarks/results/tpu_v6e_test_0.3.0.log`. Logs for earlier versions: `benchmarks/results/tpu_v6e_test_0.2.1.log`, `benchmarks/results/tpu_v6e_test_0.2.0.log`, and `benchmarks/results/tpu_v5e_test_0.2.0.log`.

## Benchmarks

### Public Model Layer Shapes (TPU v6e, 1 Chip)

Measured with version 0.3.0 on 2026-10-16 on Google Cloud TPU v6e (1 chip, batch B=1, sequence length 8,192, float32 inputs, precision `bf16`, JAX 0.11.2, libtpu 0.0.48.1; results in `benchmarks/results/public_shapes_benchmark_0.3.0.json`). Decode times report per-step time with 100 steps queued (step-by-step synchronous latency with a sync after each step in parentheses):

| Model Layer | Variant | (Hk, Hv, K, V) | Forward (B=1) | Forward + Backward | Prefill Latency (Throughput) | Decode B=1 | Decode B=8 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GDN-2 (12 key / 24 value heads) | `chunk_gdn2` | (12, 24, 128, 128) | 4.85 ms | 9.34 ms | 4.77 ms (1,716,970 tok/s) | 0.0418 ms (0.1867 ms) | 0.0411 ms (0.2123 ms) |
| Qwen3.5-2B GDN | `chunk_gdn` | (16, 16, 128, 128) | 5.63 ms | 10.93 ms | 5.61 ms (1,459,207 tok/s) | 0.0392 ms (0.1920 ms) | 0.0402 ms (0.2039 ms) |
| Qwen3.5-4B / Qwen3-Next GDN | `chunk_gdn` | (16, 32, 128, 128) | 11.37 ms | 22.52 ms | 11.29 ms (725,442 tok/s) | 0.0393 ms (0.1964 ms) | 0.0421 ms (0.2187 ms) |
| Kimi Linear KDA | `chunk_kda` | (32, 32, 128, 128) | 11.67 ms | 22.51 ms | 11.49 ms (712,746 tok/s) | 0.0392 ms (0.1923 ms) | 0.0453 ms (0.2230 ms) |
| chunked JAX fallback (`fallback_scan`) | `chunk_gdn2` | (12, 24, 128, 128) | 8.64 ms | 28.02 ms | 8.63 ms (949,305 tok/s) | - | - |

Shape (16, 16) matches Qwen3.5-2B; (16, 32) matches Qwen3.5-4B and Qwen3-Next (from published `config.json` files). On TPU v5e (measured with version 0.1.0), prefill reaches 1.31M tok/s (GDN-2), 1.15M tok/s (Qwen3.5-2B), and 516k tok/s (Qwen3.5-4B); see `benchmarks/results/public_shapes_v5e.json`.

The analytical backward pass makes it possible to train or fine-tune the linear-attention recurrence core of Gated DeltaNet hybrids such as Qwen3.5 on TPU; projections, convolution, gating and sharding come from your framework.

### Differences from Google tpu-inference GDN Kernel

1. A backward pass, so it can be used for training and fine-tuning ([tpu-inference](https://github.com/vllm-project/tpu-inference) commit 2d6bd87 is forward-only, for serving).
2. Channel-wise decay and decoupled vector erase gate b and write gate w, with grouped value heads (Hv > Hk), keeping the intra-chunk solve in VMEM (TPU on-chip vector memory).
3. Standalone pip-installable package without vLLM or serving framework dependencies.

## Limitations

- Tuned on TPU v6e (Trillium). 0.3.0 is tested on TPU v6e; 0.2.0 was also tested on TPU v5e. TPU v4, TPU v5p, and TPU v7 are untested.
- Head dimensions K and V must be 128; chunk size must be 128.
- Grouped heads require Hv to be an integer multiple of Hk.
- Per-value-head gates (Hv > Hk where g or beta has Hv heads, e.g. Qwen3.5-4B) run one kernel instance per value head instead of fusing the group (11.37 ms vs 5.63 ms for 16 key heads in the table above).
- Single-token decode (`gdn2_decode_step`) is pure vectorized JAX and is dispatch-bound (limited by kernel launch overhead on host/runtime rather than TPU compute or memory bandwidth; 0.039–0.045 ms per step across batch sizes 1 to 8).

## Acknowledgements

This package builds on chunked linear attention algorithms and architectures from:
- `fla-org/flash-linear-attention` (release v0.5.2) for the chunked WY representation (factoring intra-chunk cumulative products into compact matrix updates; see [LICENSES/flash-linear-attention-MIT.txt](https://github.com/maelie-labs/deltanet-tpu/blob/main/LICENSES/flash-linear-attention-MIT.txt)).
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length (arXiv:2406.06484).
- Gated Linear Attention Transformers with Hardware-Efficient Training (arXiv:2312.06635).
- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (arXiv:2605.22791) for the GDN-2 architecture.
- Gated Delta Networks: Improving Mamba2 with Delta Rule (arXiv:2412.06464) for the GDN architecture.
- Kimi Linear: An Expressive, Efficient Attention Architecture (arXiv:2510.26692) for Kimi Delta Attention (KDA).

Developed at [maelie-labs](https://github.com/maelie-labs).

## License

Apache-2.0. See [LICENSE](https://github.com/maelie-labs/deltanet-tpu/blob/main/LICENSE) and [NOTICE](https://github.com/maelie-labs/deltanet-tpu/blob/main/NOTICE) for details.

## Citation

Cite this repository using [CITATION.cff](https://github.com/maelie-labs/deltanet-tpu/blob/main/CITATION.cff).
