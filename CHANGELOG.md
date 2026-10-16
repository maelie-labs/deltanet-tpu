# Changelog

All notable changes to `deltanet-tpu` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-16

### Added
- `recompute_states=True` on `chunk_gdn2` and the chunked variants: the backward pass recomputes the chunk states (and the intra-chunk prepass) instead of saving them, for long sequences. Slower, but needs less memory for saved residuals.
- `streaming_prefill=True` (with `prefill_block_size`) for forward-only prefill (`is_inference=True`; ignored otherwise): the sequence is processed in blocks with the state passed between them, so intermediate buffers are bounded by the block size.
- `stability_guard` and `stability_tau` on `chunk_gdn2` and `gdn2_decode_step`, off by default. They clamp the erase gate to `b >= 0` and the log-decay to `g <= -1e-6` in every mode (with `fuse_prologue=True` only the activated decay is clamped, with zero gradient where it clamps). They also cap the Frobenius norm at `stability_tau` of the state returned by a forward-only `chunk_gdn2` call, of the state passed between streaming-prefill blocks, and of the `new_state` returned by `gdn2_decode_step`. The state inside one call and the training path are not capped.
- `initial_state_segment_id` on `chunk_gdn2`: the segment ID of the token that produced `initial_state`. If token 0 belongs to a different document, the initial state is reset at token 0. Use it when a packed sequence is split across calls or chips.
- Precision options on `chunk_gdn2`, all off by default: `mask_lambda_by_doc` (restricts the within-chunk cumulative decay to same-document pairs; avoids precision loss in a document that follows a strongly decayed one in the same chunk), `state_fp32_matmul` (full float32 precision for the state recurrence when `precision="bf16"`; slower) and `truncate_bwd_dT_bf16` (restores the bfloat16 rounding of the intra-chunk backward products used before 0.3.0).
- `DELTANET_NONPOSITIVE_GUARD=0` environment variable, for debugging only. It disables the nonpositive clamps on log-decay (on the input `g` and on cumulative decays and their differences, with their subgradients), but the gradient of the input clamp still masks positive `g`, so forward and backward become inconsistent. It is read when the function is traced.
- `ValueError` for `stability_tau <= 0` with `stability_guard`, and for `prefill_block_size <= 0` with `streaming_prefill`.

### Fixed
- `benchmarks/benchmark_public_shapes.py` stopped with a `NameError` in its chip detection before writing results.

### Changed
- Hardening: inside each 128-token chunk, the sub-block decay factors `exp(Lambda_i - Lambda_ref)` are now computed from exponents clamped to `<= 0` in the forward kernel, the backward kernel and the `fallback_scan` path, and the analytical backward applies matching subgradients. In 0.2.1 these exponents were positive only by float32 rounding, and non-finite results needed log-decay near the float32 limit. The clamp is required by `mask_lambda_by_doc`.
- With `precision="bf16"`, the Pallas intra-chunk backward keeps its intermediate products in float32 instead of rounding them to bfloat16 (`truncate_bwd_dT_bf16=False`); its matmul operands are bfloat16. On its own this moves bf16 gradients slightly closer to the fp32 reference. Forward outputs are unchanged.
- `chunk_gdn2` keeps q, k, v and the gates in their input dtype in HBM (the kernels still compute in float32), instead of upcasting them to float32 first. This saves memory but costs accuracy with bfloat16 inputs: the scaled (and, with `qk_l2norm=True`, normalized) q and k are stored in bfloat16, and gradients of broadcast or per-head gates are rounded to bfloat16 before the broadcast sum. In a CPU check (L=256, Hk=2, Hv=4, all inputs bfloat16, scalar gates at `Hv` heads, `qk_l2norm=True`, `precision="fp32"`), the relative error against the float32 reference rose from 1.7e-3 to 2.9e-3 for the output and from 2.2e-3 to 7.6e-3 for `dg`. Pass float32 inputs (or full-channel gates) if this matters.
- Grouped heads with value-head gates (`Hv > Hk` and `g` or `b` with `Hv` heads) index q and k per key head through the kernel's block mapping instead of copying them to `Hv` heads.
- `d_initial_state` is computed only when an `initial_state` is given, and the per-chunk segment-ID residual is 128 times smaller (logical size).
- README benchmark table re-measured with 0.3.0 on TPU v6e (`benchmarks/results/public_shapes_benchmark_0.3.0.json`).

## [0.2.1] - 2026-10-11

### Fixed
- Intra-chunk triangular inverse numerical stability: reduced diagonal block size from B=32 to B=16 in `_invert_unit_triangular_core` (`src/deltanet_tpu/gdn2.py`). For B=32, synthetic adversarial inputs with identical or highly aligned keys within a chunk, decay ≈ 1, and β → 1 caused polynomial doubling expansions to reach binomial coefficients C(31, 15) = 300,540,195, exceeding float32's 24-bit significand (16,777,216) and producing a residual ||I - LT||_2 ~ 4,096 that caused Newton-Schulz refinement to diverge. For B=16, the maximum binomial coefficient is C(15, 7) = 6,435, preserving 11.3 bits of headroom, converging smoothly inside the Newton-Schulz quadratic contraction basin. While not observed on the real-model activations we tested, users of 0.2.0 should upgrade to 0.2.1.

### Added
- Adversarial test suite in `tests/test_nan_adversarial.py` covering synthetic identical keys, near-identical perturbations, periodic patterns, opposite-sign keys, and direct triangular solve parity.
- Evaluated and verified strict contractive boundedness (|T_ij| <= 1.0) under exact ±1 rank-1 patterns in `tests/test_nan_adversarial.py`.
- Inverter guard in `_invert_unit_triangular_core` that raises ValueError if C > 8 * B (internal; the public API fixes chunk_size at 128).

## [0.2.0] - 2026-10-09

### Added
- Prologue fusion (`fuse_prologue=True`): computes the activations (decay exponentiation, gate sigmoids, and q/k L2-normalization) inside the kernel instead of in a separate pass, evaluating exact analytical adjoints directly in VMEM during backward execution.
- Arbitrary sequence length support under fused prologue: unactivated gates (g, b, w) at pad positions are padded with constant value -64.0; carry state is preserved across pad rows because key k is padded with 0 (and erase gate sigmoid(-64) is 1.6e-28).
- Recurrent state dtype option in decode steps: added `state_dtype` parameter to `gdn2_decode_step` and variant decode steps (`gdn_decode_step`, `kda_decode_step`, `gla_decode_step`, `delta_rule_decode_step`). Setting `state_dtype="bfloat16"` stores the state in bf16, half the bytes (validated for channel-wise GDN-2; `float32` remains the default).

### Changed
- Enforced `qk_l2norm=True` requirement when `fuse_prologue=True`, raising a `ValueError` if omitted.
- Plumbed `scale` query scaling factor through all forward and backward kernels under `fuse_prologue=True`, supporting concrete Python floats and 0-d scalar arrays.
- Decode steps default `state_dtype` to `float32` when unspecified, ensuring consistent return types across steps.

## [0.1.0] - 2026-10-08

### Added
- Initial release of Pallas TPU kernels for Gated DeltaNet-2 (GDN-2), Gated DeltaNet (GDN), Kimi Delta Attention (KDA), Gated Linear Attention (GLA), and DeltaNet in JAX.
- Chunked forward and analytical backward training kernels with custom VJP.
- Forward-only prefill mode (`is_inference=True`) and single-token decode step (`gdn2_decode_step`) for inference.
- Support for channel-wise decay, learned vector erase and write gates.
- Grouped value heads (Hv > Hk) support with integer head ratios.
- Dynamic sequence padding for arbitrary sequence lengths.
- Per-document causal state resets for packed sequences with document IDs (`segment_ids`, `cu_seqlens`).
- Multi-chip sharding support via `jax.shard_map`.
- Hardware support and validation on Google Cloud TPU v6e (Trillium) and TPU v5e.
