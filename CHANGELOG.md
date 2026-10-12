# Changelog

All notable changes to `deltanet-tpu` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
