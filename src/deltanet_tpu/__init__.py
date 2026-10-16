"""deltanet-tpu: Pallas TPU training and inference kernels for Gated DeltaNet and delta-rule linear attention in JAX."""

from deltanet_tpu.gdn2 import chunk_gdn2, gdn2_decode_step
from deltanet_tpu.variants import (
    chunk_delta_rule,
    chunk_gdn,
    chunk_gla,
    chunk_kda,
    delta_rule_decode_step,
    gdn_decode_step,
    gla_decode_step,
    kda_decode_step,
)

__version__ = "0.3.0"

__all__ = [
    "chunk_gdn2",
    "gdn2_decode_step",
    "chunk_gdn",
    "gdn_decode_step",
    "chunk_kda",
    "kda_decode_step",
    "chunk_gla",
    "gla_decode_step",
    "chunk_delta_rule",
    "delta_rule_decode_step",
    "__version__",
]
