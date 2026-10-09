"""Tests for multi-chip distributed execution using jax.shard_map.

Verifies that chunk_gdn2 runs correctly under jax.shard_map across both
batch dimensions and head dimensions, matching unsharded execution.
"""

import functools
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P

from deltanet_tpu import chunk_gdn2


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_shard_map_batch():
    """Verify chunk_gdn2 under jax.shard_map over batch dimension."""
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices), 1), ("dp", "tp"), axis_types=(AxisType.Auto, AxisType.Auto))

    B = len(devices) * 2
    L, Hk, Hv, K, V = 128, 2, 4, 128, 128
    key = jax.random.PRNGKey(1101)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k5, (B, L, Hv, V), jnp.float32) * 0.39)

    out_unsharded = chunk_gdn2(q, k, v, g, b, w, precision="fp32")

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            P("dp", None, None, None),
            P("dp", None, None, None),
            P("dp", None, None, None),
            P("dp", None, None, None),
            P("dp", None, None, None),
            P("dp", None, None, None),
        ),
        out_specs=P("dp", None, None, None),
        check_vma=False,
    )
    def sharded_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, precision="fp32")

    out_sharded = sharded_fn(q, k, v, g, b, w)
    assert out_sharded.shape == (B, L, Hv, V)
    err = rel_l2(out_sharded, out_unsharded)
    assert err < 1e-5, f"Batch sharding rel L2 error {err:.2e} exceeds 1e-5"


def test_shard_map_heads():
    """Verify chunk_gdn2 under jax.shard_map over head dimension."""
    devices = jax.devices()
    mesh = jax.make_mesh((1, len(devices)), ("dp", "tp"), axis_types=(AxisType.Auto, AxisType.Auto))

    B, L = 1, 128
    Hk, Hv = len(devices) * 2, len(devices) * 4
    K, V = 128, 128
    key = jax.random.PRNGKey(1102)
    k0, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

    q = jax.random.normal(k0, (B, L, Hk, K), jnp.float32) * 0.1
    k = jax.random.normal(k1, (B, L, Hk, K), jnp.float32) * 0.1
    v = jax.random.normal(k2, (B, L, Hv, V), jnp.float32) * 0.1
    g = -jnp.exp(jax.random.normal(k3, (B, L, Hk, K), jnp.float32) * 0.39 - 1.0)
    b = jax.nn.sigmoid(jax.random.normal(k4, (B, L, Hk, K), jnp.float32) * 0.39)
    w = jax.nn.sigmoid(jax.random.normal(k5, (B, L, Hv, V), jnp.float32) * 0.39)

    out_unsharded = chunk_gdn2(q, k, v, g, b, w, precision="fp32")

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            P(None, None, "tp", None),
            P(None, None, "tp", None),
            P(None, None, "tp", None),
            P(None, None, "tp", None),
            P(None, None, "tp", None),
            P(None, None, "tp", None),
        ),
        out_specs=P(None, None, "tp", None),
        check_vma=False,
    )
    def sharded_fn(q, k, v, g, b, w):
        return chunk_gdn2(q, k, v, g, b, w, precision="fp32")

    out_sharded = sharded_fn(q, k, v, g, b, w)
    assert out_sharded.shape == (B, L, Hv, V)
    err = rel_l2(out_sharded, out_unsharded)
    assert err < 1e-5, f"Head sharding rel L2 error {err:.2e} exceeds 1e-5"
