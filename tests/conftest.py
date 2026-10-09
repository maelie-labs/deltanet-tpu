"""Pytest configuration and test fixtures for deltanet-tpu."""

import os
import pytest

# Default to Pallas interpret mode on CPU when no TPU hardware is present
if "PALLAS_INTERPRET" not in os.environ:
    try:
        import jax
        if jax.default_backend() == "tpu":
            os.environ["PALLAS_INTERPRET"] = "0"
        else:
            os.environ["PALLAS_INTERPRET"] = "1"
    except Exception:
        os.environ["PALLAS_INTERPRET"] = "1"


def pytest_configure(config):
    config.addinivalue_line("markers", "tpu: mark test to run on TPU hardware")


def pytest_report_header(config):
    try:
        import jax
        devs = jax.devices()
        kind = devs[0].device_kind if devs else "none"
        return [
            f"JAX backend: {jax.default_backend()}",
            f"JAX devices: {devs}",
            f"TPU device kind: {kind}",
            f"PALLAS_INTERPRET: {os.environ.get('PALLAS_INTERPRET', 'unset')}",
        ]
    except Exception as e:
        return [f"JAX diagnostic error: {e}"]
