"""Minimal support utilities bundled with the anonymous review package."""

from importlib import import_module

from nqs_support_core import distributed as distributed
from nqs_support_core import hilbert as hilbert

__all__ = [
    "InfidelityOperator",
    "InfidelityUVOperator",
    "distributed",
    "hilbert",
    "infidelity",
    "jax",
    "monkeypatch",
    "serialization",
    "utils",
]


def __getattr__(name: str):
    if name in {"infidelity", "jax", "monkeypatch", "operator", "serialization", "utils"}:
        module = import_module(f"nqs_support_core.{name}")
        globals()[name] = module
        return module
    if name in {"InfidelityOperator", "InfidelityUVOperator"}:
        module = import_module("nqs_support_core.infidelity")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
