"""This module helps to resolve import
discrepancies between different Python versions.
"""
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


__all__ = [
    "Protocol",
]
