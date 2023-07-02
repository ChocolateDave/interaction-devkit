# Copyright (c) 2023, Juanwu Lu <juanwu@purdue.edu>.
# Released under the BSD-3-Clause license.
# See https://opensource.org/license/bsd-3-clause/ for licensing details.
from __future__ import annotations

from . import _version
from .dataset import INTERACTIONCase, INTERACTIONMap, INTERACTIONScenario

__docformat__ = "restructuredtext"
__doc__ = """
interaction - INTERACTION dataset development toolkit.
=============================================================

This package provides a set of tools for working with the INTERACTION dataset.
"""
__version__ = _version.version

__all__ = [
    "__version__",
    "__doc__",
    "INTERACTIONMap",
    "INTERACTIONScenario",
    "INTERACTIONCase",
]
