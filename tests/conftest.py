"""pytest config: make the parent package importable without an install.

The gtd-maei-thesis tree uses a flat-module layout (top-level .py modules),
so tests inject that folder into sys.path here.
"""

from __future__ import annotations

import os
import sys


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)
