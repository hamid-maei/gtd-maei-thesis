"""pytest config: make the parent package importable without an install.

The gtd_phd_maei folder is a flat-module layout (each .py
file at top level), so the tests inject that folder into sys.path here.
"""

from __future__ import annotations

import os
import sys


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)
