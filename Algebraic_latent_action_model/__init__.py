"""Public package name for the byte-preserved ALAM implementation."""

import sys


# Preserved source files import their historical package name internally.
# Registering the same package object keeps those files byte-identical.
sys.modules.setdefault("uni_world_model", sys.modules[__name__])
