"""Public package name for the byte-preserved ALAM implementation in pi0."""

import sys


_PACKAGE = sys.modules[__name__]
sys.modules.setdefault("openpi.models.physics_lam_model.uni_world_model", _PACKAGE)
sys.modules.setdefault("uni_world_model", _PACKAGE)
