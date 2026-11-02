"""Compatibility imports for the public adapter API.

New code imports model types from ``locomanilab.plugin`` and independent stages from
``locomanilab.workflows``. No controller runtime lives in this module.
"""
from locomanilab.plugin import REGISTRY, ModelPlugin, SimServer, StageResult, register


__all__ = ["REGISTRY", "ModelPlugin", "SimServer", "StageResult", "register"]
