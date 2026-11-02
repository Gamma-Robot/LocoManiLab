"""Model-independent pipeline operations.

Create ``Workflows(ctx, runner)`` and call ``collect``, ``prepare`` or ``replay``.
The services build commands; model and robot dependencies stay in child processes.
"""
from locomanilab.workflows.collection import CollectionWorkflow
from locomanilab.workflows.data import DataWorkflow
from locomanilab.workflows.replay import ReplayWorkflow


class Workflows(CollectionWorkflow, DataWorkflow, ReplayWorkflow):
    """Collection, dataset preparation and replay using one machine context."""


__all__ = ["Workflows", "CollectionWorkflow", "DataWorkflow", "ReplayWorkflow"]
