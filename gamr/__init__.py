"""
GAMR — Galaxy AI Memory Runtime
================================
An adaptive, block-streaming execution runtime for large AI models.

The runtime knows nothing about AI models, transformers, or attention.
It only knows: Tensors, Blocks, and ComputeRequests.

Paper: HAMR — Hierarchical Adaptive Memory Runtime for Large AI Models
"""

from gamr.core.block import WeightBlock, BlockMetadata, BlockState
from gamr.core.event import GAMREvent, EventType, EventQueue
from gamr.core.cost_model import CostModel, Cost
from gamr.core.runtime import GAMRRuntime

__version__ = "0.1.0-alpha"
__author__ = "Vishal Govindasamy"
__paper__ = "HAMR: Hierarchical Adaptive Memory Runtime for Large AI Models"

__all__ = [
    "WeightBlock",
    "BlockMetadata",
    "BlockState",
    "GAMREvent",
    "EventType",
    "EventQueue",
    "CostModel",
    "Cost",
    "GAMRRuntime",
]
