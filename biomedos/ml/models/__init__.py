"""Graph model definitions."""

from biomedos.ml.models.base import BaseGNNModel
from biomedos.ml.models.graphsage import HeteroGraphSAGE
from biomedos.ml.models.node2vec import Node2VecBaseline
from biomedos.ml.models.rgcn import RGCNModel

__all__ = ["BaseGNNModel", "HeteroGraphSAGE", "Node2VecBaseline", "RGCNModel"]
