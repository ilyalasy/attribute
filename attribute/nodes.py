from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class Node:
    id: str
    layer_index: int
    token_position: int

    @property
    def id_js(self):
        return self.id

    def to_dict(self):
        """Returns a dictionary representation of the node that can be serialized to JSON.

        Args:
            exclude_attrs: List of attribute names to exclude from the representation
        """
        exclude_attrs = ["output_vector", "input_vector", "list_contributions"]
        repr_dict = {}
        for key, value in self.__dict__.items():
            if key in exclude_attrs:
                continue

            if isinstance(value, torch.Tensor):
                temp = value.detach().cpu().to(dtype=torch.float32).numpy().tolist()
                if isinstance(temp, list):
                    repr_dict[key] = [
                        round(x, 4) if isinstance(x, float) else x for x in temp
                    ]
                else:
                    repr_dict[key] = round(temp, 4)
            else:
                repr_dict[key] = value

        repr_dict["node_type"] = self.node_type
        return repr_dict

    @property
    def node_type(self):
        return self.__class__.__name__



@dataclass
class Contribution:
    source: Node
    target: Node
    contribution: float
    vector: Optional[Tensor] = None

    def to_dict(self):
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "contribution": self.contribution,
        }


@dataclass
class OutputNode(Node):
    token_str: str
    token_idx: int
    probability: float
    logit: float
    input_vector: Tensor
    logit_idx: int


@dataclass
class InputNode(Node):
    token_str: str
    token_idx: int
    output_vector: Tensor


@dataclass
class IntermediateNode(Node):
    feature_index: int
    activation: Tensor
    input_vector: Tensor


@dataclass
class ErrorNode(Node):
    output_vector: Tensor


@dataclass
class Edge:
    @property
    def id(self):
        return f"{self.source.id} -> {self.target.id}"

    source: Node
    target: Node
    weight: float

    def to_dict(self):
        return {
            "source": self.source.id,
            "target": self.target.id,
            "weight": round(self.weight, 4),
        }
