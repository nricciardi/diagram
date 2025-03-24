from abc import ABC
from dataclasses import dataclass


@dataclass
class DiagramRepresentation(ABC):
    """
    Agnostic representation of a diagram
    """
