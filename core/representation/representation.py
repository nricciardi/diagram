from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.utils.dumpable import DumpableMixin
from core.utils.loadable import LoadableMixin


@dataclass(frozen=True)
class DiagramRepresentation(DumpableMixin, LoadableMixin, ABC):
    """
    Agnostic representation of a diagram
    """
