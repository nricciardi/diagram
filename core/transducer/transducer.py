from abc import ABC, abstractmethod
from typing import List, Type
from core.representation.representation import DiagramRepresentation
from src.unified_representation import UnifiedDiagramRepresentation
from core.transducer.outcome import Outcome


class Transducer(ABC):

    def __init__(self, identifier: str):
        self._identifier = identifier

    def compatible_representations(self) -> List[Type[UnifiedDiagramRepresentation]]:
        return [
            UnifiedDiagramRepresentation
        ]

    @abstractmethod
    def compatible_diagrams(self) -> List[str]:
        pass

    @abstractmethod
    def elaborate(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> Outcome:
        """
        Convert agnostic representation into (compilable) outcome

        :param diagram_id:
        :param diagram_representation:
        :return:
        """

