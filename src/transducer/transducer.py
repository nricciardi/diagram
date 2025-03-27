from abc import ABC, abstractmethod
from typing import List, Type
from src.representation.representation import DiagramRepresentation
from src.representation.unified_representation import UnifiedDiagramRepresentation
from src.transducer.outcome import Outcome


class Transducer(ABC):

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

