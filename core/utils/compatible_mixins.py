from abc import ABC, abstractmethod
from typing import Type, List

from core.representation.representation import DiagramRepresentation


class CompatibleRepresentationsMixin(ABC):

    @abstractmethod
    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        pass

class CompatibleDiagramsMixin(ABC):

    @abstractmethod
    def compatible_diagrams(self) -> List[str]:
        pass