from abc import ABC, abstractmethod
from typing import Type, List
from dataclasses import dataclass

from core.representation.representation import DiagramRepresentation

@dataclass
class IdentifiableMixin(ABC):
    identifier: str


class CompatibleRepresentationsMixin(ABC):

    @abstractmethod
    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        """
        List of compatible diagram representation class
        """

class CompatibleDiagramsMixin(ABC):

    @abstractmethod
    def compatible_diagrams(self) -> List[str]:
        """
        List of compatible diagram identifiers (as string)
        """

class CompatibleMarkupLanguagesMixin(ABC):

    @abstractmethod
    def compatible_markup_languages(self) -> List[str]:
        """
        List of compatible markup languages identifiers (as string)
        """