from abc import ABC, abstractmethod
from typing import List, Type
from core.representation.representation import DiagramRepresentation
from core.utils.compatible_mixins import CompatibleDiagramsMixin, CompatibleRepresentationsMixin
from src.unified_representation import UnifiedDiagramRepresentation
from core.transducer.outcome import Outcome


class Transducer(CompatibleDiagramsMixin, CompatibleRepresentationsMixin, ABC):

    def __init__(self, identifier: str):
        self._identifier = identifier

    @abstractmethod
    def trasduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> Outcome:
        """
        Convert agnostic representation into (compilable) outcome

        :param diagram_id:
        :param diagram_representation:
        :return:
        """

