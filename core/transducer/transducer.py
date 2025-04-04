from abc import ABC, abstractmethod
from typing import List, Type
from core.representation.representation import DiagramRepresentation
from core.utils.compatible_mixins import CompatibleDiagramsMixin, CompatibleRepresentationsMixin, IdentifiableMixin
from src.unified_representation import UnifiedDiagramRepresentation
from core.transducer.outcome import TransducerOutcome


class Transducer(IdentifiableMixin, CompatibleDiagramsMixin, CompatibleRepresentationsMixin, ABC):

    @abstractmethod
    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        """
        Convert agnostic representation into (compilable) outcome

        :param diagram_id:
        :param diagram_representation:
        :return:
        """

