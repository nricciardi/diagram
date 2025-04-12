from abc import ABC, abstractmethod
from typing import List, Type
from dataclasses import dataclass
from core.representation.representation import DiagramRepresentation
from core.utils.compatible_mixins import CompatibleDiagramsMixin, CompatibleRepresentationsMixin, IdentifiableMixin
from core.transducer.outcome import TransducerOutcome


@dataclass
class Transducer(IdentifiableMixin, CompatibleDiagramsMixin, CompatibleRepresentationsMixin, ABC):

    @abstractmethod
    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        """
        Convert agnostic representation into (compilable) outcome

        :param diagram_id:
        :param diagram_representation:
        :return:
        """

