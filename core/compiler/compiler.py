from abc import ABC, abstractmethod
from dataclasses import dataclass
from core.utils.compatible_mixins import CompatibleMarkupLanguagesMixin, IdentifiableMixin


@dataclass
class Compiler(IdentifiableMixin, CompatibleMarkupLanguagesMixin, ABC):

    @abstractmethod
    def compile(self, payload: str, output_path: str):
        """
         :param payload:
         :param output_path: path in which output will be dumped
         """

