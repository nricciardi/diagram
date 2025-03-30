from abc import ABC, abstractmethod

from core.utils.compatible_mixins import CompatibleMarkupLanguagesMixin


class Compiler(CompatibleMarkupLanguagesMixin, ABC):

     @abstractmethod
     def compile(self, payload: str, output_path: str):
         pass