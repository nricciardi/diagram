from abc import ABC, abstractmethod


class Compiler(ABC):

     @abstractmethod
     def compile(self, payload: str, output_path: str):
         pass