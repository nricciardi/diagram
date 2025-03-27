from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DiagramRepresentation(ABC):
    """
    Agnostic representation of a diagram
    """

    @abstractmethod
    def dump(self, output_path: str):
        """
        Dump representation into file

        :param output_path:
        :return:
        """


    @abstractmethod
    def load(self, intput_path: str):
        """
        Load representation from file

        :param intput_path:
        :return:
        """