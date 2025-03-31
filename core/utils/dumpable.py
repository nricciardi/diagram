from abc import ABC, abstractmethod


class DumpableMixin(ABC):
    @abstractmethod
    def dump(self, output_path: str):
        """
        Dump representation into file

        :param output_path:
        :return:
        """


