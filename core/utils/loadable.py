from abc import ABC, abstractmethod


class LoadableMixin(ABC):
    @abstractmethod
    def load(self, input_path: str):
        """
        Load representation from file

        :param input_path:
        :return:
        """