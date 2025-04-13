from abc import ABC, abstractmethod


class LoadableMixin(ABC):
    @staticmethod
    @abstractmethod
    async def load(input_path: str):
        """
        Load representation from file

        :param input_path:
        :return:
        """