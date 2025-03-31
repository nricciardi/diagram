from abc import ABC, abstractmethod


class DumpableMixin(ABC):
    @abstractmethod
    async def dump(self, output_path: str):
        """
        Dump representation into file

        :param output_path:
        :return:
        """


