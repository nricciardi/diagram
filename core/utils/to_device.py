from abc import ABC, abstractmethod


class ToDeviceMixin(ABC):
    @staticmethod
    @abstractmethod
    async def to_device(device: str):
        """
        Set to device

        :param device:
        :return:
        """