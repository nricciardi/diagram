from abc import ABC, abstractmethod


class ToDeviceMixin(ABC):


    @abstractmethod
    def to_device(self, device: str) -> None:
        """
        Set to device

        :param device:
        :return:
        """

    @abstractmethod
    def get_device(self) -> str:
        """
        Get device

        :return: device as str
        """