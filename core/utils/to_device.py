from abc import ABC, abstractmethod


class ToDeviceMixin(ABC):


    @abstractmethod
    def to_device(self, device: str):
        """
        Set to device

        :param device:
        :return:
        """