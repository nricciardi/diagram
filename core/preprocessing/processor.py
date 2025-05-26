from abc import ABC, abstractmethod
from core.image.image import Image
import matplotlib.pyplot as plt

class Processor(ABC):
    """
    Abstract base class for processing data.
    """

    @abstractmethod
    def process(self, image: Image) -> Image:
        """
        Process the input image.
        Args:
            image: The input image to be processed.
        Returns:
            The processed image.
        """
        pass
    
class MultiProcessor(Processor):
    
    def __init__(self, processors: list[Processor]):
        """
        Initialize the MultiProcessor with a list of processors.

        Args:
            processors: A list of Processor instances.
        """
        self.processors = processors
    
    def process(self, image: Image, debug: bool = False) -> Image:
        """
        Process the input image using all processors in the list.

        Args:
            debug:
            image: The input image to be processed.

        Returns:
            The processed image.
        """
        for processor in self.processors:
            image = processor.process(image)
            if debug:
                plt.imshow(image.tensor.squeeze(), cmap="gray")
                plt.axis("off")
                plt.show()
        return image
        