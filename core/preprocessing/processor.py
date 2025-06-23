from abc import ABC, abstractmethod
from core.image.image import Image
from core.image.tensor_image import TensorImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Processor(ABC):
    """
    Abstract base class for processing data.
    """
    
    BASE_SHAPE = (512, 512)

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
            debug: If True, display the image after each processing step.
            image: The input image to be processed.

        Returns:
            The processed image.
        """
        steps = []
        if debug:
            steps.append(image.tensor.permute(1, 2, 0).squeeze())

        for processor in self.processors:
            image = processor.process(image)
            if debug:
                steps.append(image.tensor.squeeze())
                plt.imshow(steps[-1], cmap="gray")
                plt.axis("off")
                plt.show()

        if debug:
            labels = [f"({chr(97 + i)})" for i in range(len(steps))]
            fig, axes = plt.subplots((len(steps) + 1) // 2, 2)
            axes = axes.flatten()
            for ax, img, label in zip(axes, steps, labels):
                ax.imshow(img, cmap="gray")
                ax.axis("off")
                h, w = img.shape[:2]
                rect = patches.Rectangle((0.5, 0.5), w - 1, h - 1, linewidth=2, edgecolor='black', facecolor='none', transform=ax.transData)
                ax.add_patch(rect)
                ax.text(
                    0.95, 0.05, label,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight='bold',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.6)
                )
            for ax in axes[len(steps):]:
                ax.axis("off")
            plt.tight_layout()
            plt.show()

        return TensorImage(image.as_tensor().unsqueeze(0))
        