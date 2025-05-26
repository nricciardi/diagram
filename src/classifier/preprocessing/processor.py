from core.preprocessing.processor import Processor, MultiProcessor
from core.image.image import Image
from core.image.tensor_image import TensorImage

import numpy as np
import cv2
import torch

class GrayScaleProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process(self, image: Image) -> Image:
        """
        Convert the image to grayscale.
        
        :param image: The input image to be processed.
        :return: The processed image in grayscale.
        """
        image_np = image.as_tensor().detach().cpu().numpy()
        image_np = image_np.astype(np.uint8)
        
        if len(image_np.shape) != 3:
            return image
        elif image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        assert gray_image.shape == image_np.shape[:2], f"Expected shape {image_np.shape[:2]}, but got {gray_image.shape}"
        return TensorImage(torch.from_numpy(gray_image))

class PadderProcessor(Processor):
    def __init__(self, target_size: tuple[int, int] = (1028, 1028)):
        super().__init__()
        self.target_size = target_size

    def process(self, image: Image) -> Image:
        image_np = image.as_tensor().detach().cpu().numpy()
        image_np = image_np.astype(np.uint8)
        
        h, w = image_np.shape
        assert len(image_np.shape) == 2, f"Expected 2D image, but got {image_np.shape}"
        assert h <= self.target_size[0] and w <= self.target_size[1], f"Image size {image_np.shape} is larger than target size {self.target_size}"
        diff_h = self.target_size[0] - h
        diff_w = self.target_size[1] - w

        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top
        pad_left = diff_w // 2
        pad_right = diff_w - pad_left

        padded_image = cv2.copyMakeBorder(
            image_np,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=255
        )
        assert padded_image.shape == self.target_size, f"Expected shape {self.target_size}, but got {padded_image.shape}"
        return TensorImage(torch.from_numpy(padded_image))

class MedianFilterProcessor(Processor):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def process(self, image: Image) -> Image:
        
        image_np = image.as_tensor().detach().cpu().numpy()
        image_np = image_np.astype(np.uint8)
        image_np = cv2.medianBlur(image_np, self.kernel_size)
        
        return TensorImage(torch.from_numpy(image_np))

class OtsuThresholdProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process(self, image: Image) -> Image:
        image_np = image.as_tensor().detach().cpu().numpy()
        image_np = image_np.astype(np.uint8)
        
        _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        assert thresh.shape == image_np.shape, f"Expected shape {image_np.shape}, but got {thresh.shape}"
        return TensorImage(torch.from_numpy(thresh))
        

class GNRMultiProcessor(MultiProcessor):
    def __init__(self, processors: list[Processor] = [GrayScaleProcessor(), OtsuThresholdProcessor(), PadderProcessor(), MedianFilterProcessor()]):
        super().__init__(processors)