from core.preprocessing.processor import Processor, MultiProcessor
from core.image.image import Image
from core.image.tensor_image import TensorImage

import numpy as np
import cv2
import torch

class ReverseProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process(self, image: Image) -> Image:
        """
        Reverse the image tensor.
        
        :param image: The input image to be processed.
        :return: The processed image with reversed tensor.
        """
        image_np = image.as_tensor().detach().numpy()
        if len(image_np.shape) == 2:
            return TensorImage(torch.from_numpy(image_np[::-1, ::-1].copy()))
        elif len(image_np.shape) == 3:
            return TensorImage(torch.from_numpy(image_np[:, ::-1, ::-1].copy()))
        else:
            raise ValueError(f"Unsupported image shape: {image_np.shape}")

class GrayScaleProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process(self, image: Image) -> Image:
        """
        Convert the image to grayscale.
        
        :param image: The input image to be processed.
        :return: The processed image in grayscale.
        """

        device = image.as_tensor().device
        image_np = image.as_tensor().cpu().detach().numpy()
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        
        if len(image_np.shape) != 3:
            return image
        elif image_np.shape[0] == 4:
            image_np = image_np[:3]  # Discard alpha
            image_np = np.transpose(image_np, (1, 2, 0))
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        elif image_np.shape[0] == 1:
            gray_image = image_np[0]
        elif image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            assert gray_image.shape == image_np.shape[:2], f"Expected shape {image_np.shape[:2]}, but got {gray_image.shape}"
        
        avg_brightness = gray_image.mean()
        if avg_brightness < 127:
            gray_image = 255 - gray_image

        tensor = torch.from_numpy(gray_image).to(device)
        return TensorImage(tensor)

class PadderProcessor(Processor):
    def __init__(self, target_size: tuple[int, int] = Processor.BASE_SHAPE):
        super().__init__()
        self.target_size = target_size

    def process(self, image: Image) -> Image:
        device = image.as_tensor().device
        image_np = image.as_tensor().cpu().detach().numpy()
        image_np = image_np.astype(np.uint8)
        
        if len(image_np.shape) != 2:
            if image_np.shape[0] == 1:
                image_np = image_np[0]
        h, w = image_np.shape
        
        if h > self.target_size[0] or w > self.target_size[1]:
            scale_factor = min(self.target_size[0] / h, self.target_size[1] / w)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = image_np.shape
            
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
        tensor = torch.from_numpy(padded_image).to(device)
        return TensorImage(tensor)

class MedianFilterProcessor(Processor):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def process(self, image: Image) -> Image:
        
        device = image.as_tensor().device
        image_np = image.as_tensor().cpu().detach().numpy()
        image_np = image_np.astype(np.uint8)
        image_np = cv2.medianBlur(image_np, self.kernel_size)
        
        return TensorImage(torch.from_numpy(image_np).to(device))

class OtsuThresholdProcessor(Processor):
    def __init__(self):
        super().__init__()

    def process(self, image: Image) -> Image:
        device = image.as_tensor().device
        image_np = image.as_tensor().cpu().detach().numpy()
        image_np = image_np.astype(np.uint8)
        
        _, thresh = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        assert thresh.shape == image_np.shape, f"Expected shape {image_np.shape}, but got {thresh.shape}"
        return TensorImage(torch.from_numpy(thresh).to(device))

class PerspectiveCorrectionProcessor(Processor):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = None
        if output_size is not None:
            self.output_size = (int(output_size[0]), int(output_size[1]))
        
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order the points in the following order: top-left, top-right, bottom-right, bottom-left.
        Args:
            pts: The input points to be ordered.
        Returns:
            The ordered points.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect
    
    def process(self, image: Image) -> Image:
        device = image.as_tensor().device
        gray = image.as_tensor().cpu().detach().numpy()

        thresh = gray.astype(np.uint8)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # raise ValueError("No contours found in the image")
            return image  # No contours found, return original image

        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) != 4:
            # raise ValueError(f"Expected 4 corners, got {len(approx)}")
            return image  # Not enough corners found, return original image

        pts = approx.reshape(4, 2)
        rect = self._order_points(pts)
        if self.output_size is None:
            output_1 = int(Processor.BASE_SHAPE[0] / 1.2)
            output_2 = int(output_1 * (gray.shape[0] / gray.shape[1]))
            self.output_size = (output_1, output_2)
        dst = np.array([
            [0, 0],
            [self.output_size[0] - 1, 0],
            [self.output_size[0] - 1, self.output_size[1] - 1],
            [0, self.output_size[1] - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (self.output_size[0], self.output_size[1]))

        tensor = torch.from_numpy(warped).unsqueeze(0).float()
        return TensorImage(tensor.to(device))

class GNRMultiProcessor(MultiProcessor):
    def __init__(self, processors: list[Processor] = None):
        if processors is None:
            processors = [
                GrayScaleProcessor(), 
                OtsuThresholdProcessor(), 
                MedianFilterProcessor(), 
                PerspectiveCorrectionProcessor(), 
                PadderProcessor()
            ]
        super().__init__(processors)