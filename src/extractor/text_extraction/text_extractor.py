from abc import ABC, abstractmethod
from core.image.image import Image
from core.image.bbox.bbox import ImageBoundingBox

import torch
from torchvision.transforms.functional import to_pil_image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TextExtractor(ABC):
    """
    Abstract base class for text extraction from various file formats.
    """
    
    @abstractmethod
    def extract_text(self, image: Image, text_bbox: ImageBoundingBox) -> str:
        """
        Extracts text from the given file.

        Args:
            file_path (str): The path to the file from which to extract text.

        Returns:
            str: The extracted text.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def compute_metric(self, image: Image, text_bbox: ImageBoundingBox, ground_truth: str, metric: str) -> float:
        """
        Computes a metric based on the extracted text and ground truth.

        Args:
            image (Image): The image object containing the file.
            ground_truth (str): The ground truth text to compare against.
            metric (str): The metric to compute. Can be "levenshtein" or "hamming".

        Returns:
            float: The computed metric value.
        """
        extracted_text = self.extract_text(image, text_bbox)

        if metric == "levenshtein":
            return self.levenshtein_distance(extracted_text, ground_truth)
        elif metric == "hamming":
            if len(extracted_text) != len(ground_truth):
                if len(extracted_text) > len(ground_truth):
                    ground_truth += " " * (len(extracted_text) - len(ground_truth))
                else:
                    extracted_text += " " * (len(ground_truth) - len(extracted_text))
            return self.hamming_distance(extracted_text, ground_truth)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
    def compute_metrics(self, image: list[Image], text_bboxes: list[ImageBoundingBox], ground_truth: list[str]) -> dict:
        """
        Computes metrics for multiple images and their corresponding bounding boxes.

        Args:
            image (list[Image]): List of image objects containing the files.
            text_bboxes (list[ImageBoundingBox]): List of bounding boxes for each image.
            ground_truth (list[str]): List of ground truth texts to compare against.

        Returns:
            dict: A dictionary containing the computed metric values.
        """
        metrics = {}
        for i, bbox in enumerate(text_bboxes):
            metrics[i] = {
                "levenshtein": self.compute_metric(image[i], bbox, ground_truth[i], "levenshtein"),
                "hamming": self.compute_metric(image[i], bbox, ground_truth[i], "hamming")
            }
        return metrics


    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Computes the Levenshtein distance between two strings.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            int: The Levenshtein distance.
        """
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """
        Computes the Hamming distance between two strings.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            int: The Hamming distance.
        """
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    def crop_image(self, image: Image, text_bbox: ImageBoundingBox):
        """
        Crops the image to the specified bounding box.

        Args:
            image (Image): The image object containing the file.
            text_bbox (ImageBoundingBox): The bounding box for cropping.

        Returns:
            Image: The cropped image.
        """
        tensor = image.as_tensor()  # [C, H, W], torch.Tensor

        # Get bounding box as integers
        left = int(min(text_bbox.top_left_x, text_bbox.bottom_left_x))
        right = int(max(text_bbox.top_right_x, text_bbox.bottom_right_x))
        top = int(min(text_bbox.top_left_y, text_bbox.top_right_y))
        bottom = int(max(text_bbox.bottom_left_y, text_bbox.bottom_right_y))

        _, H, W = tensor.shape
        left = max(0, left)
        right = min(W, right)
        top = max(0, top)
        bottom = min(H, bottom)

        cropped_tensor = tensor[:, top:bottom, left:right]
        if cropped_tensor.ndim == 2:
            cropped_tensor = cropped_tensor.unsqueeze(0).repeat(3, 1, 1)
        elif cropped_tensor.shape[0] == 1:
            cropped_tensor = cropped_tensor.repeat(3, 1, 1)
        return to_pil_image(cropped_tensor)

class TrOCRTextExtractorSmall(TextExtractor):
    
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def extract_text(self, image: Image, text_bbox: ImageBoundingBox) -> str:
        """
        Extracts text from the given image using the TrOCR model.

        Args:
            image (Image): The image object containing the file.

        Returns:
            str: The extracted text.
        """
        
        cropped_image = self.crop_image(image, text_bbox)
        pixel_values = self.processor(images=cropped_image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text: str = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    
class TrOCRTextExtractorBase(TextExtractor):
    
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def extract_text(self, image: Image, text_bbox: ImageBoundingBox) -> str:
        """
        Extracts text from the given image using the TrOCR model.

        Args:
            image (Image): The image object containing the file.

        Returns:
            str: The extracted text.
        """
        
        cropped_image = self.crop_image(image, text_bbox)
        pixel_values = self.processor(images=cropped_image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text: str = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    
class TrOCRTextExtractorLarge(TextExtractor):
        
        def __init__(self):
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
            self.model.eval()
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        def extract_text(self, image: Image, text_bbox: ImageBoundingBox) -> str:
            """
            Extracts text from the given image using the TrOCR model.
    
            Args:
                image (Image): The image object containing the file.
    
            Returns:
                str: The extracted text.
            """
            
            cropped_image = self.crop_image(image, text_bbox)
            pixel_values = self.processor(images=cropped_image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text: str = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text.strip()