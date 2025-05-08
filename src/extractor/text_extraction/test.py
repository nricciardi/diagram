import sys, os

sys.path.append(os.path.dirname("."))
sys.path.append(os.path.dirname("/"))

from src.extractor.text_extraction.text_extractor import TrOCRTextExtractorSmall, TrOCRTextExtractorBase, TrOCRTextExtractorLarge
from src.extractor.text_extraction.text_extractor import TextExtractor
from core.image.tensor_image import TensorImage, Image
from core.image.bbox.bbox2p import ImageBoundingBox2Points, ImageBoundingBox
import os, json, torch

if __name__ == "__main__":
    # Create instances of the text extractors
    small_extractor = TrOCRTextExtractorSmall()
    # base_extractor = TrOCRTextExtractorBase()
    # large_extractor = TrOCRTextExtractorLarge()
    
    json_path = os.path.join(os.path.dirname(__file__), "../../../dataset/source/fa/test.json")
    with open(json_path, 'r') as json_file:
        json_content = json.load(json_file)
    
    images: list[Image] = []
    text_bboxes: list[ImageBoundingBox] = []
    ground_truth_texts: list[str] = []
    dataset_path = os.path.join(os.path.dirname(__file__), "../../../dataset/source/fa/test/")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            image_id = next((item for item in json_content["images"] if item["file_name"] == file), None)["id"]
            annotations = [annotation for annotation in json_content["annotations"] if annotation["image_id"] == image_id and annotation["category"] == "text"]
            for annotation in annotations:
                x, y, w, h = annotation["bbox"]
                text_bbox = ImageBoundingBox2Points("text", torch.Tensor([x, y, x + w, y + h]), 1)
                image = TensorImage.from_str(file_path)
                images.append(image)
                text_bboxes.append(text_bbox)
                ground_truth_texts.append(annotation["text"])

    # Extract text using each extractor
    small_metrics = small_extractor.compute_metrics(images, text_bboxes, ground_truth_texts)
    # base_metrics = base_extractor.compute_metrics(images, text_bboxes, ground_truth_texts)
    # large_metrics = large_extractor.compute_metrics(images, text_bboxes, ground_truth_texts)

    print("Small Extractor Metrics:", small_metrics)
    # print("Base Extractor Metrics:", base_metrics)
    # print("Large Extractor Metrics:", large_metrics)