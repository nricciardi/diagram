import sys, os

sys.path.append(os.path.dirname("."))
sys.path.append(os.path.dirname("/"))

from src.extractor.text_extraction.text_extractor import TrOCRTextExtractorSmall, TrOCRTextExtractorBase, TrOCRTextExtractionSmallHandwritten, TrOCRTextExtractorBaseHandwritten
from src.extractor.text_extraction.text_extractor import TextExtractor
from core.image.tensor_image import TensorImage, Image
from core.image.bbox.bbox2p import ImageBoundingBox2Points, ImageBoundingBox
import json, torch

if __name__ == "__main__":
    # Create instances of the text extractors
    
    # small_extractor = TrOCRTextExtractorSmall()
    # base_extractor = TrOCRTextExtractorBase()
    # small_handwritten = TrOCRTextExtractionSmallHandwritten()
    base_handwritten = TrOCRTextExtractorBaseHandwritten()
    
    json_path = os.path.join(os.path.dirname(__file__), "../../../dataset/source/fa/test.json")
    with open(json_path, 'r') as json_file:
        json_content = json.load(json_file)
    
    # small_metrics: list[dict] = []
    base_metrics: list[dict] = []
    # large_metrics: list[dict] = []
    dataset_path = os.path.join(os.path.dirname(__file__), "../../../dataset/source/fa/test/")
    for root, dirs, files in os.walk(dataset_path):
        for i, file in enumerate(files):
            print("Processing file: " + str(i) + "/" + str(len(files)))
            file_path = os.path.join(root, file)
            image_id = next((item for item in json_content["images"] if item["file_name"] == file), None)["id"]
            annotations = [annotation for annotation in json_content["annotations"] if annotation["image_id"] == image_id and annotation["category"] == "text" and annotation["text"] != None]
            image = TensorImage.from_str(file_path)
            partial_metric = []
            for annotation in annotations:
                x, y, w, h = annotation["bbox"]
                text_bbox = ImageBoundingBox2Points("text", torch.Tensor([x, y, x + w, y + h]), 1)
                metric = base_handwritten.compute_metrics([image], [text_bbox], [annotation["text"]])
                partial_metric.append(metric)
            base_metrics.append({
                "hamming": sum([metric["hamming"] for metric in partial_metric]) / len(partial_metric),
                "cosine": sum([metric["cosine"] for metric in partial_metric]) / len(partial_metric),
                "euclidean": sum([metric["euclidean"] for metric in partial_metric]) / len(partial_metric),
            })

    # Calculate and print the average metrics for each extractor
    """
    small_avg_metrics = {
        "hamming": sum([metric["hamming"] for metric in small_metrics]) / len(small_metrics),
        "cosine": sum([metric["cosine"] for metric in small_metrics]) / len(small_metrics),
        "euclidean": sum([metric["euclidean"] for metric in small_metrics]) / len(small_metrics),
    }
    print("Small Extractor Average Metrics:" + str(small_avg_metrics))
    """
    base_avg_metrics = {
        "hamming": sum([metric["hamming"] for metric in base_metrics]) / len(base_metrics),
        "cosine": sum([metric["cosine"] for metric in base_metrics]) / len(base_metrics),
        "euclidean": sum([metric["euclidean"] for metric in base_metrics]) / len(base_metrics),
    }
    print("Base Extractor Average Metrics:" + str(base_avg_metrics))
    
    