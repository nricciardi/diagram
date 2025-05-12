import sys, os

sys.path.append(os.path.dirname("."))
sys.path.append(os.path.dirname("/"))

from src.extractor.text_extraction.text_extractor import TrOCRTextExtractorSmall, TrOCRTextExtractorBase, TrOCRTextExtractionSmallHandwritten, TrOCRTextExtractorBaseHandwritten
from src.extractor.text_extraction.text_extractor import TextExtractor
from core.image.tensor_image import TensorImage, Image
from torchvision.transforms.functional import to_pil_image
from core.image.bbox.bbox2p import ImageBoundingBox2Points, ImageBoundingBox
import json, torch
import random

import matplotlib.pyplot as plt
from PIL import Image

def show_ocr_comparison(images, ground_truths, preds: list[dict], model_names, max_width=200):
    n = len(images)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 2.5 * n))
    
    if n == 1:
        axes = [axes]  # Ensure it's iterable

    for idx in range(n):
        # Load and resize image
        img = images[idx]
        ratio = min(max_width / img.width, 1.0)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size)

        # Display image
        axes[idx][0].imshow(img)
        axes[idx][0].axis('off')

        # Show texts
        gt = ground_truths[idx]
        text_lines = [(
            f"Ground Truth: {gt}\n", "black"
        )]
        for model_name, pred in preds.items():
            text_lines.append((f"{model_name}: {pred[idx]}\n", "green" if gt == pred[idx] else "red"))
        
        ax = axes[idx][1]
        y = 0.9
        for line, color in text_lines:
            ax.text(0, y, line, fontsize=30, verticalalignment='top', fontfamily='monospace', color=color)
            y -= 0.3  # Line spacing
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create instances of the text extractors
    small_extractor = TrOCRTextExtractorSmall()
    # base_extractor = TrOCRTextExtractorBase()
    small_handwritten = TrOCRTextExtractionSmallHandwritten()
    base_handwritten = TrOCRTextExtractorBaseHandwritten()

    json_path = os.path.join(os.path.dirname(__file__), "../../../dataset/source/fa/test.json")
    with open(json_path, 'r') as json_file:
        json_content = json.load(json_file)
        
    MAX_ITEM_ROW = 4
    predictions = {
        "small": [],
        # "base": [],
        "small_handwritten": [],
        "base_handwritten": []
    }
    ground_truths_glob = []
    images_glob = []
    dataset_path = os.path.join(os.path.dirname(__file__), "../../../dataset/source/fa/test/")
    for root, dirs, files in os.walk(dataset_path):
        if len(predictions["small"]) >= MAX_ITEM_ROW:
            break
        for i, file in enumerate(files):
            if len(predictions["small"]) >= MAX_ITEM_ROW:
                break
            print("Processing file: " + str(i) + "/" + str(len(files)))
            file_path = os.path.join(root, file)
            image_id = next((item for item in json_content["images"] if item["file_name"] == file), None)["id"]
            annotations = [annotation for annotation in json_content["annotations"] if annotation["image_id"] == image_id and annotation["category"] == "text" and annotation["text"] != None]
            image = TensorImage.from_str(file_path)
            for annotation in annotations:
                if random.random() * (1 / len(annotation["text"]) ** 3) < 0.025:    
                    if ground_truths_glob.count(annotation["text"]) >= 1:
                        continue
                    x, y, w, h = annotation["bbox"]
                    text_bbox = ImageBoundingBox2Points("text", torch.Tensor([x, y, x + w, y + h]), 1)
                    predictions["base_handwritten"].append(base_handwritten.extract_text(image, text_bbox))
                    if predictions["base_handwritten"][-1] == annotation["text"]:
                        predictions["base_handwritten"].pop()
                        continue
                    images_glob.append(small_extractor.crop_image(image, text_bbox))
                    ground_truths_glob.append(annotation["text"])
                    predictions["small"].append(small_extractor.extract_text(image, text_bbox))
                    predictions["small_handwritten"].append(small_handwritten.extract_text(image, text_bbox))
                    break

    show_ocr_comparison(images_glob, ground_truths_glob, predictions, ["small", "small_handwritten", "base_handwritten"])