import os, sys
sys.path.append(os.path.dirname("."))
sys.path.append(os.path.dirname("/"))

from src.classifier.preprocessing.processor import GNRMultiProcessor
from core.image.tensor_image import TensorImage

if __name__ == "__main__":
    processor = GNRMultiProcessor()
    image = TensorImage.from_str("test_resources/handwritten_diagram.jpg")
    image = processor.process(image, debug=True)