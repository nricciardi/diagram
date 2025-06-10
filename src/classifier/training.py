import os, sys, torch
sys.path.append(os.path.dirname("."))
sys.path.append(os.path.dirname("/"))

from src.classifier.classifier import GNRClassifier
from src.classifier.dataset import DatasetClassifier

if __name__ == "__main__":
    
    """
    Training script for the GNRClassifier.

    dataset = DatasetClassifier()
    classifier = GNRClassifier(classes=dataset.classes)
    classifier.train(dataset, verbose=True, batch_size=64, epochs=10)
    classifier.save_model(path="src/classifier/classifier.pth")
    """

    dataset = DatasetClassifier()
    classifier = GNRClassifier(dataset.classes, model_path="src/classifier/classifier.pth")
    classifier.evaluate(visual_output=True, verbose=True)