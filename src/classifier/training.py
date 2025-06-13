import os, sys, torch
sys.path.append(os.path.dirname("."))
sys.path.append(os.path.dirname("/"))

from src.classifier.classifier import GNRClassifier
from src.classifier.dataset import DatasetClassifier

if __name__ == "__main__":
    
    """
    # Training script for the GNRClassifier.

    dataset = DatasetClassifier()
    classifier = GNRClassifier(classes=dataset.classes)
    classifier.train(dataset, verbose=True, batch_size=64, epochs=10)
    classifier.save_model(path="src/classifier/classifier.pth")
    
    # Evaluation script for the GNRClassifier.

    dataset = DatasetClassifier()
    classifier = GNRClassifier(dataset.classes, model_path="src/classifier/classifier.pth")
    classifier.evaluate(visual_output=True, verbose=True)

    # Finetune the GNRClassifier without the 'other' class.
    dataset = DatasetClassifier(undersample_classes=["other"])
    classifier = GNRClassifier(classes=dataset.classes, model_path="src/classifier/classifier.pth")
    classifier.train(dataset, verbose=True, batch_size=64, epochs=3, learning_rate=5e-5)
    classifier.save_model(path="src/classifier/classifier.pth")
    """

    dataset = DatasetClassifier()
    classifier = GNRClassifier(dataset.classes, model_path="src/classifier/classifier.pth")
    classifier.evaluate(visual_output=True, subset_percentage=0.25)