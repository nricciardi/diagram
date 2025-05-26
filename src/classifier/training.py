import os, sys
sys.path.append(os.path.dirname("."))
sys.path.append(os.path.dirname("/"))

from src.classifier.classifier import GNRClassifier

if __name__ == "__main__":
    
    classifier = GNRClassifier()
    classifier.train(verbose=True)
    classifier.save_model(path="src/classifier/classifier.pth")