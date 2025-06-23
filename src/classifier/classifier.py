from typing import List, override, Optional
from dataclasses import dataclass, field
from core.classifier.classifier import Classifier
from core.image.image import Image
from src.classifier.preprocessing.processor import GNRMultiProcessor, Processor
from src.classifier.model import ClassifierCNN
from src.classifier.dataset import DatasetClassifier

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.wellknown_diagram import WellKnownDiagram
import logging, sys

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(handler)
recognizable_diagrams = [WellKnownDiagram.FLOW_CHART, WellKnownDiagram.GRAPH_DIAGRAM,  WellKnownDiagram.OTHER]

@dataclass
class GNRClassifier(Classifier):
    
    """
    A classifier that employs a CNN for image classification.
    
    :param model_path: Path to the pre-trained model. If not provided, a new model will be initialized.
    :param classes: A dictionary mapping class indices to class names. Defaults to a predefined set of classes. IT'S HIGHLY RECCOMMENDED TO PASS IT AS DATASET.classes!
    :param processor: The image processor to use for preprocessing images. Defaults to GNRMultiProcessor.
    """

    classes: Optional[list[WellKnownDiagram]] = field(default=None)
    model_path: Optional[str] = field(default=None)
    processor: Processor = GNRMultiProcessor()
    model: Optional[ClassifierCNN] = None

    def __post_init__(self):
        if self.classes is None:
            self.classes = recognizable_diagrams
        else:
            for cls in self.classes:
                if cls not in recognizable_diagrams:
                    raise ValueError(f"Class {cls} is not a recognized diagram type. Use one of {recognizable_diagrams}.")

        if self.model_path is None:
            self.model = ClassifierCNN(num_classes=len(self.classes))
        else:
            self.model = ClassifierCNN(num_classes=len(self.classes))
            self.model.load(self.model_path)

    @override
    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]
    
    def classify(self, image: Image) -> Optional[str]:
        """
        Classify the image using the model and processor.
        
        :param image: The image to classify.
        :return: The classification result.
        """
        
        self.model.eval()
        self.model = self.model.to(self.device)
        image = self.processor.process(image)
        tensor = image.as_tensor().to(self.device)
        y = self.model.forward(tensor.float())
        _, predicted = torch.max(y, dim=1)
        predicted = predicted.item()

        if self.classes[predicted] == WellKnownDiagram.OTHER:
            return None

        return self.classes[predicted].value
    
    
    def train(self, dataset: DatasetClassifier, epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-4, verbose: bool = True):
        
        """
        Train the model using the provided dataset.
        """

        label_to_index = [cls.value for cls in self.classes]
        assert self.classes == dataset.classes 
        
        ### Prepare the dataset
        dataset_size = len(dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size

        ### Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        ### Prepare the optimizer and loss function
        device = self.device
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(dataset.weights, dtype=torch.float32).to(device))
        self.model.to(device)
        self.model.train()
        
        ### Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_dataloader):
                if (i + 1) % 20 == 0 and verbose:
                    print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}]")

                optimizer.zero_grad()
                
                outputs = self.model(images.float().to(device))
                labels = torch.tensor([label_to_index.index(label) for label in labels], dtype=torch.long).to(device)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / len(train_dataloader) :.4f}")
                
        ### Testing loop
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = self.model(images.float().to(device))
                labels = torch.tensor([label_to_index.index(label) for label in labels], dtype=torch.long).to(device)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        logger.info(f"Validation Accuracy: {100 * correct / total:.2f}%")
    
    
    def save_model(self, path: str = ""):
        """
        Save the model to the specified path.
        
        :param path: The path to save the model. If not provided, uses the model_path attribute.
        :raises ValueError: If no path is specified and model_path is empty.
        """
        if path == "" and self.model_path == "":
            raise ValueError("No path specified to save the model. Either provide a path or set the model_path attribute.")
        if path == "":
            path = self.model_path
        self.model.save(path)
        
        
    def load_model(self, path: str = ""):
        """
        Load the model from the specified path.
        
        :param path: The path to load the model from. If not provided, uses the model_path attribute.
        :raises ValueError: If no path is specified and model_path is empty.
        """
        if path == "" and self.model_path == "":
            raise ValueError("No path specified to load the model. Either provide a path or set the model_path attribute.")
        if path == "":
            path = self.model_path
        self.model.load(path)

    
    def evaluate(self, dataset: DatasetClassifier = None, visual_output: bool = False, subset_percentage: float = 0.1) -> float:
        """
        Evaluate the model on the provided dataset.
        
        :param dataset: The dataset to evaluate the model on.
        :param visual_output: If True, will visualize some predictions.
        :param subset_percentage: The percentage of the dataset to use for evaluation. Defaults to 10%.
        :return: The accuracy of the model on the dataset.
        """

        if dataset is None:
            dataset = DatasetClassifier()

        subset_size = int(subset_percentage * len(dataset))
        _, subset = random_split(dataset, [len(dataset) - subset_size, subset_size])
        dataloader = DataLoader(subset, batch_size=32, shuffle=True)

        device = self.device
        self.model.to(device)

        # Computw the accuracy of the model on the dataset
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for images, labels in dataloader:

                labels = torch.tensor([self.classes.index(WellKnownDiagram.from_string(label)) for label in labels], dtype=torch.long).to(device)
                images = images.float().to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

            self.last_confusion_preds = all_preds
            self.last_confusion_labels = all_labels
        
        accuracy = 100 * correct / total
        classes_names = self.classes
        print(f"Accuracy of the model on the dataset: {accuracy:.2f}%")

        # If visual_output is True, plot the confusion matrix
        if visual_output:
            import matplotlib.pyplot as plt

            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[classes_names[i].value for i in range(len(self.classes))])
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
            plt.title("Confusion Matrix")
            plt.show()
            plt.waitforbuttonpress()
        return accuracy
    
    @staticmethod
    def get_default() -> 'GNRClassifier':
        return GNRClassifier(
            classes=recognizable_diagrams,
            model_path="src/classifier/classifier.pth",
            processor=GNRMultiProcessor()
        )
    
    def to_device(self, device: str):
        """
        Set to device

        :param device:
        :return:
        """
        self.device = torch.device(device)
        self.model.to(self.device)