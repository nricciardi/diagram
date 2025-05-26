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

class GNRClassifier(Classifier):
    
    """
    A classifier that employs a CNN for image classification.
    
    :param model_path: Path to the pre-trained model. If not provided, a new model will be initialized.
    :param classes: A dictionary mapping class indices to class names. Defaults to a predefined set of classes. IT'S HIGHLY RECCOMMENDED TO PASS IT!
    :param processor: The image processor to use for preprocessing images. Defaults to GNRMultiProcessor.
    """
    def __init__(self, classes: dict[int, str] = None, model_path: str = "", processor: Processor = GNRMultiProcessor()):
        super().__init__()
        if classes is None:
            classes = {
                0: "flowchart",
                1: "graph",
                2: "other"
            }
        self.classes = classes
        self.model : ClassifierCNN = None
        if model_path == "":
            self.model = ClassifierCNN(num_classes=len(classes.keys()))
        else:
            self.model = ClassifierCNN(num_classes=len(classes.keys())).load(path=model_path)
        self.processor = processor
    
    
    def classify(self, image: Image) -> str:
        """
        Classify the image using the model and processor.
        
        :param image: The image to classify.
        :return: The classification result.
        """
        
        self.model.eval()
        image = self.processor.process(image)
        y = self.model.forward(image.as_tensor())
        _, predicted = torch.max(y, dim=1)
        return self.classes[predicted.item()]
    
    
    def train(self, dataset: DatasetClassifier = DatasetClassifier(), epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001, verbose: bool = True):
        
        """
        Train the model using the provided dataset.
        """
        
        ### Prepare the dataset
        dataset_size = len(dataset)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        label_to_index = {v: k for k, v in self.classes.items()}
        
        ### Prepare the optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        
        ### Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                outputs = self.model(images.float()).to(device)
                labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long).to(device)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 10:.4f}")
                
        ### Testing loop
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
    
    
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