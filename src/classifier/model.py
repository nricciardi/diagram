import torch
import torch.nn as nn

class ClassifierCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.
    This model consists of three convolutional layers followed by
    fully connected layers. It uses ReLU activation and dropout for
    regularization.
    """
    
    def __init__(self, num_classes: int):
        super(ClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sequential = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.conv3,
            self.relu,
            self.pool,
            nn.Flatten(),
            self.fc1,
            self.relu,
            self.dropout,
            self.fc2,
            self.relu,
            self.dropout,
            self.fc3,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        :param x: Input tensor.
        
        :return: Output tensor.
        """
        x = self.sequential(x)
        return x
    
    def load(self, path: str):
        """
        Load the model from the given path.
        
        :param path: The path to the model file.
        """
        self.load_state_dict(torch.load(path))
        
    def save(self, path: str):
        """
        Save the model to the given path.
        
        :param path: The path to save the model file.
        """
        torch.save(self.state_dict(), path)