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
        
        class ConvLayer(nn.Module):
            """
            A convolutional layer followed by ReLU activation and max pooling.
            """
            def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 1, padding: int = 1, kernel_pool_size: int = 3, pool_stride: int = 3, pool_padding: int = 0):
                super(ConvLayer, self).__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
                self.norm = nn.BatchNorm2d(out_channels)
                self.pool = nn.MaxPool2d(kernel_size=kernel_pool_size, stride=pool_stride, padding=pool_padding)
            def forward(self, x):
                x = self.conv(x)
                x = self.norm(x)
                x = self.pool(x)
                return x
        
        class LinearLayer(nn.Module):
            """
            A linear layer followed by ReLU activation.
            """
            def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
                super(LinearLayer, self).__init__()
                self.linear = nn.Linear(in_features, out_features)
                self.dropout = nn.Dropout(p=dropout_rate)
                self.relu = nn.ReLU()
            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                x = self.dropout(x)
                return x
        
        self.conv1 = ConvLayer(1, 8)
        self.conv2 = ConvLayer(8, 16)
        self.conv3 = ConvLayer(16, 24, kernel_size=3, kernel_pool_size=2)
        self.fc1 = LinearLayer(8664, 2048)
        self.fc2 = LinearLayer(2048, 128)
        self.fc3 = LinearLayer(128, num_classes)
        self.convSeq = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )
        self.linSeq = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        :param x: Input tensor.
        
        :return: Output tensor.
        """
        x = x.unsqueeze(1).permute(1, 0, 2, 3)
        x = self.convSeq(x)
        x = self.linSeq(x.view(x.size(0), -1))
        return x
    
    def load(self, path: str):
        """
        Load the model from the given path.
        
        :param path: The path to the model file.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        
    def save(self, path: str):
        """
        Save the model to the given path.
        
        :param path: The path to save the model file.
        """
        torch.save(self.state_dict(), path)