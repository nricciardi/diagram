import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def load_model(weights_path, device):
    # Load the model architecture
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=12)

    # Load the saved weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Set to evaluation mode
    model.to(device)
    model.eval()
    return model