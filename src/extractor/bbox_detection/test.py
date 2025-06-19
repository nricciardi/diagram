import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from src.dataset.extractor.dataset import ObjectDetectionDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_model(weights_path, device):
    # Load the model architecture
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=12)

    # Load the saved weights
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Set to evaluation mode
    model.to(device)
    model.eval()
    return model

def infer(model):
    annotations_file = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/labels.json"
    img_dir = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/flow_graph_diagrams"
    # Create dataset and dataloader
    dataset = ObjectDetectionDataset(annotations_file, img_dir)

    # Inference & visualization
    model.eval()
    img, target = dataset[0]
    img = img.to(device)
    with torch.no_grad():
        prediction = model([img])[0]

    # Draw predictions
    img_cpu = img.cpu()
    boxes = prediction['boxes']
    labels = prediction['labels']
    drawn = draw_bounding_boxes(img_cpu, boxes=boxes, labels=[str(l.item()) for l in labels], width=2)
    plt.imshow(to_pil_image(drawn))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('model.pth', device)
    infer(model)