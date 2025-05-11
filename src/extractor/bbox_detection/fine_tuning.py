import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.dataset.extractor.dataset import ObjectDetectionDataset

import logging

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


# Setup paths
annotations_file = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/labels.json"
img_dir = "/Users/saverionapolitano/PycharmProjects/diagram/dataset/extractor/flow_graph_diagrams"

# Create dataset and dataloader
dataset = ObjectDetectionDataset(annotations_file, img_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Load model
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # fine-tuning
# model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None) # from scratch
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 10  # number of classes (state, final state, text, arrow, connection, data, decision, process,
# terminator) + 1 (for background)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
model.train()
epochs = 100
min_loss = 10000000
for epoch in range(epochs):
    for images, targets in dataloader:
        images = [img.to(device) for img in images]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        logger.debug(f"Loss: {losses.item():.4f} at step {epoch + 1}")
        if losses.item() < min_loss:
            min_loss = losses.item()
            torch.save(model.state_dict(), "best_model.pth")  # save the best weights

logger.debug(f"Best loss: {min_loss:.4f}")

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

# Evaluation (basic AP computation with torchmetrics)
metric = MeanAveragePrecision()

# Evaluate over dataset
model.eval()
with torch.no_grad():
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)

        # Format predictions and targets for metric
        metric.update(outputs, targets)

metrics = metric.compute()
logger.debug("\nEvaluation metrics:")
for k, v in metrics.items():
    logger.debug(f"{k}: {v:.4f}")
