import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.dataset.extractor.dataset import ObjectDetectionDataset
from torch.utils.data import random_split

import logging

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return tuple(zip(*batch))


def fine_tune():
    # Setup paths
    # annotations_file = "dataset/extractor/labels.json"
    # img_dir = "dataset/extractor/flow_graph_diagrams/"

    annotations_file = "extractor/labels.json"
    img_dir = "extractor/flow_graph_diagrams/"

    # Create dataset and dataloader
    dataset = ObjectDetectionDataset(annotations_file, img_dir)

    print("Dataset loaded")

    dataset_size: int = len(dataset)
    # test_size: int = int(0.2 * dataset_size)
    val_size: int = int(0.2 * dataset_size)
    # train_size: int = dataset_size - (test_size + val_size)
    train_size: int = dataset_size - val_size

    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


    # Load model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # fine-tuning
    # model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None) # from scratch
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 10  # number of classes (state, final state, text, arrow, connection, data, decision, process,
    # terminator) + 1 (for background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print("Model loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    model.train()
    epochs = 100
    final_loss: float = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in train_dataloader:
            images = [img.to(device) for img in images]

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()


        final_loss = running_loss
        #logger.debug(f"Loss: {running_loss:.4f} at step {epoch + 1}")
        print(f"Loss: {running_loss:.4f} at step {epoch + 1}")
        #torch.save(model.state_dict(), "model.pth")

    #logger.debug(f"Final loss: {final_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    torch.save(model.state_dict(), "model.pth")  # save the final weights

    # Evaluation (basic AP computation with torchmetrics)
    metric = MeanAveragePrecision()

    # Evaluate over dataset
    model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            # Format predictions and targets for metric
            metric.update(outputs, targets)

    metrics = metric.compute()
    #logger.debug("\nEvaluation metrics:")
    print("Evaluation metrics:")
    for k, v in metrics.items():
        #logger.debug(f"{k}: {v:.4f}")
        print(f"{k}: {v:.4f}")

    """
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
    """

if __name__ == '__main__':
    fine_tune()
