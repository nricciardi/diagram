import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src.dataset.extractor.arrow_dataset import ArrowDataset
from src.extractor.arrow.arrownet import ArrowNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model: ArrowNet, loader: DataLoader[ArrowDataset], show_examples: bool = False, num_examples: int = 5):
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0

    shown = 0

    for images, heatmaps in loader:
        images = images.to(DEVICE).float()
        heatmaps = heatmaps.to(DEVICE).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        running_loss += loss.item()

        if show_examples and shown < num_examples:
            for i in range(min(images.size(0), num_examples - shown)):
                img = images[i].cpu().squeeze(0).numpy()
                pred = outputs[i].cpu().squeeze(0).numpy()
                gt = heatmaps[i].cpu().squeeze(0).numpy()

                fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                axs[0].imshow(img, cmap="gray")
                axs[0].set_title("Input Image")

                axs[1].imshow(gt, cmap="hot")
                axs[1].set_title("Ground Truth")

                axs[2].imshow(pred, cmap="hot")
                axs[2].set_title("Prediction")

                for ax in axs:
                    ax.axis("off")

                plt.tight_layout()
                plt.show()

                shown += 1

    avg_loss = running_loss / len(loader)
    print(f"\nTest Loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test ArrowNet model on test dataset")
    parser.add_argument("--info_file", required=True, help="Path to test JSON file.")
    parser.add_argument("--images_dir", required=True, help="Directory containing test images.")
    parser.add_argument("--patch_size", type=int, default=64, help="Dimensione delle patch quadrate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--arrow_type", required=True, choices=["head", "tail"], help="Arrow type.")
    parser.add_argument("--model_path", required=True, help="Path to the saved model (e.g., model.pth).")
    parser.add_argument("--show_examples", action="store_true", help="Show example predictions.")

    args = parser.parse_args()

    test_loader = DataLoader(
        ArrowDataset(args.arrow_type, args.info_file, args.images_dir, args.patch_size),
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location=DEVICE, weights_only=True)
    model.to(DEVICE)

    evaluate(model, test_loader, show_examples=args.show_examples)


if __name__ == "__main__":
    # --info_file /home/nricciardi/Repositories/diagram/dataset/source/fca/test.json --images_dir /home/nricciardi/Repositories/diagram/dataset/source/fca/test --patch_size 64 --arrow_type head --model_path /home/nricciardi/Repositories/diagram/src/extractor/arrow/test.pth
    main()
