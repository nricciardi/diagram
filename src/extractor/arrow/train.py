import argparse

import torch
import torch.nn as nn
from torch.optim.optimizer import required
from torch.utils.data import DataLoader

from src.dataset.extractor.arrow_dataset import ArrowDataset
from src.extractor.arrow.arrownet import ArrowNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model: ArrowNet, loader: DataLoader[ArrowDataset], num_epochs: int):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, heatmaps in loader:
            images = images.to(DEVICE).float()  # (B, H, W)
            heatmaps = heatmaps.to(DEVICE)  # (B, H, W)

            outputs = model(images)  # (B, H, W)
            loss = criterion(outputs, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss / len(loader):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ArrowNet by NR")
    parser.add_argument("--info_file", required=True, help="Path to information JSON file.")
    parser.add_argument("--images_dir", required=True, help="Images directory.")
    parser.add_argument("--patch_size", type=int, default=64, help="Dimensione delle patch quadrate.")
    parser.add_argument("--n_epochs", type=int, default=10, help="N. epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--arrow_type", choices=["head", "tail"], required=True, type=str, help="Arrow type 'head' or 'tail'")
    parser.add_argument("--output", required=True, type=str, help="Output file in which save weights")

    args = parser.parse_args()

    train_loader = DataLoader(ArrowDataset(args.arrow_type, args.info_file, args.images_dir, args.patch_size), batch_size=args.batch_size, shuffle=True)
    model = ArrowNet().to(DEVICE)

    train(model, train_loader, args.n_epochs)

    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':

    # --info_file /home/nricciardi/Repositories/diagram/dataset/arrow/train.json --images_dir /home/nricciardi/Repositories/diagram/dataset/arrow/train --patch_size 64 --n_epochs 10 --arrow_type head --output test.pth
    main()