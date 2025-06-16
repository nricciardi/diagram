import argparse

import torch
import torch.nn as nn
from torch.optim.optimizer import required
from torch.utils.data import DataLoader

from src.dataset.extractor.arrow_dataset import ArrowDataset, ContentType
from src.extractor.arrow.arrownet import ArrowNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_WEIGHTS = [1, 1, 1]

def train(model: ArrowNet, head_loader: DataLoader[ArrowDataset], tail_loader: DataLoader[ArrowDataset], other_loader: DataLoader[ArrowDataset], num_epochs: int):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        optimizer.zero_grad()

        for loader_index, batch in enumerate((head_loader, tail_loader, other_loader)):

            loss = torch.tensor(0.0, device=DEVICE)

            for images, heatmaps in batch:

                images = images.to(DEVICE).float()
                images = images.unsqueeze(1)
                heatmaps = heatmaps.to(DEVICE)

                outputs = model(images)  # (B, 3, H, W)
                loss += LOSS_WEIGHTS[loader_index] * criterion(outputs[:, loader_index, :, :], heatmaps)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ArrowNet by NR")
    parser.add_argument("--info_file", required=True, help="Path to information JSON file.")
    parser.add_argument("--images_dir", required=True, help="Images directory.")
    parser.add_argument("--patch_size", type=int, default=64, help="Dimensione delle patch quadrate.")
    parser.add_argument("--output_size", type=int, default=256, help="Dimensione finale dell'immagine paddata.")
    parser.add_argument("--n_epochs", type=int, default=10, help="N. epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--sigma", type=float, default=5, help="Sigma")
    parser.add_argument("--output", required=True, type=str, help="Output file in which save weights")

    args = parser.parse_args()

    train_head_loader = DataLoader(ArrowDataset(ContentType.HEAD.value, args.info_file, args.images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
    train_tail_loader = DataLoader(ArrowDataset(ContentType.TAIL.value, args.info_file, args.images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
    train_other_loader = DataLoader(ArrowDataset(ContentType.OTHER.value, args.info_file, args.images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)

    model = ArrowNet().to(DEVICE)

    train(model, train_head_loader, train_tail_loader, train_other_loader, args.n_epochs)

    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':

    # --info_file /home/nricciardi/Repositories/diagram/dataset/arrow/train.json --images_dir /home/nricciardi/Repositories/diagram/dataset/arrow/train --patch_size 64 --n_epochs 10 --arrow_type head --output test.pth
    main()