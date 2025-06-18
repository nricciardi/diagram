import argparse

import torch
import torch.nn as nn
from torch.optim.optimizer import required
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset.extractor.arrow_dataset import ArrowDataset, ContentType
from src.extractor.arrow.arrownet import ArrowNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_WEIGHTS = [1, 1, 1]
criterion = nn.MSELoss()

def train(model: ArrowNet, head_loader: DataLoader[ArrowDataset], tail_loader: DataLoader[ArrowDataset], other_loader: DataLoader[ArrowDataset], num_epochs: int):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Create iterators
        head_iter = iter(head_loader)
        tail_iter = iter(tail_loader)
        other_iter = iter(other_loader)

        while True:
            try:
                head_images, head_heatmaps = next(head_iter)
                tail_images, tail_heatmaps = next(tail_iter)
                other_images, other_heatmaps = next(other_iter)
            except StopIteration:
                break  # Exit loop when any loader runs out of data

            optimizer.zero_grad()
            total_loss = 0.0

            # HEAD
            head_images = head_images.to(DEVICE).float().unsqueeze(1)
            head_heatmaps = head_heatmaps.to(DEVICE)
            head_outputs = model(head_images)
            head_loss = LOSS_WEIGHTS[0] * criterion(head_outputs[:, 0, :, :], head_heatmaps)
            total_loss += head_loss

            # TAIL
            tail_images = tail_images.to(DEVICE).float().unsqueeze(1)
            tail_heatmaps = tail_heatmaps.to(DEVICE)
            tail_outputs = model(tail_images)
            tail_loss = LOSS_WEIGHTS[1] * criterion(tail_outputs[:, 1, :, :], tail_heatmaps)
            total_loss += tail_loss

            # OTHER
            other_images = other_images.to(DEVICE).float().unsqueeze(1)
            other_heatmaps = other_heatmaps.to(DEVICE)
            other_outputs = model(other_images)
            other_loss = LOSS_WEIGHTS[2] * criterion(other_outputs[:, 2, :, :], other_heatmaps)
            total_loss += other_loss

            # Backward and optimize once
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

@torch.no_grad()
def evaluate(model: ArrowNet, loader: DataLoader[ArrowDataset], show_examples: bool = False, num_examples: int = 5):
    model.eval()
    running_loss = 0.0

    shown = 0

    for images, heatmaps in loader:
        images = images.to(DEVICE).float().unsqueeze(1)
        heatmaps = heatmaps.to(DEVICE)

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
    parser = argparse.ArgumentParser(description="Train ArrowNet by NR")
    parser.add_argument("--train_info_file", required=True, help="Path to information JSON file.")
    parser.add_argument("--test_info_file", required=True, help="Path to information JSON file.")
    parser.add_argument("--train_images_dir", required=True, help="Images directory.")
    parser.add_argument("--test_images_dir", required=True, help="Images directory.")
    parser.add_argument("--patch_size", type=int, default=64, help="Dimensione delle patch quadrate.")
    parser.add_argument("--output_size", type=int, default=256, help="Dimensione finale dell'immagine paddata.")
    parser.add_argument("--n_epochs", type=int, default=10, help="N. epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--sigma", type=float, default=5, help="Sigma")
    parser.add_argument("--output", required=True, type=str, help="Output file in which save weights")

    args = parser.parse_args()

    train_head_loader = DataLoader(ArrowDataset(ContentType.HEAD.value, args.train_info_file, args.train_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
    train_tail_loader = DataLoader(ArrowDataset(ContentType.TAIL.value, args.train_info_file, args.train_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
    train_other_loader = DataLoader(ArrowDataset(ContentType.OTHER.value, args.train_info_file, args.train_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)

    model = ArrowNet().to(DEVICE)

    train(model, train_head_loader, train_tail_loader, train_other_loader, args.n_epochs)

    torch.save(model.state_dict(), args.output)

    test_head_loader = DataLoader(ArrowDataset(ContentType.HEAD.value, args.test_info_file, args.test_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
    test_tail_loader = DataLoader(ArrowDataset(ContentType.TAIL.value, args.test_info_file, args.test_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
    test_other_loader = DataLoader(ArrowDataset(ContentType.OTHER.value, args.test_info_file, args.test_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)

    print("Evaluate head")
    evaluate(model, loader=test_head_loader)

    print("Evaluate tail")
    evaluate(model, loader=test_tail_loader)

    print("Evaluate other")
    evaluate(model, loader=test_other_loader)

if __name__ == '__main__':

    main()