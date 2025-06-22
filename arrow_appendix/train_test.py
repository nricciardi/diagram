import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.dataset.extractor.arrow_dataset import ArrowDataset, ContentType
from arrow_appendix.arrownet import ArrowNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_WEIGHTS = [1, 1, 1]
criterion = nn.MSELoss()

def train(model: ArrowNet, head_loader: DataLoader[ArrowDataset], tail_loader: DataLoader[ArrowDataset], num_epochs: int):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Create iterators
        head_iter = iter(head_loader)
        tail_iter = iter(tail_loader)

        while True:
            try:
                head_images, head_heatmaps = next(head_iter)
                tail_images, tail_heatmaps = next(tail_iter)
            except StopIteration:
                break  # Exit loop when any loader runs out of data

            optimizer.zero_grad()
            total_loss = 0.0

            # HEAD
            head_images = head_images.to(DEVICE).float().unsqueeze(1)
            head_heatmaps = head_heatmaps.to(DEVICE)
            head_outputs = model(head_images)
            head_loss = LOSS_WEIGHTS[0] * criterion(head_outputs[:, 0, :, :], head_heatmaps) \
                        + LOSS_WEIGHTS[1] * criterion(head_outputs[:, 1, :, :], torch.zeros(head_heatmaps.shape).to(DEVICE)) \
                        + LOSS_WEIGHTS[2] * criterion(head_outputs[:, 2, :, :], torch.ones(head_heatmaps.shape).to(DEVICE) - head_heatmaps)
            total_loss += head_loss

            # TAIL
            tail_images = tail_images.to(DEVICE).float().unsqueeze(1)
            tail_heatmaps = tail_heatmaps.to(DEVICE)
            tail_outputs = model(tail_images)
            tail_loss = LOSS_WEIGHTS[0] * criterion(head_outputs[:, 0, :, :], torch.zeros(tail_heatmaps.shape).to(DEVICE)) \
                        + LOSS_WEIGHTS[1] * criterion(head_outputs[:, 1, :, :], tail_heatmaps) \
                        + LOSS_WEIGHTS[2] * criterion(head_outputs[:, 2, :, :], torch.ones(tail_heatmaps.shape).to(DEVICE) - tail_heatmaps)
            total_loss += tail_loss

            # Backward and optimize once
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}")

@torch.no_grad()
def evaluate(model: ArrowNet, class_index: int, loader: DataLoader[ArrowDataset], show_examples: bool = False, num_examples: int = 5):
    model.eval()
    running_loss = 0.0

    shown = 0

    for images, heatmaps in loader:
        images = images.to(DEVICE).float().unsqueeze(1)
        heatmaps = heatmaps.to(DEVICE)

        outputs = model(images)
        right_output_maps = outputs[:, class_index, :, :].squeeze(1)
        loss = criterion(right_output_maps, heatmaps)
        running_loss += loss.item()

        if show_examples and shown < num_examples:
            for i in range(min(images.size(0), num_examples - shown)):
                img = images[i].cpu().squeeze(0).numpy()
                pred_head = outputs[i][0].cpu().squeeze(0).numpy()
                pred_tail = outputs[i][1].cpu().squeeze(0).numpy()
                pred_other = outputs[i][2].cpu().squeeze(0).numpy()
                gt = heatmaps[i].cpu().squeeze(0).numpy()

                fig, axs = plt.subplots(1, 5, figsize=(9, 3))
                axs[0].imshow(img, cmap="gray")
                axs[0].set_title("Input Image")

                axs[1].imshow(gt, cmap="hot")
                axs[1].set_title("Ground Truth")

                axs[2].imshow(pred_head, cmap="hot")
                axs[2].set_title("Head Prediction")

                axs[3].imshow(pred_tail, cmap="hot")
                axs[3].set_title("Tail Prediction")

                axs[4].imshow(pred_other, cmap="hot")
                axs[4].set_title("Other Prediction")

                for ax in axs:
                    ax.axis("off")

                plt.tight_layout()
                plt.show()

                shown += 1

    avg_loss = running_loss / len(loader)
    print(f"\nTest Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train ArrowNet by NR")
    parser.add_argument("--train_info_file", required=False, help="Path to information JSON file.")
    parser.add_argument("--test_info_file", required=True, help="Path to information JSON file.")
    parser.add_argument("--train_images_dir", required=False, help="Images directory.")
    parser.add_argument("--weights_path", required=False, default=None, help="Weights")
    parser.add_argument("--test_images_dir", required=True, help="Images directory.")
    parser.add_argument("--patch_size", type=int, default=64, help="Dimensione delle patch quadrate.")
    parser.add_argument("--output_size", type=int, default=256, help="Dimensione finale dell'immagine paddata.")
    parser.add_argument("--n_epochs", type=int, default=10, help="N. epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--sigma", type=float, default=5, help="Sigma")
    parser.add_argument("--output", required=True, type=str, help="Output file in which save weights")

    args = parser.parse_args()

    print("Instance model...")
    model = ArrowNet().to(DEVICE)

    if args.weights_path is None:
        print("Generate dataloader (train)...")
        train_head_loader = DataLoader(ArrowDataset(ContentType.HEAD.value, args.train_info_file, args.train_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
        train_tail_loader = DataLoader(ArrowDataset(ContentType.TAIL.value, args.train_info_file, args.train_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)

        print("Start training...")
        train(model, train_head_loader, train_tail_loader, args.n_epochs)

        print("Saving model weights...")
        torch.save(model.state_dict(), args.output)

    else:
        print(f"Load model weights: {args.weights_path}")
        model.load_state_dict(torch.load(args.weights_path))

    print("Generate dataloader (test)...")
    test_head_loader = DataLoader(ArrowDataset(ContentType.HEAD.value, args.test_info_file, args.test_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)
    test_tail_loader = DataLoader(ArrowDataset(ContentType.TAIL.value, args.test_info_file, args.test_images_dir, args.patch_size, args.output_size, args.sigma), batch_size=args.batch_size, shuffle=True)

    print("Evaluate head")
    evaluate(model, class_index=0, loader=test_head_loader, show_examples=True)

    print("Evaluate tail")
    evaluate(model, class_index=1, loader=test_tail_loader, show_examples=True)


if __name__ == '__main__':
    # --train_info_file /home/nricciardi/Repositories/diagram/dataset/arrow/train.json --train_images_dir /home/nricciardi/Repositories/diagram/dataset/arrow/train --test_info_file /home/nricciardi/Repositories/diagram/dataset/arrow/test.json --test_images_dir /home/nricciardi/Repositories/diagram/dataset/arrow/test --patch_size 64 --n_epochs 10 --output test.pth --batch_size 8
    # --test_info_file /home/nricciardi/Repositories/diagram/dataset/arrow/test.json --test_images_dir /home/nricciardi/Repositories/diagram/dataset/arrow/test --patch_size 64 --n_epochs 10 --output test.pth --batch_size 8 --weights_path /home/nricciardi/Repositories/diagram/src/extractor/arrow/model1.pth
    main()