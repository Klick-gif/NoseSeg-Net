import os

import torch

from config import get_config
from data_loader import load_data_loader
from eval import calculate_metrics
from model import UNet
from train import train_model

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    config = get_config()
    torch.manual_seed(config.seed)
    os.makedirs(config.result_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(config)

    train_loader, val_loader = load_data_loader(config)

    model = UNet(
        in_channels=config.in_channels,
        out_channels=config.num_classes,
    ).to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        device,
        config,
    )

    val_metrics = calculate_metrics(
        model,
        val_loader,
        device,
        num_classes=config.num_classes,
    )
    print(f'Val Metrics: Dice: {val_metrics["Dice"]:.4f}, IOU: {val_metrics["mIoU"]:.4f}, '
          f'Accuracy: {val_metrics["Accuracy"]:.4f}, Precision: {val_metrics["Precision"]:.4f}, '
          f'Recall: {val_metrics["Recall"]:.4f}')


if __name__ == "__main__":
    main()
