from train import train_model
from eval import calculate_metrics
from model import UNet
import torch
from data_loader import load_data_loader

torch.manual_seed(42)  # 设置随机种子以确保可重复性

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    train_loader, val_loader = load_data_loader()
    
    # 初始化模型
    model = UNet(in_channels=1, out_channels=2).to(device)
    
    # 训练模型
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, num_epochs=10)

    # 计算验证集指标
    val_metrics = calculate_metrics(model, val_loader, device)
    print(f'Val Metrics: Dice: {val_metrics["Dice"]:.4f}, IOU: {val_metrics["mIoU"]:.4f}, '
          f'Accuracy: {val_metrics["Accuracy"]:.4f}, Precision: {val_metrics["Precision"]:.4f}, Recall: {val_metrics["Recall"]:.4f}')