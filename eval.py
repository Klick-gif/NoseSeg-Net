import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(model, test_loader, device, num_classes=2):
    """计算分割评估指标：mIoU、Dice、Accuracy、Precision、Recall"""
    model.eval()
    eps = 1e-8
    total_correct = 0
    total_pixels = 0
    intersections = torch.zeros(num_classes, device=device)
    pred_areas = torch.zeros(num_classes, device=device)
    target_areas = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == y).sum().item()
            total_pixels += y.numel()
            
            for class_idx in range(1, num_classes):  # 跳过背景
                pred = (preds == class_idx).float()
                target = (y == class_idx).float()
                
                intersections[class_idx] += (pred * target).sum()
                pred_areas[class_idx] += pred.sum()
                target_areas[class_idx] += target.sum()

    iou_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []

    for class_idx in range(1, num_classes):  # 跳过背景
        intersection = intersections[class_idx]
        pred_area = pred_areas[class_idx]
        target_area = target_areas[class_idx]
        union = pred_area + target_area - intersection

        if union > 0:
            iou_scores.append((intersection / (union + eps)).item())
            dice_scores.append(((2.0 * intersection) / (pred_area + target_area + eps)).item())

        if pred_area > 0:
            precision_scores.append((intersection / (pred_area + eps)).item())

        if target_area > 0:
            recall_scores.append((intersection / (target_area + eps)).item())
    
    return {
        "mIoU": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "Dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "Accuracy": total_correct / total_pixels if total_pixels > 0 else 0.0,
        "Precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "Recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


def visualize_successful_segmentation(image, true_mask, pred_mask, metrics):
    """可视化分割结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # 设置中文字体为 SimHei（黑体）
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 第一行：原始图像
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 真实标签
    axes[0, 1].imshow(true_mask, cmap='gray')
    axes[0, 1].set_title(f'真实标签\n(鼻子区域)')
    axes[0, 1].axis('off')
    
    # 预测结果
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title(f'预测结果\n(Dice={metrics["Dice"]:.4f})')
    axes[0, 2].axis('off')
    
    # 第二行：叠加显示
    # 真实+原始
    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].imshow(true_mask, cmap='Reds', alpha=0.5)
    axes[1, 0].set_title('真实标注（红色）')
    axes[1, 0].axis('off')
    
    # 预测+原始
    axes[1, 1].imshow(image, cmap='gray')
    axes[1, 1].imshow(pred_mask, cmap='Reds', alpha=0.5)
    axes[1, 1].set_title('预测结果（红色）')
    axes[1, 1].axis('off')
    
    # 差异图
    diff = (true_mask != pred_mask).astype(float)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title(f'差异区域\n(错误: {diff.sum():.0f} pixels)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from model import UNet
    from data_loader import load_data_loader

    _, val_loader = load_data_loader()  # 获取验证集 DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load("./results/best_unet_model_mIoU0.9562.pth"))  # 加载训练好的模型
    model.eval()
    
    metrics = calculate_metrics(model, val_loader, device)  # 评估模型
    print(f"mIoU: {metrics['mIoU']:.4f}, Dice: {metrics['Dice']:.4f}, "
          f"Accuracy: {metrics['Accuracy']:.4f}, Precision: {metrics['Precision']:.4f}, "
          f"Recall: {metrics['Recall']:.4f}")
    
    # 可视化成功分割结果
    image_batch, true_mask_batch = next(iter(val_loader))

    with torch.no_grad():   
        x = image_batch[0].unsqueeze(0).to(device)
        outputs = model(x)
        pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()

    image = image_batch[0].cpu().numpy().squeeze()
    true_mask = true_mask_batch[0].cpu().numpy().squeeze()
    visualize_successful_segmentation(image, true_mask, pred_mask, metrics)

    
