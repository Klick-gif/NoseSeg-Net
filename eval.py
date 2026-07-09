import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from thop import profile
from thop import clever_format


def create_metric_state(num_classes, device):
    return {
        "num_classes": num_classes,
        "total_correct": 0,
        "total_pixels": 0,
        "intersections": torch.zeros(num_classes, device=device),
        "pred_areas": torch.zeros(num_classes, device=device),
        "target_areas": torch.zeros(num_classes, device=device),
    }


def update_metric_state(state, outputs, y):
    preds = torch.argmax(outputs, dim=1)
    state["total_correct"] += (preds == y).sum().item()
    state["total_pixels"] += y.numel()

    for class_idx in range(1, state["num_classes"]):  # 跳过背景
        pred = (preds == class_idx).float()
        target = (y == class_idx).float()

        state["intersections"][class_idx] += (pred * target).sum()
        state["pred_areas"][class_idx] += pred.sum()
        state["target_areas"][class_idx] += target.sum()


def summarize_metric_state(state):
    eps = 1e-8
    iou_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []

    for class_idx in range(1, state["num_classes"]):  # 跳过背景
        intersection = state["intersections"][class_idx]
        pred_area = state["pred_areas"][class_idx]
        target_area = state["target_areas"][class_idx]
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
        "Accuracy": state["total_correct"] / state["total_pixels"] if state["total_pixels"] > 0 else 0.0,
        "Precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "Recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


def calculate_metrics(model, test_loader, device, num_classes=2):
    """计算分割评估指标：mIoU、Dice、Accuracy、Precision、Recall"""
    model.eval()
    state = create_metric_state(num_classes, device)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            update_metric_state(state, outputs, y)

    return summarize_metric_state(state)


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


def calculate_model_complexity(model,
                               input_size=(1, 1, 512, 512),
                               device="cuda"):
    """Params、FLOPs统计"""

    model.eval()

    dummy = torch.randn(input_size).to(device)

    flops, params = profile(
        model,
        inputs=(dummy,),
        verbose=False
    )

    flops, params = clever_format(
        [flops, params],
        "%.3f"
    )

    return params, flops


def calculate_fps(model, input_size=(1,1,512,512), device="cuda",
                  warmup=20, repeat=100):
    """测试推理速度FPS, latency(ms)"""

    model.eval()

    dummy = torch.randn(input_size).to(device)

    with torch.no_grad():

        # GPU预热
        for _ in range(warmup):
            _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()

        for _ in range(repeat):
            _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    total = end - start

    latency = total / repeat

    fps = repeat / total

    return fps, latency * 1000


if __name__ == "__main__":
    from model import choose_model
    from data_loader import load_data_loader
    from config import get_config
    config = get_config()

    train_loader, val_loader = load_data_loader(config=config)  # 获取验证集 DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = choose_model(config).to(device)
    model.load_state_dict(torch.load(f"./results/best_{config.model}_model.pth", weights_only=True))  # 加载训练好的模型
    model.eval()
    
    metrics = calculate_metrics(model, val_loader, device)  # 评估模型
    print(f"mIoU: {metrics['mIoU']:.4f}, Dice: {metrics['Dice']:.4f}, "
          f"Accuracy: {metrics['Accuracy']:.4f}, Precision: {metrics['Precision']:.4f}, "
          f"Recall: {metrics['Recall']:.4f}")
    
    # 可视化成功分割结果
    image_batch, true_mask_batch = next(iter(train_loader))
    for i in range(image_batch.shape[0]):
        with torch.no_grad():   
            x = image_batch[i].unsqueeze(0).to(device)
            outputs = model(x)
            pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()

        image = image_batch[i].cpu().numpy().squeeze()
        true_mask = true_mask_batch[i].cpu().numpy().squeeze()
        visualize_successful_segmentation(image, true_mask, pred_mask, metrics)


    # 计算模型复杂度
    params, flops = calculate_model_complexity(
        model,
        input_size=(1,1,config.image_size,config.image_size),
        device=device
    )
    print(f"Params: {params}, FLOPs: {flops}")
    
    # 计算推理速度
    fps, latency = calculate_fps(model, device=device)
    print(f"FPS: {fps:.2f}, Latency(ms): {latency:.2f}")


    
