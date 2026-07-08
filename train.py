import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from eval import create_metric_state, summarize_metric_state, update_metric_state
from torch.utils.tensorboard import SummaryWriter
import os
import shutil




def train_model(
    model,
    train_loader,
    val_loader,
    device,
    config,
):
    log_dir = os.path.join(config.log_root, config.model_name)
    save_path = os.path.join(config.result_root, config.save_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = SummaryWriter(log_dir)
    # 损失函数：交叉熵损失（适用于多类别分割）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_batches = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        for x, y in train_bar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batches = 0
        metric_state = create_metric_state(num_classes=config.num_classes, device=device)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
            for x, y in val_bar:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                update_metric_state(metric_state, outputs, y)
                
                val_loss += loss.item()
                val_batches += 1
                val_bar.set_postfix({'loss': loss.item()})
            
            val_metrics = summarize_metric_state(metric_state)

        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # 调整学习率
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✓ New best model saved with val_loss: {avg_val_loss:.4f}")
        
        print(f'Epoch {epoch+1}/{config.epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}',
              f'Val Metrics: Dice: {val_metrics["Dice"]:.4f}, IOU: {val_metrics["mIoU"]:.4f}',
              f'Accuracy: {val_metrics["Accuracy"]:.4f}, Precision: {val_metrics["Precision"]:.4f}, '
              f'Recall: {val_metrics["Recall"]:.4f}')
        # 记录到 TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar('Metrics/Dice', val_metrics["Dice"], epoch)
        writer.add_scalar('Metrics/mIoU', val_metrics["mIoU"], epoch)
        writer.add_scalar('Metrics/Accuracy', val_metrics["Accuracy"], epoch)
        writer.add_scalar('Metrics/Precision', val_metrics["Precision"], epoch)
        writer.add_scalar('Metrics/Recall', val_metrics["Recall"], epoch)
    
    writer.close()
    return train_losses, val_losses

