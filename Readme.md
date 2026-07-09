<div align="center">

# Medical Image Segmentation

**基于 U-Net / Attention U-Net / U-Net++ 的医学图像语义分割框架**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+cu118-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 项目简介

本项目是一个面向医学 CT/MRI 图像的语义分割框架，支持 **7 组 NIfTI 格式数据集**，提供 **U-Net**、**Attention U-Net** 和 **U-Net++** 三种经典分割网络，涵盖从数据预处理、模型训练、评估、推理测速到可视化的完整流程。

### 核心特性

- **三模型支持** — U-Net & Attention U-Net & U-Net++，一行命令切换
- **7 组医学数据集** — 覆盖鼻窦多层结构，开箱即用
- **预缓存加速** — 离线预处理为 `.pt` 张量，训练时 `--use_cache` 直接加载，大幅降低 CPU 瓶颈
- **丰富评估指标** — mIoU / Dice / Accuracy / Precision / Recall
- **模型复杂度分析** — 自动统计 Params、FLOPs
- **推理速度基准** — FPS、Latency 实测（含 GPU 预热 & CUDA 同步）
- **TensorBoard 可视化** — 训练曲线与指标实时监控
- **分割结果可视化** — 原图 / 标签 / 预测 / 差异图一键生成

---

## 项目结构

```
ITK/
├── config.py              # 超参数配置 & 命令行参数解析
├── dataset.py             # 数据集加载（在线 & 缓存两种模式）
├── data_loader.py         # DataLoader 构建 & 按 case 划分训练/验证集
├── model.py               # U-Net / Attention U-Net / U-Net++ 模型定义
├── train.py               # 训练循环（含 TensorBoard 日志）
├── eval.py                # 评估指标计算、模型复杂度分析、推理测速 & 可视化
├── main.py                # 入口脚本
├── preprocess_cache.py    # 离线预处理：NIfTI → .pt 张量缓存
├── test/
│   └── test.py            # 标签合并工具
├── dataset/               # 原始 NIfTI 数据（7 组）
│   ├── H2_4Layer/
│   ├── S_2/
│   ├── S_4/
│   ├── nose_layer4/
│   ├── nose_layer12/
│   ├── nose_layer28/
│   └── nose_layer36/
├── cache/                 # 预处理后的 .pt 缓存（可选）
├── results/               # 模型权重保存
├── runs/                  # TensorBoard 日志
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## 数据集

本项目包含 **7 组医学分割数据集**，均为 NIfTI (`.nii` / `.nii.gz`) 格式，每组包含原图与标注：

| 数据集 | 描述 | 原图文件 | 标注文件 |
|--------|------|----------|----------|
| **H2_4Layer** | H2 四层结构 | `H2-4Layer.nii` | `H2-4Layer_Seg.nii.gz` |
| **S_2** | S-2 数据 | `S-2.nii` | `S_2_Seg.nii.gz` |
| **S_4** | S-4 数据 | `S-4.nii` | `S-4-seg.nii` |
| **nose_layer4** | 鼻窦 4 层 | `nose_layer4.nii` | `nose_layer4_Seg.nii.gz` |
| **nose_layer12** | 鼻窦 12 层 | `nose_layer12.nii` | `nose_layer12-seg.nii.gz` |
| **nose_layer28** | 鼻窦 28 层 | `nose_layer28.nii` | `nose_layer28_Seg.nii.gz` |
| **nose_layer36** | 鼻窦 36 层 | `nose_layer36.nii` | `nose_layer36_Seg.nii` |

> 数据集自动扫描 `dataset/` 目录，按文件名中是否包含 `seg` 区分原图与标注。

---

## 模型

### U-Net

经典编码器-解码器结构，4 层下采样 + 瓶颈层 + 4 层上采样，跳跃连接通过 `torch.cat` 拼接。

> 论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

```
编码器: 64 → 128 → 256 → 512
瓶颈层: 1024
解码器: 512 → 256 → 128 → 64
```

### Attention U-Net

在 U-Net 基础上，每个解码器跳跃连接处增加 **注意力门（Attention Gate）**，自动学习关注目标区域、抑制无关背景，提升小目标分割精度。

> 论文：[Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)

### U-Net++

在 U-Net 基础上引入 **嵌套跳跃连接（Nested Skip Pathways）**，通过额外的跨层上采样-拼接-卷积路径，使解码器在不同语义层级上融合编码器特征，减少编码器-解码器之间的语义鸿沟。

> 论文：[U-Net++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)

---

## 基准测试

以下结果在 **RTX 4090 Ti** 上测得，输入分辨率 512×512，batch size = 1。

### 分割精度

| 模型 | mIoU | Dice | Accuracy | Precision | Recall |
|------|------|------|----------|-----------|--------|
| **Attention U-Net** | 0.9580 | 0.9785 | 0.9865 | 0.9806 | 0.9765 |
| **U-Net++** | 0.9556 | 0.9773 | 0.9857 | 0.9755 | 0.9790 |
| **U-Net** | 0.9552 | 0.9771 | 0.9856 | 0.9781 | 0.9760 |

### 模型复杂度 & 推理速度

| 模型 | Params | FLOPs | FPS | Latency (ms) |
|------|--------|-------|-----|--------------|
| **U-Net** | 31.042M | 218.666G | 117.52 | 8.51 |
| **Attention U-Net** | 31.394M | 223.104G | 103.13 | 9.70 |
| **U-Net++** | 33.338M | 256.280G | 105.73 | 9.46 |

> FPS / Latency 测量方式：GPU 预热 20 次，连续推理 100 次取平均，含 `torch.cuda.synchronize()` 同步。

---

## 快速开始

### 环境配置

```bash
conda create -n itk python=3.11
conda activate itk
pip install -r requirements.txt
```

> **核心依赖**：PyTorch 2.7.1 + CUDA 11.8、nibabel、tensorboard、matplotlib、thop

### 直接训练

```bash
python main.py --batch_size 12 --epochs 50 --lr 1e-4 --num_workers 8
```

### 使用缓存加速训练（推荐）

当 CPU 处理速度成为瓶颈、拖慢 GPU 显存利用率时，可提前将 NIfTI 数据预处理为 `.pt` 张量缓存：

```bash
# 第一步：离线预处理（归一化、Resize、转张量，存为 .pt）
python preprocess_cache.py

# 第二步：启用缓存模式训练，跳过在线预处理
python main.py --use_cache --batch_size 4 --epochs 50 --lr 1e-4 --num_workers 8
```

> **原理**：`preprocess_cache.py` 将每个切片的归一化、Resize、Tensor 转换一次性完成，保存至 `cache/` 目录。训练时 `--use_cache` 直接 `torch.load` 加载，消除 CPU 瓶颈，显著提升数据吞吐。

### 选择模型

```bash
# U-Net
python main.py --model unet --batch_size 12 --epochs 50 --lr 1e-4 --num_workers 8

# Attention U-Net
python main.py --model attunet --batch_size 12 --epochs 50 --lr 1e-4 --num_workers 8 --use_cache

# U-Net++
python main.py --model unet++ --batch_size 12 --epochs 50 --lr 1e-4 --num_workers 8 --use_cache
```

### 评估、推理测速 & 可视化

```bash
python eval.py
```

将输出：
- **分割指标**：mIoU、Dice、Accuracy、Precision、Recall
- **模型复杂度**：Params、FLOPs
- **推理速度**：FPS、Latency (ms)
- **可视化窗口**：原图 / 真实标签 / 预测结果 / 差异图

### TensorBoard 监控

```bash
tensorboard --logdir runs
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 42 | 随机种子 |
| `--data_root` | `dataset` | 数据集根目录 |
| `--cache_root` | `cache` | 缓存根目录 |
| `--use_cache` | `False` | 启用缓存模式（需先运行预处理） |
| `--image_size` | 512 | 输入图像尺寸 |
| `--skip_empty` | `False` | 跳过无标注的空白切片 |
| `--train_ratio` | 0.8 | 训练集比例 |
| `--batch_size` | 4 | 批大小 |
| `--no_shuffle` | `False` | 禁用训练集打乱 |
| `--num_workers` | 6 | DataLoader 工作线程数 |
| `--no_pin_memory` | `False` | 禁用 pin_memory |
| `--in_channels` | 1 | 输入通道数（灰度图=1） |
| `--num_classes` | 2 | 分类数（背景+前景=2） |
| `--epochs` | 20 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--scheduler_patience` | 5 | ReduceLROnPlateau patience |
| `--scheduler_factor` | 0.5 | 学习率衰减因子 |
| `--model` | `unet` | 模型选择：`unet` / `attunet` / `unet++` |
| `--log_root` | `runs` | TensorBoard 日志目录 |
| `--result_root` | `results` | 模型权重保存目录 |

---

## 训练流程

1. **数据加载** — 自动扫描 `dataset/` 下所有子目录，配对原图与标注
2. **数据划分** — 按 case 内部 80/20 随机划分训练/验证集，确保每个 case 均有训练和验证样本
3. **预处理** — 归一化至 [0,1]、Resize 至 512×512、二值化标签（>0 为前景）
4. **训练** — CrossEntropyLoss + Adam + ReduceLROnPlateau，自动保存最优模型
5. **评估** — 计算 mIoU / Dice / Accuracy / Precision / Recall
6. **推理分析** — 统计 Params、FLOPs，测量 FPS、Latency
7. **可视化** — TensorBoard 曲线 + 分割结果对比图

---

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。