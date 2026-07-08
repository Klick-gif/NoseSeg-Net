from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

from dataset import NoseDataset


def split_indices_by_case(dataset, train_ratio=0.8, seed=42):
    """在每个 case 内部按比例划分 train/val，避免某个数据集完全缺失。"""
    case_to_indices = defaultdict(list)
    for idx, (case_idx, _) in enumerate(dataset.slices):
        case_to_indices[case_idx].append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []

    for case_idx in sorted(case_to_indices):
        indices = case_to_indices[case_idx]
        perm = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[i] for i in perm]

        train_size = int(len(shuffled) * train_ratio)
        train_size = max(1, min(train_size, len(shuffled) - 1))

        train_indices.extend(shuffled[:train_size])
        val_indices.extend(shuffled[train_size:])

    return train_indices, val_indices


def load_data_loader(config):
    dataset = NoseDataset(
        data_root=config.data_root,
        size=(config.image_size, config.image_size),
        skip_empty=config.skip_empty,
    )
    train_indices, val_indices = split_indices_by_case(
        dataset,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Train slices: {len(train_dataset)}")
    print(f"Val slices: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader
