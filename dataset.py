import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def is_nifti_file(filename):
    return filename.endswith(".nii") or filename.endswith(".nii.gz")


def is_label_file(filename):
    name = filename.lower()
    return "seg" in name


def get_data_paths(data_root="dataset"):
    """自动读取 dataset 下所有子数据集，并返回 image/label 配对路径。"""
    pairs = []

    for case_name in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, case_name)

        nii_files = [
            f for f in sorted(os.listdir(case_dir))
            if is_nifti_file(f)
        ]
        image_files = [f for f in nii_files if not is_label_file(f)]
        label_files = [f for f in nii_files if is_label_file(f)]

        image_path = os.path.join(case_dir, image_files[0])
        label_path = os.path.join(case_dir, label_files[0])
        pairs.append((image_path, label_path))

    return pairs


class NoseDataset(Dataset):
    def __init__(self, data_root="dataset", size=(512, 512), skip_empty=False):
        self.size = size
        self.skip_empty = skip_empty
        self.samples = []
        self.slices = []

        pairs = get_data_paths(data_root)
        print(f"Found {len(pairs)} cases:")

        for case_idx, (image_path, label_path) in enumerate(pairs):
            image_nii = nib.load(image_path)
            label_nii = nib.load(label_path)

            self.samples.append({
                "image_path": image_path,
                "label_path": label_path,
                "image": image_nii,
                "label": label_nii,
            })

            depth = image_nii.shape[2]
            valid_count = 0
            for z in range(depth):
                if self.skip_empty:
                    label_slice = np.asarray(label_nii.dataobj[:, :, z])
                    if label_slice.max() <= 0:
                        continue

                self.slices.append((case_idx, z))
                valid_count += 1

            print(
                f"  {os.path.basename(os.path.dirname(image_path))}: "
                f"shape={image_nii.shape}, slices={valid_count}/{depth}"
            )


    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        case_idx, z = self.slices[idx]
        sample = self.samples[case_idx]

        image = np.asarray(sample["image"].dataobj[:, :, z], dtype=np.float32)
        label = np.asarray(sample["label"].dataobj[:, :, z])

        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min + 1e-8)

        # 当前模型是二分类：0=背景，1=目标区域。
        label = (label > 0).astype(np.int64)

        x = torch.from_numpy(image).float().unsqueeze(0)
        y = torch.from_numpy(label).long()

        x = F.interpolate(
            x.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        y = F.interpolate(
            y.unsqueeze(0).unsqueeze(0).float(),
            size=self.size,
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        return x, y


class CachedNoseDataset(Dataset):
    def __init__(self, cache_root="cache"):
        self.cache_root = cache_root
        self.samples = []
        self.slices = []

        if not os.path.isdir(cache_root):
            raise ValueError(f"缓存目录不存在: {cache_root}，请先运行 preprocess_cache.py")

        case_names = [
            name for name in sorted(os.listdir(cache_root))
            if os.path.isdir(os.path.join(cache_root, name))
        ]

        if not case_names:
            raise ValueError(f"缓存目录中没有 case 数据: {cache_root}")

        print(f"Found {len(case_names)} cached cases:")

        for case_idx, case_name in enumerate(case_names):
            case_dir = os.path.join(cache_root, case_name)
            slice_files = [
                os.path.join(case_dir, f)
                for f in sorted(os.listdir(case_dir))
                if f.endswith(".pt")
            ]

            if not slice_files:
                continue

            self.samples.append({
                "case_name": case_name,
                "slice_files": slice_files,
            })

            for slice_idx in range(len(slice_files)):
                self.slices.append((case_idx, slice_idx))

            print(f"  {case_name}: cached slices={len(slice_files)}")

        if not self.slices:
            raise ValueError(f"缓存目录中没有 .pt 切片: {cache_root}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        case_idx, slice_idx = self.slices[idx]
        slice_path = self.samples[case_idx]["slice_files"][slice_idx]
        item = torch.load(slice_path, map_location="cpu", weights_only=True)
        return item["image"], item["label"]


if __name__ == "__main__":
    dataset = NoseDataset()
    print(f"Total training slices: {len(dataset)}")
    x, y = dataset[0]
    print("image:", x.shape, x.dtype, x.min().item(), x.max().item())
    print("label:", y.shape, y.dtype, torch.unique(y))
    print(dataset.samples[0])
