import os
import shutil

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import get_config
from dataset import get_data_paths


def preprocess_slice(image, label, size):
    image = image.astype(np.float32)
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-8)
    label = (label > 0).astype(np.int64)

    x = torch.from_numpy(image).float().unsqueeze(0)
    y = torch.from_numpy(label).long()

    x = F.interpolate(
        x.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    y = F.interpolate(
        y.unsqueeze(0).unsqueeze(0).float(),
        size=size,
        mode="nearest",
    ).squeeze(0).squeeze(0).long()

    return x, y


def build_cache(config):
    size = (config.image_size, config.image_size)

    if os.path.exists(config.cache_root):
        shutil.rmtree(config.cache_root)
    os.makedirs(config.cache_root, exist_ok=True)

    pairs = get_data_paths(config.data_root)
    print(f"Building cache in {config.cache_root}")
    print(f"Found {len(pairs)} cases")

    for image_path, label_path in pairs:
        case_name = os.path.basename(os.path.dirname(image_path))
        case_cache_dir = os.path.join(config.cache_root, case_name)
        os.makedirs(case_cache_dir, exist_ok=True)

        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)

        if image_nii.shape != label_nii.shape:
            raise ValueError(
                f"原图和标签 shape 不一致:\n"
                f"image: {image_path}, shape={image_nii.shape}\n"
                f"label: {label_path}, shape={label_nii.shape}"
            )

        depth = image_nii.shape[2]
        saved = 0

        for z in tqdm(range(depth), desc=f"Caching {case_name}"):
            label_slice = np.asarray(label_nii.dataobj[:, :, z])
            if config.skip_empty and label_slice.max() <= 0:
                continue

            image_slice = np.asarray(image_nii.dataobj[:, :, z])
            x, y = preprocess_slice(image_slice, label_slice, size)

            save_path = os.path.join(case_cache_dir, f"slice_{z:04d}.pt")
            torch.save({"image": x, "label": y}, save_path)
            saved += 1

        print(f"  {case_name}: saved {saved}/{depth} slices")


if __name__ == "__main__":
    config = get_config()
    build_cache(config)
