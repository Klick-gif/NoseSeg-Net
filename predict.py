import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from PIL import Image
from model import choose_model
from config import get_config


SUPPORTED_2D_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_label_file(filename):
    name = filename.lower()
    return "seg" in name


def load_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = choose_model(config).to(device)
    model_path = f"./results/best_{config.model}_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def preprocess_slice(image_2d, image_size):
    image_min = image_2d.min()
    image_max = image_2d.max()
    image_norm = (image_2d - image_min) / (image_max - image_min + 1e-8)

    x = torch.from_numpy(image_norm).float().unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


def save_overlay(image_gray, pred_mask, output_path, alpha=0.4):
    h, w = image_gray.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = (image_gray * 255).astype(np.uint8)
    overlay[:, :, 1] = (image_gray * 255).astype(np.uint8)
    overlay[:, :, 2] = (image_gray * 255).astype(np.uint8)

    mask_bool = pred_mask > 0
    overlay[mask_bool, 0] = np.clip(
        overlay[mask_bool, 0] * (1 - alpha) + 255 * alpha, 0, 255
    ).astype(np.uint8)
    overlay[mask_bool, 1] = np.clip(
        overlay[mask_bool, 1] * (1 - alpha) + 50 * alpha, 0, 255
    ).astype(np.uint8)
    overlay[mask_bool, 2] = np.clip(
        overlay[mask_bool, 2] * (1 - alpha) + 50 * alpha, 0, 255
    ).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def predict_single(config):
    model, device = load_model(config)
    image_path = config.image
    output_path = config.output

    if image_path.endswith((".nii", ".nii.gz")):
        nii = nib.load(image_path)
        volume = np.asarray(nii.dataobj, dtype=np.float32)
        if volume.ndim == 3:
            z = volume.shape[2] // 2
            image_2d = volume[:, :, z]
        else:
            image_2d = volume
    else:
        image_2d = np.array(Image.open(image_path), dtype=np.float32)
        if image_2d.ndim == 3:
            image_2d = image_2d[:, :, 0]

    x = preprocess_slice(image_2d, config.image_size).to(device)

    with torch.no_grad():
        outputs = model(x)
        pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()

    image_display = x[0, 0].cpu().numpy()
    save_overlay(image_display, pred_mask, output_path)
    print(f"单张图片预测结果已保存至: {output_path}")


def predict_batch(config):
    model, device = load_model(config)
    input_dir = config.input_dir
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    nii_files = []
    for f in sorted(os.listdir(input_dir)):
        if is_label_file(f):
            continue
        if f.endswith(".nii.gz") or f.endswith(".nii"):
            nii_files.append(os.path.join(input_dir, f))

    img_files = []
    for ext in SUPPORTED_2D_EXTS:
        img_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        img_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    img_files = sorted(set(f for f in img_files if not is_label_file(f)))

    total = 0

    for nii_path in nii_files:
        basename = os.path.basename(nii_path)
        if basename.endswith(".nii.gz"):
            case_name = basename.replace(".nii.gz", "")
        else:
            case_name = basename.replace(".nii", "")

        nii = nib.load(nii_path)
        volume = np.asarray(nii.dataobj, dtype=np.float32)
        if volume.ndim == 2:
            slices = [volume]
        else:
            slices = [volume[:, :, z] for z in range(volume.shape[2])]

        print(f"[{case_name}] 共 {len(slices)} 张切片，开始推理...")

        for z, image_2d in enumerate(slices):
            x = preprocess_slice(image_2d, config.image_size).to(device)

            with torch.no_grad():
                outputs = model(x)
                pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()

            image_display = x[0, 0].cpu().numpy()
            output_path = os.path.join(output_dir, f"{case_name}_slice_{z:04d}.png")
            save_overlay(image_display, pred_mask, output_path)
            total += 1

        print(f"  [{case_name}] 完成，已保存 {len(slices)} 张")

    for img_path in img_files:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        image_2d = np.array(Image.open(img_path), dtype=np.float32)
        if image_2d.ndim == 3:
            image_2d = image_2d[:, :, 0]

        x = preprocess_slice(image_2d, config.image_size).to(device)

        with torch.no_grad():
            outputs = model(x)
            pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()

        image_display = x[0, 0].cpu().numpy()
        output_path = os.path.join(output_dir, f"{filename}.png")
        save_overlay(image_display, pred_mask, output_path)
        total += 1

    print(f"批量推理完成，共保存 {total} 张结果至 {output_dir}")


if __name__ == "__main__":
    config = get_config()

    if os.path.isdir(config.input_dir):
        predict_batch(config)
    elif os.path.isfile(config.image):
        predict_single(config)
    else:
        print(f"请指定有效的输入路径: --image <文件路径> 或 --input_dir <文件夹路径>")