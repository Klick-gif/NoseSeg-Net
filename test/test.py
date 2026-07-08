import nibabel as nib
import numpy as np

# 所有待合并文件列表，按顺序：靠前的先合并，后面的标签会覆盖前面重叠区域
file_list = ["001.nii.gz", "002.nii.gz", "003.nii.gz", "004.nii.gz"]
out_path = "nose_layer20_Seg.nii.gz"

# 加载第一个文件作为基础基底
base_img = nib.load(file_list[0])
merged_data = base_img.get_fdata()
base_affine = base_img.affine
base_header = base_img.header

# 循环遍历剩余所有文件，后文件覆盖重叠区域
for file in file_list[1:]:
    curr_img = nib.load(file)
    curr_data = curr_img.get_fdata()
    
    # 校验尺寸统一
    if curr_data.shape != merged_data.shape:
        raise Exception(f"文件 {file} 尺寸和前面文件不一致，无法合并，请先配准！")
    
    # 核心逻辑：当前文件有标签的位置，全部覆盖基底；无标签则保留原有
    mask_curr_has_label = curr_data > 0
    merged_data = np.where(mask_curr_has_label, curr_data, merged_data)

# 保存合并结果
merged_img = nib.Nifti1Image(merged_data, base_affine, base_header)
nib.save(merged_img, out_path)

print(f"全部文件合并完成，重叠区域优先保留列表靠后的标签，输出：{out_path}")