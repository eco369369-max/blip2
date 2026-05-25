import os
import shutil
import pandas as pd

# ====== 配置路径 ======
original_dataset_dir = "blip2-main/data/flickr8k_200"  # 请改成你的实际路径
output_dir = "data/flickr8k_200"

# ====== 创建输出目录 ======
images_out_dir = os.path.join(output_dir, "Images")
os.makedirs(images_out_dir, exist_ok=True)

# ====== 读取原始标注文件（第一行为列名） ======
captions_file = os.path.join(original_dataset_dir, "captions.txt")
df = pd.read_csv(captions_file)  # 默认逗号分隔，header=0 表示第一行是列名

# 确认列名
image_col = 'image'
caption_col = 'caption'

# ====== 获取前200张图片的唯一名称 ======
unique_images = df[image_col].unique()[:200]
print(f"将要保存的图片数量: {len(unique_images)}")

# ====== 筛选出这些图片的所有 caption ======
mask = df[image_col].isin(unique_images)
subset_df = df[mask].copy()

# ====== 复制图片到新目录 ======
src_img_dir = os.path.join(original_dataset_dir, "Images")
for img_name in unique_images:
    src_path = os.path.join(src_img_dir, img_name)
    dst_path = os.path.join(images_out_dir, img_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"警告: 图片 {img_name} 不存在于源目录中")

# ====== 保存新的 captions 文件 ======
subset_captions_path = os.path.join(output_dir, "captions_200.txt")
subset_df.to_csv(subset_captions_path, index=False)

print(f"处理完成！图片已保存到: {images_out_dir}")
print(f"新标注文件: {subset_captions_path}")
print(f"前200张图片的总描述数量: {len(subset_df)}")