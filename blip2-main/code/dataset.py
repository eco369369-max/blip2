import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class Flickr8kDataset(Dataset):
    def __init__(self, data_dir, tokenizer, image_processor, max_length=30):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        # 修复：路径拼接（安全写法）
        self.captions_file = os.path.join(data_dir, "captions_200.txt")
        self.df = pd.read_csv(self.captions_file)

        self.image_dir = os.path.join(data_dir, "images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 读取图片
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 图像预处理
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # 读取caption
        caption = str(self.df.iloc[idx, 1])

        # ===================== 核心修复 =====================
        # 1. 关闭自动添加特殊符号（避免EOS出现在开头）
        encoding = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_length - 1,  # 留位置给EOS
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False  # 🔥 关键：关闭自动特殊标记
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 2. 手动在【末尾】添加EOS（正确格式：文本 + EOS）
        eos_token = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        # 3. 拼接并修正长度
        input_ids = torch.cat([input_ids, eos_token])[:self.max_length]
        attention_mask = torch.cat([attention_mask, torch.tensor([1])])[:self.max_length]
        # ====================================================

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }