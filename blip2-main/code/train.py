import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor
from torch.optim import AdamW
from tqdm import tqdm

# 导入自定义模块
from dataset import Flickr8kDataset
from model import MiniBLIP2  # 使用精简版模型

# ------------------ 配置超参数 ------------------
DATA_DIR = r"D:\blip2-main\blip2-main\data\flickr8k_200"        # 200张图片的数据集路径
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 准备数据 ------------------
print("加载分词器和图像处理器...")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # OPT 没有 pad_token，用 eos 代替

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("加载数据集...")
dataset = Flickr8kDataset(
    data_dir=DATA_DIR,
    tokenizer=tokenizer,
    image_processor=image_processor,
    max_length=30,          # caption 最大长度
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,          # Windows 下建议设为0
    pin_memory=True if DEVICE.type == "cuda" else False,
)

# ------------------ 初始化模型 ------------------
print("初始化模型...")
model = MiniBLIP2()
model.set_tokenizer(tokenizer)  # 为生成时使用
model.to(DEVICE)

# 确保视觉编码器和语言解码器被冻结
for param in model.vision.parameters():
    param.requires_grad = False
for param in model.opt.parameters():
    param.requires_grad = False

# 只训练 Q-Former 和投影层
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"可训练参数量: {sum(p.numel() for p in trainable_params)}")

# ------------------ 优化器 ------------------
optimizer = AdamW(trainable_params, lr=LEARNING_RATE)

# ------------------ 训练循环 ------------------
print("开始训练...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch_idx, batch in enumerate(dataloader):
        # 数据转移到设备
        pixel_values = batch["pixel_values"].to(DEVICE)  # [B, 3, H, W]
        caption_ids = batch["input_ids"].to(DEVICE)        # [B, max_len]
        # 在第一次迭代时打印一个样本
        if batch_idx == 0:
            sample_ids = caption_ids[0].cpu()
            print("Caption IDs:", sample_ids)
            print("Decoded text:", tokenizer.decode(sample_ids, skip_special_tokens=False))
        # 前向传播
        loss = model(pixel_values, caption_ids)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - 平均 Loss: {avg_loss:.4f}")

# ------------------ 保存模型 ------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "qformer_state_dict": model.qformer.state_dict(),
    "proj_state_dict": model.proj.state_dict(),
}, "checkpoints/mini_blip2_200.pth")
print("训练完成，模型已保存。")