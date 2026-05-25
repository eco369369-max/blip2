import os
# 离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings
warnings.filterwarnings("ignore")

import sys
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from model import MiniBLIP2

# ---------- 初始化 ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/mini_blip2_200.pth"

# 加载分词器/处理器
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

# 模型
model = MiniBLIP2()
model.set_tokenizer(tokenizer)
model.to(DEVICE)

# 加载权重
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
model.qformer.load_state_dict(checkpoint["qformer_state_dict"])
model.proj.load_state_dict(checkpoint["proj_state_dict"])
model.eval()

# ========================
# 🔍 诊断 1：随机前缀测试 OPT 是否正常
# ========================
print("\n=== 诊断：用随机向量测试 OPT 生成 ===")
with torch.no_grad():
    dummy_q = torch.randn(1, 32, 768).to(DEVICE)
    dummy_attn = torch.ones(1, 32, dtype=torch.long).to(DEVICE)
    gen_test = model.opt.generate(
        inputs_embeds=dummy_q,
        attention_mask=dummy_attn,
        max_length=32 + 20,
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("随机前缀生成原始 ID:", gen_test)
    print("解码（保留特殊符号）:", tokenizer.batch_decode(gen_test[:, 32:], skip_special_tokens=False))
    print("解码（正常）:", tokenizer.batch_decode(gen_test[:, 32:], skip_special_tokens=True))
print("==============================\n")

# ---------- 单张图片生成（带详细诊断） ----------
def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        # 手动走一遍前向，获取中间结果
        vis = model.vision(pixel_values).last_hidden_state
        q = model.qformer(vis)
        q = model.proj(q)

        # 打印 Q-Former 输出的统计量
        print(f"Q mean/std: {q.mean().item():.4f} / {q.std().item():.4f}")

        # 构建 attention mask
        prefix_len = q.shape[1]
        attn = torch.ones(1, prefix_len, dtype=torch.long, device=DEVICE)

        # 用 OPT 生成（完全绕过你的 generate 方法，直接观察）
        gen_ids = model.opt.generate(
            inputs_embeds=q,
            attention_mask=attn,
            max_length=prefix_len + 20,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print("生成 ID:", gen_ids)

        print("解码（保留特殊符号）:",
              tokenizer.batch_decode(gen_ids, skip_special_tokens=False))

        captions = tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True
        )
    return captions[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python test.py <图片路径>")
        sys.exit(1)

    img_path = sys.argv[1]
    caption = caption_image(img_path)
    print("🎯 最终字幕:", caption)