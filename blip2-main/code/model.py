import copy
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, OPTForCausalLM


class MiniQFormer(nn.Module):

    def __init__(self, query_len=32, hidden_size=768, num_layers=6):
        super().__init__()

        self.query = nn.Parameter(
            torch.zeros(1, query_len, hidden_size)
        )

        nn.init.normal_(
            self.query,
            std=hidden_size ** -0.5
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=8,
            batch_first=True,
            dim_feedforward=3072
        )

        # 修复：不能共享同一个 layer
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer)
            for _ in range(num_layers)
        ])

    def forward(self, image_features):

        B = image_features.shape[0]

        q = self.query.expand(B, -1, -1)

        for layer in self.layers:
            q = layer(
                tgt=q,
                memory=image_features
            )

        return q


class MiniBLIP2(nn.Module):

    def __init__(self):
        super().__init__()

        # ------------------------------------------------
        # Vision Encoder
        # ------------------------------------------------
        self.vision = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        for p in self.vision.parameters():
            p.requires_grad = False

        # ------------------------------------------------
        # QFormer
        # ------------------------------------------------
        self.qformer = MiniQFormer()

        # ------------------------------------------------
        # Projection
        # ------------------------------------------------
        self.proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768)
        )

        # ------------------------------------------------
        # OPT
        # ------------------------------------------------
        self.opt = OPTForCausalLM.from_pretrained(
            "facebook/opt-125m"
        )

        for p in self.opt.parameters():
            p.requires_grad = False

        self.tokenizer = None

        print("Q-Former hidden:", self.qformer.query.size(-1))
        print("OPT hidden:", self.opt.config.hidden_size)

    # ------------------------------------------------
    # tokenizer
    # ------------------------------------------------
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    # ------------------------------------------------
    # train forward
    # ------------------------------------------------
    def forward(self, images, caption_ids):

        with torch.no_grad():
            vis = self.vision(images).last_hidden_state

        q = self.qformer(vis)

        q = self.proj(q)

        text_emb = self.opt.get_input_embeddings()(
            caption_ids
        )

        inputs_emb = torch.cat([
            q,
            text_emb
        ], dim=1)

        batch_size = images.shape[0]

        Q = q.shape[1]

        T = caption_ids.shape[1]

        total = Q + T

        attention_mask = torch.ones(
            (batch_size, total),
            device=inputs_emb.device
        )

        labels = torch.cat([
            torch.full(
                (batch_size, Q),
                -100,
                device=caption_ids.device,
                dtype=torch.long
            ),
            caption_ids
        ], dim=1)

        outputs = self.opt(
            inputs_embeds=inputs_emb,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs.loss

    # ------------------------------------------------
    # generate
    # ------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        images,
        max_len=30,
        num_beams=3,
        temperature=1.0,
        do_sample=False,
        top_p=0.9,
    ):

        if self.tokenizer is None:
            raise ValueError("Tokenizer not set.")

        self.eval()
        vis = self.vision(images).last_hidden_state
        q = self.qformer(vis)
        q = self.proj(q)

        prefix_len = q.shape[1]

        attn_mask = torch.ones(
            q.size(0),
            prefix_len,
            device=q.device,
            dtype=torch.long
        )

        gen_ids = self.opt.generate(
            inputs_embeds=q,
            attention_mask=attn_mask,
            max_new_tokens=max_len,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # ⚠️ 关键修复：
        # 不要再切 prefix_len
        captions = self.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True
        )

        return captions