import timm
import torch.nn as nn
import torch

model = timm.create_model('vit_small_patch16_224', num_classes=10)

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = model
    
    def forward(self, x):
        x = self.backbone(x)
        return x


class VisionTransformer42(nn.Module):
    def __init__(self, patch_size=3, num_blocks=12, emb_dim=216, num_heads=6, mlp_dim=432, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, emb_dim)
        self.transformer = Transformer(emb_dim, num_blocks, num_heads, mlp_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        cls_token = x[:, 0]
        x = self.classifier(cls_token)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, mlp_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(dim, num_heads),
            nn.LayerNorm(dim),
            FeedForward(dim, mlp_dim)
            )
        
        self.encoder = nn.ModuleList([])
        for _ in range(num_blocks):
            self.encoder.append(self.block)

    def forward(self, x):
        for block in self.encoder:
            ln1, attn, ln2, ffn = block
            x = x + attn(ln1(x))
            x = x + ffn(ln2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, D = x.shape   # batch, sequence, embedding dim

        q = self.W_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, N, head_dim)
        k = self.W_k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, N, head_dim)
        v = self.W_v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # (B, num_heads, N, head_dim)

        scores = torch.matmul(q, k.transpose(3, 2) / (self.head_dim ** 0.5))    # (B, num_heads, N, N)
        weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).flatten(2)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.dense0 = nn.Linear(dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense1 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        x = self.dense0(x)
        x = self.gelu(x)
        x = self.dense1(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, emb_dim):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, 197, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)

        x = self.embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([cls_token_expanded, x], dim=1)
        x += self.pos_emb
        return x