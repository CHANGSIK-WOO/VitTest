import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import einops
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size : int, patch_size : int, in_channels : int, embed_dim : int = 768):
        super().__init__()
        self.img_size = img_size # 256 * 256
        self.patch_size = patch_size # 16 * 16
        self.embed_dim = embed_dim # 768
        self.num_patch = (img_size // patch_size) ** 2 # 16 ** 2 = 256
        
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride = patch_size) 
        # (bs, C, H, W) --> (bs, embed_dim, H', W')
        # (bs, 3, 256, 256) --> (bs, 768, 16, 16)

    def forward(self, x):
        print(f"Input : {x.shape}") # (bs, 3, 256, 256)
        x = self.projection(x) 
        print(f"After Projection : {x.shape}") # (bs, 3, 256, 256) --> (bs, 768, 16, 16)
        x = x.flatten(2).transpose(1, 2)
        # x = einops.rearrange(x, "b c h w -> b (h w) c") # (bs, 768, 16, 16) --> (bs, 256, 768)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim : int, num_heads : int = 12, qkv_bias : bool = True, attn_p : float = 0.0, proj_p : float = 0.0):
        self.embed_dim = embed_dim # input embedding dimension
        self.num_heads = num_heads # Multi Head Number

        self.head_dim = embed_dim // num_heads # Multi-Head Dimension
        assert self.embed_dim == self.head_dim * self.num_heads, "embed_dim must be divisible by num_heads"

        self.scale = math.sqrt(self.head_dim) # attention score scale
        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim * 3, bias = qkv_bias) # (bs, 256, 768) --> (bs, 256, 768 * 3)
        self.attn_drop = nn.Dropout(p = attn_p)
        self.projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj_drop = nn.Dropout(p = proj_p)

    def forward(self, x):
        bs, num_tokens, embed_dim = x.shape # (bs, patch token 256 + cls token 1=257, 768)
        if embed_dim != self.embed_dim:
            raise ValueError(f"Input Dimension of x {embed_dim} does not equal to the expected dimension {self.dim}")
        
        qkv = self.qkv(x) # (bs, num_tokens + 1, dim *3) +1 means [CLS] token in 1st position of embedding for classification.
        qkv = qkv.reshape(bs, num_tokens, 3, self.num_heads, self.head_dim) # (bs, 257, 768 * 3) --> (bs, 257, 3, 12, 64)
        qkv =  # (bs, 257, 3, 12, 64) --> (3, bs, 12, 257, 64)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = q @ k.tranpose(-2, -1) * self.scale # (3, bs, 12, 257, 64) --> (bs, 12, 257, 257)
        attn_scores = attn_scores.softmax(dim=-1) # (bs, 12, 257, 257) --> (bs, 12, 257, 257)
        attn_scores = self.attn_drop(attn_scores)

        mha = attn_scores @ v # (bs, 12, 257, 257) --> (bs, 12, 257, 64)
        mha = mha.transpose(1, 2).flatten(2) # (bs, 12, 257, 64) --> (bs, 257, 12, 64) --> (bs, 257, 768)

        x = self.projection(mha)
        x = self.proj_drop(x)

        return mha 



x = torch.rand([32, 3, 256, 256])
bs, c, h, w = x.shape
pe = PatchEmbedding(img_size = h, patch_size = 16, in_channels = c, embed_dim = 768)
x = pe(x)
print(f"After Patch Embedding : {x.shape}")

mha = MultiHeadAttention()


        






        self.qkv_bias = qkv_bias