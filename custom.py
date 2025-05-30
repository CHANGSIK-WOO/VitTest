import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import einops

class PatchEmbedding(nn.Module):
    def __init__(self, img_size : int, patch_size : int, in_channels : int = 3, embed_dim : int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels = embed_dim, kernel_size = patch_size, stride = patch_size) # (3, 256, 256) --> (16, 768, 16, 16)

    def forward(self, x):
        x = self.projection(x) # (bs, embed_dim, h, w)
        x = einops.rearrange(x, "b c h w -> b (h w) c") # (bs, num_patches, embed_dim), equal to x.flatten(2) --> x.transpose(1, 2)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim : int, num_heads : int = 12, qkv_bias : bool = True, attn_p : float = 0.0, proj_p : float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim # Patch embedding dimension
        self.head_dim = dim // num_heads # Multi-head attetion head dimension

        assert self.head_dim * self.num_heads == self.dim, "dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=qkv_bias)  # Query, Key, Value linear transformation
        self.attn_drop = nn.Dropout(p = attn_p)
        self.projection = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(p = proj_p)

    def forward(self, x):
        bs, num_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError(f"Input Dimension of x {dim} does not equal to the expected dimension {self.dim}")
        
        qkv = self.qkv(x) # (bs, num_tokens + 1, dim *3) +1 means [CLS] token in 1st position of embedding for classification.
        qkv = qkv.reshape(bs, num_tokens, 3, self.num_heads, self.head_dim)  # (bs, num_tokens + 1, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, bs, num_heads, num_tokens + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (bs, num_heads, num_tokens+1, num_tokens+1)
        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = self.attn_drop(attention_scores)

        weighted_avg = attention_scores @ v # (bs, num_heads, num_tokens + 1, head_dim)
        weighted_avg = einops.rearrange(weighted_avg, "b n_h n_t h_d -> b n_t (n_h n_d)") # (bs, num_tokens + 1 , dim) equal to x.transpose(1, 2) -> x.flatten(2)

        x = self.projection(weighted_avg)
        x= self.proj_drop(x)

        return x 

class MLP(nn.Module):
    def __init__(self, input_dim : int, h_dim : int, output_dim : int, dropout : float):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=h_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(in_features=h_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x) 

        return x
         
class Block(nn.Module): # Transformer Block Lx
    def __init__(self, dim : int, num_heads : int, p : float, mlp_ratio : float = 4.0, qkv_bias : bool = True, attn_p : float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps = 10**-6)
        self.attn = MultiHeadAttention(dim = dim, num_heads = num_heads, qkv_bias = True, attn_p = attn_p, proj_p = p)
        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps = 10**-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(input_dim = dim, h_dim = hidden_features, output_dim = dim, dropout = p)
    
    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x) + x
        x = self.norm2(x)
        x = self.mlp(x) + x

        return x
    
class VisionTransformer(nn.Module):
    # 1. patch Embedding + position Embedding
    # 2. Transformer Encoder Blocks
    
    def __init__(
            self,
            img_size : int = 384, patch_size : int = 16, 
            in_channels : int = 3, embed_dim : int = 768, 
            num_heads : int = 12, qkv_bias : bool = True, 
            mlp_ratio : float = 4.0, 
            p : float = 0.0, attn_p : float = 0.0, 
            num_layers : int = 12, 
            num_classes : int = 1000):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size = img_size, patch_size = patch_size, in_channels = in_channels, embed_dim = embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # shape : [1, 1, embed_dim]``
        # nn.Parameter is a tensor that is a parameter of the model, it will be optimized during training.
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.num_patches, embed_dim)) # shape : [1, 1 + num_patches, embed_dim]
        self.pos_drop = nn.Dropout(p = p)
        self.blocks = nn.ModuleList([
            Block(dim = embed_dim, num_heads = num_heads, p = p, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, attn_p = attn_p)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(normalized_shape=embed_dim, eps = 10**-6)
        self.head = nn.Linear(in_features = embed_dim, out_features =  num_classes)

    def forward(self, x):
        batch_size = x.shape[0] 
        x = self.patch_embedding(x) # (bs, num_patches, embed_dim)
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (bs, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (bs, num_patches + 1, embed_dim)
        x = x + self.pos_embedding
        x = self.pos_drop(x)

        for block in self.blocks :
            x = block(x)
        
        x = self.norm(x)
        cls_token = x[:, 0] # (bs, embed_dim)
        x = self.head(cls_token) #  (bs, num_classes) 

        return x 