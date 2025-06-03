import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import einops

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
        print(x.shape) # (bs, 3, 256, 256)
        x = self.projection(x) 
        print(x.shape) # (bs, 3, 256, 256) --> (bs, 768, 16, 16)
        x = einops.rearrange(x, "b c h w -> b (h w) c") # (bs, 768, 16, 16) --> (bs, 256, 768)
        return x


x = torch.rand([32, 3, 256, 256])
print(x.shape)
bs, c, h, w = x.shape
pe = PatchEmbedding(img_size = h, patch_size = 16, in_channels = c, embed_dim = 768)
x = pe(x)
print(x.shape)

