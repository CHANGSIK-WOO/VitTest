import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import einops

class PatchEmbedding(nn.Module):
    def __init__(self, img_size : int, patch_size : int, in_channels : int, embed_dim : int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patch = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride = patch_size) # (bs, C, H, W) --> (bs, embed_dim, H, W)


    def forward(self, x):
        print(x)
        x = self.projection(x)
        print(x)
        x = x.reshape(x.shape[0], -1, self.embed_dim) # (bs, embed_dim, H, W) --> (bs, (HW), embed_dim)
        return x


x = torch.rand([32, 3, 256, 256])
print(x.shape)
pe = PatchEmbedding(x.shape[2], 16, x.shape[1], 768)
x = pe(x)
print(x.shape)

