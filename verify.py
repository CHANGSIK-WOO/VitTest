import numpy as np
import timm # timm is "PyTorch Image Models" library
import torch
from custom import VisionTransformer

#helper
def get_num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)

model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

custom_config = {
    "img_size": 384,
    "patch_size": 16,
    "in_channels": 3,
    "embed_dim": 768,
    "num_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4.0,
    "p": 0.0,
    "attn_p": 0.0,
    "num_layers": 12,
    "num_classes": 1000
}

model_custom = VisionTransformer(**custom_config)
model_custom.eval()

for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")

    p_c.data[:] = p_o.data

    assert_tensors_equal(p_o.data, p_c.data)

inp = torch.rand(1, 3, 384, 384)
res_c = model_custom(inp)
res_o = model_official(inp)

assert get_num_params(model_official) == get_num_params(model_custom)
assert_tensors_equal(res_c, res_o)


torch.save(model_custom, "model.pth")