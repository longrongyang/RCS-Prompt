import torch
import torch.nn as nn
import torch.nn.functional as F

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            s1, s2, s3, s4 = key_prefix.size()
            k_mean = k.reshape(B, -1).mean(dim=1).unsqueeze(1)
            k_std = k.reshape(B, -1).std(dim=1).unsqueeze(1) + 1e-6
            p_k_mean = key_prefix.reshape(B, -1).mean(dim=1).unsqueeze(1)
            p_k_std = key_prefix.reshape(B, -1).std(dim=1).unsqueeze(1) + 1e-6
            key_prefix = ((key_prefix.reshape(B, -1) - p_k_mean) / p_k_std) * k_std + k_mean
            key_prefix = key_prefix.reshape(s1, s2, s3, s4)

            v_mean = v.reshape(B, -1).mean(dim=1).unsqueeze(1)
            v_std = v.reshape(B, -1).std(dim=1).unsqueeze(1) + 1e-6
            p_v_mean = value_prefix.reshape(B, -1).mean(dim=1).unsqueeze(1)
            p_v_std = value_prefix.reshape(B, -1).std(dim=1).unsqueeze(1) + 1e-6
            value_prefix = ((value_prefix.reshape(B, -1) - p_v_mean) / p_v_std) * v_std + v_mean
            value_prefix = value_prefix.reshape(s1, s2, s3, s4)

            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

