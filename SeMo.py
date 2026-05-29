# @Time 2025/1/26
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import init

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print(m.__class__.__name__)
            if init_type == 'gaussian':
                init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)

    return init_fun

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.initialize()

    def initialize(self):
        self.qkv.reset_parameters()

    def forward(self, x, src_mask, M=None):
        B, N, C = x.shape
        # print("N", N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)    # B, heads, N, d_model/heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, heads, seq_len, seq_len]

        # causal mask
        mask = torch.ones_like(attn)
        mask = torch.tril(mask)     # [B, heads, seq_len=2N, seq_len]
        if src_mask:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        # self attention
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.mlp_ = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=0.)
        self.Attn = Attention(dim,
                              num_heads=2,
                              qkv_bias=False,
                              qk_scale=None,
                              attn_drop=0.,
                              proj_drop=0.)

    def forward(self, xs):
        fuse_fea = torch.stack(xs, dim=1)
        fuse_fea = fuse_fea + self.Attn(self.norm1(fuse_fea), src_mask=True)
        fuse_fea = fuse_fea + self.mlp_(self.norm2(fuse_fea))

        return fuse_fea

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_layers, nclass, dropout):
        super(Classifier, self).__init__()
        layers = []
        last_size = input_size

        for hidden in hidden_layers:
            layers.append(nn.Linear(last_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_size = hidden

        layers.append(nn.Linear(last_size, nclass))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def msp(input, thr=0.95)->bool:
    # MSP the higher the best
    logits = F.softmax(input, dim=1)
    value, _ = torch.max(logits, dim=1)
    return value.item() >= thr

class SMiLE(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num):
        super(SMiLE, self).__init__()
        # self.clfs = []
        self.encoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            # self.clfs.appened(Classifier(feature_dim, [], class_num, dropout=0.))
        self.encoders = nn.ModuleList(self.encoders)
        self.view = view
        # self.apply(weights_init('kaiming'))
        self.attention = AttentionBlock(dim=feature_dim)
        self.clf = nn.Linear(feature_dim, class_num)

    def forward(self, xs):
        zs, outs = [], []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            # output = self.clfs[v](z)
            output = self.clf(z)
            outs.append(output)
        fuse_fea = self.attention(zs)
        fea = torch.flatten(fuse_fea.mean(1), start_dim=1)
        out = self.clf(fea)

        return out, outs

    def forward_test(self, xs, conf_thr):
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)

        use_modal = 0
        while use_modal < len(zs): # 确保不会超出模态数量
            use_modal += 1
            # Attention 融合与分类
            fuse_fea = self.attention(zs[:use_modal])
            fea = torch.flatten(fuse_fea.mean(1), start_dim=1)
            out = self.clf(fea)
            # 判断是否满足阈值条件
            if msp(out, thr=conf_thr):
                return use_modal, out
        return use_modal, out




