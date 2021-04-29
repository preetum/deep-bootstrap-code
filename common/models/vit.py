import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)



def reinit_head(vit_model, targ_nclasses):
    '''
        Replaces (and zero-inits) the last linear layer.
    '''
    dim = vit_model.cls_token.size(-1)
    mlp_dim = vit_model.mlp_head[1].out_features
    clf = nn.Linear(mlp_dim, targ_nclasses, bias=False)
    torch.nn.init.zeros_(clf.weight)
    ## strip entire head off
    # vit_model.mlp_head = nn.Sequential(
    #     nn.LayerNorm(dim),
    #     clf
    # )
    vit_model.mlp_head[-1] = clf # just re-init the final linear layer
    return vit_model

def vit4(patch_size=4,
        num_classes=10,
        image_size=32,
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
        dropout=0,
        emb_dropout=0,
        **kwargs):
    return ViT(**locals())

def vit8(patch_size=8,
        num_classes=10,
        image_size=32,
        dim = 768,
        depth = 12,
        heads = 12,
        mlp_dim = 3072,
        dropout=0,
        emb_dropout=0,
        **kwargs):
    return ViT(**locals())

# def vitB(patch_size=4, num_classes=10):
#     return ViT(
#         image_size = 32,
#         patch_size = patch_size,
#         num_classes = num_classes,
#         dim = 768,
#         depth = 12,
#         heads = 12,
#         mlp_dim = 3072
#     )

# def vitB_pre(pretrained_path : str = None, patch_size=4, orig_nclasses=1000, num_classes=10, **kwargs):
#     if pretrained_path is None:
#         return vitB(patch_size = patch_size, num_classes=num_classes, **kwargs)
#     else:
#         from common import load_state_dict
#         model = vitB(patch_size = patch_size, num_classes=orig_nclasses, **kwargs)
#         load_state_dict(model, pretrained_path)
#         model = reinit_head(model, num_classes)
#         return model


# def vit4(pretrained_path : str = None, num_classes=10, **kwargs):
#     return vitB_pre(pretrained_path=pretrained_path, patch_size=4, num_classes=num_classes, **kwargs)

# def vit8(pretrained_path : str = None, num_classes=10, **kwargs):
#     return vitB_pre(pretrained_path=pretrained_path, patch_size=8, num_classes=num_classes, **kwargs)
