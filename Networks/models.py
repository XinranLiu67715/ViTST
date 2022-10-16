
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_






class VisionTransformer_gap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)
        '''
        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128), #(in_dim = 14400, n_hidden_1)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) #(n_hidden_1, out_dim)
        )
        '''

        self.features1 = torch.nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(6912 * 4, 128), #(in_dim = 14400, n_hidden_1)
                            nn.ReLU(),
                            )
        self.features2 = torch.nn.Sequential(
                            
                            nn.Dropout(0.5),
                            nn.Linear(128, 1) #(n_hidden_1, out_dim)
                            )
        print('self._init_weights',self._init_weights)
        self.features2.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = F.adaptive_avg_pool1d(x, (48))
        x = x.view(x.shape[0], -1)
        x1 = self.features1(x)
        x2 = self.features2(x1)
        return (x1,x2)






@register_model
def base_vit_384(pretrained=True, **kwargs):
    model = VisionTransformer_gap(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained-----------------")
    return model




