import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as onp

from typing import Type
from functools import partial

from depthjax.mlp import Mlp
from depthjax.attention import Attention
from depthjax.block import Block
from depthjax.patch_embed import PatchEmbed
from flax import serialization

class DinoViT(nn.Module):
    img_size: int = 518
    in_channels: int = 3

    patch_size: int = 14
    embed_dim: int = 384

    depth: int = 12

    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0

    BlockClass: Type[nn.Module] = Block
    AttentionClass: Type[nn.Module] = Attention
    FfnClass: Type[nn.Module] = Mlp
    EmbedLayer: Type[nn.Module] = PatchEmbed
    
    def setup(self):
        self.patch_embed = self.EmbedLayer(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            name="patch_embed",
        )
        self.cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.embed_dim)
        )
        num_patches = 1369
        num_tokens = 1
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, num_patches + num_tokens, self.embed_dim),
        )
        self.blocks = [
            self.BlockClass(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                drop_path_rate=self.drop_path_rate,
                AttentionClass=self.AttentionClass,
                FfnClass=self.FfnClass,
                name=f"blocks.{i}"
            )
            for i in range(self.depth)
        ]
        self.norm = nn.LayerNorm(name="norm")
    
    def _interpolate_pos_encoding(
        self, x: jnp.ndarray, w: int, h: int
    ):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        #w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = jax.image.resize(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim),
            (1, w0, h0, dim),
            method="bicubic",
        )
        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, -1, dim))

        return jnp.concatenate((class_pos_embed[None], patch_pos_embed), axis=1).astype(
            previous_dtype
        )
        
    def __call__(self, x, n=4, encoder: bool = True, training: bool = False):
        B, H, W, C = x.shape

        x = self.patch_embed(x)
        cls_token = jnp.broadcast_to(self.cls_token, (x.shape[0], *self.cls_token.shape[1:]))
        x = jnp.concatenate((cls_token, x), axis=1)
        
        x = x + self._interpolate_pos_encoding(
            x, H, W
        )
        if encoder:
            for i in range(self.depth):
                x = self.blocks[i](x, training=training)
        
            output = self.norm(x)
            output = output[:, 1:]
            return output
        else:
            blocks_to_take = range(self.depth - n, self.depth)
            outputs = []
            for i in range(self.depth):
                x = self.blocks[i](x, training=training)
                if i in blocks_to_take:
                    outputs.append(x)
            assert len(outputs) == len(blocks_to_take), f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
            outputs = [self.norm(out) for out in outputs]
            class_tokens = [out[:, 0] for out in outputs]
            outputs = [out[:, 1 :] for out in outputs]
            return tuple(zip(outputs, class_tokens))
            
def depth_encoder_loader(params, path):
    model = DinoViT()
    key = jax.random.PRNGKey(42)
    model_variables = model.init(key,jnp.ones((1, 518, 518, 3)))['params']
    with open(path, 'rb') as f:
        params_bytes = f.read()
        
    model_variables = serialization.from_bytes(model_variables,params_bytes)
    
    replaced = False
    
    def find_and_replace(params, key, replacement):
        nonlocal replaced
        for k in params.keys():
            if k == key:
                params[k] = replacement
                print(f"Replaced {key} in params")
                replaced = True
                return
            if isinstance(params[k], type(params)):
                find_and_replace(params[k], key, replacement)

    find_and_replace(params, "DinoViT_0", model_variables)
    assert replaced, "Failed to load weights"
    return params