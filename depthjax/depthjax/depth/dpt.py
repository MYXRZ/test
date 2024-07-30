import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
from depthjax.depth.blocks import FeatureFusionBlock, scratch ,Identity
from depthjax.vit import DinoViT


class DPTHead(nn.Module):
    #out_channels: list
    nclass: int = 1
    in_channels: int = 384
    features: int = 64
    use_bn: bool = False
    use_clstoken: bool = False

    def setup(self):
        self.out_channels = [48, 96, 192, 384]
        self.projects = [nn.Conv(features=out_channel, kernel_size=(1, 1), strides=(1, 1), padding='VALID', name=f'projects.{i}') for i, out_channel in enumerate(self.out_channels)]
        self.resize_layers = [
            nn.ConvTranspose(features=self.out_channels[0], kernel_size=(4, 4), strides=(4, 4), padding='VALID', transpose_kernel=True, name='resize_layers.0'),
            nn.ConvTranspose(features=self.out_channels[1], kernel_size=(2, 2), strides=(2, 2), padding='VALID', transpose_kernel=True, name='resize_layers.1'),
            Identity(name='resize_layers.2'),
            nn.Conv(features=self.out_channels[3], kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='resize_layers.3')
        ]

        if self.use_clstoken:
            self.readout_projects = [nn.Sequential([
                nn.Dense(features=self.in_channels),
                nn.gelu
            ]) for _ in range(len(self.projects))]

        self.scratch = scratch(groups=1, expand=False)
        
    def __call__(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = jnp.expand_dims(cls_token, 1).repeat(x.shape[1], axis=1)
                x = jnp.concatenate((x, readout), -1)
                x = self.readout_projects[i](x)
            else:
                x = x[0]
            
            x = x.reshape((x.shape[0], patch_h, patch_w, x.shape[-1]))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
    
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[1:3])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[1:3])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[1:3])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = jax.image.resize(out, (out.shape[0], int(patch_h * 14), int(patch_w * 14), out.shape[-1]), 'bilinear',antialias=False)
        out = self.scratch.output_conv2(out)
        return out
    

class DPT_DINOv2_jax(nn.Module):
    #out_channels: list
    encoder: str = 'vits'
    features: int = 64
    use_bn: bool = False
    use_clstoken: bool = False
    localhub: bool = True

    def setup(self):
        assert self.encoder in ['vits', 'vitb', 'vitl']
        
        self.pretrained = DinoViT(num_heads=6, embed_dim=384, mlp_ratio=4, depth=12, img_size=518)
        dim = 384
        self.depth_head = DPTHead(1, dim, self.features, self.use_bn, use_clstoken=self.use_clstoken)

    def __call__(self, x):
        b, h, w, c= x.shape
        features = self.pretrained(x, encoder=False)
        patch_h, patch_w = h // 14, w // 14
        depth = self.depth_head(features, patch_h, patch_w)
        depth = jax.image.resize(depth, (depth.shape[0], h, w, depth.shape[-1]), 'bilinear', antialias=False)
        depth = jnp.maximum(depth, 0.0)

        return depth.squeeze(-1)
    
def Depth_model_weights_loader(params, path):
    model = DPT_DINOv2_jax()
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

    find_and_replace(params, "DPT_DINOv2_jax_0", model_variables)
    assert replaced, "Failed to load weights"
    return params

    