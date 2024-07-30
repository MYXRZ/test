import jax
from flax import linen as nn
from typing import Any

class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class scratch(nn.Module):
    groups: int = 1
    expand: bool = False

    def setup(self):
        self.layer1_rn = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, feature_group_count=self.groups)
        self.layer2_rn = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, feature_group_count=self.groups)
        self.layer3_rn = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, feature_group_count=self.groups)
        self.layer4_rn = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False, feature_group_count=self.groups)
            
        self.stem_transpose=None
        self.refinenet1 = FeatureFusionBlock(64, nn.relu)
        self.refinenet2 = FeatureFusionBlock(64, nn.relu)
        self.refinenet3 = FeatureFusionBlock(64, nn.relu)
        self.refinenet4 = FeatureFusionBlock(64, nn.relu)
        self.output_conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
        self.output_conv2 = nn.Sequential([
            nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='output_conv2.0'),
            nn.relu,
            nn.Conv(features=1, kernel_size=(1, 1), strides=(1, 1), padding='VALID', name='output_conv2.2'),
            nn.relu,
            Identity(),
        ])
        

class ResidualConvUnit(nn.Module):
    features: int
    activation: Any
    bn: bool

    @nn.compact
    def __call__(self, x):
        out = self.activation(x)
        conv1 = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True, feature_group_count=1, name='conv1')
        out = conv1(out)
        
        out = self.activation(out)
        conv2 = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=True, feature_group_count=1, name='conv2')
        out = conv2(out)
        
        #_activation_post_process = activation_post_process()
        out = out + x
        
        return out
    
class FeatureFusionBlock(nn.Module):
    features: int
    activation: Any
    deconv: bool = False
    bn: bool = False
    expand: bool = False
    align_corners: bool = True
    size: Any = None

    def setup(self):
        self.groups = 1
        out_features = self.features
        if self.expand:
            out_features = self.features // 2
        
        self.out_conv = nn.Conv(features=out_features, kernel_size=(1, 1), strides=(1, 1), padding='VALID', use_bias=True, feature_group_count=1)
        
        self.resConfUnit1 = ResidualConvUnit(features=self.features, activation=self.activation, bn=self.bn)
        self.resConfUnit2 = ResidualConvUnit(features=self.features, activation=self.activation, bn=self.bn)

    def __call__(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res
            
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {"size": (296, 296)}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = jax.image.resize(output,(output.shape[0], modifier['size'][0], modifier['size'][1], output.shape[-1]), 'bilinear', antialias=False)
        output = self.out_conv(output)
        return output

