
import torch.nn as nn

from backbones.our.attention_blocks import AttentionGuidedFeatureSelection


class MultiScaleFeatureExtractor(nn.Module):    
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        # Lightweight backbone using depthwise separable convolutions
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Multi-scale blocks
        self.scale1 = self._make_scale_block(base_channels, base_channels * 2, stride=2)
        self.scale2 = self._make_scale_block(base_channels * 2, base_channels * 4, stride=2)
        self.scale3 = self._make_scale_block(base_channels * 4, base_channels * 8, stride=2)
        
        # Attention modules for each scale
        self.attention1 = AttentionGuidedFeatureSelection(base_channels * 2)
        self.attention2 = AttentionGuidedFeatureSelection(base_channels * 4)
        self.attention3 = AttentionGuidedFeatureSelection(base_channels * 8)
        
    def _make_scale_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x):
        # Extract multi-scale features
        x = self.stem(x)
        
        scale1 = self.scale1(x)
        scale1 = self.attention1(scale1)
        
        scale2 = self.scale2(scale1)
        scale2 = self.attention2(scale2)
        
        scale3 = self.scale3(scale2)
        scale3 = self.attention3(scale3)
        
        return scale1, scale2, scale3