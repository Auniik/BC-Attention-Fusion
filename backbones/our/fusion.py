import torch
import torch.nn as nn
import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class CrossMagnificationFusion(nn.Module):    
    def __init__(self, channels_list, output_channels=256):
        super().__init__()
        
        # Adaptive pooling to handle different spatial dimensions
        # self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_pool = GeM()
        
        # Feature alignment
        self.align_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True)
            ) for channels in channels_list
        ])
        
        # Cross-magnification attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_channels * 4, output_channels),  # 4 magnifications
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
    def forward(self, *features_list):
        fused_feats = []
        for features in features_list:
            pooled = [self.adaptive_pool(f).flatten(1) for f in features]
            aligned = [align(p) for p, align in zip(pooled, self.align_blocks)]
            agg = torch.stack(aligned, dim=1)
            fused_feats.append(torch.mean(agg, dim=1))
        
        stacked_mags = torch.stack(fused_feats, dim=1)  # [B, 4, 256]
        attended, _ = self.cross_attention(stacked_mags, stacked_mags, stacked_mags)
        
        # Then fuse the attended features
        fusion_input = attended.flatten(1)  # [B, 4*256]
        output = self.fusion(fusion_input)
        return output