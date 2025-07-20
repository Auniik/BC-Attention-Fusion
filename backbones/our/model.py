import timm
from backbones.our.fusion import CrossMagnificationFusion
import torch.nn as nn
import torch

class LightweightMultiMagNet(nn.Module):
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, num_tumor_types=8):
        super().__init__()
        
        self.magnifications = magnifications
        
        # Upgrade to EfficientNet-B2 for better feature extraction
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model('efficientnet_b2', pretrained=True, num_classes=0, global_pool='')
            for mag in magnifications
        })

        # Infer the feature dimension from feature maps (not pooled)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            sample_feat = self.extractors['extractor_40x'](dummy_input)
            feat_channels = sample_feat.shape[1]  # Channel dimension

        # Advanced cross-magnification fusion with attention
        # Use adaptive global pooling to convert feature maps to vectors
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Cross-attention fusion for feature vectors
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feat_channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(feat_channels * len(magnifications), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Enhanced classification heads with better regularization
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.tumor_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_tumor_types)
        )
    
    def forward(self, images_dict, return_features=False):
        # Extract feature maps from each magnification (no global pooling)
        feature_maps = [
            self.extractors[f'extractor_{mag}x'](images_dict[f'mag_{mag}'])
            for mag in self.magnifications
        ]
        
        # Apply adaptive pooling to get feature vectors
        feature_vectors = [
            self.adaptive_pool(fm).flatten(1)  # [B, C]
            for fm in feature_maps
        ]
        
        # Stack for cross-attention: [B, num_mags, feat_dim]
        stacked_features = torch.stack(feature_vectors, dim=1)
        
        # Apply cross-magnification attention
        attended_features, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Flatten for final fusion
        fusion_input = attended_features.flatten(1)  # [B, num_mags * feat_dim]
        fused_features = self.fusion(fusion_input)
        
        # Classification
        class_logits = self.classifier(fused_features)
        tumor_logits = self.tumor_classifier(fused_features)
        
        if return_features:
            return class_logits, tumor_logits, fused_features
        
        return class_logits, tumor_logits