from backbones.our.backbones import MultiScaleFeatureExtractor
from backbones.our.fusion import CrossMagnificationFusion
import torch.nn as nn

class LightweightMultiMagNet(nn.Module):    
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, num_tumor_types=8):
        super().__init__()
        
        self.magnifications = magnifications
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': MultiScaleFeatureExtractor(base_channels=32)
            for mag in magnifications
        })
        
        # Cross-magnification fusion
        channels_list = [64, 128, 256]  # channels at each scale
        self.fusion = CrossMagnificationFusion(channels_list, output_channels=256)
        
        # Classification heads
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.tumor_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_tumor_types)
        )
    
    def forward(self, images_dict, return_features=False):
        # Extract features from each magnification
        features_list = [
            self.extractors[f'extractor_{mag}x'](images_dict[f'mag_{mag}'])
            for mag in self.magnifications
        ]
        
        # Fuse features
        fused_features = self.fusion(*features_list)
        
        # Classification
        class_logits = self.classifier(fused_features)
        tumor_logits = self.tumor_classifier(fused_features)
        
        if return_features:
            return class_logits, tumor_logits, fused_features
        
        return class_logits, tumor_logits