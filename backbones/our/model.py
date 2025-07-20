import timm
from backbones.our.fusion import CrossMagnificationFusion
import torch.nn as nn
import torch

class LightweightMultiMagNet(nn.Module):
    def __init__(self, magnifications=['40', '100', '200', '400'], num_classes=2, num_tumor_types=8):
        super().__init__()
        
        self.magnifications = magnifications
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')
            for mag in magnifications
        })

        # Infer the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            sample_feat = self.extractors['extractor_40x'](dummy_input)
            feat_dim = sample_feat.shape[1]

        # Cross-magnification fusion
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * len(magnifications), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Classification heads
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.tumor_classifier = nn.Sequential(
            nn.Linear(512, 128),
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
        
        # Concatenate features
        fused_features = self.fusion(torch.cat(features_list, dim=1))
        
        # Classification
        class_logits = self.classifier(fused_features)
        tumor_logits = self.tumor_classifier(fused_features)
        
        if return_features:
            return class_logits, tumor_logits, fused_features
        
        return class_logits, tumor_logits