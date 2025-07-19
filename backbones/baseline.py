import torch
import torch.nn as nn
import timm

class BaselineSingleMag(nn.Module):
    def __init__(self, 
                 backbone_name='mobilenetv3_small_100',
                 num_classes=2):
        super().__init__()
        self.backbone_name = backbone_name

        # Use pretrained backbone without the classifier
        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,         # removes classification head
            global_pool='avg'      # use global average pooling
        )

        # Infer the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feat_dim = self.feature_extractor(dummy_input).shape[1]

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)  # shape: [B, feat_dim]
        return self.classifier(features)