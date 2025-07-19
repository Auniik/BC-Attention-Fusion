import torch
import torch.nn as nn
import timm

class BaselineConcatNet(nn.Module):
    def __init__(self, 
                 magnifications=['40', '100', '200', '400'],
                 backbone_name='mobilenetv3_small_100',
                 num_classes=2):
        super().__init__()
        self.magnifications = magnifications
        self.backbone_name = backbone_name

        # Initialize one CNN extractor per magnification
        self.extractors = nn.ModuleDict({
            mag: timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')
            for mag in magnifications
        })

        # Infer the feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            sample_feat = self.extractors[magnifications[0]](dummy_input)
            feat_dim = sample_feat.shape[1]

        # Total input to classifier after concatenation
        concat_dim = feat_dim * len(magnifications)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images_dict):
        features = []
        for mag in self.magnifications:
            img = images_dict[f'mag_{mag}']
            feat = self.extractors[mag](img)  # shape: [B, feat_dim]
            features.append(feat)

        # Concatenate features across all magnifications
        concat_feat = torch.cat(features, dim=1)  # shape: [B, feat_dim * N]
        return self.classifier(concat_feat)
    



# # Baseline models
# class SingleMagBaseline(nn.Module):
#     def __init__(self, backbone_name='efficientnet_b0', num_classes=2):
#         super().__init__()
#         self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=num_classes)
    
#     def forward(self, x):
#         return self.backbone(x)

# class SimpleConcatBaseline(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0)
#         with torch.no_grad():
#             dummy_input = torch.randn(1, 3, 224, 224)
#             feature_dim = self.backbone(dummy_input).shape[1]
#         self.classifier = nn.Linear(feature_dim * 4, num_classes)  # 4 magnifications

#     def forward(self, images_dict):
#         feats = [self.backbone(images_dict[f'mag_{m}']) for m in ['40', '100', '200', '400']]
#         concat_feat = torch.cat(feats, dim=1)
#         return self.classifier(concat_feat)