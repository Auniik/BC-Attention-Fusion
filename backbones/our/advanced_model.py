#!/usr/bin/env python3
"""
Advanced Attention-Guided Multi-Magnification Network

Implements state-of-the-art hierarchical attention mechanism for
multi-magnification histology classification with:

1. Spatial Attention within each magnification
2. Channel Attention for feature importance
3. Hierarchical Magnification Attention (40x -> 100x -> 200x -> 400x)
4. Cross-Magnification Fusion with learned importance
5. Multi-Scale Attention Pooling

This is the SOTA version for journal publication.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.our.attention_modules import (
    SpatialAttention,
    ChannelAttention, 
    HierarchicalMagnificationAttention,
    CrossMagnificationFusion,
    MultiScaleAttentionPool,
    AttentionVisualization
)


class AdvancedMultiMagAttentionNet(nn.Module):
    """
    State-of-the-Art Attention-Guided Multi-Magnification Network
    
    Architecture:
    1. EfficientNet-B2 feature extractors for each magnification
    2. Multi-scale spatial attention pooling  
    3. Channel attention for feature refinement
    4. Hierarchical magnification attention (40x->100x->200x->400x)
    5. Cross-magnification fusion with learned importance
    6. Dual classification heads (binary + tumor type)
    """
    
    def __init__(self, magnifications=['40', '100', '200', '400'], 
                 num_classes=2, num_tumor_types=8, 
                 backbone='efficientnet_b2'):
        super(AdvancedMultiMagAttentionNet, self).__init__()
        
        self.magnifications = magnifications
        self.num_mags = len(magnifications)
        
        print(f"🚀 Initializing AdvancedMultiMagAttentionNet with {self.num_mags} magnifications")
        
        # Feature extractors for each magnification
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model(
                backbone, 
                pretrained=True, 
                num_classes=0, 
                global_pool=''  # No global pooling - we'll handle this
            )
            for mag in magnifications
        })
        
        # Infer feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            sample_features = self.extractors['extractor_40x'](dummy_input)
            self.feat_channels = sample_features.shape[1]  # Channel dimension
            self.feat_spatial = sample_features.shape[2]   # Spatial dimension
            
        print(f"📊 Feature dimensions: {self.feat_channels} channels, {self.feat_spatial}x{self.feat_spatial} spatial")
        
        # 1. Multi-Scale Spatial Attention Pooling for each magnification
        self.spatial_attention_pools = nn.ModuleDict({
            f'spatial_pool_{mag}x': MultiScaleAttentionPool(
                self.feat_channels, scales=[1, 2, 4]
            )
            for mag in magnifications
        })
        
        # 2. Channel Attention for each magnification
        self.channel_attentions = nn.ModuleDict({
            f'channel_att_{mag}x': ChannelAttention(
                self.feat_channels, reduction=16
            )
            for mag in magnifications
        })
        
        # 3. Hierarchical Magnification Attention
        self.hierarchical_attention = HierarchicalMagnificationAttention(
            feat_dim=self.feat_channels, num_heads=8
        )
        
        # 4. Cross-Magnification Fusion
        self.cross_mag_fusion = CrossMagnificationFusion(
            feat_dim=self.feat_channels, num_mags=self.num_mags
        )
        
        # 5. Enhanced classification heads with attention
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
        
        self.tumor_classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_tumor_types)
        )
        
        # 6. Attention visualization module
        self.attention_viz = AttentionVisualization()
        
        print(f"✅ Model initialized successfully!")
        
    def forward(self, images_dict, return_attention=False, return_features=False):
        """
        Forward pass with advanced attention mechanisms
        
        Args:
            images_dict: Dict[str, torch.Tensor] - {f'mag_{mag}': tensor[B,3,H,W]}
            return_attention: bool - whether to return attention maps
            return_features: bool - whether to return intermediate features
            
        Returns:
            class_logits: torch.Tensor [B, num_classes]
            tumor_logits: torch.Tensor [B, num_tumor_types]
            attention_data: Dict (if return_attention=True)
            features: torch.Tensor [B, feat_channels] (if return_features=True)
        """
        
        batch_size = list(images_dict.values())[0].shape[0]
        
        # Step 1: Extract feature maps from each magnification
        feature_maps = {}
        for mag in self.magnifications:
            feature_maps[mag] = self.extractors[f'extractor_{mag}x'](
                images_dict[f'mag_{mag}']
            )  # [B, feat_channels, H, W]
        
        # Step 2: Apply multi-scale spatial attention pooling
        spatial_features = {}
        spatial_attention_maps = {}
        
        for mag in self.magnifications:
            spatial_feat, spatial_att = self.spatial_attention_pools[f'spatial_pool_{mag}x'](
                feature_maps[mag]
            )
            spatial_features[mag] = spatial_feat  # [B, feat_channels]
            spatial_attention_maps[mag] = spatial_att  # [B, 1, H, W]
        
        # Step 3: Apply channel attention
        channel_features = {}
        channel_attention_weights = {}
        
        for mag in self.magnifications:
            channel_feat, channel_att = self.channel_attentions[f'channel_att_{mag}x'](
                spatial_features[mag]
            )
            channel_features[mag] = channel_feat  # [B, feat_channels]
            channel_attention_weights[mag] = channel_att  # [B, feat_channels]
        
        # Step 4: Hierarchical magnification attention
        hierarchical_features, hierarchical_attention_maps = self.hierarchical_attention(
            channel_features
        )
        
        # Step 5: Cross-magnification fusion
        fused_features, cross_mag_attention = self.cross_mag_fusion(
            hierarchical_features
        )
        
        # Step 6: Classification
        class_logits = self.classifier(fused_features)
        tumor_logits = self.tumor_classifier(fused_features)
        
        # Prepare return values
        outputs = [class_logits, tumor_logits]
        
        if return_attention:
            attention_data = self.attention_viz(
                spatial_attention_maps, 
                cross_mag_attention,
                self.cross_mag_fusion.mag_importance
            )
            outputs.append(attention_data)
            
        if return_features:
            outputs.append(fused_features)
            
        return tuple(outputs) if len(outputs) > 2 else (outputs[0], outputs[1])
    
    def get_attention_maps(self, images_dict):
        """
        Get all attention maps for visualization and interpretation
        
        Returns:
            Dict with all attention information for analysis
        """
        with torch.no_grad():
            _, _, attention_data = self.forward(images_dict, return_attention=True)
            return attention_data
    
    def get_magnification_importance(self):
        """
        Get learned magnification importance weights
        
        Returns:
            Dict[str, float] - importance weight for each magnification
        """
        importance_weights = F.softmax(self.cross_mag_fusion.mag_importance, dim=0)
        return {
            mag: float(importance_weights[i]) 
            for i, mag in enumerate(self.magnifications)
        }
    
    def print_model_summary(self):
        """Print detailed model architecture summary"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n🏗️  AdvancedMultiMagAttentionNet Architecture Summary")
        print(f"=" * 70)
        print(f"📊 Input: {self.num_mags} magnifications ({', '.join([f'{mag}x' for mag in self.magnifications])})")
        print(f"🧠 Backbone: EfficientNet-B2 per magnification")
        print(f"🎯 Feature dimension: {self.feat_channels}")
        print(f"⚡ Total parameters: {total_params:,}")
        print(f"🔧 Trainable parameters: {trainable_params:,}")
        print(f"")
        print(f"🔍 Attention Mechanisms:")
        print(f"   • Multi-scale spatial attention (1x, 2x, 4x scales)")
        print(f"   • Channel attention (reduction=16)")
        print(f"   • Hierarchical magnification attention (8 heads)")
        print(f"   • Cross-magnification fusion attention")
        print(f"   • Learned magnification importance weights")
        print(f"")
        print(f"📋 Classification Heads:")
        print(f"   • Binary classification: {self.classifier[-1].out_features} classes")
        print(f"   • Tumor type classification: {self.tumor_classifier[-1].out_features} classes")
        print(f"=" * 70)


# Alias for backwards compatibility
LightweightMultiMagNet = AdvancedMultiMagAttentionNet


def test_advanced_model():
    """Test the advanced model with dummy data"""
    print("🧪 Testing AdvancedMultiMagAttentionNet...")
    
    model = AdvancedMultiMagAttentionNet()
    model.print_model_summary()
    
    # Create dummy input
    dummy_images = {
        'mag_40': torch.randn(2, 3, 224, 224),
        'mag_100': torch.randn(2, 3, 224, 224),
        'mag_200': torch.randn(2, 3, 224, 224),
        'mag_400': torch.randn(2, 3, 224, 224),
    }
    
    # Test forward pass
    class_logits, tumor_logits = model(dummy_images)
    print(f"✅ Forward pass successful!")
    print(f"   Class logits shape: {class_logits.shape}")
    print(f"   Tumor logits shape: {tumor_logits.shape}")
    
    # Test attention maps
    attention_data = model.get_attention_maps(dummy_images)
    print(f"✅ Attention maps generated!")
    
    # Test magnification importance
    mag_importance = model.get_magnification_importance()
    print(f"✅ Magnification importance: {mag_importance}")
    
    print(f"🎉 All tests passed!")


if __name__ == "__main__":
    test_advanced_model()