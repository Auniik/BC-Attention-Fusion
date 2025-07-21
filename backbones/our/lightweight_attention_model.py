#!/usr/bin/env python3
"""
Lightweight Attention-Guided Multi-Magnification Network

A balanced approach that maintains attention mechanisms while being
appropriately sized for the dataset (840 training samples).

Uses EfficientNet-B0 backbone to reduce parameters from 79M to ~25M.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.our.attention_modules import (
    ChannelAttention,
    CrossMagnificationFusion,
    AttentionVisualization
)


class LightweightAttentionNet(nn.Module):
    """
    Lightweight Attention Network - Balanced complexity for dataset size
    
    Key changes from AdvancedMultiMagAttentionNet:
    - EfficientNet-B0 backbone (vs B2) - 25M vs 79M parameters
    - Simplified spatial attention (single-scale vs multi-scale)
    - Streamlined hierarchical attention
    - Better suited for 840 training samples
    """
    
    def __init__(self, magnifications=['40', '100', '200', '400'], 
                 num_classes=2, num_tumor_types=8, 
                 backbone='efficientnet_b0'):
        super(LightweightAttentionNet, self).__init__()
        
        self.magnifications = magnifications
        self.num_mags = len(magnifications)
        
        print(f"🚀 Initializing LightweightAttentionNet with {self.num_mags} magnifications")
        
        # Feature extractors for each magnification (EfficientNet-B0)
        self.extractors = nn.ModuleDict({
            f'extractor_{mag}x': timm.create_model(
                backbone, 
                pretrained=True, 
                num_classes=0, 
                global_pool='avg'  # Use global average pooling
            )
            for mag in magnifications
        })
        
        # Infer feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            sample_features = self.extractors['extractor_40x'](dummy_input)
            self.feat_channels = sample_features.shape[1]  # Should be 1280 for EfficientNet-B0
            
        print(f"📊 Feature dimensions: {self.feat_channels} channels")
        
        # Lightweight channel attention for each magnification
        self.channel_attentions = nn.ModuleDict({
            f'channel_att_{mag}x': ChannelAttention(
                self.feat_channels, reduction=8  # Less reduction for smaller model
            )
            for mag in magnifications
        })
        
        # Simplified cross-magnification fusion
        self.cross_mag_fusion = CrossMagnificationFusion(
            feat_dim=self.feat_channels, num_mags=self.num_mags
        )
        
        # Simpler classification heads
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.tumor_classifier = nn.Sequential(
            nn.Linear(self.feat_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_tumor_types)
        )
        
        # Attention visualization
        self.attention_viz = AttentionVisualization()
        
        print(f"✅ Lightweight model initialized successfully!")
        
    def forward(self, images_dict, return_attention=False, return_features=False):
        """
        Forward pass with lightweight attention
        """
        batch_size = list(images_dict.values())[0].shape[0]
        
        # Step 1: Extract features from each magnification
        raw_features = {}
        for mag in self.magnifications:
            raw_features[mag] = self.extractors[f'extractor_{mag}x'](
                images_dict[f'mag_{mag}']
            )  # [B, feat_channels] after global pooling
        
        # Step 2: Apply channel attention
        channel_features = {}
        channel_attention_weights = {}
        
        for mag in self.magnifications:
            channel_feat, channel_att = self.channel_attentions[f'channel_att_{mag}x'](
                raw_features[mag]
            )
            channel_features[mag] = channel_feat  # [B, feat_channels]
            channel_attention_weights[mag] = channel_att  # [B, feat_channels]
        
        # Step 3: Cross-magnification fusion
        fused_features, cross_mag_attention = self.cross_mag_fusion(
            channel_features
        )
        
        # Step 4: Classification
        class_logits = self.classifier(fused_features)
        tumor_logits = self.tumor_classifier(fused_features)
        
        # Prepare return values
        outputs = [class_logits, tumor_logits]
        
        if return_attention:
            # Create dummy spatial attention for compatibility
            spatial_attention_maps = {
                mag: torch.ones(batch_size, 1, 7, 7) * 0.5  # Uniform attention
                for mag in self.magnifications
            }
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
        """Get attention maps for visualization"""
        with torch.no_grad():
            _, _, attention_data = self.forward(images_dict, return_attention=True)
            return attention_data
    
    def get_magnification_importance(self):
        """Get learned magnification importance weights"""
        importance_weights = F.softmax(self.cross_mag_fusion.mag_importance, dim=0)
        return {
            mag: float(importance_weights[i]) 
            for i, mag in enumerate(self.magnifications)
        }
    
    def print_model_summary(self):
        """Print detailed model architecture summary"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n🏗️  LightweightAttentionNet Architecture Summary")
        print(f"=" * 70)
        print(f"📊 Input: {self.num_mags} magnifications ({', '.join([f'{mag}x' for mag in self.magnifications])})")
        print(f"🧠 Backbone: EfficientNet-B0 per magnification")
        print(f"🎯 Feature dimension: {self.feat_channels}")
        print(f"⚡ Total parameters: {total_params:,}")
        print(f"🔧 Trainable parameters: {trainable_params:,}")
        print(f"📊 Params per training sample: {total_params/840:.1f}")  # 840 training samples
        print(f"")
        print(f"🔍 Attention Mechanisms:")
        print(f"   • Channel attention (reduction=8)")
        print(f"   • Cross-magnification fusion attention")
        print(f"   • Learned magnification importance weights")
        print(f"")
        print(f"📋 Classification Heads:")
        print(f"   • Binary classification: {self.classifier[-1].out_features} classes")
        print(f"   • Tumor type classification: {self.tumor_classifier[-1].out_features} classes")
        print(f"=" * 70)


def test_lightweight_model():
    """Test the lightweight model"""
    print("🧪 Testing LightweightAttentionNet...")
    
    model = LightweightAttentionNet()
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
    test_lightweight_model()