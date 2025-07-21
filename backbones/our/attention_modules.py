#!/usr/bin/env python3
"""
Advanced Attention Modules for Multi-Magnification Histology Classification

Implements:
1. Spatial Attention (within each magnification)
2. Channel Attention (feature importance)  
3. Cross-Magnification Attention (between magnifications)
4. Hierarchical Attention (magnification hierarchy)
5. Multi-Scale Fusion Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module - focuses on important regions within each magnification
    """
    def __init__(self, in_channels, reduction=16):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, H, W]
        attention_map = self.conv1(x)
        attention_map = F.relu(attention_map, inplace=True)
        attention_map = self.conv2(attention_map)
        attention_map = self.sigmoid(attention_map)  # [B, 1, H, W]
        
        # Apply spatial attention and global average pooling
        attended_features = x * attention_map
        pooled = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)
        
        return pooled, attention_map


class ChannelAttention(nn.Module):
    """
    Channel Attention Module - learns importance of different feature channels
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, C]
        attention_weights = self.fc(x)  # [B, C]
        return x * attention_weights, attention_weights


class HierarchicalMagnificationAttention(nn.Module):
    """
    Hierarchical Attention for Multi-Magnification Learning
    
    Hierarchy: 40x (global context) -> 100x -> 200x -> 400x (fine details)
    Each magnification can attend to lower magnifications for context
    """
    def __init__(self, feat_dim, num_heads=8):
        super(HierarchicalMagnificationAttention, self).__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        
        # Magnification hierarchy: 40x -> 100x -> 200x -> 400x
        self.mag_hierarchy = ['40', '100', '200', '400']
        
        # Learnable magnification embeddings
        self.mag_embeddings = nn.Parameter(torch.randn(4, feat_dim))
        
        # Hierarchical attention layers
        self.attention_layers = nn.ModuleDict({
            '100': nn.MultiheadAttention(feat_dim, num_heads, batch_first=True),  # 100x attends to 40x
            '200': nn.MultiheadAttention(feat_dim, num_heads, batch_first=True),  # 200x attends to 40x, 100x
            '400': nn.MultiheadAttention(feat_dim, num_heads, batch_first=True),  # 400x attends to 40x, 100x, 200x
        })
        
        # Layer normalization for residual connections
        self.layer_norms = nn.ModuleDict({
            mag: nn.LayerNorm(feat_dim) for mag in ['100', '200', '400']
        })
        
    def forward(self, mag_features):
        """
        Args:
            mag_features: Dict[str, torch.Tensor] - {mag: features[B, feat_dim]}
        
        Returns:
            hierarchical_features: Dict[str, torch.Tensor] - attended features
            attention_maps: Dict[str, torch.Tensor] - attention weights
        """
        
        # Add magnification embeddings
        enhanced_features = {}
        for i, mag in enumerate(self.mag_hierarchy):
            enhanced_features[mag] = mag_features[mag] + self.mag_embeddings[i]
        
        hierarchical_features = {'40': enhanced_features['40']}  # 40x is the root
        attention_maps = {}
        
        # Process hierarchy: each mag attends to all previous mags
        for i, mag in enumerate(self.mag_hierarchy[1:], 1):  # Start from 100x
            # Get context from all previous magnifications
            context_mags = self.mag_hierarchy[:i]  # e.g., for 200x: [40x, 100x]
            
            # Stack context features: [B, num_context_mags, feat_dim]
            context_features = torch.stack([
                hierarchical_features[ctx_mag] for ctx_mag in context_mags
            ], dim=1)
            
            # Current magnification as query: [B, 1, feat_dim]
            query = enhanced_features[mag].unsqueeze(1)
            
            # Hierarchical attention
            attended, attention_weights = self.attention_layers[mag](
                query, context_features, context_features
            )
            
            # Residual connection + layer norm
            hierarchical_features[mag] = self.layer_norms[mag](
                enhanced_features[mag] + attended.squeeze(1)
            )
            
            attention_maps[mag] = attention_weights
        
        return hierarchical_features, attention_maps


class CrossMagnificationFusion(nn.Module):
    """
    Cross-Magnification Fusion with Advanced Attention
    
    Combines all magnifications with learned importance weights
    """
    def __init__(self, feat_dim, num_mags=4):
        super(CrossMagnificationFusion, self).__init__()
        self.feat_dim = feat_dim
        self.num_mags = num_mags
        
        # Cross-attention for final fusion
        self.cross_attention = nn.MultiheadAttention(
            feat_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Magnification importance weights (learnable)
        self.mag_importance = nn.Parameter(torch.ones(num_mags))
        
        # Final fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * num_mags, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, hierarchical_features):
        """
        Args:
            hierarchical_features: Dict[str, torch.Tensor] - {mag: features[B, feat_dim]}
        
        Returns:
            fused_features: torch.Tensor [B, feat_dim]
            fusion_weights: torch.Tensor [B, num_mags, num_mags]
        """
        
        # Stack all magnification features: [B, num_mags, feat_dim]
        mag_order = ['40', '100', '200', '400']
        stacked_features = torch.stack([
            hierarchical_features[mag] for mag in mag_order
        ], dim=1)
        
        # Apply magnification importance weights
        importance_weights = F.softmax(self.mag_importance, dim=0)
        weighted_features = stacked_features * importance_weights.view(1, -1, 1)
        
        # Cross-magnification attention
        attended_features, fusion_weights = self.cross_attention(
            weighted_features, weighted_features, weighted_features
        )
        
        # Final fusion
        fusion_input = attended_features.flatten(1)  # [B, num_mags * feat_dim]
        fused_features = self.fusion(fusion_input)
        
        return fused_features, fusion_weights


class MultiScaleAttentionPool(nn.Module):
    """
    Multi-Scale Attention Pooling
    
    Instead of simple global average pooling, use attention at multiple scales
    """
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super(MultiScaleAttentionPool, self).__init__()
        self.scales = scales
        
        # Attention modules for each scale
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 1, 1),
                nn.Sigmoid()
            ) for scale in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Conv2d(len(scales), 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        
        Returns:
            pooled_features: [B, C]
            attention_map: [B, 1, H, W]
        """
        B, C, H, W = x.shape
        scale_maps = []
        
        # Generate attention at each scale
        for i, (scale, attention_module) in enumerate(zip(self.scales, self.scale_attentions)):
            # Get attention at this scale
            scale_attention = attention_module(x)  # [B, 1, scale, scale]
            
            # Upsample to original resolution
            if scale != H:  # Assuming H == W for simplicity
                scale_attention = F.interpolate(
                    scale_attention, size=(H, W), mode='bilinear', align_corners=False
                )
            
            scale_maps.append(scale_attention)
        
        # Fuse multi-scale attention maps
        stacked_scales = torch.cat(scale_maps, dim=1)  # [B, num_scales, H, W]
        fused_attention = self.sigmoid(self.scale_fusion(stacked_scales))  # [B, 1, H, W]
        
        # Apply attention and pool
        attended_features = x * fused_attention
        pooled_features = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)
        
        return pooled_features, fused_attention


class AttentionVisualization(nn.Module):
    """
    Module for visualizing attention maps for interpretability
    """
    def __init__(self):
        super(AttentionVisualization, self).__init__()
        
    def forward(self, spatial_attention_maps, fusion_weights, mag_importance):
        """
        Prepare attention visualizations for analysis
        
        Returns:
            visualization_data: Dict with attention maps for plotting
        """
        return {
            'spatial_attention': spatial_attention_maps,  # Dict[mag, [B, 1, H, W]]
            'cross_mag_attention': fusion_weights,        # [B, num_mags, num_mags]  
            'magnification_importance': mag_importance,   # [num_mags]
        }