from backbones.baseline_concat import BaselineConcatNet
from backbones.our.model import LightweightMultiMagNet


default_backbone_list = [
    # 'mobilenetv3_small_100', 
    # 'efficientnet_b0',
    # 'resnet18',
    # 'resnet50',
    # 'densenet121',
    # 'convnext_tiny'
]

default_mags = ['40', '100', '200', '400']


def get_all_backbones(mags=default_mags, backbones=default_backbone_list):

    models = {
        name: BaselineConcatNet(backbone_name=name, 
                                magnifications=mags, 
                                num_classes=2)
        for name in backbones
    }

    models['our_model'] = LightweightMultiMagNet(
        magnifications=mags,
        num_classes=2,
        num_tumor_types=8
    )

    return models

