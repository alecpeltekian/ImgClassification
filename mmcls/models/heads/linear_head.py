import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from ..utils import is_tracing
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses

    
    
@HEADS.register_module()
class SiameseLinearHead(ClsHead):
    """Distance Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 distance='euclid',
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(SiameseLinearHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.distance = distance

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def simple_test(self, img1, img2):
        """Test without augmentation."""
        x1 = self.sig(img1)
        x2 = self.sig(img2)
        if self.distance == 'euclid':
            out = torch.cdist(x1, x2, p=2)
        elif self.distance == 'abs':
            out = torch.abs(x1 - x2)
            
        pred = self.fc(out)

        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x1, x2, gt_label):
        x1 = self.sig(x1)
        x2 = self.sig(x2)
        if self.distance == 'euclid':
            out = torch.cdist(x1, x2, p=2)
        elif self.distance == 'abs':
            out = torch.abs(x1 - x2)
            
        cls_score = self.fc(out)
        losses = self.loss(cls_score, gt_label.view_as(cls_score).type_as(cls_score))
        return losses
