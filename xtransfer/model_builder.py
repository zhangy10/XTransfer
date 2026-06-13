"""Minimal source-model builder.

Extracted from pretrainer.engine.trainer.BaseNet so the method package no
longer drags the (pretraining-only) dataset/optim/tensorboard machinery.
Behaviour is identical to the original BaseNet for the backbones used here
(only `.backbone` is consumed downstream).
"""
import torch.nn as nn

from modeling import build_backbone


class BaseNet(nn.Module):
    """A CNN backbone with an optional classification head."""

    def __init__(self, model_name, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(model_name, **kwargs)
        fdim = self.backbone.out_features
        self.model_name = model_name
        self.head = None

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y
