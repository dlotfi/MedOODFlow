import torch
import torch.nn as nn


class IdentityNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(IdentityNetwork, self).__init__()
        self.num_classes = num_classes
        self.feature_size = None  # Will be set dynamically based on input

    def forward(self, x, return_feature=False):
        # Flatten the input to create the feature vector
        feature = x.view(x.size(0), -1)

        # Set feature_size if not already set
        if self.feature_size is None:
            self.feature_size = feature.size(1)

        # Create zero logits tensor with appropriate shape
        logits_cls = torch.zeros(x.size(0), self.num_classes, device=x.device)

        # Handle different return options
        if return_feature:
            return logits_cls, feature
        else:
            return logits_cls
