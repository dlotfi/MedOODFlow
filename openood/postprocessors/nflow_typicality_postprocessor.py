from typing import Any

import torch

from .base_postprocessor import BasePostprocessor


class NormalizingFlowTypicalityPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NormalizingFlowTypicalityPostprocessor, self).__init__(config)

    def postprocess(self, net, data: Any):
        # images input
        if data.shape[-1] > 1 and data.shape[1] in {1, 3}:
            with torch.no_grad():
                data = torch.Tensor(data)
                output, feats_orig = net['backbone'](data, return_feature=True)
                score = torch.softmax(output, dim=1)
                _, pred = torch.max(score, dim=1)
                if 'feat_agg' in net:
                    feats_orig = net['feat_agg'](feats_orig)
            # Enable gradient computation for this part
            feats = feats_orig.clone().detach().requires_grad_(True)
            log_prob = net['nflow'].log_prob(feats)
            grad = torch.autograd.grad(log_prob.sum(), feats)[0]
            grad_norm = grad.view(data.size(0), -1).norm(dim=1)
            conf = -grad_norm.detach().cpu()
        # feature input
        elif data.shape[-1] == 1 and data.shape[-2] == 1:
            with torch.no_grad():
                feats_orig = data.flatten(1)
                if 'feat_agg' in net:
                    feats_orig = net['feat_agg'](feats_orig)
            # Enable gradient computation for this part
            feats = feats_orig.clone().detach().requires_grad_(True)
            log_prob = net['nflow'].log_prob(feats)
            grad = torch.autograd.grad(log_prob.sum(), feats)[0]
            grad_norm = grad.norm(dim=1)
            conf = -grad_norm.detach().cpu()
            pred = torch.ones_like(conf)  # dummy predictions
        else:
            raise ValueError('Unsupported input type!')

        return pred, conf
