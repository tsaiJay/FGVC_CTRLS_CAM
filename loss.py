# import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(object):
    # https://github.com/haitongli/knowledge-distillation-pytorch/blob/9937528f0be0efa979c745174fbcbe9621cea8b7/model/net.py#L105
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    def __init__(self, T):
        self.T = T

    def __call__(self, s_out, t_out):
        kd_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(s_out / self.T, dim=1),
            F.softmax(t_out / self.T, dim=1)
        ) * (self.T ** 2)

        return kd_loss
