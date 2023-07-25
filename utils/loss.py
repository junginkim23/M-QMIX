import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()

    @staticmethod
    def calculate_loss(pred, true):
        pred = F.normalize(pred, dim=-1)
        true = F.normalize(true, dim=-1)

        return 2 - 2 * (pred * true).sum(dim=-1)
