import torch
import torch.nn as nn

from dice_loss import DiceLoss

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(DiceBCELoss,self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):

        dice = self.dice(inputs,targets)

        bce = self.bce(inputs,targets)

        return bce * self.bce_weight + dice * self.dice_weight