import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss,self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        batch_size = inputs.size(0)

        inputs = inputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = (inputs * targets).sum(dim=1)

        dice_coefficient = (2.*intersection + self.smooth)/ (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1 - dice_coefficient.mean()
