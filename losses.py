import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class PixelWiseCrossEntropy(_Loss):
    def __init__(self):
        super(PixelWiseCrossEntropy, self).__init__()

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure that the input has the same number of channels as the target
        if inp.size(1) != target.size(1):
            raise ValueError("Input and target must have the same number of channels.")

        criterion = torch.nn.CrossEntropyLoss()
        # target = target.type(torch.LongTensor)
        loss = criterion(inp, target)


        return loss

class DiceLoss(_Loss):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure that the input has the same number of channels as the target
        if inp.size(1) != target.size(1):
            raise ValueError("Input and target must have the same number of channels.")

        # Cross-Entropy Loss
        criterion_ce = torch.nn.CrossEntropyLoss()
        ce_loss = criterion_ce(inp, target)

        # Dice Loss
        probs = F.softmax(inp, dim=1)
        target_one_hot = F.one_hot(target, num_classes=inp.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(probs * target_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))

        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = torch.mean(dice_loss)

        # Combine Cross-Entropy and Dice Loss
        loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss

        return loss
