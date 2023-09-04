
import torch.nn.functional as F
import torch
from torch import nn

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', alpha=0.25):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = alpha

    def forward(self, input, target):
        ## input: B, 2, L
        ## target: B, L
        target = F.one_hot(target, 2).float().view([-1,2])
        input = input.permute(0,2,1).reshape([-1, 2])
        ## input = target: B*L, 2

        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=self.weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


def dice_coeff(input:torch.Tensor, target:torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but gottorch.Tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]

def multiclass_dice_coeff(input:torch.Tensor, target:torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

class dice_loss(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
    def forward(self, input:torch.Tensor, target:torch.Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        fn = multiclass_dice_coeff if multiclass else dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

class CE_dice(nn.Module):
    def __init__(self, weight = None) -> None:
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)
        self.dice = dice_loss()
    def forward(self, input:torch.Tensor, target:torch.Tensor):
        ce = self.ce(input, target)
        dice = self.dice(F.softmax(input, dim=-2), F.one_hot(target, 2).permute(0,2,1).float())
        return ce + dice

class CE_L1(nn.Module):
    def __init__(self, ignore_index=5, alpha=0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.psg_epoch = 1260
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.l1 = torch.nn.L1Loss()
    def forward(self, input:torch.Tensor, target:torch.Tensor):
        # input shape: [bs, len_psg, n_class]
        loss1 = self.ce(input.reshape([-1,input.shape[-1]]), target.view(-1))
        loss2 = self.l1(input[:, 1:, :], input[:, :-1, :])
        return loss1 + self.alpha * loss2

class FocalLoss_L2(nn.Module):
    def __init__(self, weight=None, alpha=.25, gamma=2, alpha_L2 = .2) -> None:
        super().__init__()
        self.focal = FocalLoss(weight=weight, alpha=alpha, gamma=gamma)
        self.alpha_L2 = alpha_L2
    def forward(self, input, target):
        FL = self.focal(input, target)
        L2 = torch.mean(torch.diff(input)**2)
        return FL + self.alpha_L2 * L2
