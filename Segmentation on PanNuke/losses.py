import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        num_classes = output.shape[1]
        focal_losses = []

        for class_idx in range(num_classes):
            class_output = output[:, class_idx, :, :]
            class_target = (target == class_idx).float()
            ce_loss = F.binary_cross_entropy_with_logits(class_output, class_target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            focal_losses.append(focal_loss)

        focal_losses = torch.stack(focal_losses, dim=1)  # Stack along class dimension
        focal_losses = torch.mean(focal_losses, dim=1)  # Average over classes

        if self.reduction == 'mean':
            return focal_losses.mean()
        elif self.reduction == 'sum':
            return focal_losses.sum()
        else:
            return focal_losses.mean()
        

#class FocalLoss(nn.Module):
#    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
#        super(FocalLoss, self).__init__()
#        self.alpha = alpha
#        self.gamma = gamma
#        self.reduction = reduction
#
#    def forward(self, output, target):
#        output = torch.sigmoid(output)
#        num_classes = output.shape[1]
#        focal_losses = []
#
#        for class_idx in range(num_classes):
#            class_output = output[:, class_idx, :, :]
#            class_target = (target == class_idx).float()
#            ce_loss = F.binary_cross_entropy(class_output, class_target, reduction='none')
#            pt = torch.exp(-ce_loss)
#            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#            focal_losses.append(focal_loss)
#
#        focal_losses = torch.stack(focal_losses, dim=1)  # Stack along class dimension
#        focal_losses = torch.mean(focal_losses, dim=1)  # Average over classes
#
#        if self.reduction == 'mean':
#            return focal_losses.mean()
#        elif self.reduction == 'sum':
#            return focal_losses.sum()
#        else:
#            return focal_losses.mean()