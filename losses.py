import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanTeacherLoss(nn.Module):
    """ Mean Teacher loss object that manages accuracy and consistency loss """
    def __init__(self,
                 accuracy_loss, consistency_loss,
                 alpha=1.0, beta=10.0):
        """  Constructor for Mean Teacher loss
        
        :param accuracy_loss: accuracy loss object (nn.Module)
        :param consistency_loss: consistency loss object (nn.Module)
        :param alpha: weighting for accuracy loss
        :param beta: weighting for consistency loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self._acc_loss = accuracy_loss
        self._con_loss = consistency_loss
        
    def forward(self, student_predictions, teacher_predictions, targets):
        """ Calculate loss
        
        :param student_predictions: predictions from student network
        :param teacher_predictions: predictions from teacher network
        :param targets: annotation targets (from dataloader)
        :returns: loss value
        """
        acc = self.alpha * self._acc_loss(student_predictions, targets)
        con = self.beta * self._con_loss(student_predictions['prediction'], teacher_predictions['prediction'])
        return acc + con


class LRIZZLoss(nn.Module):
    """ L-RIZZ loss object """
    def __init__(self, l=0.5, weight_eq=1.0, weight_ineq=1.0):
        """ Constructs L-RIZZ object
        
        :param l: margin hyperparameter
        :param weight_eq: weight for equality labels
        :param weight_ineq: weight for inequality labels
        """
        super().__init__()
        self._s = 1.0
        self._l = l
        self._weight_eq = weight_eq
        self._weight_ineq = weight_ineq
    
    def forward(self, predictions, targets):
        """ Calculates L-RIZZ loss
        
        :param predictions: predictions from network, which should be a dictionary with a
                            key "prediction" that contains a tensor of shape [B, 2, H, W]
        :param targets: annotation target (tensor of shape [B, n_annos, 7])
        """
        
        # Expect predictions['prediction'] to be of shape [B, 2, H, W]
        B = predictions['prediction'].shape[0]
        pa = torch.stack([predictions['prediction'][b,targets[b,:,0],targets[b,:,2],targets[b,:,1]] for b in range(B)], dim=0)
        pb = torch.stack([predictions['prediction'][b,targets[b,:,3],targets[b,:,5],targets[b,:,4]] for b in range(B)], dim=0)
        diff = pb - pa
        
        # Compute loss
        loss_ineq = (torch.square(F.relu((self._s * self._l) - (self._s * diff * targets[:,:,-1]))) * (targets[:,:,-1] != 0)).sum()
        loss_eq = ((torch.square(diff * self._s)) * (targets[:,:,-1] == 0)).sum()
        
        num_ineq = (targets[:,:,-1] != 0).sum()
        norm_ineq = 1.0 / num_ineq if num_ineq > 0 else 0.0
        
        num_eq = (targets[:,:,-1] == 0).sum()
        norm_eq = 1.0 / num_eq if num_eq > 0 else 0.0
        
        return (self._weight_ineq * norm_ineq * loss_ineq) + (self._weight_eq * norm_eq * loss_eq)


def get_loss(name, **kwargs):
    """ Helper function for getting a named loss """
    if name == "lrizz":
        return LRIZZLoss(**kwargs)
    else:
        raise ValueError("Invalid loss was provided")
    