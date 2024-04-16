import torch
import torch.nn as nn


class SparseDisagreementScore(nn.Module):
    """ Sparse Disagreement Metric """
    def __init__(self, threshold, mode=None):
        """ Constructor for HDR metric
        
        :param threshold: threshold to use (tau)
        :param mode: what to measure on (None=all, 'eq'=equality only, 'neq'=inequality only)
        """
        super().__init__()
        self._threshold = threshold
        # These are used for keeping track of score over multiple batches
        self._err = None
        self._tot = None
        if mode not in [None, 'eq', 'neq']:
            raise ValueError("Invalid mode for SparseDisagreementScore")
        self._mode = mode
        
    def reset(self):
        """ Reset the metric """
        self._err = None
        self._tot = None
    
    def compute(self):
        """ Returns the computed metric value 
        
        Raises ValueError if no data was provided to metric.
        """
        if self._err is None or self._tot is None:
            raise ValueError("Metric can not be computed - no data")
        
        return self._err / self._tot
    
    def forward(self, predictions, targets):
        """ Update metric value
        
        Note: in addition to returning prediction value from the provided set of predictions and targets,
        the metric is updated to incorporate the new data to its running metric value.
        
        :param predictions: predictions from network (tensor of shape [B, 2, H, W])
        :param targets: annotation targets (tensor of shape [B, n_annos, 7])
        :returns: metric value for given predictions and targets
        """
        # expect predictions to be [B, 2, H, W]
        B = predictions.shape[0]
        pa = torch.stack([predictions[b,targets[b,:,0],targets[b,:,2],targets[b,:,1]] for b in range(B)], dim=0)
        pb = torch.stack([predictions[b,targets[b,:,3],targets[b,:,5],targets[b,:,4]] for b in range(B)], dim=0)
        diff = pb - pa
        
        # Create ordinal labels
        po = torch.zeros_like(targets[:,:,-1])
        po[diff >  self._threshold] = 1
        po[diff < -self._threshold] = -1
        
        # Create mask for eq/neq
        mask = torch.ones_like(targets[:,:,-1])
        if self._mode is None:
            pass
        elif self._mode == 'eq':
            mask[targets[:,:,-1] != 0] = 0
        else: #neq
            mask[targets[:,:,-1] == 0] = 0
        
        # Compute error
        err = ((po != targets[:,:,-1]) * mask).sum()
        tot = mask.sum()
        
        self._err = err if self._err is None else err + self._err
        self._tot = tot if self._tot is None else tot + self._tot
        
        if tot > 0:
            return err / tot
        else:
            return None
