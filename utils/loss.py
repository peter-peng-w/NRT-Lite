import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, cross_entropy, nll_loss, binary_cross_entropy_with_logits

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def mask_nll_loss(inp, target, mask):
    inp = inp.view(-1, inp.size(-1))
    target = target.view(-1)

    mask = mask.view(-1)

    loss = nll_loss(inp, target, reduction='none').masked_select(mask).mean()

    return loss


def review_loss(c_hat, c):
    assert c_hat.size() == c.size(), '{} != {}'.format(c_hat.size(), c.size())
    return torch.mul(c_hat, c.float()).sum()


def mask_ce_loss(inp, target, mask):
    '''
    Calculate the cross entropy loss and a binary mask tensor describing the padding of the target tensor.

    Inputs:
        inp: tensor of raw values; size=(seq, batch, classes)
        target: tensor of classes; size=(seq, batch)
        mask: binary mask; size=(seq, batch)

    Outputs:
        loss: scalar
    '''

    # format the sizes
    inp = inp.view(-1, inp.size(-1))
    target = target.view(-1)

    mask = mask.view(-1)

    loss = cross_entropy(inp, target, reduction='none').masked_select(mask).mean()

    return loss


def rmse_loss(preds, truth):
    return mse_loss(preds, truth).sqrt()


def bpr_loss(preds, truth, weight=None):
    # assert preds.size(-1) == 2
    # assert truth.size(-1) == 2
    assert preds.size(-1) == truth.size(-1)

    loss_all = None
    n_samples = preds.size(-1)

    for k in range(1, n_samples):
        # truth_ = torch.zeros_like(truth)
        # truth_[truth > 3] = 1
        # truth[truth <= 3] = 0

        diff = preds[..., 0] - preds[..., k]
        target = truth[..., 0] - truth[..., k]

        target[target > 0] = 1
        target[target == 0] = .5
        target[target < 0] = 0

        # only select diff pairs
        diff = diff[target != 0.5]
        target = target[target != 0.5]

        loss = binary_cross_entropy_with_logits(diff, target, weight=weight, reduction='none')

        loss_all = loss if loss_all is None else torch.cat([loss_all, loss])

    return loss_all.mean()
