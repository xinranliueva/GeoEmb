import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Mask corruption
# ============================================================

def mask_features(x, obs_mask, ratio, epoch, base_seed):
    g = torch.Generator(device=x.device)
    g.manual_seed(base_seed + epoch)
    rand = torch.rand(x.shape, generator=g, device=x.device)
    corr_mask = (rand < ratio).float() * obs_mask
    x_corr = x.clone()
    x_corr[corr_mask.bool()] = 0
    return x_corr, corr_mask


# ============================================================
# Loss
# ============================================================

def masked_loss(x_hat, x, obs_mask, corr_mask):
    mask = obs_mask * corr_mask
    diff = F.smooth_l1_loss(x_hat, x, reduction="none")
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-8)
