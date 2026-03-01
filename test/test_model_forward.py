import torch

from pretrain.models import SharedMaskedGraphAutoEncoder
from dataloader import load_data
from utils import mask_features, masked_loss, set_seed


def test_model_forward():

    set_seed(42)

    path = "data/region_graph_with_features_and_targets.npz"

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    xw, obs_w, edge_index, edge_attr, vertices, labels = load_data(path, "wind")
    xa, obs_a, _, _, _, _ = load_data(path, "aq")

    num_states = labels.max().item() + 1

    # --------------------------------------------------
    # Mask features (CRITICAL: match training)
    # --------------------------------------------------

    mask_ratio = 0.3
    epoch = 0
    seed = 42

    xw_corr, corr_w = mask_features(xw, obs_w, mask_ratio, epoch, seed)
    xa_corr, corr_a = mask_features(xa, obs_a, mask_ratio, epoch, seed)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SharedMaskedGraphAutoEncoder(
        in_dims=[xw.shape[1], xa.shape[1]],
        emb_dims=[8, 8],
        hidden_dim=32,
        num_states=num_states,
        label_dim=4,
        num_layers=2,
        dropout=0.1,
        heads=2,
        edge_dim=1,
    )

    model.train()

    # --------------------------------------------------
    # Forward pass (use masked inputs)
    # --------------------------------------------------

    z, z_list, x_hat_list = model(
        [xw_corr, xa_corr],
        edge_index,
        edge_attr,
        vertices,
        labels
    )

    # --------------------------------------------------
    # Compute reconstruction loss (same as training)
    # --------------------------------------------------

    loss_rec_w = masked_loss(x_hat_list[0], xw, obs_w, corr_w)
    loss_rec_a = masked_loss(x_hat_list[1], xa, obs_a, corr_a)

    loss = loss_rec_w + loss_rec_a

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    loss.backward()

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert z.shape[0] == xw.shape[0]

    assert not torch.isnan(loss)

    for param in model.parameters():
        assert param.grad is not None