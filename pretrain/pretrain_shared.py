# ============================================================
# Shared Masked Graph Autoencoder training
# ============================================================

import os
import argparse
import torch
import wandb

from models import SharedMaskedGraphAutoEncoder

from dataloader import load_data
from utils import mask_features, masked_loss, set_seed


# ============================================================
# Training
# ============================================================

def train(cfg):

    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg.project,
               name=f"shared_AE_{cfg.emb_dim_w+cfg.emb_dim_a}_{cfg.seed}",
               config=vars(cfg))


    xw, obs_w, edge_index, edge_attr, vertices, labels = load_data(cfg.data, "wind")
    xa, obs_a, _, _, _, _ = load_data(cfg.data, "aq")

    xw = xw.to(device)
    xa = xa.to(device)

    obs_w = obs_w.to(device)
    obs_a = obs_a.to(device)

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    vertices = vertices.to(device)
    labels = labels.to(device)

    num_states = labels.max().item() + 1


    model = SharedMaskedGraphAutoEncoder(
        in_dims=[xw.shape[1], xa.shape[1]],
        emb_dims=[cfg.emb_dim_w, cfg.emb_dim_a],
        hidden_dim=cfg.hidden_dim,
        num_states=num_states,
        label_dim=cfg.label_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        heads=cfg.heads,
        edge_dim=1,
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )


    os.makedirs(cfg.out, exist_ok=True)


    for epoch in range(cfg.epochs):

        model.train()
        optimizer.zero_grad()

        xw_corr, corr_w = mask_features(xw, obs_w, cfg.mask_ratio, epoch, cfg.seed)
        xa_corr, corr_a = mask_features(xa, obs_a, cfg.mask_ratio, epoch, cfg.seed)

        z, z_list, x_hat_list = model([xw_corr, xa_corr],
                                      edge_index,
                                      edge_attr,
                                      vertices,
                                      labels)

        zw = z_list[0]
        za = z_list[1]

        xw_hat = x_hat_list[0]
        xa_hat = x_hat_list[1]

        loss_rec_w = masked_loss(xw_hat, xw, obs_w, corr_w)
        loss_rec_a = masked_loss(xa_hat, xa, obs_a, corr_a)

        loss = cfg.lambda_w * loss_rec_w + cfg.lambda_a * loss_rec_a

        loss.backward()
        optimizer.step()

        wandb.log({
            "loss_total": loss.item(),
            "loss_rec_wind": loss_rec_w.item(),
            "loss_rec_aq": loss_rec_a.item(),
            "epoch": epoch
        })

        if epoch % 10 == 0:
            print(epoch, loss.item())


    model.eval()

    with torch.no_grad():

        z, z_list, _ = model([xw, xa],
                             edge_index,
                             edge_attr,
                             vertices,
                             labels)

        zw = z_list[0]
        za = z_list[1]
        embedding = z


    torch.save(model.state_dict(),
               os.path.join(cfg.out,
               f"shared_AE_{cfg.emb_dim_w+cfg.emb_dim_a}.pt"))

    torch.save(zw.cpu(),
               os.path.join(cfg.out,
               f"shared_wind_emb_{cfg.emb_dim_w}.pt"))

    torch.save(za.cpu(),
               os.path.join(cfg.out,
               f"shared_aq_emb_{cfg.emb_dim_a}.pt"))

    torch.save(embedding.cpu(),
               os.path.join(cfg.out,
               f"shared_final_emb_{cfg.emb_dim_w+cfg.emb_dim_a}.pt"))

    wandb.finish()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="../data/region_graph_with_features_and_targets.npz")

    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=512)

    parser.add_argument("--emb_dim_w", type=int, default=64)
    parser.add_argument("--emb_dim_a", type=int, default=64)

    parser.add_argument("--label_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=10000)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--lambda_w", type=float, default=1)
    parser.add_argument("--lambda_a", type=float, default=1)

    parser.add_argument("--mask_ratio", type=float, default=0.3)

    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out", default="checkpoints")

    parser.add_argument("--project", default="graph_ssl_shared")

    cfg = parser.parse_args()

    train(cfg)