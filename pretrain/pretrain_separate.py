import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from models import MaskedGraphAutoEncoder
from dataloader import load_data
from utils import mask_features, masked_loss, set_seed


# ============================================================
# Training
# ============================================================

def train(cfg):

    set_seed(cfg.seed)

    if cfg.cuda >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.cuda}")
    else:
        device = torch.device("cpu")
        
    wandb.init(project=cfg.project, name=f"{cfg.modality}_pretrain_{cfg.emb_dim}", config=vars(cfg))

    x, obs_mask, edge_index, edge_attr, vertices, labels = load_data(cfg.data, cfg.modality)

    x = x.to(device)
    obs_mask = obs_mask.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    vertices = vertices.to(device)
    labels = labels.to(device)

    num_states = labels.max().item() + 1

    model = MaskedGraphAutoEncoder(
        x.shape[1],
        cfg.hidden_dim,
        cfg.emb_dim,
        num_states,
        cfg.label_dim,
        cfg.num_layers,
        cfg.dropout,
        cfg.heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    os.makedirs(cfg.out, exist_ok=True)

    base_seed = cfg.seed

    for epoch in range(cfg.epochs):

        model.train()

        optimizer.zero_grad()

        x_corr, corr_mask = mask_features(
            x,
            obs_mask,
            cfg.mask_ratio,
            epoch,
            base_seed
        )


        z, x_hat = model(
            x_corr,
            edge_index,
            edge_attr,
            vertices,
            labels
        )

        loss = masked_loss(x_hat, x, obs_mask, corr_mask)

        loss.backward()

        optimizer.step()
        scheduler.step()

        wandb.log({
            "loss": loss.item(),
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "mask_ratio_actual": corr_mask.mean().item()
        })

        if epoch % 20 == 0:
            print(epoch, loss.item())

    torch.save(model.state_dict(), f"{cfg.out}/{cfg.modality}_AE_{cfg.emb_dim}.pt")

    model.eval()

    with torch.no_grad():

        z, _ = model(
            x,
            edge_index,
            edge_attr,
            vertices,
            labels
        )

    torch.save(z.cpu(), f"{cfg.out}/{cfg.modality}_emb_{cfg.emb_dim}.pt")

    wandb.finish()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/region_graph_with_features_and_targets.npz")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index, use -1 for CPU")
    parser.add_argument("--label_dim", type=int, default=16)
    parser.add_argument("--modality", default="wind")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--project", default="graph_pretrain")
    
    cfg = parser.parse_args()
    
    train(cfg)