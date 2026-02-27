import numpy as np
import torch

def load_data(path, modality, pretrain=True):
    data = np.load(path)
    edges = data["edges"]
    lon = data["lon"]
    lat = data["lat"]
    vertices = np.stack([lat, lon], axis=1)
    labels = data["state_labels"]

    if modality == "wind":
        X = data["X_wind"]

    elif modality == "aq":
        X = data["X_aq"]
    else:
        raise ValueError("Unknown modality")

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - X_mean) / X_std

    x = torch.tensor(X, dtype=torch.float32)
    obs_mask = torch.ones_like(x)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # compute edge_attr FIRST
    row, col = edge_index
    lat1, lon1 = vertices[row,0], vertices[row,1]
    lat2, lon2 = vertices[col,0], vertices[col,1]
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlon/2)**2
    c = 2*torch.arcsin(torch.sqrt(a + 1e-12))
    edge_attr = c.unsqueeze(-1)

    # normalize vertices AFTER
    v_mean = vertices.mean(dim=0, keepdim=True)
    v_std = vertices.std(dim=0, keepdim=True) + 1e-6
    vertices = (vertices - v_mean) / v_std
    
    if not pretrain:
        Y_r = data["Y_r"]
        Y_e = data["Y_e"]
        return x, obs_mask, edge_index, edge_attr, vertices, labels, Y_r, Y_e

    return x, obs_mask, edge_index, edge_attr, vertices, labels