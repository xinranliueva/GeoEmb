import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


# ============================================================
# Encoder (Graph Transformer)
# ============================================================

class GraphEncoder(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        emb_dim,
        num_layers=4,
        dropout=0.1,
        heads=4,
        edge_dim=1,
    ):

        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first layer
        self.convs.append(
            TransformerConv(
                in_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                beta=True,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim))


        # middle layers
        for _ in range(num_layers - 2):

            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    beta=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))


        # final layer
        self.convs.append(
            TransformerConv(
                hidden_dim,
                emb_dim // heads,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                beta=True,
            )
        )
        self.norms.append(nn.LayerNorm(emb_dim))


        self.dropout = nn.Dropout(dropout)



    def forward(self, x, edge_index, edge_attr):

        for i in range(len(self.convs)):

            x_new = self.convs[i](x, edge_index, edge_attr)

            x_new = self.norms[i](x_new)

            if i != len(self.convs) - 1:

                x_new = F.relu(x_new)

                x_new = self.dropout(x_new)


            # residual connection
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new


        return x



# ============================================================
# Decoder
# ============================================================

class GraphDecoder(nn.Module):

    def __init__(self, emb_dim, hidden_dim, out_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )


    def forward(self, z):

        return self.net(z)



# ============================================================
# Masked Graph Autoencoder
# ============================================================

class MaskedGraphAutoEncoder(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        emb_dim,
        num_states,
        label_dim,
        num_layers,
        dropout,
        heads=4,
        edge_dim=1,
    ):

        super().__init__()

        self.label_emb = nn.Embedding(num_states, label_dim)

        self.encoder = GraphEncoder(
            in_dim + 2 + label_dim,
            hidden_dim,
            emb_dim,
            num_layers,
            dropout,
            heads,
            edge_dim,
        )

        self.decoder = GraphDecoder(
            emb_dim,
            hidden_dim,
            in_dim
        )



    def forward(
        self,
        x_corr,
        edge_index,
        edge_attr,
        vertices,
        labels,
    ):

        label_vec = self.label_emb(labels)

        x_input = torch.cat(
            [
                x_corr,
                vertices,
                label_vec
            ],
            dim=1,
        )


        z = self.encoder(
            x_input,
            edge_index,
            edge_attr,
        )


        x_hat = self.decoder(z)


        return z, x_hat
    
    
    

##============================================================
# Shared Masked Graph Autoencoder
# ============================================================


class SharedMaskedGraphAutoEncoder(nn.Module):

    def __init__(
        self,
        in_dims,
        emb_dims,
        hidden_dim,
        num_states,
        label_dim,
        num_layers,
        dropout,
        heads=4,
        edge_dim=1,
    ):

        super().__init__()

        assert len(in_dims) == len(emb_dims)

        self.in_dims = in_dims
        self.emb_dims = emb_dims

        self.emb_dim_total = sum(emb_dims)

        self.label_emb = nn.Embedding(num_states, label_dim)


        encoder_input_dim = sum(in_dims) + 2 + label_dim

        self.encoder = GraphEncoder(
            encoder_input_dim,
            hidden_dim,
            self.emb_dim_total,
            num_layers,
            dropout,
            heads,
            edge_dim,
        )


        self.decoders = nn.ModuleList([
            GraphDecoder(emb_dim, hidden_dim, in_dim)
            for emb_dim, in_dim in zip(emb_dims, in_dims)
        ])


    def split_embedding(self, z):

        return torch.split(z, self.emb_dims, dim=1)


    def encode(
        self,
        x_list,
        edge_index,
        edge_attr,
        vertices,
        labels,
    ):

        label_vec = self.label_emb(labels)

        x_input = torch.cat(
            x_list + [vertices, label_vec],
            dim=1,
        )

        z = self.encoder(
            x_input,
            edge_index,
            edge_attr,
        )

        return z


    def reconstruct(self, z):

        z_list = self.split_embedding(z)

        x_hat_list = [
            decoder(z_m)
            for decoder, z_m in zip(self.decoders, z_list)
        ]

        return x_hat_list


    def forward(
        self,
        x_corr_list,
        edge_index,
        edge_attr,
        vertices,
        labels,
    ):

        z = self.encode(
            x_corr_list,
            edge_index,
            edge_attr,
            vertices,
            labels,
        )

        z_list = self.split_embedding(z)

        x_hat_list = self.reconstruct(z)

        return z, z_list, x_hat_list