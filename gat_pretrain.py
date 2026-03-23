"""
Unsupervised GAT auto-encoder pretraining for the DTODRL encoder.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv

from Graph_policy import build_graph_inputs_from_adj
from dtodrl_backbone import DTODRLGATEncoder


class GATDecoder(nn.Module):
    """Mirror the encoder with GAT layers and reconstruct node attributes."""

    def __init__(self, hidden_dim: int = 128, out_dim: int = 6, heads: int = 3, num_layers: int = 3):
        super().__init__()
        out_per_head = (hidden_dim + heads - 1) // heads
        mid_dim = out_per_head * heads
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, out_per_head, heads=heads, add_self_loops=True, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(mid_dim, out_per_head, heads=heads, add_self_loops=True, concat=True))
            self.convs.append(GATConv(mid_dim, out_dim, heads=1, add_self_loops=True, concat=False))
        else:
            self.convs.append(GATConv(hidden_dim, out_dim, heads=1, add_self_loops=True, concat=False))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < self.num_layers - 1:
                x = F.leaky_relu(x, 0.2)
        return x


class GATAutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dim: int = 128, heads: int = 3, num_layers: int = 3):
        super().__init__()
        self.encoder = DTODRLGATEncoder(in_dim, hidden_dim, heads, num_layers)
        self.decoder = GATDecoder(hidden_dim, in_dim, heads, num_layers)

    def forward(self, x: Tensor, edge_index: Tensor, adj: Tensor, batch: Optional[Tensor] = None):
        h = self.encoder(x, edge_index, batch)
        h_recon = self.decoder(h, edge_index)
        return h, h_recon


def feature_loss(h_orig: Tensor, h_recon: Tensor) -> Tensor:
    return F.mse_loss(h_recon, h_orig)


def structure_loss(h: Tensor, adj: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    adj_ = adj[0] if adj.dim() == 3 else adj
    n = h.shape[0]
    if batch is not None:
        n = (batch == batch[0]).sum().item()
        h = h[:n]
        adj_ = adj_[:n, :n] if adj_.shape[0] >= n else adj_

    loss = 0.0
    count = 0
    for i in range(min(n, h.shape[0])):
        neighbors = (adj_[i] > 0).nonzero(as_tuple=True)[0]
        if len(neighbors) == 0:
            continue
        neighbors = neighbors[neighbors < h.shape[0]]
        if len(neighbors) == 0:
            continue
        sim = torch.sigmoid((h[i : i + 1] @ h[neighbors].T).squeeze())
        if sim.dim() == 0:
            sim = sim.unsqueeze(0)
        loss = loss - torch.log(sim.clamp(min=1e-9)).mean()
        count += 1
    return loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=h.device)


def pretrain_gat_on_obs(obs_list: List[dict], epochs: int = 50, lr: float = 1e-3, device: str = "cpu") -> dict:
    model = GATAutoEncoder(in_dim=6, hidden_dim=128, heads=3, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        total_fl, total_sl = 0.0, 0.0
        n_batch = 0
        for obs in obs_list:
            node_features = torch.stack(
                [
                    torch.as_tensor(obs["nodes_C"], dtype=torch.float32),
                    torch.as_tensor(obs["nodes_D"], dtype=torch.float32),
                    torch.as_tensor(obs["nodes_in_degree"], dtype=torch.float32),
                    torch.as_tensor(obs["nodes_out_degree"], dtype=torch.float32),
                    torch.as_tensor(obs["nodes_loc"], dtype=torch.float32),
                    torch.as_tensor(obs["nodes_ava"], dtype=torch.float32),
                ],
                dim=-1,
            ).to(device)
            adj = torch.as_tensor(obs["adj"], dtype=torch.float32).to(device)
            edge_attr = torch.as_tensor(obs["edge_attr"], dtype=torch.float32).to(device)

            edge_index, _, batch, _ = build_graph_inputs_from_adj(adj, edge_attr)
            edge_index = edge_index.to(device)

            h, h_recon = model(node_features, edge_index, adj.unsqueeze(0) if adj.dim() == 2 else adj, batch)
            fl = feature_loss(node_features, h_recon)
            sl = structure_loss(h, adj, batch)
            loss = fl + sl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_fl += fl.item()
            total_sl += sl.item()
            n_batch += 1

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep + 1} FL={total_fl / n_batch:.4f} SL={total_sl / n_batch:.4f}")

    return model.encoder.state_dict()


def run_pretrain_and_save(
    env_controller,
    save_path: str = "gat_pretrained.pt",
    n_dags: int = 100,
    epochs: int = 50,
    lr: float = 1e-3,
):
    obs_list = []
    for _ in range(n_dags):
        env = env_controller()
        obs, _ = env.reset()
        obs_list.append(obs)
    encoder_sd = pretrain_gat_on_obs(obs_list, epochs=epochs, lr=lr)
    torch.save(encoder_sd, save_path)
    print(f"GAT pretraining finished, saved to {save_path}")
    return save_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from final_training import DAGConfig, make_dto_env_controller

    dag_cfg = DAGConfig()
    ctrl = make_dto_env_controller(
        ue_number=dag_cfg.ue_numbers,
        es_number=dag_cfg.es_numbers,
        n_compute_nodes_per_ue=20,
        start_count_max=3,
        f_ue=dag_cfg.f_ue,
        f_es=dag_cfg.f_es,
        es_processors=4,
        tr_ue_es=dag_cfg.tr_ue_es,
        tr_es_es=dag_cfg.tr_es_es,
    )
    run_pretrain_and_save(ctrl, n_dags=50, epochs=30)
