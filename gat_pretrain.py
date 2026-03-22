"""
论文 GAT 无监督预训练

论文 4.1 节: GAT Auto-Encoder
- Feature Loss: MSE(原特征, 重建特征)
- Structure Loss: 邻接节点表示相似
- 仅保留 Encoder 用于 DTODRL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
import numpy as np

from Graph_policy import build_graph_inputs_from_adj
from dtodrl_backbone import DTODRLGATEncoder


class GATDecoder(nn.Module):
    """论文 Decoder: 逆向结构重建节点特征"""
    def __init__(self, hidden_dim: int = 128, in_dim: int = 6, num_layers: int = 3):
        super().__init__()
        layers = []
        cur = hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            cur = hidden_dim
        layers.append(nn.Linear(cur, in_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class GATAutoEncoder(nn.Module):
    """论文 GAT Auto-Encoder for 无监督预训练"""
    def __init__(self, in_dim: int = 6, hidden_dim: int = 128, heads: int = 3, num_layers: int = 3):
        super().__init__()
        self.encoder = DTODRLGATEncoder(in_dim, hidden_dim, heads, num_layers)
        self.decoder = GATDecoder(hidden_dim, in_dim, num_layers)

    def forward(self, x: Tensor, edge_index: Tensor, adj: Tensor, batch: Optional[Tensor] = None):
        h = self.encoder(x, edge_index, batch)
        h_recon = self.decoder(h)
        return h, h_recon


def feature_loss(h_orig: Tensor, h_recon: Tensor) -> Tensor:
    """论文 Eq.9: Feature Loss"""
    return F.mse_loss(h_recon, h_orig)


def structure_loss(h: Tensor, adj: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """论文 Eq.10: Structure Loss, 邻接节点表示应相似"""
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
        sim = torch.sigmoid((h[i:i+1] @ h[neighbors].T).squeeze())
        if sim.dim() == 0:
            sim = sim.unsqueeze(0)
        loss = loss - torch.log(sim.clamp(min=1e-9)).mean()
        count += 1
    return loss / max(count, 1) if count > 0 else torch.tensor(0.0, device=h.device)


def pretrain_gat_on_obs(obs_list: List[dict], epochs: int = 50, lr: float = 1e-3, device: str = "cpu") -> dict:
    """
    在观测数据上预训练 GAT
    obs_list: 多个 DAG 的观测 (从 env.build_obs 或 reset 获取)
    返回: encoder state_dict (可加载到 DTODRLBackbone.gat_encoder)
    """
    model = GATAutoEncoder(in_dim=6, hidden_dim=128, heads=3, num_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        total_fl, total_sl = 0.0, 0.0
        n_batch = 0
        for obs in obs_list:
            node_features = torch.stack([
                torch.as_tensor(obs["nodes_C"], dtype=torch.float32),
                torch.as_tensor(obs["nodes_D"], dtype=torch.float32),
                torch.as_tensor(obs["nodes_in_degree"], dtype=torch.float32),
                torch.as_tensor(obs["nodes_out_degree"], dtype=torch.float32),
                torch.as_tensor(obs["nodes_loc"], dtype=torch.float32),
                torch.as_tensor(obs["nodes_ava"], dtype=torch.float32),
            ], dim=-1).to(device)
            adj = torch.as_tensor(obs["adj"], dtype=torch.float32).to(device)
            edge_attr = torch.as_tensor(obs["edge_attr"], dtype=torch.float32).to(device)

            edge_index, _, batch, _ = build_graph_inputs_from_adj(adj, edge_attr)
            edge_index = edge_index.to(device)

            h, h_recon = model(node_features, edge_index, adj.unsqueeze(0) if adj.dim() == 2 else adj, batch)
            fl = feature_loss(node_features, h_recon)
            sl = structure_loss(h, adj, batch)
            loss = fl + sl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_fl += fl.item()
            total_sl += sl.item()
            n_batch += 1
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1} FL={total_fl/n_batch:.4f} SL={total_sl/n_batch:.4f}")

    return model.encoder.state_dict()


def run_pretrain_and_save(
    env_controller,
    save_path: str = "gat_pretrained.pt",
    n_dags: int = 100,
    epochs: int = 50,
    lr: float = 1e-3,
):
    """收集 DAG 观测并预训练 GAT，保存 encoder 权重"""
    obs_list = []
    for i in range(n_dags):
        env = env_controller()
        obs, _ = env.reset()
        obs_list.append(obs)
    encoder_sd = pretrain_gat_on_obs(obs_list, epochs=epochs, lr=lr)
    torch.save(encoder_sd, save_path)
    print(f"GAT 预训练完成，已保存到 {save_path}")
    return save_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from final_training import make_dto_env_controller, DAGConfig

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
