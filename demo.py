import torch
from Graph_policy import GraphBackbone
from joint_policy import JointActor, JointCritic

device = "cuda" if torch.cuda.is_available() else "cpu"

N = 10
K = 4
hidden_dim = 108
raw_loc_feature_dim = 7

obs = {
    "nodes_C": torch.randn(N),
    "nodes_D": torch.randn(N),
    "nodes_ue_id": torch.randint(0, 2, (N,)),
    "nodes_in_degree": torch.randint(0, 3, (N,)),
    "nodes_out_degree": torch.randint(0, 3, (N,)),
    "nodes_loc": torch.randint(-1, 2, (N,)),
    "nodes_ava": torch.randint(0, 2, (N,)),
    "adj": torch.randint(0, 2, (N, N)),
    "edge_attr": torch.rand(N, N),
    "loc_cpu_speed": torch.rand(K),
    "loc_min_processor_EAT": torch.rand(K),
    "loc_num_processor": torch.randint(1, 4, (K,))
}
obs = {k: v.to(device) for k, v in obs.items()}

backbone = GraphBackbone(hidden_dim=hidden_dim).to(device)
actor = JointActor(hidden_dim=hidden_dim, raw_loc_feature_dim=raw_loc_feature_dim).to(device)
critic = JointCritic(hidden_dim=hidden_dim).to(device)

out = backbone.forward_all(obs)
loc_raw_features = torch.zeros((N, K, raw_loc_feature_dim), device=device)

logits = actor(
    node_embs=out["node_embs"],
    loc_embs=out["loc_embs"],
    loc_raw_features=loc_raw_features,
    graph_emb=out["graph_emb"],
    loc_global_emb=out["loc_global_emb"],
)

value = critic(
    graph_emb=out["graph_emb"],
    loc_global_emb=out["loc_global_emb"],
)

print("logits shape:", logits.shape)
print("value shape:", value.shape)

assert logits.shape == (N * K,), f"Expected {(N*K,)}, got {tuple(logits.shape)}"
assert value.shape == (1,), f"Expected (1,), got {tuple(value.shape)}"

import torch

# logits: (N*K,)
# 这里 N=10, K=4
N = 10
K = 4

# 造一个 pair_mask: (N, K)
# 例如只允许前 3 个 node，且 loc 只能选 0 和 2
pair_mask = torch.zeros(N, K, dtype=torch.bool, device=logits.device)
pair_mask[:3, 0] = True
pair_mask[:3, 2] = True

print("pair_mask shape:", pair_mask.shape)
print("num valid actions:", pair_mask.sum().item())

# flatten，必须和 logits 对齐
flat_mask = pair_mask.reshape(-1)

# mask logits
masked_logits = logits.masked_fill(~flat_mask, -1e9)

# 构造分布
dist = torch.distributions.Categorical(logits=masked_logits)

# sample 多次看看
print("\nSampled actions:")
for _ in range(10):
    action = dist.sample()
    node_idx = (action // K).item()
    loc_idx = (action % K).item()
    is_valid = pair_mask[node_idx, loc_idx].item()
    print(f"action={action.item():2d}, node={node_idx}, loc={loc_idx}, valid={is_valid}")

    assert is_valid, f"Sampled invalid action: node={node_idx}, loc={loc_idx}"
