"""
Critical Path (CP) 计算 —— 可落地版本

公式:
  CP(i) = w̄_i + max_{j∈Succ(i)} (c̄_ij + CP(j))
  出口节点: CP(i) = w̄_i

  w̄_i = (1/K) * Σ t_{i,k}  节点 i 在 K 个位置的平局执行时间
  c̄_ij = data_ij / avg_bandwidth  边通信开销
"""
from typing import Dict, List, Tuple
from collections import deque


def topological_sort(
    n: int,
    succ: Dict[int, List[int]],
) -> List[int]:
    """Kahn 拓扑排序，返回拓扑序"""
    in_deg = [0] * n
    for i in range(n):
        for j in succ.get(i, []):
            in_deg[j] += 1
    q = deque(i for i in range(n) if in_deg[i] == 0)
    topo = []
    while q:
        i = q.popleft()
        topo.append(i)
        for j in succ.get(i, []):
            in_deg[j] -= 1
            if in_deg[j] == 0:
                q.append(j)
    return topo


def compute_cp(
    node_ids: List[int],
    id2idx: Dict[int, int],
    nodes: Dict,
    edges_data: Dict[Tuple[int, int], float],
    locations: List,
    exec_model,
    trans_model,
) -> Dict[int, float]:
    """
    计算每个节点的 Critical Path 值（按 index 索引）

    Returns:
        CP: Dict[index -> float]，index 为 node_ids 中的索引 0..N-1
    """
    N = len(node_ids)
    if N == 0:
        return {}

    # 1. 构建邻接表 (index -> [successor indices])
    succ: Dict[int, List[int]] = {}
    for i, node_id in enumerate(node_ids):
        node = nodes[node_id]
        succ[i] = [id2idx[v] for v in node.succ if v in id2idx]

    # 2. node_cost: w̄_i = (1/K) * Σ t_{i,k}
    K = len(locations)
    node_cost: Dict[int, float] = {}
    for i, node_id in enumerate(node_ids):
        node = nodes[node_id]
        total = sum(exec_model.exec_time(node, loc) for loc in locations)
        node_cost[i] = total / K if K > 0 else 0.0

    # 3. edge_cost: c̄_ij = data_ij / avg_bandwidth
    avg_bw = (trans_model.ue_es_transmission_rate + trans_model.es_es_transmission_rate) / 2.0
    if avg_bw <= 0:
        avg_bw = 1e-9

    edge_cost: Dict[Tuple[int, int], float] = {}
    for (u, v), data in edges_data.items():
        if u in id2idx and v in id2idx:
            i, j = id2idx[u], id2idx[v]
            edge_cost[(i, j)] = float(data) / avg_bw

    # 4. 拓扑排序
    topo = topological_sort(N, succ)
    if len(topo) != N:
        raise ValueError("Graph has cycle or disconnected components")

    # 5. 逆拓扑序计算 CP
    CP: Dict[int, float] = {}
    for i in reversed(topo):
        succlist = succ.get(i, [])
        if len(succlist) == 0:
            CP[i] = node_cost[i]
        else:
            best = max(
                edge_cost.get((i, j), 0.0) + CP[j]
                for j in succlist
            )
            CP[i] = node_cost[i] + best

    return CP


def normalize_cp(cp_dict: Dict[int, float], method: str = "max") -> Dict[int, float]:
    """
    归一化 CP 值
    method: "max" -> CP(i)/max(CP)  |  "minmax" -> (CP-min)/(max-min)
    """
    if not cp_dict:
        return {}
    vals = list(cp_dict.values())
    mx = max(vals)
    if mx <= 0:
        return {k: 0.0 for k in cp_dict}
    if method == "max":
        return {k: v / mx for k, v in cp_dict.items()}
    mn = min(vals)
    if mx == mn:
        return {k: 1.0 for k in cp_dict}
    return {k: (v - mn) / (mx - mn) for k, v in cp_dict.items()}
