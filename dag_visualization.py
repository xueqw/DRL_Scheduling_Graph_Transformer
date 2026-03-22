from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


# ---------------- 单位换算（你现在 edges_data 是 bits） ----------------
def bits_to_kb(bits: float) -> float:
    return bits / (8.0 * 1024.0)


# ---------------- 纯 Python 分层布局（从上到下） ----------------
def layered_layout(subG: nx.DiGraph, end_set: set[int]) -> dict[int, tuple[float, float]]:
    """
    对一个 DAG 子图做分层布局：
      - start(入度=0 且非 End) 在第 0 层
      - 其它节点层 = max(pred层)+1
      - End 强制放在最底层
    返回 pos: node -> (x, y)
    """
    topo = list(nx.topological_sort(subG))
    level: dict[int, int] = {}

    # 先算每个节点的层级
    for n in topo:
        if n in end_set:
            continue
        preds = list(subG.predecessors(n))
        if not preds:
            level[n] = 0
        else:
            level[n] = max(level[p] for p in preds if p in level) + 1 if level else 0

    # End 放到最后一层（更像汇聚）
    max_lv = max(level.values()) if level else 0
    for e in end_set:
        if e in subG:
            level[e] = max_lv + 1

    # 分层
    layers: dict[int, List[int]] = defaultdict(list)
    for n, lv in level.items():
        layers[lv].append(n)

    # 层内排序：按 id 排（也可以换成按 indeg/outdeg）
    for lv in layers:
        layers[lv].sort()

    # 生成坐标：y = -lv，从上到下；x 在层内均匀排开并居中
    pos: dict[int, tuple[float, float]] = {}
    x_gap = 1.6
    y_gap = 1.8

    for lv in sorted(layers.keys()):
        ns = layers[lv]
        k = len(ns)
        for i, n in enumerate(ns):
            x = (i - (k - 1) / 2.0) * x_gap
            y = -lv * y_gap
            pos[n] = (x, y)

    return pos


def _try_graphviz_layout(subG: nx.DiGraph) -> Optional[dict[int, tuple[float, float]]]:
    """
    如果本机有 graphviz + pydot，就用 dot 画分层布局（更漂亮）；
    否则返回 None，外部 fallback 到 layered_layout。
    """
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        # dot 默认是分层布局（top-down）
        pos = graphviz_layout(subG, prog="dot")
        # graphviz_layout 可能返回 int/float 混合，统一成 float
        return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    except Exception:
        return None


# ---------------- 主可视化函数 ----------------
def visualize_dag(
    nodes: Dict[int, Any],
    edges_data: Dict[Tuple[int, int], float],
    end_nodes: List[int],
    *,
    title: str = "Generated DAG",
    show_edge_labels: bool = False,
    edge_label_unit: str = "KB",          # "KB" 或 "bits"
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
):
    """
    可视化 generator 生成的 DAG（支持多 UE 分组展示，分层布局从上到下）

    约定：
      - start 节点：len(node.pred)==0 且不在 end_nodes
      - end 节点：node_id in end_nodes
      - download 边：任何 u -> End 的边（v in end_nodes）
    """
    end_set = set(end_nodes)

    # 1) 构图
    G = nx.DiGraph()
    for nid, node in nodes.items():
        ue_id = getattr(node, "ue_id", 0)
        G.add_node(nid, ue_id=ue_id)

    for (u, v), bits in edges_data.items():
        G.add_edge(u, v, bits=bits)

    # 2) 分类节点（全局）
    start_nodes: List[int] = []
    normal_nodes: List[int] = []
    for nid, node in nodes.items():
        if nid in end_set:
            continue
        pred = getattr(node, "pred", [])
        if pred is None:
            pred = []
        if len(pred) == 0:
            start_nodes.append(nid)
        else:
            normal_nodes.append(nid)

    # 3) 边分类
    download_edges = [(u, v) for (u, v) in G.edges() if v in end_set]
    other_edges = [(u, v) for (u, v) in G.edges() if v not in end_set]

    # 4) 多 UE：按 ue_id 分组做子图分层布局，然后横向拼接
    ue_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for nid in G.nodes():
        ue_to_nodes[G.nodes[nid].get("ue_id", 0)].append(nid)
    ue_ids = sorted(ue_to_nodes.keys())

    pos: dict[int, tuple[float, float]] = {}
    x_offset = 0.0
    ue_gap = 10.0  # UE 之间的横向间隔（越大越分散）

    for ue_id in ue_ids:
        sub_nodes = ue_to_nodes[ue_id]
        subG = G.subgraph(sub_nodes).copy()

        # 优先 graphviz(dot) 分层，否则用纯 python 分层
        sub_pos = _try_graphviz_layout(subG)
        if sub_pos is None:
            sub_pos = layered_layout(subG, end_set)

        # 归一化并平移：让每个 UE 子图不会重叠
        xs = [sub_pos[n][0] for n in sub_pos]
        if xs:
            min_x, max_x = min(xs), max(xs)
            width = max(max_x - min_x, 1.0)
        else:
            min_x, width = 0.0, 1.0

        for n, (x, y) in sub_pos.items():
            pos[n] = ((x - min_x) + x_offset, y)

        x_offset += width + ue_gap

    # 5) 开始画图
    plt.figure(figsize=figsize)
    plt.title(title)
    ax = plt.gca()

    # 先画普通边，再画 download 边（更粗更显眼）
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, arrows=True, arrowstyle="-|>", width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=download_edges, arrows=True, arrowstyle="-|>", width=2.6, ax=ax)

    # 画节点：normal / start / end
    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=520, node_shape="o", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=start_nodes, node_size=650, node_shape="s", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(end_set), node_size=760, node_shape="D", ax=ax)

    # 节点标签：id + UE
    labels = {}
    for nid in G.nodes():
        ue_id = G.nodes[nid].get("ue_id", None)
        labels[nid] = f"{nid}\nUE{ue_id}" if ue_id is not None else str(nid)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    # 可选：边标签（显示数据量）
    if show_edge_labels:
        edge_labels = {}
        for (u, v) in G.edges():
            bits = G.edges[u, v].get("bits", 0.0)
            if edge_label_unit.lower() == "kb":
                edge_labels[(u, v)] = f"{bits_to_kb(bits):.0f}KB"
            else:
                edge_labels[(u, v)] = f"{bits:.0f}b"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    # 图例说明（中文）
    ax.text(
        0.01, 0.01,
        "图例：方形=start(pred=[]), 圆形=normal, 菱形=End； 指向End的边=download边（加粗）",
        transform=ax.transAxes,
        fontsize=9
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# ---- 示例用法（你按自己的 generator 改 import） ----
if __name__ == "__main__":
    from dag_generator import generate_multi_ue_dag

    nodes, edges_data, end_nodes, download_nodes, order = generate_multi_ue_dag(ue_number=3,n_compute_nodes_per_ue=5,seed=0)

    visualize_dag(
        nodes,
        edges_data,
        end_nodes,
        title="UE=2, N=15, seed=0",
        show_edge_labels=False,
        save_path="dag.png"
    )