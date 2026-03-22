# dag_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import random
from collections import defaultdict, deque


# ---------------- 与调度器一致的 Node 结构 ----------------
@dataclass
class Node:
    id: int
    C: float                 # 计算量（CPU cycles）
    D: float                 # UE -> ES 上传数据量（bits）
    pred: List[int]          # 前驱节点 id 列表（不能为空，用空 list 表示无前驱）
    succ: List[int]          # 后继节点 id 列表
    ue_id: int               # 该节点属于哪个 UE



@dataclass(frozen=True)
class DAGCase:
    """
    给所有的实验模型统一dag图
    """
    nodes: Dict[int, Node]
    edges_data: Dict[Tuple[int, int], float]
    end_nodes: List[int]
    download_nodes: List[int]


# ---------------- 单位工具（方案 U1：数据用 bits，速率用 bps） ----------------
def kb_to_bits(kb: float) -> float:
    return kb * 1024.0 * 8.0


def rand_bits_from_kb(kb_low: int, kb_high: int) -> float:
    """在 [kb_low, kb_high] KB 中随机取一个值，并转成 bits"""
    return kb_to_bits(float(random.randint(kb_low, kb_high)))


# ---------------- 建图辅助函数 ----------------
def add_edge(
    u: int,
    v: int,
    nodes: Dict[int, Node],
    edges_data: Dict[Tuple[int, int], float],
    bits: float,
    out_degree: Dict[int, int],
    max_out_degree: int,
):
    """
    添加一条有向边 u -> v，并同步维护 pred / succ
    同时限制每个节点的最大出度
    """
    if (u, v) in edges_data:
        return
    if out_degree[u] >= max_out_degree:
        return

    nodes[u].succ.append(v)
    nodes[v].pred.append(u)
    edges_data[(u, v)] = bits
    out_degree[u] += 1


# ---------------- 主 DAG 生成函数 ----------------
def generate_multi_ue_dag(
    ue_number: int,                   # UE 数量
    n_compute_nodes_per_ue: int,       # 每个 UE 的“计算节点”数量（不含 End）
    start_count_max: int = 3,          # 每个 UE 的 start（入口节点）最大数量，实际会随机在 [1, start_count_max]
    max_in_degree: int = 3,            # 每个节点最大前驱数
    max_out_degree: int = 3,           # 每个节点最大后继数
    extra_edge_prob: float = 0.35,     # 额外加边概率，用于调节 DAG 稠密度

    # 下面全部按论文给的范围（单位：KB / cycles），内部统一转成 bits
    C_range: Tuple[float, float] = (1e8, 1e9),          # 节点计算量范围（cycles）
    upload_kb_range: Tuple[int, int] = (20, 200),       # Node.D（UE->ES 上传量）
    edge_kb_range: Tuple[int, int] = (100, 500),        # 普通依赖边数据量
    download_kb_range: Tuple[int, int] = (100, 500),    # pred(End) -> End 的 download 数据量
    seed: Optional[int] = None,
):
    """
    生成多 UE DAG，每个 UE：
      - 有 n_compute_nodes_per_ue 个“计算节点”
      - 有多个 start（pred=[] 的入口节点，数量随机）
      - 有 1 个 End 节点（C=D=0，只用于收集结果，不参与计算）
      - 所有“无后继的计算节点（sink）”都会只连接到 End
        -> 从而满足你 scheduler 中的 download 触发条件：succ=[End]

    返回：
        nodes         : Dict[node_id, Node]
        edges_data    : Dict[(u,v), 数据量(bits)]
        end_nodes     : 每个 UE 的 End 节点 id
        download_nodes: 所有会触发 download 的节点（即 End 的前驱）
        order         : 拓扑序（不包含 End 节点，可直接送给 scheduler）
    """
    if seed is not None:
        random.seed(seed)

    nodes: Dict[int, Node] = {}
    edges_data: Dict[Tuple[int, int], float] = {}
    end_nodes: List[int] = []
    download_nodes: List[int] = []

    next_id = 0
    order: List[int] = []

    for ue_id in range(ue_number):

        # --- 随机决定这个 UE 有多少个 start 节点 ---
        n_start = random.randint(1, min(start_count_max, n_compute_nodes_per_ue))

        # --- 创建计算节点 ---
        compute_ids: List[int] = []
        for _ in range(n_compute_nodes_per_ue):
            nid = next_id
            next_id += 1

            C = random.uniform(C_range[0], C_range[1])
            D_bits = rand_bits_from_kb(upload_kb_range[0], upload_kb_range[1])

            nodes[nid] = Node(
                id=nid,
                C=C,
                D=D_bits,
                pred=[],
                succ=[],
                ue_id=ue_id
            )
            compute_ids.append(nid)

        # --- 创建 End 节点（只收集，不计算） ---
        end_id = next_id
        next_id += 1
        nodes[end_id] = Node(id=end_id, C=0.0, D=0.0, pred=[], succ=[], ue_id=ue_id)
        end_nodes.append(end_id)

        # 前 n_start 个节点作为入口节点（pred=[]）
        start_ids = compute_ids[:n_start]
        non_start_ids = compute_ids[n_start:]

        out_degree = defaultdict(int)

        # --- 确保每个非 start 节点至少有 1 个前驱 ---
        for j, nid in enumerate(non_start_ids, start=n_start):
            # 可选前驱只能来自“编号在它之前的计算节点”，保证无环
            candidates = compute_ids[:j]
            pred1 = random.choice(candidates)

            add_edge(
                pred1, nid, nodes, edges_data,
                bits=rand_bits_from_kb(edge_kb_range[0], edge_kb_range[1]),
                out_degree=out_degree, max_out_degree=max_out_degree
            )

            # 随机再加一些前驱（控制最大前驱数）
            extra_candidates = [c for c in candidates if c != pred1]
            random.shuffle(extra_candidates)
            for c in extra_candidates:
                if len(nodes[nid].pred) >= max_in_degree:
                    break
                if random.random() < extra_edge_prob:
                    add_edge(
                        c, nid, nodes, edges_data,
                        bits=rand_bits_from_kb(edge_kb_range[0], edge_kb_range[1]),
                        out_degree=out_degree, max_out_degree=max_out_degree
                    )

        # --- 再随机补一些前向边，让 DAG 更均衡 ---
        extra_edges_target = max(0, n_compute_nodes_per_ue // 2)
        trials = 0
        while extra_edges_target > 0 and trials < n_compute_nodes_per_ue * 10:
            trials += 1
            u_pos = random.randint(0, n_compute_nodes_per_ue - 2)
            v_pos = random.randint(u_pos + 1, n_compute_nodes_per_ue - 1)
            u = compute_ids[u_pos]
            v = compute_ids[v_pos]

            if out_degree[u] >= max_out_degree:
                continue
            if len(nodes[v].pred) >= max_in_degree:
                continue
            if (u, v) in edges_data:
                continue

            if random.random() < extra_edge_prob:
                add_edge(
                    u, v, nodes, edges_data,
                    bits=rand_bits_from_kb(edge_kb_range[0], edge_kb_range[1]),
                    out_degree=out_degree, max_out_degree=max_out_degree
                )
                extra_edges_target -= 1

        # --- 找到所有 sink（没有后继的计算节点），让它们只连 End ---
        sinks = [nid for nid in compute_ids if len(nodes[nid].succ) == 0]
        if not sinks:
            sinks = [compute_ids[-1]]

        for s in sinks:
            add_edge(
                s, end_id, nodes, edges_data,
                bits=rand_bits_from_kb(download_kb_range[0], download_kb_range[1]),
                out_degree=out_degree, max_out_degree=max_out_degree
            )
            download_nodes.append(s)

        # 当前 UE 的拓扑序（因为只从小下标连到大下标）
        order.extend(compute_ids)

    return nodes, edges_data, end_nodes, download_nodes, order

def assert_consistency(
    nodes: Dict[int, Node],
    edges_data: Dict[Tuple[int, int], float],
    end_nodes: List[int],
    ue_number: Optional[int] = None,
) -> None:
    """
    DAG 生成完成后的强一致性检查（任何不合法直接抛异常）：

    检查内容：
    1）edges_data / pred / succ 三者是否完全一致
    2）是否存在自环、重复边、缺失节点
    3）是否存在跨 UE 的计算依赖
       - 计算节点 → 计算节点：必须属于同一 UE
       - 计算节点 → End：必须连到自身 UE 的 End
    4）Download 语义是否满足：
       - 只要连了 End，则 succ 必须且只能是 [End]
    5）图必须是 DAG（无环，拓扑排序可完成）
    6）End 节点合法性检查（C=0，D=0，无后继）
    """
    if ue_number is None:
        ue_number = len(end_nodes)

    # ---------- 基础合法性 ----------
    assert isinstance(nodes, dict) and len(nodes) > 0, "nodes 必须是非空字典"
    assert isinstance(edges_data, dict), "edges_data 必须是字典"
    assert isinstance(end_nodes, list) and len(end_nodes) == ue_number, \
        f"end_nodes 数量 ({len(end_nodes)}) 与 ue_number ({ue_number}) 不一致"

    node_ids: Set[int] = set(nodes.keys())
    end_set: Set[int] = set(end_nodes)
    assert len(end_set) == len(end_nodes), "end_nodes 中存在重复的 End 节点"

    # UE -> End 节点映射
    end_by_ue = {ue: end_nodes[ue] for ue in range(ue_number)}

    # ---------- End 节点检查 ----------
    for ue, end_id in end_by_ue.items():
        assert end_id in nodes, f"未找到 UE={ue} 的 End 节点 {end_id}"
        n = nodes[end_id]
        assert n.ue_id == ue, f"End 节点 {end_id} 的 ue_id={n.ue_id}，但期望为 {ue}"
        assert n.C == 0.0 and n.D == 0.0, \
            f"End 节点 {end_id} 的 C、D 应为 0，实际为 C={n.C}, D={n.D}"
        assert len(n.succ) == 0, f"End 节点 {end_id} 不应有后继，但发现 succ={n.succ}"

    # ---------- 边级别检查（edges_data 必须与 pred / succ 一致） ----------
    for (u, v), w in edges_data.items():
        assert u in node_ids and v in node_ids, \
            f"边 ({u} -> {v}) 的端点不存在于 nodes 中"
        assert u != v, f"检测到自环边 ({u} -> {v})"
        assert w is not None and w >= 0, \
            f"边 ({u} -> {v}) 的数据量非法：{w}"

        assert v in nodes[u].succ, \
            f"edges_data 中存在 ({u}->{v})，但 v 不在 nodes[{u}].succ 中"
        assert u in nodes[v].pred, \
            f"edges_data 中存在 ({u}->{v})，但 u 不在 nodes[{v}].pred 中"

        u_is_end = u in end_set
        v_is_end = v in end_set

        # End 节点不能作为边的起点
        if u_is_end:
            raise AssertionError(f"非法边：End 节点 {u} 不能作为起点 ({u}->{v})")

        # 指向 End 的边检查
        if v_is_end:
            assert v == end_by_ue[nodes[u].ue_id], \
                f"节点 {u}(ue={nodes[u].ue_id}) 错误地连到了 End {v}"
        else:
            # 计算节点之间禁止跨 UE 依赖
            assert nodes[u].ue_id == nodes[v].ue_id, \
                f"检测到跨 UE 计算依赖：{u}(ue={nodes[u].ue_id}) -> {v}(ue={nodes[v].ue_id})"

    # ---------- pred / succ 内部一致性 ----------
    for nid, n in nodes.items():
        if len(n.pred) != len(set(n.pred)):
            raise AssertionError(f"节点 {nid} 的 pred 中存在重复项：{n.pred}")
        if len(n.succ) != len(set(n.succ)):
            raise AssertionError(f"节点 {nid} 的 succ 中存在重复项：{n.succ}")

        for p in n.pred:
            assert p in node_ids, f"节点 {nid} 的 pred {p} 不存在"
            assert (p, nid) in edges_data, \
                f"缺少 pred 边对应的 edges_data：({p}->{nid})"

        for s in n.succ:
            assert s in node_ids, f"节点 {nid} 的 succ {s} 不存在"
            assert (nid, s) in edges_data, \
                f"缺少 succ 边对应的 edges_data：({nid}->{s})"

        # Download 语义：一旦连 End，就只能连 End
        succ_end = [x for x in n.succ if x in end_set]
        if len(succ_end) > 0:
            assert len(succ_end) == 1, \
                f"节点 {nid} 同时连到了多个 End：{succ_end}"
            assert len(n.succ) == 1, \
                f"节点 {nid} 已连 End，但仍存在其他 succ：{n.succ}"
            assert succ_end[0] == end_by_ue[n.ue_id], \
                f"节点 {nid}(ue={n.ue_id}) 错误地连到了 End {succ_end[0]}"

    # ---------- DAG 无环性检查（拓扑排序） ----------
    indeg = {nid: 0 for nid in node_ids}
    adj = defaultdict(list)
    for (u, v) in edges_data.keys():
        adj[u].append(v)
        indeg[v] += 1

    q = deque([nid for nid, d in indeg.items() if d == 0])
    visited = 0

    while q:
        x = q.popleft()
        visited += 1
        for y in adj.get(x, []):
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)

    assert visited == len(node_ids), \
        f"检测到环或非法依赖：拓扑排序仅访问 {visited}/{len(node_ids)} 个节点"

    # 全部检查通过
    return

def make_dag_case(
    ue_number: int,
    n_compute_nodes_per_ue: int,
    start_count_max: int,
    seed: int,
) -> DAGCase:
    nodes, edges_data, end_nodes, download_nodes, _ = generate_multi_ue_dag(
        ue_number=ue_number,
        n_compute_nodes_per_ue=n_compute_nodes_per_ue,
        start_count_max=start_count_max,
        seed=seed,
    )

    return DAGCase(nodes=nodes, edges_data=edges_data, end_nodes=end_nodes, download_nodes=download_nodes)