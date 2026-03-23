from dataclasses import dataclass
from typing import Set, Dict, List, Tuple, Optional

import numpy as np
from gensim.models import FastText

from GNNDAG.DTO_scheduler import Node
import gymnasium as gym
from gymnasium import spaces

from cp_utils import compute_cp, normalize_cp

@dataclass
class StepInfo:
    # 用于日志和调试
    scheduled_node: int
    ue_id: int
    chosen_loc_id: int
    chosen_loc_ue_id: Optional[int]
    chosen_processor_id: int
    finish_time: float
    makespan_by_ue: Dict[int, float]

class DTOEnv(gym.Env):
    def __init__(
        self,
        scheduler,
        *,
        reward_oracle: str = "local",
        reward_scale: bool = False,
    ):
        """
        reward_oracle: "local" (baseline 全部本地) | "greedy" (每步选最优 loc)
        reward_scale: 是否对 reward 做相对缩放 (prev-curr)/max(prev, 1e-9)
        """
        super(DTOEnv).__init__()

        self.scheduler = scheduler
        self.es_numbers = scheduler.es_numbers
        self.reward_oracle = reward_oracle
        self.reward_scale = reward_scale

        """以下为内部状态(内部频繁更改)"""
        self._unscheduled: set[int] = set()
        self._pred_remaining: Dict[int, int] = {}
        self._ready_list: List[int] = []

        # 一张dag用的值 不随reset而变化
        self.end_nodes = set(getattr(scheduler, "end_nodes", []))

        # 此处这三者都是占位用 数值在reset中会更新
        self.node_ids = sorted(nid for nid in self.scheduler.nodes.keys() if nid not in self.end_nodes)

        self.N = len(self.node_ids)

        # 构造二维矩阵 给每个坐标编号
        self.action_space = spaces.Discrete(self.N * (self.es_numbers + 1))

        # ===== 构造观测空间（论文 6 维节点特征: Ci, Di, in_degree, out_degree, loci, avai）=====
        self.observation_space = spaces.Dict({
            # ---- nodes (论文 6 维 + ue_id 用于候选位置构建) ----
            "nodes_C": spaces.Box(0, np.inf, shape=(self.N,), dtype=np.float32),
            "nodes_D": spaces.Box(0, np.inf, shape=(self.N,), dtype=np.float32),
            "nodes_ue_id": spaces.Box(0, self.scheduler.ue_numbers - 1, shape=(self.N,), dtype=np.int64),
            "nodes_in_degree": spaces.Box(0, self.N, shape=(self.N,), dtype=np.int64),
            "nodes_out_degree": spaces.Box(0, self.N, shape=(self.N,), dtype=np.int64),
            "nodes_loc": spaces.Box(-1, self.es_numbers, shape=(self.N,), dtype=np.int64),
            "nodes_ava": spaces.Box(0, 1, shape=(self.N,), dtype=np.int8),

            # ---- locations ----
            "loc_cpu_speed": spaces.Box(0, np.inf, shape=(len(self.scheduler.locations),), dtype=np.float32),
            "loc_min_processor_EAT": spaces.Box(0, np.inf, shape=(len(self.scheduler.locations),), dtype=np.float32),
            "loc_num_processor": spaces.Box(1, np.inf, shape=(len(self.scheduler.locations),), dtype=np.int64),

            # ---- ue ----
            "ue_upload_EAT": spaces.Box(0, np.inf, shape=(self.scheduler.ue_numbers,), dtype=np.float32),
            "ue_download_EAT": spaces.Box(0, np.inf, shape=(self.scheduler.ue_numbers,), dtype=np.float32),

            # ---- 邻接矩阵（喂给GAT） ----
            "adj": spaces.Box(0, 1, shape=(self.N, self.N), dtype=np.int8),
            # ---- edge ----
            "edge_attr": spaces.Box(0, np.inf, shape=(self.N, self.N), dtype=np.float32),
            # ---- Critical Path 归一化特征 ----
            "nodes_cp": spaces.Box(0, 1, shape=(self.N,), dtype=np.float32),
        })

        self.prev_mean_eft = 0.0

    def build_action_mask(self):
        """
        联合mask
        规定action的限制
        - node (N,) True表示节点可选
        - loc (N, M+1) True表示节点在当前位置合法 (UE=0 ES=1...M+1)
        """
        scheduler = self.scheduler
        # 呈现形式为 {真实id: 索引id}
        id_to_index = {nid: i for i, nid in enumerate(self.node_ids)}
        N = len(self.node_ids)

        # ES的数量
        M = scheduler.es_numbers

        node_mask = [False] * N
        loc_mask = [[False] * (M + 1) for _ in range(N)]

        for nid in self._ready_list:
            if nid not in id_to_index:
                continue  # end 或被过滤的，不该出现在obs
            i = id_to_index[nid]
            node_mask[i] = True
            for j in range(M + 1):
                loc_mask[i][j] = True

        return {"node": node_mask, "loc": loc_mask}

    def get_ready_node_mask(self):
        """
        two-stage使用的mask限制
        """
        mask = np.zeros(self.N, dtype=bool)
        ready_set = set(self._ready_list)

        for idx, node_id in enumerate(self.node_ids):
            if node_id in ready_set:
                mask[idx] = True

        return mask

    def get_loc_mask_for_node(self, node_id: int):
        """
        two-stage PPO 使用：
        在给定 node_id 后，返回该节点可选 loc 的 mask
        当前版本先默认所有 loc_choice 都合法
        长度为 es_numbers + 1
        """
        if node_id not in self.node_ids:
            raise ValueError(f"Invalid node_id={node_id}")

        if node_id not in self._ready_list:
            raise ValueError(f"Node {node_id} is not ready")

        return np.ones(self.es_numbers + 1, dtype=bool)

    def build_obs(self, nodes: Dict[int, Node]) -> Dict:
        """观测环境，节点特征按论文: Ci, Di, in_degree, out_degree, loci, avai"""
        scheduler = self.scheduler

        # ---------- nodes ------------
        nodes_ids = self.node_ids
        N = len(nodes_ids)
        adj = self.adj  # (N,N) adj[i,j]=1 表示边 i->j

        C = []
        D = []
        ue_id = []
        in_degree = []
        out_degree = []
        loc = []
        ava = []

        for idx, node_id in enumerate(nodes_ids):
            node = nodes[node_id]
            C.append(node.C)
            D.append(node.D)
            ue_id.append(node.ue_id)

            # 论文: in-degree 指向该节点的边数, out-degree 从该节点出发的边数
            in_degree.append(int(np.sum(adj[:, idx])))
            out_degree.append(int(np.sum(adj[idx, :])))

            # loci: 已分配则为 loc.id，未分配为 -1
            loc_obj = scheduler.node_loc.get(node_id, None)
            loc.append(-1 if loc_obj is None else loc_obj.id)

            # avai: 1 表示可调度(ready)，0 表示不可
            ava.append(1 if node_id in self._ready_list else 0)

        # ---------- Locations ------------
        cpu_speed = []
        min_processor_EAT = []
        num_processor = []

        for loc_obj in scheduler.locations:
            cpu_speed.append(loc_obj.cpu_speed)
            min_processor_EAT.append(min(p.EAT for p in loc_obj.processors))
            num_processor.append(len(loc_obj.processors))

        # ---------- ue queues ------------
        upload_EAT = [scheduler.upload_EAT[ue] for ue in range(scheduler.ue_numbers)]
        download_EAT = [scheduler.download_EAT[ue] for ue in range(scheduler.ue_numbers)]

        observation = {
            # ===== nodes（论文 6 维: Ci, Di, in_deg, out_deg, loci, avai）=====
            "nodes_C": np.asarray(C, dtype=np.float32),
            "nodes_D": np.asarray(D, dtype=np.float32),
            "nodes_ue_id": np.asarray(ue_id, dtype=np.int64),
            "nodes_in_degree": np.asarray(in_degree, dtype=np.int64),
            "nodes_out_degree": np.asarray(out_degree, dtype=np.int64),
            "nodes_loc": np.asarray(loc, dtype=np.int64),
            "nodes_ava": np.asarray(ava, dtype=np.int8),

            # ===== locations =====
            "loc_cpu_speed": np.asarray(cpu_speed, dtype=np.float32),
            "loc_min_processor_EAT": np.asarray(min_processor_EAT, dtype=np.float32),
            "loc_num_processor": np.asarray(num_processor, dtype=np.int64),

            # ===== ue =====
            "ue_upload_EAT": np.asarray(upload_EAT, dtype=np.float32),
            "ue_download_EAT": np.asarray(download_EAT, dtype=np.float32),

            # 图结构特征
            "adj": self.adj,
            "edge_attr": self.edge_attr,
            # Critical Path 归一化
            "nodes_cp": self.nodes_cp,
        }

        return observation


    def reset(self, *, seed=None, options=None):
        """
        function:
            每个episode需要重置的变量
            end_nodes
            prev_makespan
            _unscheduled
            _pred_remaining
            _ready_list
            aft
            node_loc
            upload_EAT
            download_EAT
            processor.EAT
            trace
            download_trace
            _step_counter
        """
        scheduler = self.scheduler

        # 未调度节点初始化 去除end_nodes
        self._unscheduled = {node for node in scheduler.nodes.keys() if node not in self.end_nodes}

        # 统计节点的pred是否全都完成
        self._pred_remaining = {}
        for node_id in self._unscheduled:
            # 避免end的前驱是另一个end
            pred = [p for p in scheduler.nodes[node_id].pred if p not in self.end_nodes]
            self._pred_remaining[node_id] = len(pred)

        self._ready_list = [node_id for node_id in self._unscheduled if self._pred_remaining[node_id] == 0]
        self._ready_list.sort()

        # 以下为 单个episode 需要清空的变量
        scheduler.aft.clear()
        scheduler.node_loc.clear()

        # ue queues
        for ue in scheduler.upload_EAT:
            scheduler.upload_EAT[ue] = 0.0
            scheduler.download_EAT[ue] = 0.0

        # processors (❗关键)
        for loc in scheduler.locations:
            for p in loc.processors:
                p.EAT = 0.0

        # traces / counters
        scheduler.trace.clear()
        scheduler.download_trace.clear()

        # 初始化eft 即全部本地部署的特殊情况
        self.prev_mean_eft = self.scheduler.estimate_complete_mean_eft_by_copy(
            unscheduled=self._unscheduled,
        )

        # ===== 每个 episode 换 DAG：重建 node_ids / adj =====
        self.end_nodes = set(getattr(scheduler, "end_nodes", []))
        self.node_ids = sorted(nid for nid in scheduler.nodes.keys() if nid not in self.end_nodes)
        self.N = len(self.node_ids)

        self.id2idx = {nid: i for i, nid in enumerate(self.node_ids)}

        adj = np.zeros((self.N, self.N), dtype=np.int8)
        for u in self.node_ids:
            ui = self.id2idx[u]
            for v in scheduler.nodes[u].succ:
                if v in self.id2idx:
                    vi = self.id2idx[v]
                    adj[ui, vi] = 1

        edge_attr = np.zeros((self.N, self.N), dtype=np.float32)

        for (u, v), bits in scheduler.edges_data.items():
            if u in self.id2idx and v in self.id2idx:
                ui = self.id2idx[u]
                vi = self.id2idx[v]
                edge_attr[ui, vi] = float(bits)

        self.adj = adj
        self.edge_attr = edge_attr

        # Critical Path 计算并归一化 (CP/max_CP)
        cp_raw = compute_cp(
            node_ids=self.node_ids,
            id2idx=self.id2idx,
            nodes=scheduler.nodes,
            edges_data=scheduler.edges_data,
            locations=scheduler.locations,
            exec_model=scheduler.exec_model,
            trans_model=scheduler.trans_model,
        )
        cp_norm = normalize_cp(cp_raw, method="max")
        self.nodes_cp = np.asarray([cp_norm.get(i, 0.0) for i in range(self.N)], dtype=np.float32)

        observation = self.build_obs(scheduler.nodes)

        return observation, {}

    # 给外部调用使用
    def ready_nodes(self) -> List[int]:
        return list(self._ready_list)

    def done(self) -> bool:
        return len(self._unscheduled) == 0

    # 名字不能修改 专给PPO用
    def action_masks(self):
        K = self.es_numbers + 1
        mask = [False] * (self.N * K)

        for node_id in self._ready_list:
            node_index = self.node_ids.index(node_id)
            for loc_choice in range(K):
                a = node_index * K + loc_choice
                mask[a] = True

        return mask

    def step(self, action: int):
        """

        step分为两部分
        一种是走联合编码的action 也就是作为一个latent传给PPO
        另一种就是我们的two-stage PPO 此时action无需decode 直接传就行

        """
        if self.done():
            raise RuntimeError("episode已结束 此step应停止 reset()进行重置")

        # action已经编号成了整数形式 采回来的时候映射成我们理解的索引方便处理
        K = self.es_numbers + 1

        node_index = action // K
        loc_choice = action % K

        if not (0 <= node_index < self.N):
            raise ValueError(f"Invalid action={action}: node_index={node_index} out of range")

        node_id = self.node_ids[node_index]

        return self.step_with_decision(node_id, loc_choice)

    def _compute_reward(self):
        if self.reward_oracle == "greedy":
            mean_eft = self.scheduler.estimate_complete_mean_eft_by_copy_greedy(self._unscheduled)
        else:
            mean_eft = self.scheduler.estimate_complete_mean_eft_by_copy(self._unscheduled)

        reward = self.prev_mean_eft - mean_eft
        if self.reward_scale:
            scale = max(self.prev_mean_eft, 1e-9)
            reward = reward / scale
        self.prev_mean_eft = mean_eft
        return reward

    def _apply_decision(self, node_id: int, loc_choice: int):
        loc, processor, finish_time = self.scheduler.schedule_node_at(node_id, loc_choice)

        self._unscheduled.remove(node_id)

        for succ_id in self.scheduler.nodes[node_id].succ:
            if succ_id in self._pred_remaining:
                self._pred_remaining[succ_id] -= 1
                if self._pred_remaining[succ_id] == 0:
                    self._ready_list.append(succ_id)

        if node_id in self._ready_list:
            self._ready_list.remove(node_id)

        self._ready_list.sort()

        info = StepInfo(
            scheduled_node=node_id,
            ue_id=self.scheduler.nodes[node_id].ue_id,
            chosen_loc_id=loc.id,
            chosen_loc_ue_id=loc.ue_id,
            chosen_processor_id=processor.id,
            finish_time=finish_time,
            makespan_by_ue=dict(self.scheduler.download_EAT),
        )
        return info

    def step_with_decision(self, node_id: int, loc_choice: int):
        """
        根据选择的node_id和loc_choice直接决策
        two-stage PPO使用
        """
        if self.done():
            raise RuntimeError("Episode is done. Please reset() before calling step().")

        if node_id not in self._ready_list:
            raise ValueError(f"Node {node_id} is not ready")

        info = self._apply_decision(node_id=node_id, loc_choice=loc_choice)
        reward = self._compute_reward()

        observation = self.build_obs(self.scheduler.nodes)
        terminated = self.done()
        truncated = False

        return observation, reward, terminated, truncated, {"step_info": info}

    def step_greedy(self, action: int):
        # 对于greedy来说 输入只有node_id
        # 所以就委屈一下把id当成action了
        # 按理来说runner里面应该负责location的选择
        # 但如果用了greedy 相当于在这里直接处理掉了 自主分配location
        node_id = action
        loc, finish_time, processor = self.scheduler.schedule_node(node_id)

        self._unscheduled.remove(node_id)

        # 更新节点succ对应的pred_remaining
        for s in self.scheduler.nodes[node_id].succ:
            if s in self._unscheduled:
                self._pred_remaining[s] -= 1
                if self._pred_remaining[s] == 0:
                    self._ready_list.append(s)

        self._ready_list.remove(node_id)
        # 用于调用的稳定性 实际ready list的顺序并不重要
        self._ready_list.sort()

        makespan = max(self.scheduler.aft.values())
        reward = self.prev_makespan - makespan
        self.prev_makespan = makespan

        info = StepInfo(scheduled_node=node_id,
                        ue_id=self.scheduler.nodes[node_id].ue_id,
                        chosen_loc_id=loc.id,
                        chosen_loc_ue_id=loc.ue_id,
                        chosen_processor_id=processor.id,
                        finish_time=finish_time,
                        makespan_by_ue=dict(self.scheduler.download_EAT)
                        )
        # print(info)
        # print("makespan", makespan)
        # print("reward", reward)

        observation = self.build_obs(self.scheduler.nodes)

        terminated = self.done()

        truncated = False

        return observation, reward, terminated, truncated, {"step_info": info}
