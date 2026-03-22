import copy
from typing import Dict, List, Tuple
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TraceEntry:
    # identity
    step: int
    node_id: int
    ue_id: int

    # decision
    loc_id: int
    loc_ue_id: Optional[int]  # local 才有
    cpu_speed: float
    processor_id: int

    # times (all in seconds)
    uplink_EAT_before: float
    upload_time: float
    upload_finish: float

    dependency_ready: float
    exec_ready: float

    start_time: float
    finish_time: float

    # convenience
    did_upload: bool
    is_local: bool


@dataclass
class Node:
    id: int
    C: float  # computation cycles
    D: float  # 节点上传数据量
    pred: List[int]
    succ: List[int]
    ue_id: int  # 记录节点属于哪个UE


@dataclass
class Processor:
    id: int
    EAT: float = 0.0


@dataclass
class Location:
    id: int
    # 表示本地/es
    # 本地：0 es：按照es编号
    # 服从于本地系统
    ue_id: int | None
    # ue编号
    # es则无该值
    # 服从整个坐标体系
    cpu_speed: float
    processors: list[Processor]  # 单个ue/es的若干个processors ue即为1个


class ExecutionModel:
    def exec_time(self, node: Node, loc: Location):
        return node.C / loc.cpu_speed


class UploadModel:
    def __init__(self, transmission_rate: float, local: int = 0):
        self.transmission_rate = transmission_rate
        self.local = local

    def upload_time(self, node: Node, loc: Location):
        if loc.id == self.local:  # 判断是否本地
            return 0.0
        return node.D / self.transmission_rate


class TransmissionModel:
    def __init__(self,
                 ue_es_transmission_rate: float,
                 es_es_transmission_rate
                 ):
        self.ue_es_transmission_rate = ue_es_transmission_rate
        self.es_es_transmission_rate = es_es_transmission_rate

    def trans_time(self, u: int, v: int, loc_u: Location, loc_v: Location,
                   edge_data: Dict[Tuple[int, int], float]) -> float:
        # 前驱es则id相同即可
        # 如果本地 判断ue_id是否相同
        if loc_u.id != 0:
            if loc_u.id == loc_v.id:
                return 0.0
        if loc_u.id == 0:
            if loc_u.ue_id == loc_v.ue_id:
                return 0.0

        data_u_v = edge_data[(u, v)]

        if (u, v) not in edge_data:
            raise KeyError(f"Missing edge_data for ({u},{v})")

        # 只要其中一端是UE 用ue_es 否则es_es
        if (loc_u.ue_id is not None) or (loc_v.ue_id is not None):
            return data_u_v / self.ue_es_transmission_rate
        return data_u_v / self.es_es_transmission_rate


class DTOScheduler:
    """
    分成两部分：
    1. 当前node分配到location
    2. 预测整个dag完成的EFT（用于计算reward）
    """
    def __init__(
            self,
            nodes: Dict[int, Node],
            edges_data: Dict[Tuple[int, int], float],
            locations: List[Location],
            exec_model: ExecutionModel,
            upload_model: UploadModel,
            trans_model: TransmissionModel,
            end_nodes: List[int],
            download_nodes: List[int],
            ue_number: int,
            es_number: int
    ):
        self.nodes = nodes
        self.edges_data = edges_data
        self.locations = locations
        self.exec_model = exec_model
        self.upload_model = upload_model
        self.trans_model = trans_model
        self.end_nodes = end_nodes
        # 创建时确定ue的总数
        self.ue_numbers = ue_number
        # es总数
        self.es_numbers = es_number
        # download_nodes存储所有需要download的节点
        self.download_nodes = download_nodes

        """以下为调度器自用变量"""
        self.aft: Dict[int, float] = {}
        self.node_loc: Dict[int, Location] = {}

        # 检查download节点时 需要检查end所在ue
        self.ue_loc = {}
        for loc in self.locations:
            if loc.id == 0:
                self.ue_loc[loc.ue_id] = loc

        # 为每条ue建立独立的upload和download
        # 记得ue_id需要从0开始
        self.upload_EAT: Dict[int, float] = {ue: 0.0 for ue in range(self.ue_numbers)}
        self.download_EAT: Dict[int, float] = {ue: 0.0 for ue in range(self.ue_numbers)}

        for loc in self.locations:
            for p in loc.processors:
                p.EAT = 0.0

        """以下为调度器自检变量"""
        self.trace: list[TraceEntry] = []
        self._step_counter = 0
        self.download_trace: list[tuple[int, int, float, float]] = []
        # (ue_id, node_id, download_start, download_finish)

    def physical_eat(self, loc, node, node_id, ue_id):
        """

        基本物理逻辑
        无论什么卸载方案都需要满足如下条件
        即upload队列完成 前驱节点完成 当前节点计算完成
        max(upload_finish, dependency_ready, exec_ready)

        """
        # 1) 上传队列
        # 如果本地处理则不参与max约束 但又不能直接清零 因为一个uplink上会处理多个nodes
        # 所以upload_finish放到后面 分类处理
        upload_time = self.upload_model.upload_time(node, loc)
        uplink_EAT = self.upload_EAT[ue_id]
        # 2) 依赖最晚完成
        dependency_ready = 0.0
        for pred in node.pred:
            loc_pred = self.node_loc[pred]
            # 无线链路 不考虑传输链路的条数（本文的建模仅有一条upload 一条download 不考虑传输的条数）
            transmission_time = self.trans_model.trans_time(pred, node_id, loc_pred, loc, self.edges_data)
            dependency_ready = max(dependency_ready, self.aft[pred] + transmission_time)
        # 3) 当前loc什么时候空闲
        # 每个es中有若干个处理器 取所有处理器的最小完成时间即可
        processor = min(loc.processors, key=lambda p: p.EAT)
        exec_ready = processor.EAT
        # 4) 获取最早分配时间
        if upload_time == 0.0:
            start_time = max(dependency_ready, exec_ready)
            upload_finish = uplink_EAT
            did_upload = False
        else:
            upload_finish = uplink_EAT + upload_time
            start_time = max(upload_finish, dependency_ready, exec_ready)
            did_upload = True
        exec_time = self.exec_model.exec_time(node, loc)
        finish_time = start_time + exec_time
        return dependency_ready, did_upload, exec_ready, finish_time, processor, start_time, uplink_EAT, upload_finish, upload_time

    def download_function(self, location, node, node_id, ue_id, download_trace_open):
        """
        判断当前节点为end的前驱时
        直接进入download队列计算传输
        原因是这个end仅用于收集数据而非计算
        """
        if len(node.succ) == 1 and node.succ[0] in self.end_nodes:
            start_id = node_id
            end_id = node.succ[0]
            start_loc = self.node_loc[start_id]
            end_loc = self.ue_loc[ue_id]

            download_start = max(self.aft[node_id], self.download_EAT[ue_id])
            # 在本地ue执行无需download
            if location.id == 0:
                download_time = 0
            else:
                download_time = self.trans_model.trans_time(start_id, end_id, start_loc, end_loc, self.edges_data)
            download_finish = download_start + download_time

            self.aft[node_id] = max(self.aft[node_id], download_finish)

            if download_trace_open:
                self.download_trace.append((ue_id, node_id, download_start, download_finish))

            self.download_EAT[ue_id] = download_finish

    def action_mapping(self, ue_id: int, loc_choice: int) -> Location:
        """
        function：
        将输入的action(loc_choice)转化为scheduler可以识别的Location(dataclass)
        主要是方便处理
        """
        if loc_choice == 0:
            return self.ue_loc[ue_id]

        for loc in self.locations:
            if loc.id == loc_choice and loc.ue_id is None:
                return loc

        raise ValueError(f"Invalid loc_choice={loc_choice}. Cannot find matching ES location.")

    def schedule_node_at(self, node_id: int, loc_choice: int) -> Tuple[Location, Processor, float]:
        """
        强制将node放到指定location
        """

        node = self.nodes[node_id]
        ue_id = node.ue_id

        loc = self.action_mapping(ue_id, loc_choice)

        # 禁止跨ue
        if loc.id == 0 and loc.ue_id != ue_id:
            raise ValueError(f"Cross-UE local execution: node ue={ue_id}, loc ue_id={loc.ue_id}")

        dependency_ready, did_upload, exec_ready, finish_time, processor, start_time, uplink_EAT, upload_finish, upload_time = self.physical_eat(loc, node, node_id, ue_id)

        self.node_loc[node_id] = loc
        self.aft[node_id] = finish_time

        processor.EAT = finish_time
        if did_upload:
            self.upload_EAT[ue_id] = upload_finish

        self.download_function(loc, node, node_id, ue_id, download_trace_open=False)

        return loc, processor, finish_time

    def estimate_complete_mean_eft_by_copy(self, unscheduled):
        """
        论文 baseline：对剩余节点全部使用本地执行 (loc=0) 预估 mean_eft
        """
        scheduler_copy = copy.deepcopy(self)
        unscheduled_copy = set(unscheduled)

        while unscheduled_copy:
            for node_id in list(unscheduled_copy):
                preds = [p for p in scheduler_copy.nodes[node_id].pred if p not in scheduler_copy.end_nodes]
                if all(p in scheduler_copy.aft for p in preds):
                    scheduler_copy.schedule_node_at(node_id, 0)
                    unscheduled_copy.remove(node_id)
                    break

        return sum(scheduler_copy.download_EAT.values()) / scheduler_copy.ue_numbers

    def estimate_complete_mean_eft_by_copy_greedy(self, unscheduled):
        """
        改进 oracle：对剩余节点使用 greedy（每步选 EFT 最小的 loc）预估 mean_eft
        与 schedule_node 逻辑一致，更接近真实下界
        """
        scheduler_copy = copy.deepcopy(self)
        unscheduled_copy = set(unscheduled)

        while unscheduled_copy:
            for node_id in list(unscheduled_copy):
                preds = [p for p in scheduler_copy.nodes[node_id].pred if p not in scheduler_copy.end_nodes]
                if all(p in scheduler_copy.aft for p in preds):
                    scheduler_copy.schedule_node(node_id)
                    unscheduled_copy.remove(node_id)
                    break

        return sum(scheduler_copy.download_EAT.values()) / scheduler_copy.ue_numbers


    def schedule_node(self, node_id: int):
        """
        input：
        节点编号
        node location

        function:
        判断是否为download节点
        如果需要download无需比较loc download到ue 用download_EAT即可
        正常节点遍历loc获取节点在所有location中最小的EFT位置 并分配

        greedy逻辑
        对每个节点求最小EAT的location进行分配
        """
        node = self.nodes[node_id]
        ue_id = node.ue_id

        best_finish_time = float("inf")
        best_loc = None
        best_processor = None
        best_upload_EAT = self.upload_EAT[ue_id]

        best_start_time = None
        best_dependency_ready = None
        best_exec_ready = None
        best_upload_time = None
        best_uplink_before = None
        best_did_upload = None

        for loc in self.locations:
            is_local = (loc.id == 0)
            # 禁止跨ue
            if is_local and loc.ue_id != ue_id:
                continue
            dependency_ready, did_upload, exec_ready, finish_time, processor, start_time, uplink_EAT, upload_finish, upload_time = self.physical_eat(
                loc, node, node_id, ue_id)

            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_loc = loc
                best_processor = processor

                best_start_time = start_time
                best_dependency_ready = dependency_ready
                best_exec_ready = exec_ready
                best_upload_time = upload_time
                best_uplink_before = uplink_EAT
                best_did_upload = did_upload
                best_upload_EAT = upload_finish

        self.node_loc[node_id] = best_loc

        self.aft[node_id] = best_finish_time

        best_processor.EAT = best_finish_time

        if best_did_upload:
            self.upload_EAT[ue_id] = best_upload_EAT

        self.trace.append(
            TraceEntry(
                step=self._step_counter,
                node_id=node_id,
                ue_id=ue_id,
                loc_id=best_loc.id,
                loc_ue_id=best_loc.ue_id,
                cpu_speed=best_loc.cpu_speed,
                processor_id=best_processor.id,
                uplink_EAT_before=best_uplink_before,
                upload_time=best_upload_time,
                upload_finish=best_upload_EAT if best_did_upload else best_uplink_before,
                dependency_ready=best_dependency_ready,
                exec_ready=best_exec_ready,
                start_time=best_start_time,
                finish_time=best_finish_time,
                did_upload=best_did_upload,
                is_local=(best_loc.id == 0),
            )
        )
        self._step_counter += 1

        self.download_function(best_loc, node, node_id, ue_id, download_trace_open=True)

        return best_loc, best_finish_time, best_processor