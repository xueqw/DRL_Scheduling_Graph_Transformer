# validator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
from collections import defaultdict

# 直接复用你 DTO_scheduler.py 里的类型
# from DTO_scheduler import Node, Location, TransmissionModel, DTOScheduler, TraceEntry

"""
检查调度器的功能

禁止跨 UE 本地执行
遵守 DAG 拓扑顺序
依赖传输必须等待
传输速率计算正确
执行完成时间正确
处理器不允许时间重叠
本地任务不占用 uplink
UE 的 uplink 必须串行
Download 只能来自 End 的前驱
UE 的 downlink 必须串行
makespan 定义一致
"""
EPS = 1e-9

@dataclass
class ValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message

class DTOValidator:
    def __init__(self, eps: float = EPS, strict_download_nodes: bool = True):
        self.eps = eps
        self.strict_download_nodes = strict_download_nodes

    def validate(self, scheduler) -> None:
        """
        scheduler: 你的 DTOScheduler 实例
        需要它具备：
          - nodes, edges_data, trans_model
          - aft, node_loc
          - trace: List[TraceEntry]
          - download_trace: List[(ue_id, node_id, start, finish)]
          - end_nodes, download_nodes（可选严格校验）
        """
        self._validate_trace_fields_present(scheduler)
        self._validate_local_cross_ue(scheduler)
        self._validate_finish_time_consistency(scheduler)
        self._validate_dependency_constraints(scheduler)
        self._validate_processor_no_overlap(scheduler)
        self._validate_uplink_serial(scheduler)
        self._validate_downlink_serial(scheduler)
        if self.strict_download_nodes:
            self._validate_download_trigger(scheduler)

    def _fail(self, msg: str) -> None:
        raise ValidationError(msg)

    def _validate_trace_fields_present(self, scheduler) -> None:
        if not hasattr(scheduler, "trace"):
            self._fail("DTOScheduler missing attribute `trace`. Add TraceEntry logging in schedule_node().")
        if not hasattr(scheduler, "download_trace"):
            self._fail("DTOScheduler missing attribute `download_trace`. Add download event logging.")

    def _validate_local_cross_ue(self, scheduler) -> None:
        for e in scheduler.trace:
            if e.is_local:
                if e.loc_ue_id != e.ue_id:
                    self._fail(
                        f"[Cross-UE local] step={e.step} node={e.node_id} ue={e.ue_id} "
                        f"scheduled to local loc_ue={e.loc_ue_id}"
                    )

    def _validate_finish_time_consistency(self, scheduler) -> None:
        nodes = scheduler.nodes
        for e in scheduler.trace:
            C = nodes[e.node_id].C
            expected = e.start_time + (C / e.cpu_speed if e.cpu_speed != 0 else 0.0)
            if abs(e.finish_time - expected) > self.eps:
                self._fail(
                    f"[Finish mismatch] step={e.step} node={e.node_id}: "
                    f"finish={e.finish_time:.9f} expected={expected:.9f} "
                    f"(start={e.start_time:.9f}, C={C}, cpu={e.cpu_speed})"
                )

    def _validate_dependency_constraints(self, scheduler) -> None:
        nodes = scheduler.nodes
        trans_model = scheduler.trans_model
        edges_data = scheduler.edges_data
        aft = scheduler.aft
        node_loc = scheduler.node_loc

        # 需要假设：pred 已经调度过（拓扑序）
        for e in scheduler.trace:
            for p in nodes[e.node_id].pred:
                if p not in aft or p not in node_loc:
                    self._fail(
                        f"[Topo/order] node={e.node_id} scheduled before pred={p} was scheduled."
                    )
                t = trans_model.trans_time(p, e.node_id, node_loc[p], node_loc[e.node_id], edges_data)
                ready = aft[p] + t
                if e.start_time + self.eps < ready:
                    self._fail(
                        f"[Dependency violated] step={e.step} node={e.node_id} pred={p}: "
                        f"start={e.start_time:.9f} < aft[p]+trans={ready:.9f} "
                        f"(aft[p]={aft[p]:.9f}, trans={t:.9f})"
                    )

    def _validate_processor_no_overlap(self, scheduler) -> None:
        # key: (loc_id, loc_ue_id, proc_id)
        intervals = defaultdict(list)
        for e in scheduler.trace:
            key = (e.loc_id, e.loc_ue_id, e.processor_id)
            intervals[key].append((e.start_time, e.finish_time, e))

        for key, segs in intervals.items():
            segs.sort(key=lambda x: x[0])
            for i in range(1, len(segs)):
                prev_s, prev_f, prev_e = segs[i - 1]
                cur_s, cur_f, cur_e = segs[i]
                if cur_s + self.eps < prev_f:
                    self._fail(
                        f"[Processor overlap] loc={key[0]} loc_ue={key[1]} proc={key[2]}: "
                        f"prev(node={prev_e.node_id}, {prev_s:.9f}-{prev_f:.9f}) "
                        f"cur(node={cur_e.node_id}, {cur_s:.9f}-{cur_f:.9f})"
                    )

    def _validate_uplink_serial(self, scheduler) -> None:
        # 每个 UE uplink 只对 did_upload 的节点形成区间 [uplink_before, upload_finish]
        by_ue = defaultdict(list)
        for e in scheduler.trace:
            if e.did_upload:
                by_ue[e.ue_id].append((e.uplink_EAT_before, e.upload_finish, e))

        for ue, segs in by_ue.items():
            segs.sort(key=lambda x: x[0])
            for i in range(1, len(segs)):
                prev_s, prev_f, prev_e = segs[i - 1]
                cur_s, cur_f, cur_e = segs[i]
                if cur_s + self.eps < prev_f:
                    self._fail(
                        f"[Uplink overlap] ue={ue}: "
                        f"prev(node={prev_e.node_id}, {prev_s:.9f}-{prev_f:.9f}) "
                        f"cur(node={cur_e.node_id}, {cur_s:.9f}-{cur_f:.9f})"
                    )
            # 额外：本地不应产生 uplink 区间
        for e in scheduler.trace:
            if e.is_local and (abs(e.upload_time) > self.eps):
                self._fail(f"[Local upload_time nonzero] step={e.step} node={e.node_id} upload_time={e.upload_time}")

    def _validate_downlink_serial(self, scheduler) -> None:
        by_ue = defaultdict(list)
        for (ue, node, s, f) in getattr(scheduler, "download_trace", []):
            by_ue[ue].append((s, f, node))
        for ue, segs in by_ue.items():
            segs.sort(key=lambda x: x[0])
            for i in range(1, len(segs)):
                prev_s, prev_f, prev_node = segs[i - 1]
                cur_s, cur_f, cur_node = segs[i]
                if cur_s + self.eps < prev_f:
                    self._fail(
                        f"[Downlink overlap] ue={ue}: "
                        f"prev(node={prev_node}, {prev_s:.9f}-{prev_f:.9f}) "
                        f"cur(node={cur_node}, {cur_s:.9f}-{cur_f:.9f})"
                    )

    def _validate_download_trigger(self, scheduler) -> None:
        # 可选严格：download_trace 里的 node 必须满足 Pred(End)
        nodes = scheduler.nodes
        end_nodes = set(getattr(scheduler, "end_nodes", []))
        download_nodes = set(getattr(scheduler, "download_nodes", []))

        for (ue, node_id, s, f) in scheduler.download_trace:
            # 1) node_id 是否在 download_nodes（如果你维护了这个集合）
            if download_nodes and node_id not in download_nodes:
                self._fail(f"[Download node mismatch] ue={ue} node={node_id} not in scheduler.download_nodes")

            # 2) 是否满足 len(succ)==1 and succ[0] in end_nodes（与你 scheduler 的触发条件一致）
            succ = nodes[node_id].succ
            if not (len(succ) == 1 and succ[0] in end_nodes):
                self._fail(
                    f"[Download trigger violated] ue={ue} node={node_id} succ={succ} end_nodes={list(end_nodes)}"
                )
