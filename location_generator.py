# location_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union


@dataclass
class Processor:
    id: int
    EAT: float = 0.0


@dataclass
class Location:
    id: int                  # 本地: 0；ES: 1..M
    ue_id: Optional[int]     # 本地属于哪个 UE；ES 为 None
    cpu_speed: float         # cycles per second
    processors: List[Processor]


def build_locations(
    ue_number: int,
    es_number: int,
    ue_cpu_speed: float,
    es_cpu_speeds: Union[float, Sequence[float]],
    es_proc_nums: Union[int, Sequence[int]],
) -> List[Location]:
    """
    生成 locations 列表：
      - 每个 UE 一个本地 Location：Location(id=0, ue_id=k, processors=1)
      - 每个 ES 一个 Location：Location(id=1..M, ue_id=None, processors=可变)

    参数：
      ue_number: UE 数量 K
      es_number: ES 数量 M
      ue_cpu_speed: UE 本地 CPU 速度（cycles/s），例如 1e9
      es_cpu_speeds:
        - 若传 float：所有 ES 用同一个 cpu_speed
        - 若传 list：长度必须为 M，对应每个 ES 的 cpu_speed
      es_proc_nums:
        - 若传 int：所有 ES 用同一个处理器数量
        - 若传 list：长度必须为 M，对应每个 ES 的处理器数量

    返回：
      locations: List[Location]
    """
    if ue_number <= 0:
        raise ValueError("ue_number must be >= 1")
    if es_number < 0:
        raise ValueError("es_number must be >= 0")

    # 规范化 es_cpu_speeds
    if isinstance(es_cpu_speeds, (int, float)):
        es_cpu_speeds_list = [float(es_cpu_speeds)] * es_number
    else:
        es_cpu_speeds_list = [float(x) for x in es_cpu_speeds]
        if len(es_cpu_speeds_list) != es_number:
            raise ValueError("es_cpu_speeds length must equal es_number")

    # 规范化 es_proc_nums
    if isinstance(es_proc_nums, int):
        es_proc_nums_list = [int(es_proc_nums)] * es_number
    else:
        es_proc_nums_list = [int(x) for x in es_proc_nums]
        if len(es_proc_nums_list) != es_number:
            raise ValueError("es_proc_nums length must equal es_number")
        if any(p <= 0 for p in es_proc_nums_list):
            raise ValueError("each ES processor number must be >= 1")

    locations: List[Location] = []

    # UE 本地 locations：id=0，但 ue_id 不同（你现在的 TransmissionModel 也按 ue_id 区分本地）
    for ue_id in range(ue_number):
        locations.append(
            Location(
                id=0,
                ue_id=ue_id,
                cpu_speed=float(ue_cpu_speed),
                processors=[Processor(0)]  # UE 固定 1 个 processor
            )
        )

    # ES locations：id=1..M，ue_id=None
    for i in range(es_number):
        es_id = i + 1
        proc_num = es_proc_nums_list[i]
        locations.append(
            Location(
                id=es_id,
                ue_id=None,
                cpu_speed=es_cpu_speeds_list[i],
                processors=[Processor(processor_id) for processor_id in range(proc_num)]
            )
        )

    return locations


def reset_locations(locations: List[Location]) -> None:
    """把所有 processor 的 EAT 清零（可用于每次 episode 开始）"""
    for loc in locations:
        for p in loc.processors:
            p.EAT = 0.0
