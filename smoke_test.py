import random

from GNNDAG import dag_generator, DTO_scheduler, location_generator, DTO_env
from GNNDAG.validator import DTOValidator

EPS = 1e-9

def build_models(ue_es_rate=2e6, es_es_rate=20e6):
    exec_model = DTO_scheduler.ExecutionModel()
    upload_model = DTO_scheduler.UploadModel(ue_es_rate)
    trans_model = DTO_scheduler.TransmissionModel(ue_es_rate, es_es_rate)
    return exec_model, upload_model, trans_model

def build_locations(ue_number=3, es_number=2):
    return location_generator.build_locations(
        ue_number=ue_number,
        es_number=es_number,
        ue_cpu_speed=1e9,
        es_cpu_speeds=[8e9, 12e9],
        es_proc_nums=[2, 4],
    )

def run_eft_record(nodes, edges_data, end_nodes, download_nodes, order, ue_number):
    """Run A: 用原 scheduler 的 greedy/EFT 规则跑一遍，并记录每个 node 的 loc_choice（0 or ES id）。"""
    exec_model, upload_model, trans_model = build_models()
    locations = build_locations(ue_number=ue_number, es_number=2)

    sched = DTO_scheduler.DTOScheduler(
        nodes, edges_data, locations,
        exec_model, upload_model, trans_model,
        end_nodes, download_nodes, ue_number
    )

    loc_choice_map = {}  # node_id -> loc_choice (0 or ES id)
    order_run = list(order)

    for node_id in order_run:
        # 旧逻辑：scheduler 自动选 best loc + best processor
        sched.schedule_node(node_id)

        # 可选：每步验证（慢一点，但最稳）
        DTOValidator().validate(sched)

        chosen_loc = sched.node_loc[node_id]  # Location
        loc_choice_map[node_id] = chosen_loc.id  # 本地=0, ES=j

    makespan = max(sched.download_EAT.values()) if sched.download_EAT else 0.0
    return loc_choice_map, sched.download_EAT, makespan

def run_replay_with_env(nodes, edges_data, end_nodes, download_nodes, order, ue_number, loc_choice_map):
    """Run B: 用 env + schedule_node_at 按 RunA 记录的 loc_choice replay。"""
    exec_model, upload_model, trans_model = build_models()
    locations = build_locations(ue_number=ue_number, es_number=2)

    sched = DTO_scheduler.DTOScheduler(
        nodes, edges_data, locations,
        exec_model, upload_model, trans_model,
        end_nodes, download_nodes, ue_number
    )

    env = DTO_env.DTOEnv(sched)

    # reset 返回 ready，但这里我们严格按 order replay（需保证 order 是合法拓扑序）
    env.reset(nodes)

    order_run = list(order)
    done = False
    for node_id in order_run:
        if done:
            break
        loc_choice = loc_choice_map[node_id]
        _, done, _ = env.step((node_id, loc_choice), nodes)

    # 如果你的 env.done 只在末尾置 True，这里再兜底一下
    makespan = max(sched.download_EAT.values()) if sched.download_EAT else 0.0
    return sched.download_EAT, makespan

def assert_close_dict(d1, d2, eps=1e-9):
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    for k in keys:
        v1 = float(d1.get(k, 0.0))
        v2 = float(d2.get(k, 0.0))
        if abs(v1 - v2) > eps:
            raise AssertionError(f"Mismatch at key={k}: {v1} vs {v2}")

def main():
    ue_number = 3
    nodes, edges_data, end_nodes, download_nodes, order = dag_generator.generate_multi_ue_dag(
        ue_number=ue_number,
        n_compute_nodes_per_ue=5,
        seed=0
    )

    # Run A
    loc_choice_map, eat_A, ms_A = run_eft_record(
        nodes, edges_data, end_nodes, download_nodes, order, ue_number
    )
    print("RunA (EFT) download_EAT:", eat_A)
    print("RunA (EFT) makespan:", ms_A)

    # Run B
    eat_B, ms_B = run_replay_with_env(
        nodes, edges_data, end_nodes, download_nodes, order, ue_number, loc_choice_map
    )
    print("RunB (Replay) download_EAT:", eat_B)
    print("RunB (Replay) makespan:", ms_B)

    # Assert
    assert_close_dict(eat_A, eat_B, EPS)
    if abs(ms_A - ms_B) > EPS:
        raise AssertionError(f"Makespan mismatch: {ms_A} vs {ms_B}")

    print("✅ EFT -> Replay 对齐成功")

def set_env():
    ue_number = 3
    es_number = 2
    nodes, edges_data, end_nodes, download_nodes, order = dag_generator.generate_multi_ue_dag(
        ue_number=ue_number,
        n_compute_nodes_per_ue=5,
        seed=0
    )

    locations = build_locations(ue_number=ue_number, es_number=es_number)
    exec_model, upload_model, trans_model = build_models()

    scheduler = DTO_scheduler.DTOScheduler(
        nodes, edges_data, locations,
        exec_model, upload_model, trans_model,
        end_nodes, download_nodes, ue_number,
        es_number
    )

    env = DTO_env.DTOEnv(scheduler)

    return env

def test_for_observation():
    env, scheduler = set_env()

    # --- reset ---
    observation = env.reset()

    # 如果你的 build_obs 还没返回 node_ids，这里兜底构造一个
    if "node_ids" not in observation:
        # 建议排除 end_nodes（你现在已经决定 obs 不包含 end）
        # 如果 observation 里有 end_nodes，你也可以用 observation["end_nodes"]
        end_nodes = getattr(env, "end_nodes", set())
        observation["node_ids"] = sorted(nid for nid in nodes.keys() if nid not in end_nodes)

    while True:
        mask = observation["mask"]
        node_ids = observation["node_ids"]

        # 1) 选一个合法 node（按 index）
        valid_i = [i for i, ok in enumerate(mask["node"]) if ok]
        if not valid_i:
            raise RuntimeError("No valid nodes in node mask. ready_list/mask 可能没对齐。")

        i = random.choice(valid_i)
        node_id = node_ids[i]

        # 2) 选一个合法 loc_choice（取索引 j，而不是 True/False）
        loc_row = mask["loc"][i]  # 注意：用 i（index），不要用 node_id
        valid_locs = [j for j, ok in enumerate(loc_row) if ok]
        if not valid_locs:
            raise RuntimeError(f"No valid loc for node index={i}, node_id={node_id}. loc_mask 可能没对齐。")

        loc_choice = random.choice(valid_locs)

        # 3) step：用真实 node_id 调 env
        observation, reward, done, info = env.step((node_id, loc_choice), nodes)

        if done:
            break

    print("makespan", max(env.scheduler.download_EAT.values()))

def test_for_reset():
    env, scheduler = set_env()

    # ---------- 第一次 episode ----------
    obs = env.reset()

    # 选一个确定性的 action（第一个 ready node + loc=0）
    first_node = env.ready_nodes()[0]
    action = (first_node, 0)

    observation1, reward1, terminated1, truncated1, info1 = env.step(action)

    # 记录 scheduler 的关键时间状态
    snapshot1 = {
        "upload_EAT": dict(env.scheduler.upload_EAT),
        "download_EAT": dict(env.scheduler.download_EAT),
        "processor_EAT": [
            p.EAT for loc in env.scheduler.locations for p in loc.processors
        ],
    }

    # ---------- 第二次 episode ----------
    obs = env.reset()

    first_node = env.ready_nodes()[0]
    action = (first_node, 0)

    observation2, reward2, terminated2, truncated2, info2 = env.step(action)

    snapshot2 = {
        "upload_EAT": dict(env.scheduler.upload_EAT),
        "download_EAT": dict(env.scheduler.download_EAT),
        "processor_EAT": [
            p.EAT for loc in env.scheduler.locations for p in loc.processors
        ],
    }

    # ---------- 对比 ----------
    print("reward1 =", reward1)
    print("reward2 =", reward2)

    print("snapshot1 =", snapshot1)
    print("snapshot2 =", snapshot2)

    assert reward1 == reward2, "❌ reward 不一致，scheduler 没 reset 干净"
    assert snapshot1 == snapshot2, "❌ scheduler 时间状态泄漏"

    print("✅ reset 是干净的，scheduler runtime 状态已正确清空")



if __name__ == "__main__":
    test_for_reset()
