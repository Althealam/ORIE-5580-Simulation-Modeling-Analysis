# simulation.py
from __future__ import annotations

import heapq
import random
from typing import List, Dict, Tuple

from basic_types import Event, EventType, Query, QueryPhase
from service_model import GPUWorker
from scheduler import (
    SystemQueues,
    CompleteToEndScheduler,
    PrefillFirstScheduler,
    HybridScheduler,
    ChunkedPrefillScheduler,
)


def generate_next_arrival(current_time: float, lam: float) -> float:
    """泊松到达：下一次到达时间 = 当前时间 + Exp(λ)."""
    return current_time + random.expovariate(lam)


def build_scheduler(scheduler_name: str):
    """
    - "complete"         → CompleteToEndScheduler
    - "prefill_first"    → PrefillFirstScheduler
    - "hybrid"           → HybridScheduler
    - "chunked_prefill"  → ChunkedPrefillScheduler
    """
    name = scheduler_name.lower()
    if name in ["complete", "complete_to_end", "baseline"]:
        return CompleteToEndScheduler()
    elif name in ["prefill_first", "prefill"]:
        return PrefillFirstScheduler()
    elif name in ["hybrid"]:
        return HybridScheduler()
    elif name in ["chunked_prefill", "chunked"]:
        return ChunkedPrefillScheduler()
    else:
        raise ValueError(f"Unknown scheduler_name: {scheduler_name}")


def run_simulation(
    lam: float = 0.5,
    num_queries: int = 200,
    L_fixed: int = 64,
    B_fixed: int = 16,
    gpu_c: float = 0.0455,
    gpu_a: float = 0.0003,
    gpu_b0: int = 64,
    seed: int = 0,
    scheduler_name: str = "complete",
) -> Dict:
    """
    单 GPU、单 job 类型（固定 L, B）的离散事件仿真。

    满足 demo 要求：
        - single GPU worker
        - single type of jobs (fixed L_i, B_i)
        - 比较不同 scheduler（complete / prefill_first / ...）
        - 输出 average throughput, TTFT, TBT

    返回:
        {
          "scheduler_name": str,
          "finished_queries": List[Query],
          "avg_ttft": float,
          "avg_latency": float,
          "avg_tbt": float,
          "throughput": float,
          "sim_end_time": float,
        }
    """
    random.seed(seed)

    current_time = 0.0
    next_query_id = 0
    total_arrivals = 0
    finished_queries: List[Query] = []

    first_arrival_time: float | None = None

    # GPU + 队列 + 调度器
    gpu = GPUWorker(c=gpu_c, a=gpu_a, b0=gpu_b0)
    queues = SystemQueues(prefill_queue=[], decode_queue=[])
    scheduler = build_scheduler(scheduler_name)

    # 事件列表（最小堆）
    event_list: List[Event] = []

    # 初始化第一条到达事件
    first_arrival_time_sim = generate_next_arrival(0.0, lam)
    heapq.heappush(
        event_list,
        Event(
            time=first_arrival_time_sim,
            priority=0,
            event_type=EventType.ARRIVAL,
            payload={},
        ),
    )

    # ------------ 主循环：离散事件仿真 ------------ #
    while event_list and len(finished_queries) < num_queries:
        event = heapq.heappop(event_list)
        current_time = event.time

        # ------ 到达事件 ------ #
        if event.event_type == EventType.ARRIVAL:
            q = Query(
                id=next_query_id,
                arrival_time=current_time,
                L=L_fixed,
                B=B_fixed,
            )
            if first_arrival_time is None:
                first_arrival_time = current_time

            next_query_id += 1
            total_arrivals += 1
            queues.prefill_queue.append(q)

            # 安排下一次到达
            if total_arrivals < num_queries:
                t_next = generate_next_arrival(current_time, lam)
                heapq.heappush(
                    event_list,
                    Event(
                        time=t_next,
                        priority=0,
                        event_type=EventType.ARRIVAL,
                        payload={},
                    ),
                )

            # 如果 GPU 空闲，尝试派一个 batch
            if not gpu.busy:
                batch = scheduler.select_next_batch(queues, gpu_idle=True)
                if batch is not None:
                    finish_time = gpu.assign_batch(batch, current_time)
                    heapq.heappush(
                        event_list,
                        Event(
                            time=finish_time,
                            priority=1,
                            event_type=EventType.BATCH_DONE,
                            payload={},
                        ),
                    )

        # ------ batch 完成事件 ------ #
        elif event.event_type == EventType.BATCH_DONE:
            batch = gpu.finish_batch()
            if batch is None:
                continue

            if batch.mode == "prefill":
                # prefill 完成 → 进入 decode 阶段，同时视为“第一个 token 开始输出”的时间
                for q in batch.jobs:
                    q.phase = QueryPhase.DECODING
                    q.decoded = 0
                    q.ttft = current_time
                    queues.decode_queue.append(q)

            elif batch.mode == "decode":
                done_list = []
                for q in batch.jobs:
                    q.decoded += 1
                    if q.decoded >= q.B:
                        q.phase = QueryPhase.DONE
                        q.finish_time = current_time
                        done_list.append(q)

                for q in done_list:
                    finished_queries.append(q)
                    if q in queues.decode_queue:
                        queues.decode_queue.remove(q)

            # batch 完成后，尝试派下一个 batch
            if not gpu.busy:
                batch = scheduler.select_next_batch(queues, gpu_idle=True)
                if batch is not None:
                    finish_time = gpu.assign_batch(batch, current_time)
                    heapq.heappush(
                        event_list,
                        Event(
                            time=finish_time,
                            priority=1,
                            event_type=EventType.BATCH_DONE,
                            payload={},
                        ),
                    )

    # ------------ 仿真结束：统计指标 ------------ #
    if not finished_queries:
        raise RuntimeError("No queries finished in the simulation; check parameters.")

    sim_end_time = current_time
    if first_arrival_time is None:
        first_arrival_time = 0.0

    # TTFT
    ttfts = [
        q.ttft - q.arrival_time
        for q in finished_queries
        if q.ttft is not None
    ]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None

    # 总 latency
    latencies = [
        q.finish_time - q.arrival_time
        for q in finished_queries
        if q.finish_time is not None
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    # TBT（平均每个 token 间隔时间）
    # 近似定义：TBT_i = (finish - ttft) / max(B - 1, 1)
    tbts = []
    for q in finished_queries:
        if q.ttft is None or q.finish_time is None:
            continue
        if q.B <= 1:
            continue
        tbts.append((q.finish_time - q.ttft) / (q.B - 1))
    avg_tbt = sum(tbts) / len(tbts) if tbts else None

    # throughput：单位时间完成的 query 数
    effective_time = max(sim_end_time - first_arrival_time, 1e-9)
    throughput = len(finished_queries) / effective_time

    print(f"[Single GPU][{scheduler_name}] finished {len(finished_queries)} queries.")
    print(f"Simulation time window : {first_arrival_time:.3f} → {sim_end_time:.3f}")
    print(f"Average throughput     : {throughput}")
    print(f"Average TTFT           : {avg_ttft}")
    print(f"Average TBT            : {avg_tbt}")
    print(f"Average total latency  : {avg_latency}")

    return {
        "scheduler_name": scheduler_name, # 目前仿真使用的是哪一个scheduler，比如complete, prefill_first
        "finished_queries": finished_queries, # 完成的query对象，包含arrival_time, ttft, finish_time, decoded
        "avg_ttft": avg_ttft, # 所有完成的query的平均TTFT
        "avg_latency": avg_latency, # 平均的延迟时间
        "avg_tbt": avg_tbt, # 平均每个query的token间隔时间
        "throughput": throughput, # 吞吐量=完成的query数/有效的仿真时间窗口
        "sim_end_time": sim_end_time, # 仿真结束的时间戳
    }


# ------------------ M/M/1 验证用仿真 ------------------ #

def run_mm1_validation(
    lam: float = 0.5,
    mu: float = 1.0,
    num_customers: int = 5000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    用于验证的 M/M/1 仿真：
        - 无 batching
        - 无 setup time (c = 0)
        - 服务时间 ~ Exp(μ)

    这里不区分 prefill/decode，只是一个单阶段 server。
    返回:
        (avg_waiting_time, avg_system_time)
    可以和理论结果 W_q = λ / (μ(μ-λ)), W = 1 / (μ-λ) 对比。
    """
    random.seed(seed)

    current_time = 0.0
    total_arrivals = 0
    finished = 0

    # 事件类型仍然复用 ARRIVAL / BATCH_DONE
    event_list: List[Event] = []

    # queue: 等待服务的顾客（只存 arrival_time 即可）
    queue: List[float] = []

    server_busy = False

    # 指标
    waiting_times: List[float] = []
    system_times: List[float] = []

    # 第一个到达事件
    first_arrival_time = generate_next_arrival(0.0, lam)
    heapq.heappush(
        event_list,
        Event(
            time=first_arrival_time,
            priority=0,
            event_type=EventType.ARRIVAL,
            payload={},
        ),
    )

    while event_list and finished < num_customers:
        event = heapq.heappop(event_list)
        current_time = event.time

        if event.event_type == EventType.ARRIVAL:
            arrival_time = current_time
            total_arrivals += 1
            queue.append(arrival_time)

            # 安排下一次到达
            if total_arrivals < num_customers:
                t_next = generate_next_arrival(current_time, lam)
                heapq.heappush(
                    event_list,
                    Event(
                        time=t_next,
                        priority=0,
                        event_type=EventType.ARRIVAL,
                        payload={},
                    ),
                )

            # 如果 server 空闲，立刻开始服务一个顾客
            if not server_busy:
                start_time = current_time
                arrival = queue.pop(0)
                waiting_times.append(start_time - arrival)
                service_time = random.expovariate(mu)
                finish_time = current_time + service_time
                server_busy = True
                heapq.heappush(
                    event_list,
                    Event(
                        time=finish_time,
                        priority=1,
                        event_type=EventType.BATCH_DONE,
                        payload={"arrival": arrival},
                    ),
                )

        elif event.event_type == EventType.BATCH_DONE:
            finished += 1
            arrival = event.payload["arrival"]
            system_times.append(current_time - arrival)

            # 看队列有没有下一个顾客
            if queue:
                arrival_next = queue.pop(0)
                start_time = current_time
                waiting_times.append(start_time - arrival_next)
                service_time = random.expovariate(mu)
                finish_time = current_time + service_time
                heapq.heappush(
                    event_list,
                    Event(
                        time=finish_time,
                        priority=1,
                        event_type=EventType.BATCH_DONE,
                        payload={"arrival": arrival_next},
                    ),
                )
                server_busy = True
            else:
                server_busy = False

    avg_wait = sum(waiting_times) / len(waiting_times)
    avg_sys = sum(system_times) / len(system_times)

    print(f"[M/M/1] λ={lam}, μ={mu}, ρ={lam/mu:.3f}")
    print(f"Average waiting time W_q (sim) : {avg_wait}")
    print(f"Average system time  W   (sim) : {avg_sys}")
    return avg_wait, avg_sys


# # ------------------ 简单命令行测试 ------------------ #

# if __name__ == "__main__":
#     # 1) LLM-serving 仿真：比较两个 scheduler
#     for name in ["complete", "prefill_first"]:
#         print("=" * 70)
#         print(f"Running LLM simulation with scheduler = {name}")
#         run_simulation(
#             lam=0.5,
#             num_queries=500,
#             L_fixed=64,
#             B_fixed=16,
#             scheduler_name=name,
#             seed=0,
#         )

#     # 2) M/M/1 验证例子
#     print("=" * 70)
#     print("Running M/M/1 validation example")
#     run_mm1_validation(lam=0.5, mu=1.0, num_customers=5000, seed=0)
