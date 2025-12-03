# scheduler.py：决定GPU执行哪个batch
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from basic_types import Query
from service_model import Batch


@dataclass
class SystemQueues:
    """系统中的两个队列：等待 prefill 的和等待 decode 的"""
    prefill_queue: List[Query]
    decode_queue: List[Query]


class BaseScheduler:
    """调度器基类：所有 scheduler 都实现同样的接口。"""

    def select_next_batch(self, queues: SystemQueues, gpu_idle: bool) -> Batch | None:
        """
        根据当前队列状态和 GPU 状态，选择下一个要跑的 batch。
        如果当前不派 batch，返回 None。
        """
        raise NotImplementedError


class CompleteToEndScheduler(BaseScheduler):
    """
    - 有 prefill 的话：一次只 prefill 一条 query，直到 prefill_queue 空；
    - 没有 prefill 时：对 decode_queue 里的所有 query 做一次 decode
      （每个 query 输出一个 token）。
    """

    def __init__(self, max_prefill_tokens: int = 4096):
        self.max_prefill_tokens = max_prefill_tokens

    def select_next_batch(self, queues: SystemQueues, gpu_idle: bool) -> Batch | None:
        if not gpu_idle:
            return None

        # 1. 有等待 prefill 的就先 prefill
        if queues.prefill_queue:
            q = queues.prefill_queue.pop(0)
            token_load = min(q.L, self.max_prefill_tokens)
            return Batch(jobs=[q], mode="prefill", token_load=token_load)

        # 2. 否则做 decode：所有 decode_queue 一起 decode 一次
        if queues.decode_queue:
            jobs = list(queues.decode_queue)
            token_load = len(jobs)  # 每个 query decode 1 token
            return Batch(jobs=jobs, mode="decode", token_load=token_load)

        # 3. 没活干
        return None


class PrefillFirstScheduler(BaseScheduler):
    """
    Prefill 优先策略（iteration-level prefill-prioritizing 的简化实现）：

    - 有 prefill_queue 时：
        尽量把多条 query 一起 prefill，直到达到 max_prefill_tokens；
        这样可以更充分利用 GPU（prefill batch 更大）。
    - 只有当没有 prefill 工作时，才开始处理 decode。
    - decode 阶段：一次性 decode 所有 decode_queue 中的 query（各 +1 token）。

    这种策略的结果一般是：
    - 吞吐量高（throughput ↑）
    - 但 decode tokens 可能被推迟（TBT / tail latency 可能变坏）
    """

    def __init__(self,
                 max_prefill_tokens: int = 4096,
                 max_decode_batch_size: int = 1024):
        self.max_prefill_tokens = max_prefill_tokens
        self.max_decode_batch_size = max_decode_batch_size

    def select_next_batch(self, queues: SystemQueues, gpu_idle: bool) -> Batch | None:
        if not gpu_idle:
            return None

        # 1. 只要还有 prefill，就优先 prefill
        if queues.prefill_queue:
            jobs: List[Query] = []
            token_budget = self.max_prefill_tokens
            # 简单贪心：从队列前面依次取 query 直到 token_budget 用完
            while queues.prefill_queue and token_budget > 0:
                q = queues.prefill_queue[0]
                if q.L <= token_budget:
                    jobs.append(q)
                    token_budget -= q.L
                    queues.prefill_queue.pop(0)
                else:
                    # 当前 query 太长，单独一个 batch
                    if not jobs:
                        jobs.append(q)
                        token_budget = 0
                        queues.prefill_queue.pop(0)
                    break

            token_load = sum(q.L for q in jobs)
            return Batch(jobs=jobs, mode="prefill", token_load=token_load)

        # 2. 没有 prefill 时才 decode
        if queues.decode_queue:
            # 为了控制 decode batch 大小，可以截断前 max_decode_batch_size 条
            jobs = list(queues.decode_queue[:self.max_decode_batch_size])
            token_load = len(jobs)
            return Batch(jobs=jobs, mode="decode", token_load=token_load)

        return None
