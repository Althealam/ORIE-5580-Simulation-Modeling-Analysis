# scheduler.py
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
    非常简单的 baseline 策略：
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


class HybridScheduler(BaseScheduler):
    """
    一个简单的“混合调度”示例：

    - 目标：在 throughput 和 decode 延迟之间取得折中。
    - 策略：
        * 如果 decode 队列太长（超过 decode_queue_threshold），优先 decode；
        * 否则在 prefill 和 decode 之间“交替”（round-robin）：
              - 上一次做了 prefill → 这次优先 decode（如果有）
              - 上一次做了 decode → 这次优先 prefill（如果有）
        * 如果某一类队列为空，就退化为另一类。

    注意：这只是一个示例，你可以根据项目需求自己调参数或改策略。
    """

    def __init__(
        self,
        max_prefill_tokens: int = 4096,
        max_decode_batch_size: int = 1024,
        decode_queue_threshold: int = 32,
    ):
        self.max_prefill_tokens = max_prefill_tokens
        self.max_decode_batch_size = max_decode_batch_size
        self.decode_queue_threshold = decode_queue_threshold

        # 记录上一次运行的模式："prefill" / "decode" / None
        self._last_mode: str | None = None

    def _build_prefill_batch(self, queues: SystemQueues) -> Batch | None:
        if not queues.prefill_queue:
            return None
        jobs: List[Query] = []
        token_budget = self.max_prefill_tokens
        while queues.prefill_queue and token_budget > 0:
            q = queues.prefill_queue[0]
            if q.L <= token_budget:
                jobs.append(q)
                token_budget -= q.L
                queues.prefill_queue.pop(0)
            else:
                if not jobs:
                    jobs.append(q)
                    queues.prefill_queue.pop(0)
                break
        token_load = sum(q.L for q in jobs)
        return Batch(jobs=jobs, mode="prefill", token_load=token_load)

    def _build_decode_batch(self, queues: SystemQueues) -> Batch | None:
        if not queues.decode_queue:
            return None
        jobs = list(queues.decode_queue[:self.max_decode_batch_size])
        token_load = len(jobs)
        return Batch(jobs=jobs, mode="decode", token_load=token_load)

    def select_next_batch(self, queues: SystemQueues, gpu_idle: bool) -> Batch | None:
        if not gpu_idle:
            return None

        has_prefill = bool(queues.prefill_queue)
        has_decode = bool(queues.decode_queue)

        if not has_prefill and not has_decode:
            return None

        # 1) 如果 decode 队列已经很长，优先 decode，防止 decode starvation
        if has_decode and len(queues.decode_queue) >= self.decode_queue_threshold:
            batch = self._build_decode_batch(queues)
            self._last_mode = "decode"
            return batch

        # 2) 否则在 prefill / decode 之间交替
        if self._last_mode == "prefill":
            # 上一次 prefill，这次优先 decode
            if has_decode:
                batch = self._build_decode_batch(queues)
                self._last_mode = "decode"
                return batch
            elif has_prefill:
                batch = self._build_prefill_batch(queues)
                self._last_mode = "prefill"
                return batch

        elif self._last_mode == "decode":
            # 上一次 decode，这次优先 prefill
            if has_prefill:
                batch = self._build_prefill_batch(queues)
                self._last_mode = "prefill"
                return batch
            elif has_decode:
                batch = self._build_decode_batch(queues)
                self._last_mode = "decode"
                return batch

        # 3) 如果还没有历史（_last_mode is None），或者某一侧队列为空
        #    → 先尝试 prefill，再 decode
        if has_prefill:
            batch = self._build_prefill_batch(queues)
            self._last_mode = "prefill"
            return batch
        if has_decode:
            batch = self._build_decode_batch(queues)
            self._last_mode = "decode"
            return batch

        return None



class ChunkedPrefillScheduler(BaseScheduler):
    """
    简化版的“chunked prefill”调度器。

    说明（非常重要）：
    ------------------
    严格意义上的 chunked prefill 需要：
        - 在 Query 里记录“剩余 prefill tokens”（比如 remaining_L）；
        - 每个 prefill batch 只对部分 tokens 做 prefill；
        - 直到某个 Query 的所有 prompt token 都 prefill 完成，
          才能进入 decode 阶段。

    目前你的 simulation.py 里：
        - 一旦一个 Query 出现在 prefill batch 中，
          就认为它已经完成了全部 prefill，并立刻转入 decode_queue。

    为了让这段代码 *立即可运行* 而不用改其他文件，
    这里给出的 ChunkedPrefillScheduler 做的是一个 **近似版本**：

        - prefill 阶段仍然一次性完成每个 Query 的 prefill；
        - 不是真正的“按 Query 内部 token 分块”，
        - 但我们在 batch 层面限制了单次 prefill 的总 token 数
          （prefill_chunk_tokens），并在 prefill / decode 之间更积极地切换，
          从而模拟出“不会让长 prompt 一直占满 GPU”的效果。

    如果你之后要实现“真正的 chunked prefill”，可以：
        1. 在 Query 里加 remaining_L 字段；
        2. 在 simulation.py 的 prefill 处理部分修改逻辑；
        3. 在 Batch 里增加 per-query token 计数。
    """

    def __init__(
        self,
        prefill_chunk_tokens: int = 1024,
        max_decode_batch_size: int = 1024,
        decode_bias: float = 0.3,
    ):
        """
        prefill_chunk_tokens: 每个 prefill batch 的 token 上限（模拟 chunk 大小）
        max_decode_batch_size: decode batch 中最多包含多少个 query
        decode_bias: 0~1，越大表示越偏向 decode
                     （例如 0.3 表示 roughly 70% 时间在 prefill，30% 在 decode）
        """
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.max_decode_batch_size = max_decode_batch_size
        self.decode_bias = decode_bias

        self._last_mode: str | None = None
        self._prefill_counter: int = 0
        self._decode_counter: int = 0

    def _build_prefill_batch(self, queues: SystemQueues) -> Batch | None:
        if not queues.prefill_queue:
            return None
        jobs: List[Query] = []
        token_budget = self.prefill_chunk_tokens
        while queues.prefill_queue and token_budget > 0:
            q = queues.prefill_queue[0]
            if q.L <= token_budget:
                jobs.append(q)
                token_budget -= q.L
                queues.prefill_queue.pop(0)
            else:
                if not jobs:
                    jobs.append(q)
                    queues.prefill_queue.pop(0)
                break
        token_load = sum(q.L for q in jobs)
        return Batch(jobs=jobs, mode="prefill", token_load=token_load)

    def _build_decode_batch(self, queues: SystemQueues) -> Batch | None:
        if not queues.decode_queue:
            return None
        jobs = list(queues.decode_queue[:self.max_decode_batch_size])
        token_load = len(jobs)
        return Batch(jobs=jobs, mode="decode", token_load=token_load)

    def select_next_batch(self, queues: SystemQueues, gpu_idle: bool) -> Batch | None:
        if not gpu_idle:
            return None

        has_prefill = bool(queues.prefill_queue)
        has_decode = bool(queues.decode_queue)

        if not has_prefill and not has_decode:
            return None

        # 简单的“比例控制”：根据历史 prefill / decode 次数，
        # 尝试维持 prefill : decode ≈ (1 - decode_bias) : decode_bias
        total = self._prefill_counter + self._decode_counter + 1e-9
        current_decode_ratio = self._decode_counter / total

        # 如果 decode 比例太低 → 优先 decode（只要有 decode 工作）
        if has_decode and current_decode_ratio < self.decode_bias:
            batch = self._build_decode_batch(queues)
            self._decode_counter += 1
            self._last_mode = "decode"
            return batch

        # 否则优先 prefill（只要有 prefill 工作）
        if has_prefill:
            batch = self._build_prefill_batch(queues)
            self._prefill_counter += 1
            self._last_mode = "prefill"
            return batch

        # 如果没有 prefill，但有 decode，就 decode
        if has_decode:
            batch = self._build_decode_batch(queues)
            self._decode_counter += 1
            self._last_mode = "decode"
            return batch

        return None
