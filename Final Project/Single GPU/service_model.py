# Batch+单个GPU
import random
from dataclasses import dataclass
from typing import List
from basic_types import Query

@dataclass
class Batch:
    """
    一个批次里面的工作
    - jobs：包含哪些query
    - mode：prefill或者decode
    - token_load：该batch的token数
    """
    jobs: List[Query]
    mode: str  # "prefill" or "decode"
    token_load: int

class GPUWorker:
    """
    单个GPU的简单模型
    - 一次只能处理一个batch
    - busy_until：当前batch结束的时间
    """
    def __init__(self, c: float, a: float, b0: int):
        """
        c: setup cost (ms)
        a: per-token cost beyond b0 (ms/token)
        b0: threshold
        """
        self.c = c
        self.a = a
        self.b0 = b0

        self.busy: bool = False
        self.current_batch: Batch | None = None
        self.busy_until: float = 0.0

    def service_time_deterministic(self, b: int) -> float:
        """确定性的 S(b)，先用简单版：不加随机。"""
        return self.c + self.a * max(0, b - self.b0)

    def assign_batch(self, batch: Batch, current_time: float) -> float:
        """GPU 开始处理一个 batch，返回完成时间。"""
        assert not self.busy, "GPU should be idle when assigning new batch."
        self.busy = True
        self.current_batch = batch

        b = batch.token_load
        s = self.service_time_deterministic(b)  # 后面你可以改成随机 Exp 分布

        self.busy_until = current_time + s
        return self.busy_until

    def finish_batch(self):
        """batch 完成时调用：返回刚刚完成的 batch。"""
        finished = self.current_batch
        self.busy = False
        self.current_batch = None
        return finished
