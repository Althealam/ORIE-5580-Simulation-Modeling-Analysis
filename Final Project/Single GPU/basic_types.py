
from dataclasses import dataclass, field
import heapq
from enum import Enum, auto

class EventType(Enum):
    ARRIVAL = auto()       # 新 query 到达
    BATCH_DONE = auto()    # 一个 batch 完成（prefill 或 decode）
    SIM_END = auto()       # 结束模拟（可选）

@dataclass(order=True)
class Event:
    """
    离散事件仿真中的事件
    - time：事件发生的事件
    - priority：同一个时间多个事件的时候，用priority决定顺序
    - event_type：事件种类
    - payload：附加信息（batch或者其他对象）
    """
    time: float
    priority: int
    event_type: EventType = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)
    # payload 里可以放: {"query": q} 或 {"batch": batch}

class QueryPhase(Enum):
    """
    一条query在系统里面的阶段
    """
    WAIT_PREFILL = auto() 
    DECODING = auto()
    DONE = auto()

@dataclass
class Query:
    """
    we can see the notation from the project description
    LLM请求对象：
    - id：唯一编号
    - arrival_time：到达系统的事件
    - L：prompt长度（prefill tokens）
    - B：decode需要生成的token数
    - decoded：已经生成的token数
    - phase：当前阶段
    - ttft：Time to first token
    - finish_time：完成时间
    """
    id: int 
    arrival_time: float 
    L: int          # prompt length (prefill tokens)
    B: int          # total decode tokens needed
    decoded: int = 0
    phase: QueryPhase = QueryPhase.WAIT_PREFILL # wait_prefill/decoding/done
    ttft: float | None = None   # time to first token
    finish_time: float | None = None
