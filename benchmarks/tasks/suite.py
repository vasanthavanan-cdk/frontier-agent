from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    coding = "coding"
    reasoning = "reasoning"
    research = "research"


@dataclass(frozen=True)
class BenchTask:
    id: str
    category: Category
    workflow: str
    prompt: str
    # keywords that a good answer should contain (used by automated judge)
    quality_hints: tuple[str, ...]


TASKS: list[BenchTask] = [
    # ── Coding (10) ──────────────────────────────────────────────────────────
    BenchTask(
        id="code-01",
        category=Category.coding,
        workflow="coding",
        prompt="Write a Python function that implements binary search on a sorted list. Include type hints and docstring.",
        quality_hints=("binary search", "def ", "mid", "low", "high", "return"),
    ),
    BenchTask(
        id="code-02",
        category=Category.coding,
        workflow="coding",
        prompt="Write a Python context manager that measures and logs block execution time.",
        quality_hints=("__enter__", "__exit__", "time", "contextmanager", "def "),
    ),
    BenchTask(
        id="code-03",
        category=Category.coding,
        workflow="coding",
        prompt="Implement a simple LRU cache in Python without using functools.lru_cache.",
        quality_hints=("OrderedDict", "capacity", "get", "put", "evict"),
    ),
    BenchTask(
        id="code-04",
        category=Category.coding,
        workflow="coding",
        prompt="Write a Python function that flattens a deeply nested list of arbitrary depth.",
        quality_hints=("isinstance", "list", "recursive", "def flatten", "yield"),
    ),
    BenchTask(
        id="code-05",
        category=Category.coding,
        workflow="coding",
        prompt="Write a Python async function that fetches multiple URLs concurrently using asyncio.gather and returns results.",
        quality_hints=("async def", "asyncio", "gather", "await", "aiohttp"),
    ),
    BenchTask(
        id="code-06",
        category=Category.coding,
        workflow="coding",
        prompt="Implement a thread-safe singleton pattern in Python.",
        quality_hints=("lock", "Lock", "_instance", "__new__", "thread"),
    ),
    BenchTask(
        id="code-07",
        category=Category.coding,
        workflow="coding",
        prompt="Write a Python function to find all permutations of a string without using itertools.",
        quality_hints=("def ", "permutation", "recursive", "swap", "result"),
    ),
    BenchTask(
        id="code-08",
        category=Category.coding,
        workflow="coding",
        prompt="Implement a simple publish-subscribe event system in Python.",
        quality_hints=("subscribe", "publish", "listener", "event", "dict"),
    ),
    BenchTask(
        id="code-09",
        category=Category.coding,
        workflow="coding",
        prompt="Write a Python dataclass for a paginated API response with helper methods to check if there are more pages.",
        quality_hints=("@dataclass", "page", "total", "has_next", "def "),
    ),
    BenchTask(
        id="code-10",
        category=Category.coding,
        workflow="coding",
        prompt="Write a Python function that validates an email address using a regex without external libraries.",
        quality_hints=("re.", "pattern", "match", "def ", "@", "domain"),
    ),
    # ── Reasoning (5) ────────────────────────────────────────────────────────
    BenchTask(
        id="reason-01",
        category=Category.reasoning,
        workflow="research",
        prompt="Explain why a hash map lookup is O(1) average but O(n) worst case. What causes the worst case and how do real implementations avoid it?",
        quality_hints=("hash collision", "load factor", "rehash", "O(1)", "O(n)"),
    ),
    BenchTask(
        id="reason-02",
        category=Category.reasoning,
        workflow="research",
        prompt="A REST API endpoint returns 200 OK but clients report intermittent data corruption. List the most likely causes in order of probability and how to diagnose each.",
        quality_hints=("race condition", "cache", "serialization", "concurrent", "diagnos"),
    ),
    BenchTask(
        id="reason-03",
        category=Category.reasoning,
        workflow="research",
        prompt="Compare mutex locks vs semaphores: when should you choose one over the other? Give concrete examples.",
        quality_hints=("mutex", "semaphore", "binary", "counting", "ownership"),
    ),
    BenchTask(
        id="reason-04",
        category=Category.reasoning,
        workflow="research",
        prompt="Explain the CAP theorem. Can a distributed system ever be CA (consistent and available but not partition-tolerant)? Why or why not?",
        quality_hints=("CAP", "consistency", "availability", "partition", "network"),
    ),
    BenchTask(
        id="reason-05",
        category=Category.reasoning,
        workflow="research",
        prompt="Why does TCP use a three-way handshake instead of a two-way handshake? What specific problem does the third message solve?",
        quality_hints=("SYN", "ACK", "sequence number", "synchronize", "two-way"),
    ),
    # ── Research (5) ─────────────────────────────────────────────────────────
    BenchTask(
        id="research-01",
        category=Category.research,
        workflow="research",
        prompt="Explain the difference between horizontal and vertical scaling. When is each approach the right choice?",
        quality_hints=("horizontal", "vertical", "scale out", "scale up", "stateless"),
    ),
    BenchTask(
        id="research-02",
        category=Category.research,
        workflow="research",
        prompt="Summarise the key differences between OAuth 2.0 and API keys for service authentication. Which is preferred for server-to-server calls and why?",
        quality_hints=("OAuth", "token", "scope", "API key", "server-to-server"),
    ),
    BenchTask(
        id="research-03",
        category=Category.research,
        workflow="research",
        prompt="What is the difference between a message queue and a message broker? Give two real-world examples of each.",
        quality_hints=("queue", "broker", "RabbitMQ", "Kafka", "SQS"),
    ),
    BenchTask(
        id="research-04",
        category=Category.research,
        workflow="research",
        prompt="Explain Blue-Green deployment vs Canary deployment. What are the cost and risk trade-offs of each?",
        quality_hints=("blue-green", "canary", "rollback", "traffic", "risk"),
    ),
    BenchTask(
        id="research-05",
        category=Category.research,
        workflow="research",
        prompt="What is connection pooling in databases and why does it matter? What happens if the pool is too small or too large?",
        quality_hints=("connection pool", "overhead", "latency", "timeout", "max connections"),
    ),
]
