import json
import logging
from copy import deepcopy
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ContextHelper:
    @staticmethod
    def serialize_items(items: list) -> list:
        # 尝试使用模型API将模型响应进行序列化
        return [
            item.model_dump(mode="json")
            if hasattr(item, "model_dump")
            else deepcopy(item)
            for item in items
        ]

    @staticmethod
    def serialize_items_to_str(items: list) -> str:
        serializable = ContextHelper.serialize_items(items)
        return json.dumps(serializable, ensure_ascii=False)

    @staticmethod
    def estimate_tokens_str(content: str) -> int:
        return len(content) // 3

    @staticmethod
    def estimate_tokens(items: list) -> int:
        content = ContextHelper.serialize_items_to_str(items)
        # TODO: 目前只是估算，后续需要用模型tokenizer进行纠偏
        return ContextHelper.estimate_tokens_str(content)


@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int = 1024

    @property
    def available_input_tokens(self) -> int:
        return self.max_context_tokens - self.reserved_output_tokens - self.safety_margin_tokens


class ContextManager:

    def __init__(self, context_budget: ContextBudget, compaction_ratio: float):
        self.context_budget = context_budget
        self._cached_tokens: int | None = None
        self.compaction_ratio = compaction_ratio

    def get_compaction_threshold(self):
        return int(self.compaction_ratio * self.context_budget.available_input_tokens)

    def should_compact(self, history: list) -> bool:
        if self._cached_tokens is None:
            self._cached_tokens = ContextHelper.estimate_tokens(history)
        return self._cached_tokens >= self.get_compaction_threshold()

    def invalidate(self):
        self._cached_tokens = None

    def append(self, full_history: list, new_items: list):
        estimated_tokens = ContextHelper.estimate_tokens(new_items)
        if self._cached_tokens is None:
            # cache miss，需要重新建立baseline
            self._cached_tokens = ContextHelper.estimate_tokens(full_history)
        else:
            self._cached_tokens += estimated_tokens

    def compact(self, new_history: list):
        # TODO: implement
        logger.info("Compaction executed")
        self.invalidate()
        self._cached_tokens = ContextHelper.estimate_tokens(new_history)
