import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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

    @staticmethod
    def estimate_tokens(items: list) -> int:
        # 尝试使用模型API将模型响应进行序列化
        serializable = [item.model_dump(mode="json") if hasattr(item, "model_dump")
                        else item
                        for item in items]
        content = json.dumps(serializable, ensure_ascii=False)
        # TODO: 目前只是估算，后续需要用模型tokenizer进行纠偏
        return len(content) // 3

    def should_compact(self, history: list) -> bool:
        if self._cached_tokens is None:
            self._cached_tokens = self.estimate_tokens(history)
        threshold = self.compaction_ratio * self.context_budget.available_input_tokens
        return self._cached_tokens >= int(threshold)

    def invalidate(self):
        self._cached_tokens = None

    def append(self, full_history: list, new_items: list):
        estimated_tokens = self.estimate_tokens(new_items)
        if self._cached_tokens is None:
            # cache miss，需要重新建立baseline
            self._cached_tokens = self.estimate_tokens(full_history)
        else:
            self._cached_tokens += estimated_tokens

    def compact(self, new_history: list):
        # TODO: implement
        logger.info("Compaction executed")
        self.invalidate()
        self._cached_tokens = self.estimate_tokens(new_history)
