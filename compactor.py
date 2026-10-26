import logging

from openai import OpenAI

from context_manager import ContextHelper
from session import AgentSession

logger = logging.getLogger(__name__)

MODEL = "mini-qwen3-16k"
COMPACTION_MAX_INPUT_TOKENS = 8000
SUMMARY_MAX_OUTPUT_TOKENS = 1024

COMPACTION_INSTRUCTIONS = """
You are summarizing the history of a coding agent
so it can continue working.

Keep:
- The user's goal and constraints.
- Important tool call results, including discovered files,
  code findings, errors, and test results.
- Decisions made and work already completed.
- Unresolved problems and next steps.

Rules:
- Preserve exact file paths, identifiers, and important values.
- Distinguish confirmed results from assumptions.
- Do not invent information.
- Remove redundant conversation and unimportant details.
- Keep the summary concise, but do not omit important findings.
"""


def summarize_history(client: OpenAI, old_history: list) -> str:
    if not old_history:
        raise ValueError("Nothing to compact!")
    serialized = ContextHelper.serialize_items(old_history)
    before_token_size = ContextHelper.estimate_tokens_str(serialized)
    if before_token_size >= COMPACTION_MAX_INPUT_TOKENS:
        raise ValueError(f"Estimated compaction token size {before_token_size} exceeds "
                         f"max size {COMPACTION_MAX_INPUT_TOKENS}")
    response = client.responses.create(
        model=MODEL,
        instructions=COMPACTION_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": (
                    "Summarize the following historical "
                    "agent interactions:\n\n"
                    "<history>\n"
                    f"{serialized}\n"
                    "</history>"
                )
            }
        ],
        max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        extra_body={"think": False}
    )
    summary = response.output_text.strip()
    if not summary:
        raise ValueError("Compaction produced empty summary")
    return summary


def calculate_new_step_ends(old_step_ends: list[int], old_count: int, keep_recent_steps: int) -> list[int]:
    new_start_idx = len(old_step_ends) - keep_recent_steps
    offset = old_count - 1
    new_step_ends = [step_end_idx - offset for step_end_idx in old_step_ends[new_start_idx:]]
    return new_step_ends


def validate_candidate(before_token_size: int, after_token_size: int, compaction_threshold: int,
                       candidate_history: list, new_step_ends: list[int]):
    # 校验压缩真实减少了token
    if after_token_size >= before_token_size:
        raise ValueError(f"After token size {after_token_size} exceeds before token size {before_token_size}")
    # 校验压缩后的token小于trigger
    if after_token_size >= compaction_threshold:
        raise ValueError(f"After token size {after_token_size} exceeds compaction threshold {compaction_threshold}")
    # 校验new_step_ends单调递增且index合法
    if not new_step_ends:
        raise ValueError("No completed steps")
    if new_step_ends[-1] != len(candidate_history):
        raise ValueError("Final step boundary mismatch")
    prev = 0
    for end in new_step_ends:
        if not prev < end <= len(candidate_history):
            raise ValueError(f"Invalid step boundary: {end}")
        prev = end


def compact_session(session: AgentSession, client: OpenAI):
    keep_recent_steps = 2
    # 0. Record Original History Token Size
    before_token_size = ContextHelper.estimate_tokens(session.get_history())
    # 1. Split History
    pinned, old, recent = session.split_for_compaction(keep_recent_steps=keep_recent_steps)
    # 2. Summarize Old History
    summary = summarize_history(client, old)
    # 3. Assemble New Candidate History
    summary_item = {
        "role": "assistant",
        "content": (
                "[Historical context summary]\n" + summary
        )
    }
    candidate_history = pinned + [summary_item] + recent
    # 4. Recalculate New Step Ends
    old_step_ends = session.get_step_ends()
    new_step_ends = calculate_new_step_ends(old_step_ends, len(old), keep_recent_steps)
    # 5. Validate candidate tokens
    after_token_size = ContextHelper.estimate_tokens(candidate_history)
    validate_candidate(before_token_size, after_token_size, session.get_compaction_threshold(), candidate_history,
                       new_step_ends)
    # 6. Replace history
    session.replace_history(candidate_history, new_step_ends)
    logger.info("[Compaction] before_token_size=%d after_token_size=%d old_step_ends=%s new_step_ends=%s",
                before_token_size, after_token_size, old_step_ends, new_step_ends)
