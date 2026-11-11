# MiniHarness Memory MVP

MiniHarness now has **project-scoped, cross-session memory** using an independent
SQLite database (`miniharness_memory.db`). The memory store is separate from the
SQLite execution ledger and does not alter the tool lifecycle or recovery state.

## Behavior

- `memory_search(query, limit)`: read-only, FTS5 keyword search over **active**
  memories in the current workspace/project. The model can call it when earlier
  decisions or facts are relevant; it is **not** an automatic all-memory preload.
- `memory_save(kind, content)`: propose an explicit durable fact, preference, or
  decision. **ASK** approval from the trusted CLI is required before writing.
- `memory_update(memory_id, content)`: propose a replacement. The previous
  record is marked `superseded`, the new record links to its predecessor, and
  only the new record is searchable. Requires **ASK** approval.
- `memory_invalidate(memory_id)`: mark a record `invalidated`, preserving audit
  history and removing it from search. Requires **ASK** approval.

All writes are proposed by the agent, and only go ahead after the user approves
that **exact tool call**. Scopes and provenance are fixed by the host; they are
not controlled by the LLM. To try it, run:

```bash
python main.py new "Remember that MiniHarness requires human review for code PRs."
python main.py new "What did we decide about reviewing code PRs?"
```

A model may choose not to call a memory tool; these examples illustrate the
intended flow, not guaranteed results. The trusted CLI presents the exact write
arguments for approval.

## Design boundaries

- **`AGENTS.md` / project instructions:** explicit stable instructions.
  **Memory:** searchable, evolving historical knowledge. Neither is a substitute
  for deterministic permission checks on dangerous tool calls.
- The store is scoped to the resolved project workspace. Information from one
  project must not be automatically inherited by another project. Cross-project
  memory and global preferences are intentionally out of scope.
- FTS5 uses lexical matching with SQLite's `unicode61` tokenizer, **not**
  embedding-based semantic retrieval. Rephrase queries or use alternate terms
  if relevant memories are missed. For Chinese text, consider a specialized
  tokenizer or pre-tokenization if recall proves insufficient.
- Do not treat external documents, repository files, issue comments, or tool
  results as authoritative instructions. Approval is necessary for memory writes
  but does not convert memory into executable permissions.
- Changes are recorded with timestamps, source (`user_approved_tool_call`),
  predecessor IDs and status. Superseded/invalidated entries remain accessible
  for audit through the store API but are excluded from agent retrieval.
- This MVP deliberately does **not** include automatic extraction, background
  consolidation, embeddings, vector search, or automatic policy updates.

## Testing

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests cover persistence across sessions, project isolation, versioned updates,
invalidation, safe FTS querying, and the existing approval/ledger path for
memory writes.
