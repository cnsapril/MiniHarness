# Mid-turn steering MVP

MiniHarness is still a small synchronous Python agent, but this MVP adds a
**durable Turn control queue**. Model responses remain non-streaming.

It deliberately does **not** implement Codex's full app-server protocol,
a terminal TUI, automatic rollback, or semantic tool-call classification.

## Usage (two terminals)

Terminal A:

```bash
python main.py new "Refactor PaymentService"
```

The agent prints a `Session ID` when the turn starts. While it is running,
submit a control from **Terminal B**, using that exact ID:

```bash
python main.py steer --session <SESSION_ID> "Stop deleting old code; inspect changed files"
python main.py interrupt --session <SESSION_ID>
```

`steer` delivers user input at the next safe processing boundary, and the
agent replans with the new instructions. `interrupt` terminates the turn
before starting further tools. If a tool is already executing, the harness
**cannot guarantee immediate cancellation**; existing tool behavior and
side effects remain intact. The control command reports that the request was
**queued**, not that the action has already taken effect.

The CLI still uses a trusted blocking approval prompt for ASK tools. You can
send a control from Terminal B while the prompt is open; after a control has
arrived, the ledger blocks any *new* stale tool admission. Terminal A still
needs to finish the outstanding approval prompt before it can process the
control. This MVP does not implement asynchronous prompt cancellation.

## Design

- **Turn control state**: `agent_turns` tracks the one active turn for this
  invocation (one turn per session in v1); `turn_controls` persists queued
  STEER / INTERRUPT requests.
- **Steering**: safe-boundary consumption appends a durable user message to
  the same session history. A stale model response returned after a new input
  arrives is discarded before being committed.
- **Tool calls**: if a control arrives between Tool Calls, remaining RECORDED
  calls are closed with `CANCELLED_BY_CONTROL` function outputs. Existing
  terminal ledger state `DENIED` is reused to avoid breaking old databases;
  the structured function output distinguishes cancellation from policy denial.
- **Execution fence**: SQLite `BEGIN IMMEDIATE` serializes control submission,
  turn completion, and tool admission. Once a new control is durably queued,
  the old batch cannot start another tool. Already STARTED operations are not
  retroactively cancelled and may have side effects.
- **Interrupt**: separate from Steering, persisted and finalized as
  `INTERRUPTED`. Any unprocessed steering remains marked `UNHANDLED`.
- **Turn events**: lightweight lifecycle events (such as `turn_started`,
  `steering_applied`, `tools_cancelled`) remain for CLI observability.
  No token-level output streaming is implemented.
- **Recovery**: serialized history, tool outputs, and control events remain
  inspectable. This PR does not auto-resume or retry tools after a crash.

## Limitations

- Turn status is recorded under the session ID, with one turn per invocation.
  This is not multi-turn conversational session management.
- This uses two CLI processes. It does not support typing steering into the
  same terminal while an agent turn is running.
- **No immediate preemption** of a blocking model call, tool handler,
  or approval prompt. Steering is only guaranteed to take effect at a
  subsequent safe boundary. When a blocking LLM call returns, its stale
  response is discarded if a control arrived while it was executing.
- The database has no worker heartbeat or lease; an unclean process death may
  leave a stale RUNNING turn until it is inspected/reconciled.
- Side-effecting operations that end with an unknown result keep their
  STARTED state, and require manual reconciliation; they are not blindly
  retried or rolled back.
- Tool execution is single-worker in this MVP. Distributed execution,
  backpressure, resumable event subscriptions and event-stream persistence
  are intentionally out of scope.

## Test

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests cover durable control replay, steering vs interrupt priority,
no stale tool execution, approval-control race fencing, blocking LLM
stale-response discard, and the completion/arrival boundary.
