# MiniHarness PR #4 — Docker sandbox

This initial sandbox runs **one fixed command**: `python -m unittest discover -s tests -p 'test_*.py'`.
The LLM cannot choose a command, image, mount, environment variables, or container flags.

## Build the local test image

From the MiniHarness repository root:

```bash
docker build -t miniharness-tests:local -f sandbox/Dockerfile sandbox
```

The build downloads packages (network is available **at build time**). Runtime uses
`--pull=never --network none` to prevent unexpected image pulling/network access.

## Run

The registered `run_tests()` Tool uses `PermissionRequirement.ASK`: the
trusted CLI must approve it. The tool then runs the existing unittest suite
inside a Docker container with:

- read-only container root and read-only `/workspace` bind mount
- no network; non-root UID/GID; all capabilities dropped
- no-new-privileges, Docker's default seccomp profile (unless daemon customized)
- CPU = 1, memory = 512 MiB, PID limit = 64
- writable 64 MiB tmpfs at `/tmp`, with Python bytecode caching disabled
- a 90-second wall-clock timeout and 32 KiB host-side output cap

A unique container name is created per invocation. On timeout/output limit
the runner kills the Docker CLI and requests `docker rm -f <name>` to stop and
remove the container. Failure to verify cleanup is raised, not silently ignored.

Normal test assertion failures return `status: FAILED`, with exit code and
bounded output. That is a *successful tool invocation* but **not a successful
test run**. `TIMED_OUT`, `OUTPUT_LIMIT_EXCEEDED`, and `DOCKER_UNAVAILABLE`
are separate structured statuses.

## Tests

Unit tests use a fake Docker executable, **no Docker daemon required**. These
host-side runner tests are skipped *inside the sandbox* because the container
does not contain a Docker daemon or executable Docker CLI. Other project tests
still run normally in the container.

Tests that create SQLite files use temporary directories under `/tmp` instead
of writing to the read-only `/workspace`. On macOS, the test compares Docker's
mount path against its resolved real path (`/var` may resolve to `/private/var`).

Unit tests use a fake Docker executable, **no Docker daemon required**:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

After building the image, opt in to a real Docker smoke test:

```bash
MINIHARNESS_DOCKER_TEST=1 python -m unittest discover -s tests -p 'test_sandbox_runner.py'
```

## Security boundaries / limitations

- **Do not put credentials in the workspace.** A read-only mount prevents
  modification, not reading. Untrusted tests could print readable secrets
  to tool output. The CLI limits output size, not the information exposed.
  A curated/ephemeral source snapshot is a possible later enhancement.
- Docker Desktop on macOS runs Linux containers inside a Linux VM. Linux
  containers are **not a replacement for a strong VM trust boundary**.
- The host Docker daemon, installed image, SandboxConfig and Tool Registry
  must be trusted. Do **not** mount the Docker socket into the sandbox.
- Resource controls are Docker daemon limits, not a rigorous guarantee
  against all denial-of-service cases. Abrupt host/daemon crashes may prevent
  cleanup; there is no external janitor in v0.1.
- This is an isolated test runner, not a general-purpose shell executor.
  `run_tests` remains `ToolSafety.SIDE_EFFECTING` for conservative retry
  behavior: the previous outcome may be unknown after transport failure.
