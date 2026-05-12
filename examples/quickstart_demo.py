"""Quickstart demo — verifies the README Quickstart code actually runs.

Run:
    python examples/quickstart_demo.py

This file IS the README Quickstart, lifted verbatim. Tests import + execute
it (see ``tests/test_readme_quickstart.py``) so the README cannot drift
from runnable code.
"""

from __future__ import annotations


def main() -> None:
    from agent_amplifier import AgentAmplifier

    # No adapter required for a basic envelope. Pass `adapter=` to bind
    # CLAUDE.md / Cursor MDC / Copilot instructions / etc. as memory.
    amp = AgentAmplifier()
    try:
        envelope = amp.before_step("refactor the auth module")

        # `envelope.envelope` carries the amplified system prompt:
        # effort routing, goal anchor, persona, recalled patterns.
        assert isinstance(envelope.envelope, str)
        assert "refactor" in envelope.envelope.lower()

        # ...run your agent step using envelope.envelope as system prompt...

        # Persist the outcome (no-op without a memory provider, but the
        # call is always safe).
        decision = amp.after_step(envelope, "auth refactor done")
        assert decision["action"] in {"continue", "stop", "re_anchor"}

        print(f"OK: classification={envelope.classification.complexity.value} "
              f"phase={envelope.phase} action={decision['action']}")
    finally:
        amp.close()


if __name__ == "__main__":
    main()
