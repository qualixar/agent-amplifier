# Contributing to agent-amplifier

Thanks for the interest. This is a beta-stage open-source project and we
welcome PRs that match the quality bar already set by the existing code.

## Quick start

```bash
git clone https://github.com/qualixar/agent-amplifier
cd agent-amplifier
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/                  # 1,100+ tests, 100% line + branch coverage
```

The full quality matrix (run before every PR):

```bash
.venv/bin/python -m pytest tests/ --cov=src/agent_amplifier --cov-branch --cov-fail-under=100
.venv/bin/python -m mypy --strict src/agent_amplifier
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m bandit -r src/agent_amplifier -ll
.venv/bin/pip-audit
```

CI runs the same matrix on Linux, macOS, and Windows across Python
3.11 / 3.12 / 3.13 / 3.14.

## What we accept

- **Bug fixes** with a regression test that fails before the fix and
  passes after.
- **New adapters** following [`docs/adapter-spec.md`](docs/adapter-spec.md).
  Twelve community-PR slots are open: LlamaIndex, AutoGen, Pydantic AI,
  Anthropic Agent SDK, OpenAI Agents Session, Semantic Kernel, Aider,
  Cline, Continue.dev, Windsurf, Antigravity, DSPy.
- **Docs improvements** — typo fixes, broken links, clearer phrasing.
- **CI / packaging** changes that strengthen the release process.

## What we do NOT accept

- Adding a runtime dependency without a strong justification.  The
  amplifier ships with a single dep (`anyio`) and we keep it that way.
- Architectural rewrites without a prior issue / discussion.
- Code that broadens the public API without test + docstring + spec.
- Changes that lower coverage below 100% line + branch.

## Adapter contract

Read [`docs/adapter-spec.md`](docs/adapter-spec.md) before writing one.
Key non-negotiables:

1. `default_memory_recall` and `default_memory_remember` MUST NOT raise.
2. Framework imports MUST be lazy (inside methods, never at module
   top-level) so the package imports cleanly without the framework.
3. Provide tests against mock framework objects — real-framework
   integration tests live in optional CI extras.
4. Set `INSTALL_PERSISTENT = True` on the class only if `install()`
   writes persistent state (most file-based adapters leave the default
   `False`).

## Pull request workflow

1. Fork, branch, write code + tests.
2. Run the full quality matrix locally.
3. Open the PR with a clear description of the *why*, not just the
   *what*.  Include a `CHANGELOG.md` entry under `[Unreleased]`.
4. CI must be green before review.
5. Review may take a few days.  We are small.

## Reporting security issues

Security issues SHOULD NOT be opened as public issues.  Email
`hi@qualixar.com` with the details.  We respond within five business
days.

## Code of conduct

Be professional.  This is a senior-engineer-to-senior-engineer
collaboration; we have no tolerance for harassment or low-effort
trolling and will close PRs / issues that depart from that standard.

## License

By contributing, you agree your contribution is licensed under
Apache-2.0 (see [`LICENSE`](LICENSE)).
