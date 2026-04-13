# Copilot Instructions for WozClaw

## Build, Test, and Lint Commands

- **Install dependencies:**
  ```bash
  python -m pip install -e .[dev]
  ```
- **Run the development server:**
  ```bash
  python -m uvicorn wozclaw.app:app --reload
  ```
- **Run all tests:**
  ```bash
  pytest -q
  ```
- **Run a single test:**
  ```bash
  pytest -q tests/test_agent_prompt.py::test_name
  ```
- **Linting:**
  Ruff is referenced in .gitignore, but no explicit config found. If used, run:
  ```bash
  ruff check .
  ```

## High-Level Architecture

- **Core:**
  - Built on FastAPI (see `src/wozclaw/app.py`).
  - Main entry: `wozclaw.app:app`.
  - Memory system is central, with layered storage: long-term, session, daily, and logs (see `memory/` and `src/wozclaw/memory_store.py`).
  - Skills system: global/user skills configured via YAML in `.wozclaw/skills/`.
- **API Endpoints:**
  - `/api/chat`, `/api/conversations`, `/api/settings`, `/api/approvals/decide`, etc.
- **Frontend:**
  - Single-page chat interface (see README for details).

## Key Conventions

- **Memory Storage:**
  - Layered by user: `memory/{user_id}/` with subfolders for sessions, daily logs, and LLM dialogue.
  - Message format is JSONL with unified fields (`ts`, `role`, `content`, `tags`, `meta`, `message_id`).
- **Skills:**
  - Global: `.wozclaw/skills/global/skills.yaml`
  - User: `.wozclaw/skills/{user_id}/skills.yaml`
  - Only skills with `enabled: true` and a `SKILL.md` are loaded.
  - User config overrides global for same-named skills.
- **Bash Command Policy:**
  - Read operations allowed by default.
  - Write/delete operations require human approval unless explicitly allowed.
- **Testing:**
  - Tests are in `tests/`, use `pytest`.
  - Test names and structure follow standard pytest conventions.

## Integration with Other AI Assistants

No CLAUDE.md, AGENTS.md, .cursorrules, .windsurfrules, or similar config files found. No special integration rules detected.

---

This file summarizes build/test/lint commands, architecture, and key conventions for Copilot and other AI assistants. Would you like to adjust anything or add coverage for other areas (e.g., deployment, advanced skills, or API usage)?
