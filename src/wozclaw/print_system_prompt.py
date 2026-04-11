from __future__ import annotations

import argparse
from pathlib import Path

from wozclaw.agent import ReActMemoryAgent
from wozclaw.memory_store import MemoryStore


def create_prompt_agent(user_id: str, session_id: str) -> ReActMemoryAgent:
    store = MemoryStore(root_dir=Path("memory"))
    return ReActMemoryAgent(memory_store=store, user_id=user_id, session_id=session_id)


def build_memory_context(
    user_id: str,
    session_id: str,
    memory_root: Path | str = Path("memory"),
) -> str:
    store = MemoryStore(root_dir=memory_root)
    context = store.load_context(
        user_id,
        session_id,
        query="",
        session_limit=0,
        session_token_budget=2800,
        daily_limit=0,
    )
    return store.build_long_term_prompt_context(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the WozClaw system prompt.")
    parser.add_argument("--user-id", default="demo-user",
                        help="User ID used when rendering the prompt.")
    parser.add_argument("--session-id", default="demo-session",
                        help="Session ID used when rendering the prompt.")
    parser.add_argument("--memory-root", default="memory",
                        help="Root directory for memory files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = create_prompt_agent(args.user_id, args.session_id)
    memory_context = build_memory_context(
        args.user_id,
        args.session_id,
        args.memory_root,
    )
    print(agent.build_system_prompt(memory_context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
