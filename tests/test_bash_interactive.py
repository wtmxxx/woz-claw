#!/usr/bin/env python3
"""
Interactive bash_command tool testing script.

Usage:
    python test_bash_interactive.py

Commands:
    Type any bash command to test it.
    Type 'exit' or 'quit' to exit.
    Type 'help' to see available commands.
    Type 'pwd' to show current working directory.
"""

from pathlib import Path
from wozclaw.agent import ReActMemoryAgent
from wozclaw.memory_store import MemoryStore


def main():
    # Initialize agent with temporary memory store
    tmp_root = Path("..")
    memory_store = MemoryStore(root_dir=tmp_root)
    agent = ReActMemoryAgent(memory_store=memory_store,
                             user_id="test-user", session_id="test-session")

    # Get work directory info
    work_dir = agent._root_work_dir()
    sandbox_dir = agent._wozclaw_dir()

    print("="*80)
    print("🧪 Interactive Bash Tool Tester")
    print("="*80)
    print(f"\n📁 Work Directory (root/):     {work_dir}")
    print(f"📁 Sandbox Directory (.sandbox/): {sandbox_dir}")
    print(f"\nℹ️  Type 'help' for commands or type any bash command to test")
    print("-"*80 + "\n")

    while True:
        try:
            cmd = input("bash> ").strip()

            if not cmd:
                continue

            if cmd.lower() in ("exit", "quit"):
                print("\n👋 Goodbye!")
                break

            if cmd.lower() == "help":
                print_help()
                continue

            if cmd.lower() == "pwd":
                cmd = "pwd"

            print(f"\n🔧 Running: {cmd}")
            print("-" * 80)

            # Show work directory info
            work_dir = agent._root_work_dir()
            print(f"📁 work_dir: {work_dir}")

            # Show what the command expands to
            expanded = agent._expand_bash_aliases(cmd)
            print(f"📝 Expanded: {expanded}")
            print("-" * 80)

            result = agent._run_bash_command(cmd)

            print(f"Result:")
            print(result)
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def print_help():
    help_text = """
Available commands/shortcuts:
  help              - Show this help message
  pwd               - Show current working directory
  exit/quit         - Exit the program

Example bash commands to test:
  ✓ ls -la root/                    - List files in workdir with details
  ✓ find root/ -type f              - Find all files in workdir
  ✓ cat root/README.md              - Read a file
  ✓ find root/ -type f | head -10   - Pipe example
  ✓ ls root/a && ls root/b          - Multiple commands with &&
  ✓ pwd                             - Show current working directory
  ✓ ls .sandbox                     - List sandbox/config directory

Tips:
  • root/        = your workdir (mapped from sandbox.yaml)
  • .sandbox/    = project root directory
  • All paths starting with root/ are automatically expanded
  • Commands run with bash -lc for login shell support
"""
    print(help_text)


if __name__ == "__main__":
    main()
