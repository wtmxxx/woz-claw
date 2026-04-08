from agent_memory_demo.agent import ReActMemoryAgent
from agent_memory_demo.memory_store import MemoryStore


def test_memory_prompt_requires_full_memory_rewrite(tmp_path) -> None:
    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u1", session_id="s1")

    prompt = agent.build_system_prompt("[LONG_TERM]\n用户喜欢游泳")

    assert "完整" in prompt
    assert "不是增量" in prompt
    assert "remember_note" in prompt
    assert "旧信息" in prompt
    assert "搜索" in prompt
    assert "message_id" in prompt
    assert "get_session_window" in prompt
    assert "get_daily_window" in prompt
    assert "优先调用工具" in prompt
    assert "能调用工具就调用工具" in prompt
    assert "只有在工具确实无法解决问题时" in prompt
