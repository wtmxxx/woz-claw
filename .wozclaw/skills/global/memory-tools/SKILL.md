---
name: memory-tools
description: Use memory tools first to verify facts, prefer retrieval over guessing, and state uncertainty when evidence is missing.
---

# Memory Tools Skill

When answering user questions, follow this workflow:

1. Check whether `search_session` or `search_daily` can verify the claim.
2. If you find a hit, use `get_session_window` or `get_daily_window` to expand the context around the relevant `message_id`.
3. Base the answer on retrieved evidence only.
4. If retrieval finds nothing, say that the evidence is missing and avoid guessing.
5. When a user asks about memory itself, prefer concise, evidence-backed answers over broad summaries.
