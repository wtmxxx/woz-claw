import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback for environments without tiktoken
    tiktoken = None


# Use a real tokenizer in production; fall back to character count only if unavailable.
@lru_cache(maxsize=1)
def _get_token_encoder():
    if tiktoken is None:
        return None

    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def tokenize_len(text: str) -> int:
    encoder = _get_token_encoder()
    if encoder is None:
        return len(text)
    return len(encoder.encode(text))


def _normalize_with_index_map(text: str) -> Dict:
    """Normalize whitespace and keep mapping to original indices."""
    norm_chars: List[str] = []
    index_map: List[int] = []
    in_ws = False

    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                norm_chars.append(" ")
                index_map.append(i)
                in_ws = True
        else:
            norm_chars.append(ch)
            index_map.append(i)
            in_ws = False

    return {"text": "".join(norm_chars), "map": index_map}


def _find_best_span(text: str, anchor_text: str, cursor: int) -> Dict:
    """Find best span for anchor_text in original text.

    Tries exact match first, then relaxed whitespace-normalized match.
    """
    if not anchor_text:
        return {"start": cursor, "end": cursor}

    variants: List[str] = [anchor_text]
    if anchor_text != anchor_text.lstrip("\n"):
        variants.append(anchor_text.lstrip("\n"))
    if anchor_text != anchor_text.strip():
        variants.append(anchor_text.strip())

    seen = set()
    unique_variants: List[str] = []
    for v in variants:
        if v and v not in seen:
            unique_variants.append(v)
            seen.add(v)

    # 1) Exact match preferring positions at/after cursor.
    for v in unique_variants:
        pos = text.find(v, cursor)
        if pos >= 0:
            return {"start": pos, "end": pos + len(v)}

    # 2) Exact global match fallback.
    for v in unique_variants:
        pos = text.find(v)
        if pos >= 0:
            return {"start": pos, "end": pos + len(v)}

    # 3) Whitespace-normalized fallback.
    norm_src = _normalize_with_index_map(text)
    norm_anchor = " ".join(anchor_text.split())
    if norm_anchor:
        npos = norm_src["text"].find(norm_anchor)
        if npos >= 0:
            start = norm_src["map"][npos]
            end_norm_idx = npos + len(norm_anchor) - 1
            end = norm_src["map"][end_norm_idx] + 1
            return {"start": start, "end": end}

    # 4) Last resort: clamp to cursor with anchor length.
    start = max(0, min(cursor, len(text)))
    end = max(start, min(len(text), start + len(anchor_text)))
    return {"start": start, "end": end}


def rematch_chunk_positions(text: str, chunks: List[Dict]) -> None:
    """Re-match every chunk against original text to stabilize highlight spans."""
    cursor = 0
    for chunk in chunks:
        anchor = chunk.get("source_text") or chunk.get("text", "")
        span = _find_best_span(text, anchor, cursor)
        chunk["start_char"] = span["start"]
        chunk["end_char"] = span["end"]
        cursor = max(cursor, span["end"])


@dataclass
class Section:
    title: str
    level: int
    path: List[str]
    content_lines: List[str]


def _header_level(line: str) -> Optional[int]:
    s = line.strip()
    if not s:
        return None

    # EXCLUSION RULES: 排除数据行，不让其被识别为标题
    # 时间/日期行：2026年10月18日、08:30-16:30 等
    if re.search(r'\d{1,4}[年月日]\d{1,2}|\d{1,2}:\d{2}', s):
        return None
    # 全数字或数字+符号的行（如时间段、日期段）
    if re.match(r'^[\d\-\s:/.年月日]{8,}$', s):
        return None

    # Chinese style: 第X章 / 第X节 / 第X条
    if re.match(r"^第[一二三四五六七八九十百千万\d]+章", s):
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万\d]+节", s):
        return 2
    if re.match(r"^第[一二三四五六七八九十百千万\d]+条", s):
        return 3

    # Chinese enumeration: 一、 二、 三、 (commonly used in campus docs)
    if re.match(r"^[一二三四五六七八九十]+[、.](\s|$)", s):
        return 4

    # Numbered heading: 1 / 1.2 / 1.2.3
    m = re.match(r"^(\d+(?:\.\d+){0,5})\s+", s)
    if m:
        return m.group(1).count(".") + 1

    # Markdown heading
    m_md = re.match(r"^(#{1,6})\s+", s)
    if m_md:
        return len(m_md.group(1))

    # Bracket style section title: （一） / （二） - only for Chinese numerals, not Arabic
    # 注意：只识别中文数字编号的括号标题，例如"（一）XXX"
    # 不识别"（1）"开头的行，因为这通常是列表项
    if re.match(r"^（[一二三四五六七八九十]+）\s*.{2,}$", s):
        return 5

    # Heuristic: short, non-punctuation line with content keywords
    # RESTRICTION: 提高score阈值从3到4，减少误判
    score = 0
    if len(s) <= 40:
        score += 1
    if not s.endswith(("。", ".", "；", ";", ":", "：", ",", "，", "、")):
        score += 1
    if re.search(r"(总则|定义|范围|职责|要求|流程|附则|说明|指南|规范)", s):
        score += 1
    # 必须score >= 4以上才认为是标题（提高阈值）
    if score >= 4:
        return 6

    return None


def split_by_structure(text: str) -> List[Section]:
    lines = text.splitlines()
    sections: List[Section] = []

    root = Section(title="ROOT", level=0, path=["ROOT"], content_lines=[])
    current = root
    path_stack: List[str] = ["ROOT"]
    level_stack: List[int] = [0]

    for raw in lines:
        line = raw.rstrip("\n")
        level = _header_level(line)

        if level is None:
            current.content_lines.append(line)
            continue

        if current.content_lines:
            sections.append(current)

        while level_stack and level_stack[-1] >= level:
            level_stack.pop()
            path_stack.pop()

        level_stack.append(level)
        path_stack.append(line.strip())

        current = Section(
            title=line.strip(),
            level=level,
            path=path_stack.copy(),
            content_lines=[],
        )

    if current.content_lines:
        sections.append(current)

    return sections


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;])", text)
    return [p for p in parts if p and p.strip()]


def is_list_item(line: str) -> bool:
    s = line.strip()
    return bool(
        re.match(r"^(（\d+）|\(\d+\)|\d+\.|[-*]\s+)", s)
    )


def merge_list_with_intro(lines: List[str]) -> List[str]:
    result: List[str] = []
    intro: Optional[str] = None
    buffer: List[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            if buffer:
                if intro:
                    result.append(intro + "\n" + "\n".join(buffer))
                else:
                    result.extend(buffer)
                intro = None
                buffer = []
            elif intro:
                # Keep intro in-place when it is not followed by list items.
                result.append(intro)
                intro = None
            result.append(line)
            continue

        if is_list_item(s):
            buffer.append(line)
            continue

        if buffer:
            if intro:
                result.append(intro + "\n" + "\n".join(buffer))
            else:
                result.extend(buffer)
            intro = None
            buffer = []
        elif intro:
            # Non-list content started: flush cached intro before current line.
            result.append(intro)
            intro = None

        if s.endswith((":", "：")) or "以下" in s or "如下" in s:
            intro = line
        else:
            result.append(line)

    if buffer:
        if intro:
            result.append(intro + "\n" + "\n".join(buffer))
        else:
            result.extend(buffer)

    if intro and not buffer:
        result.append(intro)

    return result


def split_intro_list_block(text: str, max_size: int) -> List[Dict]:
    """Split an intro+list block while preserving intro in every chunk.

    Expected block shape:
    - Intro line ending with ':'/'：' or containing '以下'/'如下'
    - Followed by list-item lines only
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    intro = lines[0]
    intro_norm = intro.strip()
    if not (intro_norm.endswith((":", "：")) or "以下" in intro_norm or "如下" in intro_norm):
        return []

    items = lines[1:]
    if not items or not all(is_list_item(it) for it in items):
        return []

    chunks: List[Dict] = []
    current_items: List[str] = []
    cursor = 0
    emitted_count = 0

    def flush(items_to_flush: List[str]) -> None:
        nonlocal cursor, emitted_count
        if not items_to_flush:
            return
        chunk_text = intro + "\n" + "\n".join(items_to_flush)
        source_text = chunk_text if emitted_count == 0 else "\n" + \
            "\n".join(items_to_flush)
        start_pos = text.find(source_text, cursor)
        if start_pos < 0:
            start_pos = cursor
        end_pos = start_pos + len(source_text)
        chunks.append({
            "text": chunk_text,
            "source_text": source_text,
            "start_char": start_pos,
            "end_char": end_pos,
        })
        cursor = end_pos
        emitted_count += 1

    for item in items:
        candidate_items = current_items + [item]
        candidate_text = intro + "\n" + "\n".join(candidate_items)

        if current_items and tokenize_len(candidate_text) > max_size:
            flush(current_items)
            current_items = [item]
        else:
            current_items = candidate_items

    flush(current_items)
    return chunks


def is_table_block(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    pipe_lines = sum(1 for ln in lines if "|" in ln)
    tsv_like = sum(1 for ln in lines if "\t" in ln)
    return pipe_lines >= 2 or tsv_like >= 2


def split_table(text: str, max_size: int = 800) -> List[Dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return [{"text": text, "source_text": text, "start_char": 0, "end_char": len(text)}]

    # Markdown table: keep header row + separator row in every chunk.
    header = lines[0]
    rows = lines[1:]
    if len(lines) >= 3 and "|" in lines[0] and re.match(r"^\s*\|?[\s:\-]+\|[\s\|:\-]*$", lines[1]):
        header = lines[0] + "\n" + lines[1]
        rows = lines[2:]

    chunks: List[Dict] = []
    current_rows: List[str] = []
    current_len = tokenize_len(header)
    cursor = 0
    emitted_count = 0

    for row in rows:
        row_len = tokenize_len(row)
        if current_rows and current_len + row_len > max_size:
            chunk_text = header + "\n" + "\n".join(current_rows)
            source_text = chunk_text if emitted_count == 0 else "\n" + \
                "\n".join(current_rows)
            chunk_start = text.find(source_text, cursor)
            if chunk_start < 0:
                chunk_start = cursor
            chunk_end = chunk_start + len(source_text)
            chunks.append({
                "text": chunk_text,
                "source_text": source_text,
                "start_char": max(0, chunk_start),
                "end_char": chunk_end
            })
            cursor = chunk_end
            emitted_count += 1
            current_rows = []
            current_len = tokenize_len(header)

        current_rows.append(row)
        current_len += row_len

    if current_rows:
        chunk_text = header + "\n" + "\n".join(current_rows)
        source_text = chunk_text if emitted_count == 0 else "\n" + \
            "\n".join(current_rows)
        chunk_start = text.find(source_text, cursor)
        if chunk_start < 0:
            chunk_start = cursor
        chunk_end = chunk_start + len(source_text)
        chunks.append({
            "text": chunk_text,
            "source_text": source_text,
            "start_char": max(0, chunk_start),
            "end_char": chunk_end
        })

    return chunks


def sentence_chunk(text: str, max_size: int = 1024, overlap: int = 100) -> List[Dict]:
    """Split text into sentence chunks, returning each with its character positions in the original text."""
    sentences = split_sentences(text)
    if not sentences:
        if text.strip():
            return [{"text": text, "start_char": 0, "end_char": len(text)}]
        return []

    chunks: List[Dict] = []
    cur: List[str] = []
    cur_len = 0
    cur_start_char = 0  # Track start position in original text
    src_pos = 0  # Current position in original text

    for sent in sentences:
        l = tokenize_len(sent)
        if cur and cur_len + l > max_size:
            chunk_text = "".join(cur)
            # Find the actual start/end in original text by searching
            chunk_start = text.find(chunk_text, cur_start_char)
            if chunk_start >= 0:
                chunk_end = chunk_start + len(chunk_text)
            else:
                chunk_start = cur_start_char
                chunk_end = cur_start_char + cur_len
            chunks.append({
                "text": chunk_text,
                "start_char": chunk_start,
                "end_char": chunk_end
            })

            overlap_part: List[str] = []
            overlap_len = 0
            if overlap > 0:
                for prev in reversed(cur):
                    overlap_part.insert(0, prev)
                    overlap_len += tokenize_len(prev)
                    if overlap_len >= overlap:
                        break

            cur = overlap_part
            cur_len = sum(tokenize_len(x) for x in cur)
            # Reset start position for next chunk's overlap
            overlap_text = "".join(cur)
            cur_start_char = text.find(overlap_text, chunk_start)
            if cur_start_char < 0:
                cur_start_char = 0

        cur.append(sent)
        cur_len += l

    if cur:
        chunk_text = "".join(cur)
        chunk_start = text.find(chunk_text, cur_start_char)
        if chunk_start >= 0:
            chunk_end = chunk_start + len(chunk_text)
        else:
            chunk_start = cur_start_char
            chunk_end = cur_start_char + cur_len
        chunks.append({
            "text": chunk_text,
            "start_char": chunk_start,
            "end_char": chunk_end
        })

    return [c for c in chunks if c.get("text", "").strip()]


def split_section_recursive(text: str, max_size: int, overlap: int) -> List[Dict]:
    intro_list_chunks = split_intro_list_block(text, max_size=max_size)
    if intro_list_chunks:
        return intro_list_chunks

    if tokenize_len(text) <= max_size:
        return [{"text": text, "start_char": 0, "end_char": len(text)}] if text.strip() else []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        out: List[Dict] = []
        current_pos = 0
        for p in paragraphs:
            # Find this paragraph's position in original text
            para_start = text.find(p, current_pos)
            if para_start < 0:
                para_start = current_pos

            if tokenize_len(p) <= max_size:
                out.append({"text": p, "start_char": para_start,
                           "end_char": para_start + len(p)})
                current_pos = para_start + len(p)
            else:
                para_chunks = split_intro_list_block(p, max_size=max_size)
                if not para_chunks:
                    para_chunks = sentence_chunk(
                        p, max_size=max_size, overlap=overlap)
                for chunk in para_chunks:
                    # Adjust positions relative to the section text
                    chunk["start_char"] = para_start + chunk["start_char"]
                    chunk["end_char"] = para_start + chunk["end_char"]
                    out.append(chunk)
                if para_chunks:
                    current_pos = para_chunks[-1]["end_char"]
        return out

    return sentence_chunk(text, max_size=max_size, overlap=overlap)


def is_key_content(section_title: str, text: str) -> bool:
    content = f"{section_title}\n{text}"
    return bool(re.search(r"(必须|不得|禁止|应当|责任|处罚|违约|保密|安全|风险)", content))


def score_document_structure(text: str) -> Dict:
    """
    Analyze document structure and return scoring info.

    Scoring criteria:
    - Header count and depth (more headers = better structure)
    - Title-to-content ratio (should be 5-20%)
    - List and table presence (structured content)

    Returns:
    {
        'total_score': float,           # 0-100
        'header_count': int,            # 标题行数
        'max_depth': int,               # 最大标题层级
        'structure_quality': str,       # 'high'|'medium'|'low'
        'is_semantic_friendly': bool,   # 是否适合语义分割
    }
    """
    lines = text.splitlines()
    header_count = 0
    max_depth = 0
    list_item_count = 0
    table_block_count = 0

    for line in lines:
        level = _header_level(line)
        if level is not None:
            header_count += 1
            max_depth = max(max_depth, level)

        if is_list_item(line.strip()):
            list_item_count += 1

    # Check for table blocks (simple heuristic)
    text_blocks = text.split('\n\n')
    for block in text_blocks:
        if is_table_block(block):
            table_block_count += 1

    # Calculate scoring
    # Base score from header count: each header worth 10 points, capped at 50
    header_score = min(header_count * 10, 50)

    # Depth bonus: each level adds 5 points, capped at 20
    depth_score = min(max_depth * 5, 20)

    # Structure diversity bonus: lists + tables = 10-20 points
    diversity_score = min((list_item_count // 3) + (table_block_count * 5), 20)

    total_score = header_score + depth_score + diversity_score
    total_score = min(total_score, 100)

    # Determine structure quality
    if total_score >= 60:
        quality = 'high'
        semantic_friendly = True
    elif total_score >= 35:
        quality = 'medium'
        semantic_friendly = (header_count >= 3)
    else:
        quality = 'low'
        semantic_friendly = False

    return {
        'total_score': total_score,
        'header_count': header_count,
        'max_depth': max_depth,
        'list_count': list_item_count,
        'table_count': table_block_count,
        'structure_quality': quality,
        'is_semantic_friendly': semantic_friendly,
    }


def chunk_document(
    text: str,
    base_max_size: int = 1024,
    key_content_max_size: int = 1536,
    overlap: int = 100,
) -> Dict:
    sections = split_by_structure(text)
    doc_structure = score_document_structure(text)

    raw_chunks: List[Dict] = []
    global_offset = 0
    strategy_tags: List[str] = ["章节优先"]
    used_recursive_split = False
    used_table_split = False

    def add_strategy(tag: str) -> None:
        if tag not in strategy_tags:
            strategy_tags.append(tag)

    for sec in sections:
        content = "\n".join(sec.content_lines)
        if not content.strip():
            section_title_line = sec.title
            next_offset = text.find(section_title_line, global_offset)
            if next_offset >= 0:
                global_offset = next_offset + len(section_title_line) + 1
            continue

        merged_lines = merge_list_with_intro(content.splitlines())
        content = "\n".join(merged_lines)

        key_flag = is_key_content(sec.title, content)
        max_size = key_content_max_size if key_flag else base_max_size
        content_len = tokenize_len(content)

        # ===== 表格优先：无论大小都按表格分支处理，保证统计准确 =====
        if is_table_block(content):
            used_table_split = True
            add_strategy("表格切分")
            # 表格处理：支持“前导说明 + Markdown 表格”的混合内容
            lines = content.splitlines()
            table_start = None
            for i in range(len(lines) - 1):
                if "|" in lines[i] and re.match(r"^\s*\|?[\s:\-]+\|[\s\|:\-]*$", lines[i + 1]):
                    table_start = i
                    break

            prefix_text = ""
            table_text = content
            if table_start is not None:
                prefix_text = "\n".join(lines[:table_start]).strip()
                table_text = "\n".join(lines[table_start:]).strip()

            if prefix_text:
                pos = text.find(prefix_text, global_offset)
                if pos < 0:
                    pos = global_offset
                start_pos = pos
                end_pos = pos + len(prefix_text)
                raw_chunks.append({
                    "text": prefix_text,
                    "source_text": prefix_text,
                    "section": sec.title,
                    "section_path": " > ".join(sec.path),
                    "type": "text",
                    "is_key_content": key_flag,
                    "token_len": tokenize_len(prefix_text),
                    "start_char": start_pos,
                    "end_char": end_pos,
                })
                global_offset = end_pos

            t_chunks = split_table(table_text, max_size=max_size)
            for t_chunk in t_chunks:
                chunk_text = t_chunk["text"]
                anchor_text = t_chunk.get("source_text", chunk_text)
                pos = text.find(anchor_text, global_offset)
                if pos < 0:
                    pos = global_offset
                start_pos = pos
                end_pos = pos + len(anchor_text)

                raw_chunks.append({
                    "text": chunk_text,
                    "source_text": anchor_text,
                    "section": sec.title,
                    "section_path": " > ".join(sec.path),
                    "type": "table",
                    "is_key_content": key_flag,
                    "token_len": tokenize_len(chunk_text),
                    "start_char": start_pos,
                    "end_char": end_pos,
                })
                global_offset = end_pos
            continue

        # ===== KEY CHANGE: 章节优先策略 =====
        # 如果章节内容小于max_size，直接作为一个chunk，不再切分
        if content_len <= max_size:
            pos = text.find(content, global_offset)
            if pos < 0:
                pos = global_offset
            start_pos = pos
            end_pos = pos + len(content)

            raw_chunks.append({
                "text": content,
                "source_text": content,
                "section": sec.title,
                "section_path": " > ".join(sec.path),
                "type": "text",
                "is_key_content": key_flag,
                "token_len": content_len,
                "start_char": start_pos,
                "end_char": end_pos,
            })
            global_offset = end_pos
            continue

        # 文本递归切分（仅当超过max_size时）
        used_recursive_split = True
        add_strategy("递归切分")
        if overlap > 0:
            add_strategy("句子级 overlap")
        text_chunks = split_section_recursive(
            content, max_size=max_size, overlap=overlap)
        for t_chunk in text_chunks:
            chunk_text = t_chunk["text"]
            anchor_text = t_chunk.get("source_text", chunk_text)
            pos = text.find(anchor_text, global_offset)
            if pos < 0:
                pos = global_offset
            start_pos = pos
            end_pos = pos + len(anchor_text)

            raw_chunks.append({
                "text": chunk_text,
                "source_text": anchor_text,
                "section": sec.title,
                "section_path": " > ".join(sec.path),
                "type": "text",
                "is_key_content": key_flag,
                "token_len": tokenize_len(chunk_text),
                "start_char": start_pos,
                "end_char": end_pos,
            })
            global_offset = end_pos

    # Final pass: re-match each chunk span on source text.
    rematch_chunk_positions(text, raw_chunks)

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    total = len(raw_chunks)
    for idx, item in enumerate(raw_chunks):
        chunk_id = f"{doc_id}_{idx:04d}"
        item["chunk_id"] = chunk_id
        item["prev_chunk_id"] = f"{doc_id}_{idx-1:04d}" if idx > 0 else None
        item["next_chunk_id"] = f"{doc_id}_{idx+1:04d}" if idx < total - 1 else None
        item["weight"] = 1.5 if item["is_key_content"] else 1.0
        item["chunk_index"] = idx
        item["total_chunks"] = total

    summary = {
        "total_chunks": total,
        "text_chunks": sum(1 for c in raw_chunks if c["type"] == "text"),
        "table_chunks": sum(1 for c in raw_chunks if c["type"] == "table"),
        "avg_len": round(sum(c["token_len"] for c in raw_chunks) / total, 2) if total else 0,
        "doc_structure_score": doc_structure["total_score"],
        "structure_quality": doc_structure["structure_quality"],
        "split_strategy": " / ".join(strategy_tags),
    }

    return {
        "doc_id": doc_id,
        "summary": summary,
        "chunks": raw_chunks,
    }
