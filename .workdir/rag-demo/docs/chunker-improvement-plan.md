# 分片算法改进计划

> **Goal:** 修复结构识别误判，实现"章节优先"的分片策略，并添加文档打分系统

**Architecture:** 
1. 改进标题识别规则，避免将数据行误识别为标题
2. 重构chunk_document的分片流程：按章节优先（不切），仅超大章节才递归切分
3. 添加打分系统评估文档结构质量，决定是否启用语义分割

**Tech Stack:** Python, regex, length tokenization

---

## 文件修改计划

- **Modify:** `app/chunker.py`
  - `_header_level()` - 改进启发式规则
  - `score_document_structure()` - **新增**：打分系统
  - `chunk_document()` - 重构分片流程

---

## Task 1: 修复标题识别规则

**Problem:** "2026 年 10 月 18 日 08:30 - 16:30" 被误识别为标题

**Root Cause:** 启发式规则的score >= 3条件太宽松

**Solution:** 
1. 移除或加强启发式规则的约束条件
2. 不让数据行（包含数字、时间、日期）被识别为标题
3. 优先识别明确的标题格式（中文、数字编号、markdown）

- [ ] **Step 1: 分析当前误判的规则**
  - "2026 年 10 月 18 日 08:30 - 16:30" 符合什么条件？
  - len <= 50? Yes (26字符)
  - 不以句号结尾? Yes (以时间结尾)
  - 含关键词? No
  - score = 2 < 3，不应该被识别... 除非有其他条件

- [ ] **Step 2: 改进_header_level中的启发式规则**
  - 移除或严格化通用启发式规则
  - 只保留明确的格式识别（中文章节、数字编号、markdown）
  - 对日期/时间行添加排除规则

**新增规则：**
```python
# 排除数据行（包含时间、日期、数字密集）
if re.search(r'\d{1,4}[年月日]\d{1,2}|\d{1,2}:\d{2}', s):
    return None  # 时间/日期行不是标题

# 排除标点过多或全数字的行
if re.match(r'^[\d\-\s:]{10,}$', s):
    return None
```

- [ ] **Step 3: 弱化启发式规则的score条件**
  - 改为 score >= 4（而不是 >= 3）
  - 或完全移除启发式规则，只保留明确格式

---

## Task 2: 添加文档打分系统

**Purpose:** 评估文档结构质量，决定是否使用语义分割

**Scoring Criteria:**
- 标题数量（多标题 = 更适合语义分割）
- 标题层级结构（深层级 = 结构清晰）
- 标题与内容的比例（好的文档：标题占10-20%）
- 列表、表格的比例（结构化内容）

- [ ] **Step 1: 设计score_document_structure()函数**

```python
def score_document_structure(text: str) -> Dict:
    """
    Analyze document structure and return scoring info.
    
    Returns:
    {
        'total_score': float,      # 0-100
        'header_density': float,   # 标题行占比
        'structure_depth': int,    # 最大标题层级
        'is_semantic_friendly': bool,  # 是否适合语义分割
        'recommendation': str,     # 推荐策略
    }
    """
```

- [ ] **Step 2: 实现计算逻辑**
  - 遍历所有行，统计标题数量和层级
  - 计算标题行占比
  - 评分 = 标题数 * 10 + 层级深度 * 5 + 其他因素
  - is_semantic_friendly: score > 30 or header_count > 5

- [ ] **Step 3: 在chunk_document中使用打分**
  - 调用score_document_structure获得打分
  - 如果score低，使用fallback策略（全文句子切分）
  - 如果score高，使用语义分割

---

## Task 3: 重构分片流程

**Current Flow (Broken):**
```
split_by_structure() → 生成sections
    ↓
chunk_document() → 为每个section切分
    ↓ (递归切分过度)
生成chunks
```

**New Flow (Correct):**
```
split_by_structure() + score_document_structure()
    ↓
for each section:
    content_size = len(section)
    if content_size <= max_size:
        [section as single chunk]  ← KEY: 不再切分
    else:
        → merge_list_with_intro() ← 保护列表前导
        → is_table_block()? 
           Yes → split_table()
           No  → split_section_recursive()  ← 递归+overlap
```

- [ ] **Step 1: 修改chunk_document逻辑**
  - 计算打分信息
  - 为每个section判断是否需要切分
  - 只有content_size > max_size才进行切分

**Pseudocode:**
```python
for sec in sections:
    content = "\n".join(sec.content_lines).strip()
    if not content:
        continue
    
    # 关键改进：检查大小
    if len(content) <= max_size:
        # 直接作为一个chunk，不再切分
        raw_chunks.append({
            "text": content,
            "section": sec.title,
            "section_path": " > ".join(sec.path),
            "type": "text",
            ...
            "start_char": found_pos,
            "end_char": found_pos + len(content),
        })
        continue
    
    # 超过max_size才进行递归切分
    merged_lines = merge_list_with_intro(content.splitlines())
    content = "\n".join(merged_lines).strip()
    
    # 表格处理
    if is_table_block(content):
        t_chunks = split_table(content, max_size=max_size)
        # ...
    else:
        # 递归切分
        text_chunks = split_section_recursive(content, ...)
        # ...
```

- [ ] **Step 2: 验证overlap工作正常**
  - 检查split_section_recursive中overlap逻辑
  - 确保overlap字符正确重叠到下一个chunk

- [ ] **Step 3: 测试新流程**
  - 上传club_event.txt（小文件 < max_size）
  - 期望：生成1个chunk，内容完整
  - 上传student_affairs_faq.md（大文件 > max_size）
  - 期望：多个chunks，各有start_char/end_char，文本完整，overlap正确

---

## Task 4: 测试与验证

- [ ] **Step 1: 单元测试_header_level改进**
  - Test: "一、活动时间" → level 5 (还是有问题需要改)
  - Test: "2026 年 10 月 18 日 08:30 - 16:30" → None (不是标题)
  - Test: "# 标题" → level 1
  - Test: "1. 项目名称" → level 1

- [ ] **Step 2: 测试score_document_structure**
  - 输入: 校园通知（多标题）→ score > 50
  - 输入: 日常邮件（无标题）→ score < 20

- [ ] **Step 3: 集成测试chunk_document**
  - 上传club_event.txt → 验证1-2个chunks
  - 上传student_affairs_faq.md → 验证开始/结束字符位置正确

- [ ] **Step 4: 浏览器端验证**
  - 高亮显示应该正确对应原文位置
  - 分片序号标签应该清晰可见
  - 左右滚动同步应该正常工作

---

## 关键改变汇总

| 问题     | 原有              | 改进                  |
| -------- | ----------------- | --------------------- |
| 标题识别 | 启发式score >= 3  | 排除数据行 + 提高阈值 |
| 分片策略 | 所有section都切分 | 仅超大section才切分   |
| 文档评估 | 无                | 打分系统 + 推荐策略   |
| 列表处理 | 有                | 继续保留              |
| overlap  | 有                | 继续保留并验证        |

