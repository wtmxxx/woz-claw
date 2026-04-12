# 左右布局 + 原文高亮 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task.

**Goal:** 实现原文左侧、分片右侧的两列布局，支持在原文中按不同颜色高亮显示每个分片位置，并显示分片序号标签。

**Architecture:** 
- 后端：修改 chunker.py 的 chunk 返回值，为每个 chunk 添加 `start_char` 和 `end_char` 字段，记录在原文中的字符位置
- 前端布局：将 `originalSection` 和 `chunksSection` 改为两列布局（原文左、分片右）
- 前端高亮：解析每个 chunk 的字符范围，使用 `<mark>` 或 `<span>` 标签在原文中进行高亮，添加分片序号标签
- 交互同步：点击右侧分片卡片时，左侧原文自动滚动到对应位置并高亮

**Tech Stack:**
- Backend: Python, FastAPI, 字符位置计算
- Frontend: HTML5, Vanilla JS, CSS Grid/Flexbox

---

## 文件列表

**修改文件：**
- `app/chunker.py` - 为 chunk 添加字符位置计算
- `templates/index.html` - 改为两列布局结构
- `static/app.js` - 实现高亮逻辑和交互同步
- `static/style.css` - 新增两列布局样式和高亮标签样式

**不需要新建文件**

---

## Task 1: 后端字符位置计算

**Files:**
- Modify: `app/chunker.py` - chunk_document() 函数

**功能说明：**
- 在 chunk_document() 中，追踪每个 chunk 在原始 text 中的字符位置
- 为每个 chunk dict 添加 `"start_char"` 和 `"end_char"` 字段
- 需要在 split_by_structure()、sentence_chunk()、split_table() 等处追踪位置

**实现方式：**
- 使用 callback 或者直接在分块逻辑中维护字符偏移量
- 返回 chunk 时附带位置信息

- [ ] **Step 1: 查看当前 chunker.py 结构，理解 chunk 生成流程**
- [ ] **Step 2: 修改 chunk dict，添加 start_char 和 end_char 字段**
- [ ] **Step 3: 测试后端返回的 chunk 是否包含正确的字符位置**

---

## Task 2: 前端两列布局

**Files:**
- Modify: `templates/index.html` - 结果区域部分
- Modify: `static/style.css` - 添加两列布局样式

**功能说明：**
- 将 summarySection、originalSection、chunksSection 改为：上方是 summary KPI，下方是两列布局（originalSection 左侧、chunksSection 右侧）
- 左侧原文建议固定宽度或响应式，右侧分片列表自适应

- [ ] **Step 1: 修改 HTML 结构，调整 originalSection 和 chunksSection 的父容器**
- [ ] **Step 2: 在 CSS 中添加两列网格布局样式**
- [ ] **Step 3: 验证布局在桌面和移动端都能正常显示**

---

## Task 3: 原文高亮逻辑

**Files:**
- Modify: `static/app.js` - renderChunks() 或新增 highlightChunksInOriginal() 函数
- Modify: `static/style.css` - 高亮标签样式

**功能说明：**
- 在 renderChunks() 中，调用高亮函数处理原文
- 遍历每个 chunk，根据 start_char 和 end_char，在原文的 `<pre>` 元素中插入 `<mark>` 或 `<span>` 标签
- 每个高亮区域上方（或右上角）显示分片序号标签（圆角样式）
- 使用 8~12 种不同颜色循环高亮

- [ ] **Step 1: 设计高亮颜色数组（8-12 种颜色）**
- [ ] **Step 2: 实现 highlightChunksInOriginal(text, chunks) 函数，返回 HTML 高亮版本**
- [ ] **Step 3: 在 renderChunks() 中调用高亮函数并更新原文显示**
- [ ] **Step 4: 添加高亮标签和序号标签的 CSS 样式**

---

## Task 4: 交互同步

**Files:**
- Modify: `static/app.js` - 添加点击事件监听和滚动逻辑

**功能说明：**
- 当用户点击右侧的分片卡片时，左侧原文自动滚动到该 chunk 的位置
- 左侧原文的高亮 chunk 可以添加 "active" 状态，显示不同样式（如边框加粗、阴影）

- [ ] **Step 1: 为每个 chunk 卡片添加点击事件监听**
- [ ] **Step 2: 实现滚动逻辑：将原文滚动到对应 chunk 的高亮位置**
- [ ] **Step 3: 为活跃的高亮区域添加 "active" 样式**
- [ ] **Step 4: 验证左右同步效果**

---

## Task 5: 验证与测试

**Files:**
- Test: 手动测试，使用样例文件

- [ ] **Step 1: 启动服务器**
- [ ] **Step 2: 上传 student_affairs_faq.md，验证原文高亮正确**
- [ ] **Step 3: 点击不同分片，验证左侧滚动和高亮同步**
- [ ] **Step 4: 在不同浏览器和屏幕尺寸下测试布局**
- [ ] **Step 5: 验证各颜色高亮清晰易区分，分片序号标签清晰可见**

---

## 预估时间

- Task 1: 15-20 分钟
- Task 2: 20-25 分钟
- Task 3: 30-40 分钟
- Task 4: 20-25 分钟
- Task 5: 15-20 分钟

**总计：100-130 分钟**

---

## 注意事项

1. **字符位置精度**：需要确保后端计算的 start_char/end_char 与原文完全吻合
2. **多行文本处理**：原文中的 `\n` 字符需要正确处理，高亮不能跨越换行符边界
3. **性能考虑**：大文件（>10k 字符）的高亮处理可能较慢，可考虑虚拟滚动或按需渲染
4. **颜色视觉对比**：选择的高亮颜色需要与背景形成足够的对比度（WCAG 辅助功能）
5. **响应式设计**：在移动端（<860px）两列布局可能变为单列堆叠，需提前规划
