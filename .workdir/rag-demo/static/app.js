const form = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const loadSampleBtn = document.getElementById('loadSampleBtn');
const sampleSelect = document.getElementById('sampleSelect');
const fileInput = document.getElementById('fileInput');
const fileSummary = document.getElementById('fileSummary');
const fileDropzone = document.getElementById('fileDropzone');
const statusEl = document.getElementById('status');

const summarySection = document.getElementById('summarySection');
const summaryGrid = document.getElementById('summaryGrid');
const resultsWrapper = document.getElementById('resultsWrapper');
const originalSection = document.getElementById('originalSection');
const originalMeta = document.getElementById('originalMeta');
const originalText = document.getElementById('originalText');
const chunksSection = document.getElementById('chunksSection');
const chunksContainer = document.getElementById('chunksContainer');

let scrollSyncRafId = null;
let activeChunkIndex = null;
let chunkScrollListenerBound = false;
let manualFocusChunkIndex = null;
let manualFocusTimeoutId = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? '#b23a2f' : '#5f6a62';
}

function renderSummary(summary, filename) {
  summarySection.classList.remove('hidden');

  const cards = [
    ['文件', filename],
    ['总 Chunk', summary.total_chunks],
    ['文本 Chunk', summary.text_chunks],
    ['表格 Chunk', summary.table_chunks],
    ['平均长度', summary.avg_len],
    ['文档结构评分', summary.doc_structure_score ?? '-'],
    ['结构质量', summary.structure_quality ?? '-'],
    ['切分方式', summary.split_strategy ?? '-'],
  ];

  summaryGrid.innerHTML = cards
    .map(([title, value]) => `
      <div class="kpi">
        <div class="kpi-title">${title}</div>
        <div class="kpi-value">${value}</div>
      </div>
    `)
    .join('');
}

// Highlight chunks in original text
function highlightChunksInOriginal(originalText, chunks) {
  const colors = ['color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5', 'color-6', 'color-7', 'color-8', 'color-9', 'color-10', 'color-11'];
  
  // Create an array of highlighting regions
  const regions = chunks
    .map((chunk, idx) => ({
      start: Math.max(0, Math.min(originalText.length, Number(chunk.start_char) || 0)),
      end: Math.max(0, Math.min(originalText.length, Number(chunk.end_char) || 0)),
      chunkIndex: chunk.chunk_index,
      color: colors[idx % colors.length],
    }))
    .filter(region => region.end > region.start);

  // Sort regions by start position
  regions.sort((a, b) => (a.start - b.start) || (a.end - b.end));

  // Build highlighted text
  let result = '';
  let lastEnd = 0;

  for (const region of regions) {
    // Add text before this region
    if (lastEnd < region.start) {
      result += escapeHtml(originalText.substring(lastEnd, region.start));
    }

    // Clip overlaps so rendered highlights do not double-consume source text.
    const renderStart = Math.max(region.start, lastEnd);
    if (renderStart >= region.end) {
      continue;
    }

    // Trim leading/trailing whitespace from highlighted span,
    // but keep whitespace as plain text in the output.
    let highlightStart = renderStart;
    let highlightEnd = region.end;

    while (highlightStart < highlightEnd && /\s/.test(originalText[highlightStart])) {
      highlightStart += 1;
    }
    while (highlightEnd > highlightStart && /\s/.test(originalText[highlightEnd - 1])) {
      highlightEnd -= 1;
    }

    // Preserve leading whitespace as non-highlighted text.
    if (renderStart < highlightStart) {
      result += escapeHtml(originalText.substring(renderStart, highlightStart));
    }

    if (highlightStart >= highlightEnd) {
      // Region contains only whitespace.
      if (highlightStart < region.end) {
        result += escapeHtml(originalText.substring(highlightStart, region.end));
      }
      lastEnd = region.end;
      continue;
    }

    // Get the text for this region
    const regionText = originalText.substring(highlightStart, highlightEnd);
    const chunkLabel = `#${String(region.chunkIndex + 1).padStart(3, '0')}`;

    // Add highlighted region with label
    result += `<mark class="mark ${region.color}" data-chunk-index="${region.chunkIndex}" data-start="${highlightStart}" data-end="${highlightEnd}"><span class="chunk-label ${region.color}">${chunkLabel}</span>${escapeHtml(regionText)}</mark>`;

    // Preserve trailing whitespace as non-highlighted text.
    if (highlightEnd < region.end) {
      result += escapeHtml(originalText.substring(highlightEnd, region.end));
    }

    lastEnd = region.end;
  }

  // Add remaining text
  if (lastEnd < originalText.length) {
    result += escapeHtml(originalText.substring(lastEnd));
  }

  return result;
}

function renderChunks(chunks, originalTextContent) {
  if (manualFocusTimeoutId !== null) {
    window.clearTimeout(manualFocusTimeoutId);
    manualFocusTimeoutId = null;
  }
  manualFocusChunkIndex = null;

  chunksSection.classList.remove('hidden');
  resultsWrapper.classList.remove('hidden');

  // Highlight text first
  const highlightedHtml = highlightChunksInOriginal(originalTextContent, chunks);
  originalText.innerHTML = highlightedHtml;

  // Render chunk cards
  chunksContainer.innerHTML = chunks
    .map((c, idx) => {
      const colors = ['color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5', 'color-6', 'color-7', 'color-8', 'color-9', 'color-10', 'color-11'];
      const colorClass = colors[idx % colors.length];
      const isKeyContent = Boolean(c.is_key_content);
      const keyBadge = isKeyContent
      ? '<span class="badge key">重点内容 x1.5</span>'
        : '<span class="badge">普通内容 x1.0</span>';
      const chunkDisplayText = (c.text || '').trim();

      return `
        <article class="chunk-card" data-chunk-index="${c.chunk_index}" data-start="${c.start_char}" data-end="${c.end_char}">
          <div class="chunk-head">
            <strong><span class="chunk-label ${colorClass}">#${String(c.chunk_index + 1).padStart(3, '0')}</span> ${c.chunk_id}</strong>
            <div class="badges">
              <span class="badge">${c.type}</span>
              <span class="badge">len=${c.token_len}</span>
              ${keyBadge}
            </div>
          </div>

          <div class="meta">
            <div><strong>section:</strong> ${c.section}</div>
            <div><strong>section_path:</strong> ${c.section_path}</div>
            <div><strong>prev:</strong> ${c.prev_chunk_id || '-'}</div>
            <div><strong>next:</strong> ${c.next_chunk_id || '-'}</div>
            <div><strong>index:</strong> ${c.chunk_index + 1}/${c.total_chunks}</div>
            <div><strong>weight:</strong> ${c.weight}</div>
          </div>

          <div class="chunk-text">${escapeHtml(chunkDisplayText)}</div>
        </article>
      `;
    })
    .join('');

  // Add click handlers to chunk cards for synchronization
  document.querySelectorAll('.chunk-card').forEach(card => {
    card.addEventListener('click', () => {
      const chunkIndex = parseInt(card.dataset.chunkIndex);
      scrollToChunkInOriginal(chunkIndex);
    });
  });

  // Add click handlers to highlights for reverse synchronization
  document.querySelectorAll('.original-text .mark').forEach(mark => {
    mark.addEventListener('click', (e) => {
      e.stopPropagation();
      const chunkIndex = parseInt(mark.dataset.chunkIndex);
      scrollToChunkCard(chunkIndex);
    });
  });

  bindChunkScrollSync();
}

function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function scrollToChunkInOriginal(chunkIndex) {
  syncChunkAndOriginal(chunkIndex, { scrollOriginal: true, scrollCard: true, pinSelection: true });
}

function scrollToChunkCard(chunkIndex) {
  syncChunkAndOriginal(chunkIndex, { scrollOriginal: false, scrollCard: true, pinSelection: true });
}

function syncChunkAndOriginal(chunkIndex, { scrollOriginal, scrollCard, pinSelection = false }) {
  activeChunkIndex = chunkIndex;

  if (pinSelection) {
    manualFocusChunkIndex = chunkIndex;
    if (manualFocusTimeoutId !== null) {
      window.clearTimeout(manualFocusTimeoutId);
    }
    manualFocusTimeoutId = window.setTimeout(() => {
      manualFocusChunkIndex = null;
      manualFocusTimeoutId = null;
    }, 420);
  }

  const marks = document.querySelectorAll(`.original-text .mark[data-chunk-index="${chunkIndex}"]`);
  document.querySelectorAll('.original-text .mark.active').forEach(m => m.classList.remove('active'));

  if (marks.length > 0) {
    marks[0].classList.add('active');
    if (scrollOriginal) {
      marks[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  document.querySelectorAll('.chunk-card').forEach(card => card.classList.remove('active'));
  const card = document.querySelector(`.chunk-card[data-chunk-index="${chunkIndex}"]`);
  if (card) {
    card.classList.add('active');
    if (scrollCard) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }
}

function getMostVisibleChunkIndex() {
  const cards = Array.from(document.querySelectorAll('.chunk-card'));
  if (cards.length === 0) {
    return null;
  }

  const nearBottom = chunksContainer.scrollTop + chunksContainer.clientHeight >= chunksContainer.scrollHeight - 2;
  if (nearBottom) {
    return parseInt(cards[cards.length - 1].dataset.chunkIndex);
  }

  const viewport = chunksContainer.getBoundingClientRect();
  const viewportCenter = (viewport.top + viewport.bottom) / 2;
  let bestIndex = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const card of cards) {
    const rect = card.getBoundingClientRect();
    const cardCenter = (rect.top + rect.bottom) / 2;
    const distance = Math.abs(cardCenter - viewportCenter);

    if (rect.bottom < viewport.top || rect.top > viewport.bottom) {
      continue;
    }

    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = parseInt(card.dataset.chunkIndex);
    }
  }

  return bestIndex;
}

function syncOriginalToViewportChunk() {
  if (resultsWrapper.classList.contains('hidden') || chunksContainer.children.length === 0) {
    return;
  }

  if (scrollSyncRafId !== null) {
    return;
  }

  scrollSyncRafId = window.requestAnimationFrame(() => {
    scrollSyncRafId = null;

    const chunkIndex = getMostVisibleChunkIndex();
    if (chunkIndex === null || chunkIndex === activeChunkIndex) {
      return;
    }

    if (manualFocusChunkIndex !== null && chunkIndex !== manualFocusChunkIndex) {
      if (manualFocusTimeoutId !== null) {
        window.clearTimeout(manualFocusTimeoutId);
      }
      manualFocusTimeoutId = window.setTimeout(() => {
        manualFocusChunkIndex = null;
        manualFocusTimeoutId = null;
        syncOriginalToViewportChunk();
      }, 420);
      return;
    }

    syncChunkAndOriginal(chunkIndex, { scrollOriginal: true, scrollCard: false });
  });
}

function bindChunkScrollSync() {
  if (chunkScrollListenerBound) {
    return;
  }

  const onScroll = () => syncOriginalToViewportChunk();
  chunksContainer.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  chunkScrollListenerBound = true;
  syncOriginalToViewportChunk();
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  const units = ['KB', 'MB', 'GB'];
  let size = bytes / 1024;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }

  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

function updateFileSummary(file) {
  if (!file) {
    fileSummary.textContent = '尚未选择文件';
    return;
  }

  fileSummary.textContent = `${file.name} · ${formatBytes(file.size)} · ${file.type || 'text/plain'}`;
}

function renderOriginal(file, text) {
  originalSection.classList.remove('hidden');
  originalMeta.textContent = `${file.name} · ${text.length} 字符`;
  originalText.textContent = text;
}

async function processFile(file) {
  submitBtn.disabled = true;
  loadSampleBtn.disabled = true;
  setStatus(`正在处理 ${file.name}，请稍候...`);
  resultsWrapper.classList.add('hidden');
  summarySection.classList.add('hidden');

  const sourceText = await file.text();
  renderOriginal(file, sourceText);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('base_max_size', document.getElementById('baseMax').value);
  const keyContentMaxSize = document.getElementById('keyMax').value;
  formData.append('key_content_max_size', keyContentMaxSize);
  formData.append('overlap', document.getElementById('overlap').value);

  try {
    const res = await fetch('/api/chunk', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || '请求失败');
    }

    renderSummary(data.result.summary, data.filename);
    renderChunks(data.result.chunks, sourceText);
    setStatus(`完成：${data.filename} 生成 ${data.result.summary.total_chunks} 个 chunk。`);
  } catch (err) {
    setStatus(`失败：${err.message}`, true);
  } finally {
    submitBtn.disabled = false;
    loadSampleBtn.disabled = false;
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (!fileInput.files.length) {
    setStatus('请先选择文件。', true);
    return;
  }
  await processFile(fileInput.files[0]);
});

fileInput.addEventListener('change', () => {
  updateFileSummary(fileInput.files[0]);
});

fileDropzone.addEventListener('click', (event) => {
  if (event.target === fileInput) {
    return;
  }
  fileInput.click();
});

['dragenter', 'dragover'].forEach((eventName) => {
  fileDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDropzone.classList.add('is-dragging');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  fileDropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    fileDropzone.classList.remove('is-dragging');
  });
});

fileDropzone.addEventListener('drop', (event) => {
  const [droppedFile] = event.dataTransfer.files;
  if (!droppedFile) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(droppedFile);
  fileInput.files = transfer.files;
  updateFileSummary(droppedFile);
});

loadSampleBtn.addEventListener('click', async () => {
  const selected = sampleSelect.value;
  if (!selected) {
    setStatus('请先选择一个内置样例。', true);
    return;
  }

  try {
    const res = await fetch(selected);
    if (!res.ok) {
      throw new Error('样例文件读取失败');
    }

    const blob = await res.blob();
    const name = selected.split('/').pop() || 'sample.txt';
    const file = new File([blob], name, { type: 'text/plain' });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    updateFileSummary(file);
    await processFile(file);
  } catch (err) {
    setStatus(`失败：${err.message}`, true);
  }
});

updateFileSummary(fileInput.files[0]);
