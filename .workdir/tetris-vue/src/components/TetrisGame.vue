<template>
  <div class="tetris-container" @keydown="onKeyDown" tabindex="0" ref="containerRef">
    <!-- 游戏标题 -->
    <div class="header">
      <h1>俄罗斯方块</h1>
      <p class="subtitle">Tetris · Vue + Vite</p>
    </div>

    <div class="game-area">
      <!-- 左侧 - 游戏面板 -->
      <div class="board-wrapper">
        <canvas
          ref="boardCanvas"
          :width="BOARD_PX_W"
          :height="BOARD_PX_H"
          class="game-canvas"
        />
        <!-- 游戏结束/暂停覆盖层 -->
        <div v-if="game.gameOver" class="overlay game-over-overlay">
          <div class="overlay-content">
            <h2>GAME OVER</h2>
            <p class="final-score">得分: {{ game.score }}</p>
            <p class="hint">按 R 重新开始</p>
          </div>
        </div>
        <div v-else-if="game.paused" class="overlay pause-overlay">
          <div class="overlay-content">
            <h2>暂停</h2>
            <p class="hint">按 P 继续</p>
          </div>
        </div>
      </div>

      <!-- 右侧 - 信息面板 -->
      <div class="info-panel">
        <!-- 下一个方块预览 -->
        <div class="info-section">
          <h3>下一个</h3>
          <canvas
            ref="previewCanvas"
            :width="PREVIEW_SIZE"
            :height="PREVIEW_SIZE"
            class="preview-canvas"
          />
        </div>

        <!-- 分数 -->
        <div class="info-section">
          <h3>分数</h3>
          <p class="info-value">{{ game.score }}</p>
        </div>

        <!-- 等级 -->
        <div class="info-section">
          <h3>等级</h3>
          <p class="info-value">{{ game.level }}</p>
        </div>

        <!-- 消行 -->
        <div class="info-section">
          <h3>消行</h3>
          <p class="info-value">{{ game.linesCleared }}</p>
        </div>

        <!-- 操作提示 -->
        <div class="info-section controls-info">
          <h3>操作说明</h3>
          <div class="control-keys">
            <div class="key-row"><kbd>←</kbd><kbd>→</kbd><span>移动</span></div>
            <div class="key-row"><kbd>↑</kbd><span>旋转</span></div>
            <div class="key-row"><kbd>↓</kbd><span>加速下落</span></div>
            <div class="key-row"><kbd>空格</kbd><span>硬降</span></div>
            <div class="key-row"><kbd>P</kbd><span>暂停</span></div>
            <div class="key-row"><kbd>R</kbd><span>重新开始</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- 移动端控制按钮 -->
    <div class="mobile-controls">
      <div class="ctrl-row">
        <button class="ctrl-btn" @touchstart.prevent="moveLeft" @mousedown.prevent="moveLeft">←</button>
        <button class="ctrl-btn" @touchstart.prevent="rotatePiece" @mousedown.prevent="rotatePiece">↑</button>
        <button class="ctrl-btn" @touchstart.prevent="moveRight" @mousedown.prevent="moveRight">→</button>
      </div>
      <div class="ctrl-row">
        <button class="ctrl-btn ctrl-wide" @touchstart.prevent="moveDown" @mousedown.prevent="moveDown">↓</button>
        <button class="ctrl-btn ctrl-wide" @touchstart.prevent="hardDrop" @mousedown.prevent="hardDrop">硬降</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, watch } from 'vue'

// ===== 常量定义 =====

const COLS = 10
const ROWS = 20
const CELL_SIZE = 28
const BOARD_PX_W = COLS * CELL_SIZE
const BOARD_PX_H = ROWS * CELL_SIZE
const PREVIEW_SIZE = 4 * CELL_SIZE
const BASE_INTERVAL = 500

const COLORS = {
  I: '#00f0f0',
  O: '#f0f000',
  T: '#a000f0',
  S: '#00f000',
  Z: '#f00000',
  J: '#0000f0',
  L: '#f0a000',
}

const PIECE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']

const SHAPES = {
  I: [
    [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
    [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],
    [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
  ],
  O: [
    [[1,1],[1,1]],
  ],
  T: [
    [[0,1,0],[1,1,1],[0,0,0]],
    [[0,1,0],[0,1,1],[0,1,0]],
    [[0,0,0],[1,1,1],[0,1,0]],
    [[0,1,0],[1,1,0],[0,1,0]],
  ],
  S: [
    [[0,1,1],[1,1,0],[0,0,0]],
    [[0,1,0],[0,1,1],[0,0,1]],
    [[0,0,0],[0,1,1],[1,1,0]],
    [[1,0,0],[1,1,0],[0,1,0]],
  ],
  Z: [
    [[1,1,0],[0,1,1],[0,0,0]],
    [[0,0,1],[0,1,1],[0,1,0]],
    [[0,0,0],[1,1,0],[0,1,1]],
    [[0,1,0],[1,1,0],[1,0,0]],
  ],
  J: [
    [[1,0,0],[1,1,1],[0,0,0]],
    [[0,1,1],[0,1,0],[0,1,0]],
    [[0,0,0],[1,1,1],[0,0,1]],
    [[0,1,0],[0,1,0],[1,1,0]],
  ],
  L: [
    [[0,0,1],[1,1,1],[0,0,0]],
    [[0,1,0],[0,1,0],[0,1,1]],
    [[0,0,0],[1,1,1],[1,0,0]],
    [[1,1,0],[0,1,0],[0,1,0]],
  ],
}

const SCORE_TABLE = {
  1: 100,
  2: 300,
  3: 500,
  4: 800,
}

// ===== 游戏状态 =====

class TetrisGame {
  constructor() {
    this.board = Array.from({ length: ROWS }, () => Array(COLS).fill(null))
    this.currentPiece = null
    this.currentShape = []
    this.currentRotation = 0
    this.currentX = 0
    this.currentY = 0
    this.nextPiece = null
    this.score = 0
    this.level = 1
    this.linesCleared = 0
    this.gameOver = false
    this.paused = false
    this._bag = []
    this._spawnPiece()
  }

  _generateBag() {
    const pieces = [...PIECE_NAMES]
    for (let i = pieces.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pieces[i], pieces[j]] = [pieces[j], pieces[i]]
    }
    this._bag.push(...pieces)
  }

  _getNextFromBag() {
    if (this._bag.length === 0) this._generateBag()
    return this._bag.shift()
  }

  _spawnPiece() {
    if (this.nextPiece === null) {
      this.nextPiece = this._getNextFromBag()
    }
    this.currentPiece = this.nextPiece
    this.nextPiece = this._getNextFromBag()
    this.currentRotation = 0
    this.currentShape = SHAPES[this.currentPiece][0].map(r => [...r])
    this.currentX = Math.floor((COLS - this.currentShape[0].length) / 2)
    this.currentY = 0
    if (this._checkCollision(this.currentShape, this.currentX, this.currentY)) {
      this.gameOver = true
    }
  }

  _checkCollision(shape, x, y) {
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c]) {
          const bx = x + c
          const by = y + r
          if (bx < 0 || bx >= COLS || by >= ROWS) return true
          if (by >= 0 && this.board[by][bx] !== null) return true
        }
      }
    }
    return y < 0
  }

  _lockPiece() {
    for (let r = 0; r < this.currentShape.length; r++) {
      for (let c = 0; c < this.currentShape[r].length; c++) {
        if (this.currentShape[r][c]) {
          const bx = this.currentX + c
          const by = this.currentY + r
          if (by >= 0 && by < ROWS && bx >= 0 && bx < COLS) {
            this.board[by][bx] = this.currentPiece
          }
        }
      }
    }
    this._clearLines()
    this._spawnPiece()
  }

  _clearLines() {
    const lines = []
    for (let r = 0; r < ROWS; r++) {
      if (this.board[r].every(cell => cell !== null)) {
        lines.push(r)
      }
    }
    if (lines.length > 0) {
      for (const r of lines.reverse()) {
        this.board.splice(r, 1)
        this.board.unshift(Array(COLS).fill(null))
      }
      this.linesCleared += lines.length
      this.score += SCORE_TABLE[lines.length] || 0
      this.level = Math.floor(this.linesCleared / 10) + 1
    }
  }

  moveLeft() {
    if (this.gameOver || this.paused) return false
    if (!this._checkCollision(this.currentShape, this.currentX - 1, this.currentY)) {
      this.currentX--
      return true
    }
    return false
  }

  moveRight() {
    if (this.gameOver || this.paused) return false
    if (!this._checkCollision(this.currentShape, this.currentX + 1, this.currentY)) {
      this.currentX++
      return true
    }
    return false
  }

  moveDown() {
    if (this.gameOver || this.paused) return false
    if (!this._checkCollision(this.currentShape, this.currentX, this.currentY + 1)) {
      this.currentY++
      return true
    }
    this._lockPiece()
    return false
  }

  rotate() {
    if (this.gameOver || this.paused) return false
    const rotations = SHAPES[this.currentPiece]
    const newRot = (this.currentRotation + 1) % rotations.length
    const newShape = rotations[newRot].map(r => [...r])

    if (!this._checkCollision(newShape, this.currentX, this.currentY)) {
      this.currentShape = newShape
      this.currentRotation = newRot
      return true
    }

    // Wall Kick
    for (const kick of [-1, 1, -2, 2]) {
      if (!this._checkCollision(newShape, this.currentX + kick, this.currentY)) {
        this.currentShape = newShape
        this.currentRotation = newRot
        this.currentX += kick
        return true
      }
    }
    return false
  }

  hardDrop() {
    if (this.gameOver || this.paused) return
    while (!this._checkCollision(this.currentShape, this.currentX, this.currentY + 1)) {
      this.currentY++
    }
    this._lockPiece()
  }

  getGhostY() {
    let y = this.currentY
    while (!this._checkCollision(this.currentShape, this.currentX, y + 1)) {
      y++
    }
    return y
  }

  restart() {
    this.board = Array.from({ length: ROWS }, () => Array(COLS).fill(null))
    this.currentPiece = null
    this.currentShape = []
    this.currentRotation = 0
    this.currentX = 0
    this.currentY = 0
    this.nextPiece = null
    this.score = 0
    this.level = 1
    this.linesCleared = 0
    this.gameOver = false
    this.paused = false
    this._bag = []
    this._spawnPiece()
  }

  togglePause() {
    if (!this.gameOver) {
      this.paused = !this.paused
    }
  }
}

// ===== 组件逻辑 =====

const containerRef = ref(null)
const boardCanvas = ref(null)
const previewCanvas = ref(null)
const game = reactive(new TetrisGame())

let gameLoopId = null

function lightenColor(hex, factor) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  const nr = Math.min(255, Math.round(r + (255 - r) * factor))
  const ng = Math.min(255, Math.round(g + (255 - g) * factor))
  const nb = Math.min(255, Math.round(b + (255 - b) * factor))
  return `#${nr.toString(16).padStart(2, '0')}${ng.toString(16).padStart(2, '0')}${nb.toString(16).padStart(2, '0')}`
}

function drawBoard() {
  const canvas = boardCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')

  // 清空
  ctx.clearRect(0, 0, BOARD_PX_W, BOARD_PX_H)

  // 绘制网格
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = c * CELL_SIZE
      const y = r * CELL_SIZE
      ctx.fillStyle = '#0f0f23'
      ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE)
      ctx.strokeStyle = '#1a1a3e'
      ctx.lineWidth = 1
      ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE)
    }
  }

  // 绘已固定方块
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const piece = game.board[r][c]
      if (piece) {
        drawCell(ctx, c, r, COLORS[piece])
      }
    }
  }

  // 幽灵方块
  if (game.currentPiece && !game.gameOver) {
    const ghostY = game.getGhostY()
    if (ghostY !== game.currentY) {
      for (let r = 0; r < game.currentShape.length; r++) {
        for (let c = 0; c < game.currentShape[r].length; c++) {
          if (game.currentShape[r][c]) {
            const x = (game.currentX + c) * CELL_SIZE
            const y = (ghostY + r) * CELL_SIZE
            ctx.strokeStyle = COLORS[game.currentPiece]
            ctx.lineWidth = 2
            ctx.setLineDash([3, 3])
            ctx.strokeRect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            ctx.setLineDash([])
          }
        }
      }
    }
  }

  // 当前活动方块
  if (game.currentPiece && !game.gameOver) {
    for (let r = 0; r < game.currentShape.length; r++) {
      for (let c = 0; c < game.currentShape[r].length; c++) {
        if (game.currentShape[r][c]) {
          drawCell(ctx, game.currentX + c, game.currentY + r, COLORS[game.currentPiece])
        }
      }
    }
  }
}

function drawCell(ctx, col, row, color) {
  const x = col * CELL_SIZE
  const y = row * CELL_SIZE
  const inset = 1

  // 主方块
  ctx.fillStyle = color
  ctx.fillRect(x + inset, y + inset, CELL_SIZE - inset * 2, CELL_SIZE - inset * 2)

  // 高光
  ctx.fillStyle = lightenColor(color, 0.3)
  ctx.fillRect(x + 4, y + 4, CELL_SIZE - 10, CELL_SIZE - 10)

  // 边框
  ctx.strokeStyle = '#2a2a4e'
  ctx.lineWidth = 1
  ctx.strokeRect(x + inset, y + inset, CELL_SIZE - inset * 2, CELL_SIZE - inset * 2)
}

function drawPreview() {
  const canvas = previewCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')

  ctx.clearRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE)
  ctx.fillStyle = '#0f0f23'
  ctx.fillRect(0, 0, PREVIEW_SIZE, PREVIEW_SIZE)

  if (game.nextPiece) {
    const shape = SHAPES[game.nextPiece][0]
    const rows = shape.length
    const cols = shape[0].length
    const color = COLORS[game.nextPiece]
    const previewCellSize = 24
    const offsetX = (PREVIEW_SIZE - cols * previewCellSize) / 2
    const offsetY = (PREVIEW_SIZE - rows * previewCellSize) / 2

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (shape[r][c]) {
          const x = offsetX + c * previewCellSize
          const y = offsetY + r * previewCellSize
          ctx.fillStyle = color
          ctx.fillRect(x + 1, y + 1, previewCellSize - 2, previewCellSize - 2)
          ctx.fillStyle = lightenColor(color, 0.3)
          ctx.fillRect(x + 3, y + 3, previewCellSize - 8, previewCellSize - 8)
          ctx.strokeStyle = '#2a2a4e'
          ctx.lineWidth = 1
          ctx.strokeRect(x + 1, y + 1, previewCellSize - 2, previewCellSize - 2)
        }
      }
    }
  }
}

function render() {
  drawBoard()
  drawPreview()
}

// 操作方法
function moveLeft() { game.moveLeft(); render() }
function moveRight() { game.moveRight(); render() }
function moveDown() { game.moveDown(); render() }
function rotatePiece() { game.rotate(); render() }
function hardDrop() { game.hardDrop(); render() }

function restartGame() {
  game.restart()
  render()
}

function togglePause() {
  game.togglePause()
  render()
}

function onKeyDown(e) {
  const key = e.key
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' ', 'p', 'P', 'r', 'R'].includes(key)) {
    e.preventDefault()
  }

  switch (key) {
    case 'ArrowLeft': moveLeft(); break
    case 'ArrowRight': moveRight(); break
    case 'ArrowDown': moveDown(); break
    case 'ArrowUp': rotatePiece(); break
    case ' ': hardDrop(); break
    case 'p': case 'P': togglePause(); break
    case 'r': case 'R': restartGame(); break
  }
}

function gameLoop() {
  if (!game.gameOver && !game.paused) {
    game.moveDown()
    render()
  }
  const interval = Math.max(100, BASE_INTERVAL - (game.level - 1) * 40)
  gameLoopId = setTimeout(gameLoop, interval)
}

onMounted(() => {
  containerRef.value?.focus()
  render()
  gameLoop()
})

onUnmounted(() => {
  if (gameLoopId) clearTimeout(gameLoopId)
})
</script>

<style scoped>
.tetris-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  outline: none;
  user-select: none;
}

.header {
  text-align: center;
}

.header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: 2px;
  text-shadow: 0 0 20px rgba(100, 200, 255, 0.3);
}

.subtitle {
  font-size: 13px;
  color: #8888aa;
  margin-top: 4px;
}

.game-area {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

.board-wrapper {
  position: relative;
  border: 2px solid #4a4a6a;
  border-radius: 4px;
  box-shadow: 0 0 30px rgba(50, 50, 120, 0.4);
}

.game-canvas {
  display: block;
}

.overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  pointer-events: none;
}

.game-over-overlay {
  background: rgba(0, 0, 0, 0.7);
}

.pause-overlay {
  background: rgba(0, 0, 0, 0.5);
}

.overlay-content {
  text-align: center;
}

.overlay-content h2 {
  font-size: 32px;
  font-weight: 800;
  letter-spacing: 3px;
  margin-bottom: 10px;
}

.game-over-overlay h2 {
  color: #ff4444;
  text-shadow: 0 0 20px rgba(255, 50, 50, 0.5);
}

.pause-overlay h2 {
  color: #ffff00;
  text-shadow: 0 0 20px rgba(255, 255, 50, 0.5);
}

.final-score {
  font-size: 18px;
  color: #ffffff;
  margin-bottom: 8px;
}

.hint {
  font-size: 13px;
  color: #aaaaaa;
}

/* 右侧信息面板 */
.info-panel {
  width: 160px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.info-section {
  background: rgba(15, 15, 35, 0.6);
  border: 1px solid #3a3a5a;
  border-radius: 8px;
  padding: 12px;
}

.info-section h3 {
  font-size: 11px;
  font-weight: 600;
  color: #8888aa;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 6px;
}

.info-value {
  font-size: 22px;
  font-weight: 700;
  color: #ffffff;
  font-variant-numeric: tabular-nums;
}

.preview-canvas {
  display: block;
  margin: 0 auto;
  border: 1px solid #3a3a5a;
  border-radius: 4px;
}

.controls-info .control-keys {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.key-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #cccccc;
}

kbd {
  display: inline-block;
  min-width: 20px;
  text-align: center;
  padding: 1px 5px;
  background: #2a2a4a;
  border: 1px solid #4a4a6a;
  border-radius: 3px;
  font-family: inherit;
  font-size: 11px;
  color: #e0e0e0;
}

/* 移动端控制按钮 */
.mobile-controls {
  display: none;
  flex-direction: column;
  gap: 8px;
  margin-top: 8px;
}

.ctrl-row {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.ctrl-btn {
  width: 56px;
  height: 48px;
  background: rgba(40, 40, 80, 0.8);
  border: 1px solid #4a4a6a;
  border-radius: 8px;
  color: #e0e0e0;
  font-size: 20px;
  font-weight: bold;
  cursor: pointer;
  touch-action: manipulation;
  -webkit-tap-highlight-color: transparent;
}

.ctrl-btn:active {
  background: rgba(60, 60, 120, 0.8);
  transform: scale(0.95);
}

.ctrl-wide {
  width: 80px;
  font-size: 14px;
}

/* 响应式 - 小屏显示触摸按钮 */
@media (max-width: 600px) {
  .game-area {
    flex-direction: column;
    align-items: center;
  }

  .info-panel {
    flex-direction: row;
    flex-wrap: wrap;
    width: auto;
    justify-content: center;
  }

  .info-section {
    flex: 0 0 auto;
    min-width: 70px;
  }

  .controls-info {
    display: none;
  }

  .mobile-controls {
    display: flex;
  }
}
</style>
