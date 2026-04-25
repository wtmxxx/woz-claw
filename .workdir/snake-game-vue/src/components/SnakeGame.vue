<template>
  <div class="snake-container">
    <h1>🐍 贪吃蛇</h1>

    <div class="status-bar">
      <span class="hint">← ↑ → ↓ / WASD</span>
      <span class="score">🍎 {{ score }}</span>
    </div>

    <div v-if="gameOver" class="game-over">💀 游戏结束 · 得分 {{ score }}</div>
    <div v-else-if="victory" class="game-over victory">🏆 你赢了！满分 {{ score }}</div>

    <canvas ref="canvasRef" width="420" height="420"></canvas>

    <div class="controls">
      <button v-if="!playing && !gameOver && !victory" class="primary" @click="startGame">开始游戏</button>
      <button v-if="playing" @click="togglePause">{{ paused ? '▶ 继续' : '⏸ 暂停' }}</button>
      <button v-if="gameOver || victory" class="primary" @click="startGame">再来一局</button>
    </div>

    <!-- 手机方向键 -->
    <div class="dpad">
      <div class="dpad-grid">
        <div></div>
        <button class="dpad-btn" @touchstart.prevent="setDir('up')">▲</button>
        <div></div>
        <button class="dpad-btn" @touchstart.prevent="setDir('left')">◀</button>
        <button class="dpad-btn" @touchstart.prevent="setDir('down')">▼</button>
        <button class="dpad-btn" @touchstart.prevent="setDir('right')">▶</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// ---------- 常量 ----------
const SIZE = 20
const CELL = 21
const TICK_BASE = 150
const MIN_TICK = 80

const canvasRef = ref(null)
const playing = ref(false)
const paused = ref(false)
const gameOver = ref(false)
const victory = ref(false)
const score = ref(0)

let ctx = null
let snake = []
let food = { x: 8, y: 8 }
let dir = 'right'
let nextDir = 'right'
let timer = null

// ---------- roundRect polyfill ----------
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    if (r > w / 2) r = w / 2
    if (r > h / 2) r = h / 2
    this.moveTo(x + r, y)
    this.lineTo(x + w - r, y)
    this.quadraticCurveTo(x + w, y, x + w, y + r)
    this.lineTo(x + w, y + h - r)
    this.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
    this.lineTo(x + r, y + h)
    this.quadraticCurveTo(x, y + h, x, y + h - r)
    this.lineTo(x, y + r)
    this.quadraticCurveTo(x, y, x + r, y)
    this.closePath()
    return this
  }
}

// ---------- 辅助函数 ----------
function randomFood() {
  const occupied = new Set(snake.map((s) => `${s.x},${s.y}`))
  const free = []
  for (let i = 0; i < SIZE; i++)
    for (let j = 0; j < SIZE; j++)
      if (!occupied.has(`${i},${j}`)) free.push({ x: i, y: j })
  if (free.length === 0) return null
  return free[Math.floor(Math.random() * free.length)]
}

function initSnake() {
  const mid = Math.floor(SIZE / 2)
  return [
    { x: mid, y: mid },
    { x: mid - 1, y: mid },
    { x: mid - 2, y: mid },
  ]
}

// ---------- 绘制 ----------
function draw() {
  if (!ctx) return
  const w = 420
  ctx.clearRect(0, 0, w, w)

  // 网格
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'
  ctx.lineWidth = 1
  for (let i = 0; i <= SIZE; i++) {
    const p = i * CELL
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, w); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(w, p); ctx.stroke()
  }

  // 蛇身
  snake.forEach((seg, i) => {
    const isHead = i === 0
    const x = seg.x * CELL + 1
    const y = seg.y * CELL + 1
    const c = CELL - 2
    ctx.fillStyle = isHead ? '#7cf0a0' : '#44b86a'
    ctx.shadowColor = isHead ? 'rgba(124,240,160,0.5)' : 'rgba(68,184,106,0.3)'
    ctx.shadowBlur = isHead ? 14 : 6
    ctx.beginPath()
    ctx.roundRect(x, y, c, c, 4)
    ctx.fill()

    // 蛇眼
    if (isHead) {
      ctx.shadowBlur = 0
      ctx.fillStyle = '#0f1522'
      let ex1, ey1, ex2, ey2
      const d = 5
      const off = 5
      const cx = x + c / 2
      const cy = y + c / 2
      if (dir === 'right') {
        ex1 = x + 12; ey1 = y + off; ex2 = x + 12; ey2 = y + c - off
      } else if (dir === 'left') {
        ex1 = x + c - 12; ey1 = y + off; ex2 = x + c - 12; ey2 = y + c - off
      } else if (dir === 'up') {
        ex1 = x + off; ey1 = y + c - 12; ex2 = x + c - off; ey2 = y + c - 12
      } else {
        ex1 = x + off; ey1 = y + 12; ex2 = x + c - off; ey2 = y + 12
      }
      ctx.beginPath(); ctx.arc(ex1, ey1, d / 2, 0, Math.PI * 2); ctx.fill()
      ctx.beginPath(); ctx.arc(ex2, ey2, d / 2, 0, Math.PI * 2); ctx.fill()
    }
  })
  ctx.shadowBlur = 0

  // 食物
  const fx = food.x * CELL + 1
  const fy = food.y * CELL + 1
  ctx.shadowColor = 'rgba(255,120,100,0.6)'
  ctx.shadowBlur = 16
  ctx.fillStyle = '#ff6b4a'
  ctx.beginPath()
  ctx.arc(fx + CELL / 2 - 1, fy + CELL / 2 - 1, CELL / 2 - 3, 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowBlur = 0
  ctx.fillStyle = 'rgba(255,255,255,0.25)'
  ctx.beginPath()
  ctx.arc(fx + 5, fy + 5, 4, 0, Math.PI * 2)
  ctx.fill()
}

// ---------- 游戏逻辑 ----------
function tick() {
  if (paused.value || !playing.value) {
    schedule()
    return
  }

  dir = nextDir

  const head = snake[0]
  let nx = head.x, ny = head.y
  if (dir === 'right') nx++
  else if (dir === 'left') nx--
  else if (dir === 'up') ny--
  else if (dir === 'down') ny++

  // 撞墙
  if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) {
    endGame()
    return
  }

  // 移动
  const newHead = { x: nx, y: ny }
  snake.unshift(newHead)

  if (nx === food.x && ny === food.y) {
    score.value++
    const newFood = randomFood()
    if (!newFood) {
      victory.value = true
      playing.value = false
      clearTimeout(timer)
      draw()
      return
    }
    food = newFood
  } else {
    snake.pop()
  }

  draw()
  schedule()
}

function schedule() {
  const speed = Math.max(MIN_TICK, TICK_BASE - score.value * 3)
  timer = setTimeout(tick, speed)
}

function endGame() {
  playing.value = false
  gameOver.value = true
  paused.value = false
  clearTimeout(timer)
  draw()
}

// ---------- 公开方法 ----------
function startGame() {
  snake = initSnake()
  dir = 'right'
  nextDir = 'right'
  score.value = 0
  gameOver.value = false
  victory.value = false
  paused.value = false
  const f = randomFood()
  if (f) food = f
  playing.value = true
  clearTimeout(timer)
  draw()
  schedule()
}

function togglePause() {
  paused.value = !paused.value
}

function setDir(d) {
  if (!playing.value) return
  const opposites = { up: 'down', down: 'up', left: 'right', right: 'left' }
  if (d !== opposites[dir]) nextDir = d
}

// ---------- 键盘 ----------
function onKeyDown(e) {
  const map = {
    ArrowUp: 'up', ArrowDown: 'down',
    ArrowLeft: 'left', ArrowRight: 'right',
    w: 'up', s: 'down', a: 'left', d: 'right',
    W: 'up', S: 'down', A: 'left', D: 'right',
  }
  const d = map[e.key]
  if (d) {
    e.preventDefault()
    setDir(d)
  }
  if (e.key === ' ' || e.key === 'Enter') {
    e.preventDefault()
    if (!playing.value && !gameOver.value && !victory.value) startGame()
    else if (playing.value) togglePause()
  }
}

onMounted(() => {
  ctx = canvasRef.value.getContext('2d')
  draw()
  window.addEventListener('keydown', onKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  clearTimeout(timer)
})
</script>

<style scoped>
.snake-container {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(12px);
  border-radius: 28px;
  padding: 30px 40px 40px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.08);
  text-align: center;
}

h1 {
  color: #88d69e;
  font-weight: 600;
  font-size: 26px;
  letter-spacing: 2px;
  margin-bottom: 18px;
  text-shadow: 0 0 20px rgba(136, 214, 158, 0.3);
}

.status-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
  color: #b0c7d9;
  font-size: 15px;
}

.status-bar .score {
  background: rgba(255, 255, 255, 0.07);
  padding: 6px 18px;
  border-radius: 30px;
  font-weight: 600;
  color: #f0e68c;
  font-size: 18px;
}

.status-bar .hint {
  opacity: 0.65;
}

.game-over {
  color: #ff7b7b;
  font-weight: 600;
  font-size: 17px;
  margin-bottom: 6px;
  letter-spacing: 1px;
}

.game-over.victory {
  color: #f0e68c;
}

canvas {
  display: block;
  margin: 0 auto;
  border-radius: 14px;
  background: #0f1522;
  box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.6), 0 8px 32px rgba(0, 0, 0, 0.4);
  width: 420px;
  height: 420px;
  image-rendering: pixelated;
}

.controls {
  margin-top: 18px;
  display: flex;
  justify-content: center;
  gap: 14px;
}

.controls button {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: #d0e4f0;
  padding: 10px 28px;
  border-radius: 40px;
  font-size: 15px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  letter-spacing: 0.5px;
}

.controls button:hover {
  background: rgba(255, 255, 255, 0.16);
  border-color: rgba(255, 255, 255, 0.25);
  transform: scale(1.04);
}

.controls button:active {
  transform: scale(0.96);
}

.controls button.primary {
  background: #3b8c5e;
  border-color: #5bbf84;
  color: #fff;
}

.controls button.primary:hover {
  background: #4caa72;
}

.dpad {
  display: none;
  margin-top: 18px;
  justify-content: center;
}

.dpad-grid {
  display: grid;
  grid-template-columns: 64px 64px 64px;
  grid-template-rows: 64px 64px 64px;
  gap: 6px;
}

.dpad-btn {
  background: rgba(255, 255, 255, 0.07);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 14px;
  color: #b0c7d9;
  font-size: 24px;
  cursor: pointer;
  transition: 0.15s;
}

.dpad-btn:active {
  background: rgba(255, 255, 255, 0.18);
  transform: scale(0.92);
}

@media (max-width: 520px) {
  .snake-container {
    padding: 18px 16px 24px;
  }
  canvas {
    width: 300px;
    height: 300px;
  }
  .controls button {
    padding: 8px 18px;
    font-size: 13px;
  }
  .dpad {
    display: flex;
  }
}
</style>
