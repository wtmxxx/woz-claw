<template>
  <div class="birthday-wrapper" @keydown="onKeyDown" tabindex="0">
    <!-- 星空背景粒子 -->
    <canvas ref="starCanvas" class="star-canvas" />

    <!-- 视频播放器弹窗 -->
    <Teleport to="body">
      <div v-if="showVideoPlayer" class="video-overlay" @click.self="closeVideo">
        <div class="video-container">
          <video
            ref="birthdayVideo"
            class="birthday-video"
            :src="videoSrc"
            autoplay
            controls
            @ended="onVideoEnd"
          ></video>
          <button class="video-close-btn" @click="closeVideo">✕</button>
        </div>
      </div>
    </Teleport>

    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 8-bit 标题 -->
      <div class="title-section">
        <h1 class="pixel-title">
          <span v-for="(ch, i) in titleChars" :key="i"
                :style="{ animationDelay: (i * 0.12) + 's', color: titleColors[i % titleColors.length] }"
                class="title-char">{{ ch }}</span>
        </h1>
        <div class="subtitle-row">
          <span class="pixel-star">★</span>
          <span class="pixel-sub">Happy Birthday To You</span>
          <span class="pixel-star">★</span>
        </div>
      </div>

      <!-- 音乐状态指示器 -->
      <div class="music-indicator" v-if="pixelMusicOn">
        <span class="music-note">♪</span>
        <span class="music-text">像素音乐播放中...</span>
      </div>

      <!-- 蛋糕区域 -->
      <div class="cake-section">
        <!-- 蜡烛火焰 (可吹灭) -->
        <div class="flames-row" v-if="!cakeBlown">
          <div
            v-for="(flame, fi) in activeFlames"
            :key="fi"
            class="flame"
            :style="{ left: flame.x + 'px', animationDelay: (fi * 0.15) + 's' }"
            @click="blowCandle(fi)"
          >
            <div class="flame-inner"></div>
          </div>
        </div>
        <div class="smoke-row" v-if="cakeBlown">
          <div
            v-for="(s, si) in 8"
            :key="'smoke'+si"
            class="smoke-puff"
            :style="{ left: (40 + si * 28) + 'px', animationDelay: (si * 0.2) + 's' }"
          >💨</div>
        </div>

        <!-- 像素蛋糕 -->
        <div class="cake-pixel">
          <div class="cake-layer layer-top"></div>
          <div class="cake-layer layer-mid"></div>
          <div class="cake-layer layer-bot"></div>
          <div class="cake-deco deco-1">●</div>
          <div class="cake-deco deco-2">●</div>
          <div class="cake-deco deco-3">●</div>
          <div class="cake-deco deco-4">●</div>
          <div class="cake-candle candle-1" :class="{ lit: !cakeBlown }">🕯️</div>
          <div class="cake-candle candle-2" :class="{ lit: !cakeBlown }">🕯️</div>
          <div class="cake-candle candle-3" :class="{ lit: !cakeBlown }">🕯️</div>
          <div class="cake-candle candle-4" :class="{ lit: !cakeBlown }">🕯️</div>
          <div class="cake-candle candle-5" :class="{ lit: !cakeBlown }">🕯️</div>
        </div>

        <!-- 操作提示 -->
        <div class="cake-hint">
          <span v-if="!cakeBlown">👆 点击蜡烛吹灭许愿！</span>
          <span v-else class="wish-text">✨ 愿望已送达！Happy Birthday! ✨</span>
        </div>
      </div>

      <!-- 气球区域 -->
      <div class="balloon-area">
        <div
          v-for="(b, bi) in balloons"
          :key="b.id"
          class="balloon"
          :class="{ popped: b.popped }"
          :style="{
            left: b.x + '%',
            animationDelay: b.delay + 's',
            animationDuration: (6 + b.speed) + 's',
          }"
          @click="popBalloon(bi)"
        >
          <div class="balloon-body" :style="{ background: b.color }">
            <div class="balloon-shine"></div>
          </div>
          <div class="balloon-string"></div>
        </div>
      </div>

      <!-- 飘落的彩带 -->
      <div class="confetti-area">
        <div
          v-for="(c, ci) in confetti"
          :key="'confetti'+ci"
          class="confetti-piece"
          :style="{
            left: c.x + '%',
            background: c.color,
            animationDelay: c.delay + 's',
            animationDuration: (3 + c.speed) + 's',
            width: c.size + 'px',
            height: c.size + 'px',
          }"
        ></div>
      </div>

      <!-- 底部信息 -->
      <div class="footer-info">
        <span class="pixel-text">🎂 2026.10.05</span>
        <span class="pixel-text">✦ 21岁 ✦</span>
        <span class="pixel-text">🎮 像素生日快乐</span>
      </div>
    </div>

    <!-- 底部操作栏 -->
    <div class="action-bar">
      <button class="pixel-btn" @click="restartPage">🔄 重来</button>
      <button class="pixel-btn" @click="openVideo">🎬 看生日MV</button>
      <button class="pixel-btn" @click="triggerFireworks">🎆 放烟花</button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue'

// ===== 标题 =====
const titleChars = 'HAPPY BIRTHDAY!'.split('')
const titleColors = [
  '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff',
  '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff',
  '#ff9ff3', '#f368e0', '#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff',
]

// ===== 蜡烛 =====
const cakeBlown = ref(false)
const activeFlames = reactive([
  { x: 40, alive: true },
  { x: 68, alive: true },
  { x: 96, alive: true },
  { x: 124, alive: true },
  { x: 152, alive: true },
])

function blowCandle(idx) {
  activeFlames[idx].alive = false
  const allDead = activeFlames.every(f => !f.alive)
  if (allDead) {
    cakeBlown.value = true
    // 吹灭蜡烛 → 打开视频播放真正的生日歌
    stopPixelMusic()
    openVideo()
  }
}

// ===== 气球 =====
const balloonColors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#ff9ff3', '#f368e0', '#00d2d3', '#ff9f43']
const balloons = reactive([])

function initBalloons() {
  balloons.length = 0
  for (let i = 0; i < 10; i++) {
    balloons.push({
      id: i,
      x: 5 + Math.random() * 85,
      delay: Math.random() * 3,
      speed: Math.random() * 3,
      color: balloonColors[i % balloonColors.length],
      popped: false,
    })
  }
}

function popBalloon(idx) {
  if (!balloons[idx].popped) {
    balloons[idx].popped = true
    setTimeout(() => {
      balloons[idx].x = -20
    }, 300)
  }
}

// ===== 彩带 =====
const confettiColors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#ff9ff3', '#f368e0', '#00d2d3', '#ff9f43']
const confetti = reactive([])

function initConfetti() {
  confetti.length = 0
  for (let i = 0; i < 40; i++) {
    confetti.push({
      x: Math.random() * 100,
      delay: Math.random() * 5,
      speed: Math.random() * 3,
      color: confettiColors[i % confettiColors.length],
      size: 4 + Math.random() * 8,
    })
  }
}

// ===== 星空粒子 =====
const starCanvas = ref(null)
let starAnimId = null

class Star {
  constructor(w, h) {
    this.reset(w, h)
  }
  reset(w, h) {
    this.x = Math.random() * w
    this.y = Math.random() * h
    this.size = 1 + Math.random() * 2
    this.speed = 0.2 + Math.random() * 0.5
    this.alpha = 0.3 + Math.random() * 0.7
    this.twinkleSpeed = 0.01 + Math.random() * 0.03
    this.twinklePhase = Math.random() * Math.PI * 2
  }
}

let stars = []
function initStars() {
  const canvas = starCanvas.value
  if (!canvas) return
  canvas.width = window.innerWidth
  canvas.height = window.innerHeight
  const count = Math.floor((canvas.width * canvas.height) / 3000)
  stars = Array.from({ length: count }, () => new Star(canvas.width, canvas.height))
}

function drawStars(time) {
  const canvas = starCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // 画星星
  for (const star of stars) {
    const alpha = star.alpha * (0.5 + 0.5 * Math.sin(time * star.twinkleSpeed + star.twinklePhase))
    ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`
    ctx.fillRect(Math.floor(star.x), Math.floor(star.y), star.size, star.size)
  }

  // 更新并绘制烟花（合并到同一帧循环）
  if (fireworks.length > 0) {
    for (let i = fireworks.length - 1; i >= 0; i--) {
      const f = fireworks[i]
      f.x += f.vx * 0.016
      f.y += f.vy * 0.016
      f.vy += 60 * 0.016
      f.life -= 0.02
      f.size *= 0.995
      if (f.life <= 0 || f.size < 0.5) fireworks.splice(i, 1)
    }
    for (const f of fireworks) {
      ctx.fillStyle = f.color
      ctx.globalAlpha = Math.max(0, f.life)
      ctx.fillRect(Math.floor(f.x), Math.floor(f.y), Math.max(1, Math.floor(f.size)), Math.max(1, Math.floor(f.size)))
    }
    ctx.globalAlpha = 1
  }

  starAnimId = requestAnimationFrame(drawStars)
}

// ===== 烟花 =====
const fireworks = reactive([])
let fireworkId = null

function triggerFireworks() {
  const canvas = starCanvas.value
  if (!canvas) return
  const cx = canvas.width / 2
  const cy = canvas.height / 3

  for (let burst = 0; burst < 3; burst++) {
    setTimeout(() => {
      const colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#ff9ff3', '#f368e0']
      const color = colors[Math.floor(Math.random() * colors.length)]
      for (let i = 0; i < 24; i++) {
        const angle = (i / 24) * Math.PI * 2
        const speed = 40 + Math.random() * 60
        fireworks.push({
          x: cx + (burst - 1) * 100,
          y: cy + (burst - 1) * 50,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          color,
          life: 1,
          size: 3 + Math.random() * 3,
        })
      }
    }, burst * 400)
  }

  // 烟花由 drawStars 自动绘制
}

function animateFireworks() {
  // 烟花物理更新和绘制已合并到 drawStars 中
}

// ===== 8-bit 像素音乐 (自动循环) =====
const pixelMusicOn = ref(false)
let pixelAudioCtx = null
let pixelMusicInterval = null

function getPixelAudioCtx() {
  if (!pixelAudioCtx) {
    pixelAudioCtx = new (window.AudioContext || window.webkitAudioContext)()
  }
  return pixelAudioCtx
}

function playPixelNote(freq, duration) {
  try {
    const ctx = getPixelAudioCtx()
    // 如果 AudioContext 被浏览器暂停，跳过本次播放
    if (ctx.state === 'suspended') return
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.type = 'square'
    osc.frequency.value = freq
    gain.gain.setValueAtTime(0.07, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration)
    osc.start()
    osc.stop(ctx.currentTime + duration + 0.05)
  } catch (e) { /* no audio */ }
}

const pixelMelody = [
  [262, 0.2], [262, 0.2], [294, 0.4], [262, 0.4],
  [349, 0.4], [330, 0.8],
  [262, 0.2], [262, 0.2], [294, 0.4], [262, 0.4],
  [392, 0.4], [349, 0.8],
  [262, 0.2], [262, 0.2], [523, 0.4], [440, 0.4],
  [349, 0.4], [330, 0.4], [294, 0.4],
  [494, 0.2], [494, 0.2], [440, 0.4], [349, 0.4],
  [392, 0.4], [349, 0.8],
]

function startPixelMusic() {
  if (pixelMusicOn.value) return
  pixelMusicOn.value = true

  // 尝试恢复 AudioContext（浏览器自动播放策略）
  try {
    const ctx = getPixelAudioCtx()
    if (ctx.state === 'suspended') {
      ctx.resume()
    }
  } catch (e) { /* ignore */ }

  let idx = 0
  pixelMusicInterval = setInterval(() => {
    if (!pixelMusicOn.value) return
    if (idx >= pixelMelody.length) idx = 0
    const [freq, dur] = pixelMelody[idx]
    playPixelNote(freq, dur)
    idx++
  }, 500)
}

function stopPixelMusic() {
  pixelMusicOn.value = false
  if (pixelMusicInterval) {
    clearInterval(pixelMusicInterval)
    pixelMusicInterval = null
  }
}

// ===== 视频播放器 =====
const showVideoPlayer = ref(false)
const videoSrc = '/birthday-song.mp4'
const birthdayVideo = ref(null)

function openVideo() {
  showVideoPlayer.value = true
  // 关闭像素音乐
  stopPixelMusic()
  // 等DOM更新后自动播放
  setTimeout(() => {
    if (birthdayVideo.value) {
      birthdayVideo.value.play().catch(() => {})
    }
  }, 100)
}

function closeVideo() {
  showVideoPlayer.value = false
  if (birthdayVideo.value) {
    birthdayVideo.value.pause()
    birthdayVideo.value.currentTime = 0
  }
  // 恢复像素音乐
  startPixelMusic()
}

function onVideoEnd() {
  showVideoPlayer.value = false
  if (birthdayVideo.value) {
    birthdayVideo.value.currentTime = 0
  }
  // 视频播完后恢复像素音乐
  startPixelMusic()
}

// ===== 重置 =====
function restartPage() {
  cakeBlown.value = false
  activeFlames.forEach(f => f.alive = true)
  initBalloons()
  initConfetti()
  // 如果视频开着就关了
  if (showVideoPlayer.value) {
    showVideoPlayer.value = false
    if (birthdayVideo.value) {
      birthdayVideo.value.pause()
      birthdayVideo.value.currentTime = 0
    }
  }
  // 重启像素音乐
  startPixelMusic()
}

// ===== 键盘控制 =====
function onKeyDown(e) {
  if (e.key === 'r' || e.key === 'R') restartPage()
  if (e.key === 'f' || e.key === 'F') triggerFireworks()
  if (e.key === 'v' || e.key === 'V') openVideo()
  if (e.key === 'Escape') closeVideo()
}

// ===== 窗口自适应 =====
function onResize() {
  initStars()
}

// ===== 生命周期 =====
onMounted(() => {
  initStars()
  initBalloons()
  initConfetti()
  drawStars(0)
  window.addEventListener('resize', onResize)

  // 用户点击/按键时恢复 AudioContext（浏览器自动播放策略）
  const resumeAudio = () => {
    try {
      const ctx = getPixelAudioCtx()
      if (ctx.state === 'suspended') {
        ctx.resume()
      }
    } catch (e) { /* ignore */ }
  }
  document.addEventListener('click', resumeAudio, { once: true })
  document.addEventListener('keydown', resumeAudio, { once: true })

  // 页面加载后自动开始像素音乐
  setTimeout(() => startPixelMusic(), 500)
})

onUnmounted(() => {
  if (starAnimId) cancelAnimationFrame(starAnimId)
  if (fireworkId) cancelAnimationFrame(fireworkId)
  stopPixelMusic()
  if (pixelAudioCtx) pixelAudioCtx.close()
  window.removeEventListener('resize', onResize)
})
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

.birthday-wrapper {
  position: relative;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: linear-gradient(180deg, #0a0a2e 0%, #1a0a3e 30%, #2a1a4e 60%, #0d0d2b 100%);
  font-family: 'Press Start 2P', 'Courier New', monospace;
  cursor: default;
  outline: none;
  user-select: none;
}

/* ===== 星空 ===== */
.star-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 0;
}

/* ===== 视频播放器弹窗 ===== */
:global(.video-overlay) {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(4px);
}

:global(.video-container) {
  position: relative;
  width: 90vw;
  max-width: 960px;
  border: 3px solid #4d96ff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 0 40px rgba(77, 150, 255, 0.4), 0 0 80px rgba(77, 150, 255, 0.2);
}

:global(.birthday-video) {
  width: 100%;
  display: block;
  max-height: 80vh;
}

:global(.video-close-btn) {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.6);
  color: #fff;
  border: 2px solid rgba(255, 255, 255, 0.3);
  font-size: 1.2rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  font-family: 'Press Start 2P', monospace;
  z-index: 10;
}

:global(.video-close-btn:hover) {
  background: #ff6b6b;
  border-color: #ff6b6b;
}

/* ===== 主内容 ===== */
.main-content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 20px;
  padding-bottom: 70px;
}

/* ===== 音乐指示器 ===== */
.music-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  animation: fadeInUp 0.5s ease-out;
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

.music-note {
  color: #6bcb77;
  font-size: 1rem;
  animation: noteBounce 0.5s ease-in-out infinite alternate;
}

@keyframes noteBounce {
  from { transform: translateY(0); }
  to { transform: translateY(-5px); }
}

.music-text {
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.4rem;
  letter-spacing: 1px;
}

/* ===== 像素标题 ===== */
.title-section {
  text-align: center;
  margin-bottom: 10px;
}

.pixel-title {
  font-size: 2.2rem;
  letter-spacing: 4px;
  text-shadow: 0 0 20px rgba(255,255,255,0.3), 0 4px 0 #000;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 2px;
  line-height: 1.4;
}

.title-char {
  display: inline-block;
  animation: bounceIn 0.5s cubic-bezier(0.68, -0.55, 0.27, 1.55) both;
  text-shadow: 0 0 10px currentColor, 0 3px 0 rgba(0,0,0,0.5);
}

@keyframes bounceIn {
  0% { transform: scale(0) translateY(-30px); opacity: 0; }
  60% { transform: scale(1.2) translateY(5px); opacity: 1; }
  100% { transform: scale(1) translateY(0); opacity: 1; }
}

.subtitle-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-top: 10px;
}

.pixel-star {
  color: #ffd93d;
  font-size: 1rem;
  animation: spin 3s linear infinite;
  display: inline-block;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.pixel-sub {
  color: #ffd93d;
  font-size: 0.7rem;
  letter-spacing: 3px;
  text-shadow: 0 0 15px rgba(255,217,61,0.5);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.05); }
}

/* ===== 蛋糕区域 ===== */
.cake-section {
  position: relative;
  width: 240px;
  height: 200px;
  margin: 10px auto;
}

/* 蜡烛火焰 */
.flames-row {
  position: absolute;
  top: -10px;
  left: 50%;
  transform: translateX(-50%);
  width: 200px;
  height: 40px;
  z-index: 5;
}

.flame {
  position: absolute;
  top: 0;
  width: 20px;
  height: 30px;
  cursor: pointer;
  animation: flicker 0.3s ease-in-out infinite alternate;
}

.flame-inner {
  width: 100%;
  height: 100%;
  background: radial-gradient(ellipse at center, #fff5a0 0%, #ff9f43 40%, #ff6b6b 70%, transparent 100%);
  border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
  box-shadow: 0 0 20px 5px rgba(255, 159, 67, 0.6), 0 0 40px 10px rgba(255, 107, 107, 0.3);
}

@keyframes flicker {
  0% { transform: scaleY(1) translateY(0); }
  50% { transform: scaleY(1.1) translateY(-3px); }
  100% { transform: scaleY(0.95) translateY(-1px); }
}

/* 烟 */
.smoke-row {
  position: absolute;
  top: -20px;
  left: 50%;
  transform: translateX(-50%);
  width: 200px;
  height: 50px;
  z-index: 5;
}

.smoke-puff {
  position: absolute;
  font-size: 16px;
  animation: smokeRise 2s ease-out forwards;
  opacity: 0;
}

@keyframes smokeRise {
  0% { opacity: 0.8; transform: translateY(0) scale(0.5); }
  100% { opacity: 0; transform: translateY(-60px) scale(2); }
}

/* 像素蛋糕 */
.cake-pixel {
  position: absolute;
  bottom: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.cake-layer {
  width: 100%;
  border: 2px solid rgba(0,0,0,0.3);
  image-rendering: pixelated;
}

.layer-top {
  height: 30px;
  background: #ff9ff3;
  border-radius: 4px 4px 0 0;
  box-shadow: inset 0 -3px 0 rgba(0,0,0,0.1);
}

.layer-mid {
  height: 25px;
  background: #f368e0;
  width: 96%;
  box-shadow: inset 0 -3px 0 rgba(0,0,0,0.1);
}

.layer-bot {
  height: 35px;
  background: #d6308c;
  border-radius: 0 0 4px 4px;
  box-shadow: inset 0 -3px 0 rgba(0,0,0,0.1);
}

.cake-deco {
  position: absolute;
  font-size: 12px;
  animation: float 3s ease-in-out infinite;
}

.deco-1 { top: 5px; left: 20px; color: #ffd93d; animation-delay: 0s; }
.deco-2 { top: 5px; right: 20px; color: #6bcb77; animation-delay: 0.5s; }
.deco-3 { top: 30px; left: 50px; color: #4d96ff; animation-delay: 1s; }
.deco-4 { top: 30px; right: 50px; color: #ff6b6b; animation-delay: 1.5s; }

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

.cake-candle {
  position: absolute;
  top: -35px;
  font-size: 18px;
  filter: grayscale(1);
  transition: all 0.3s;
}

.cake-candle.lit {
  filter: grayscale(0);
}

.candle-1 { left: 15px; }
.candle-2 { left: 55px; }
.candle-3 { left: 92px; }
.candle-4 { left: 130px; }
.candle-5 { left: 168px; }

.cake-hint {
  position: absolute;
  bottom: -30px;
  left: 50%;
  transform: translateX(-50%);
  white-space: nowrap;
  font-size: 0.5rem;
  color: rgba(255,255,255,0.7);
  text-align: center;
}

.wish-text {
  color: #ffd93d;
  text-shadow: 0 0 20px rgba(255,217,61,0.8);
  animation: wishGlow 1s ease-in-out infinite alternate;
}

@keyframes wishGlow {
  from { text-shadow: 0 0 10px rgba(255,217,61,0.5); }
  to { text-shadow: 0 0 30px rgba(255,217,61,1), 0 0 60px rgba(255,217,61,0.5); }
}

/* ===== 气球 ===== */
.balloon-area {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}

.balloon {
  position: absolute;
  bottom: -80px;
  cursor: pointer;
  pointer-events: auto;
  animation: floatUp linear infinite;
  transition: opacity 0.3s;
}

.balloon.popped {
  opacity: 0;
  pointer-events: none;
}

@keyframes floatUp {
  0% { transform: translateY(0) rotate(-5deg); }
  25% { transform: translateY(-25vh) rotate(5deg); }
  50% { transform: translateY(-50vh) rotate(-3deg); }
  75% { transform: translateY(-75vh) rotate(4deg); }
  100% { transform: translateY(-105vh) rotate(-2deg); opacity: 0.7; }
}

.balloon-body {
  width: 30px;
  height: 38px;
  border-radius: 50% 50% 50% 50% / 40% 40% 60% 60%;
  position: relative;
  box-shadow: inset -5px -5px 15px rgba(0,0,0,0.2), 0 4px 8px rgba(0,0,0,0.3);
  border: 1px solid rgba(255,255,255,0.2);
}

.balloon-shine {
  position: absolute;
  top: 6px;
  left: 6px;
  width: 10px;
  height: 12px;
  background: rgba(255,255,255,0.4);
  border-radius: 50%;
  transform: rotate(-30deg);
}

.balloon-string {
  width: 1px;
  height: 25px;
  background: rgba(200,200,200,0.4);
  margin: 0 auto;
}

/* ===== 彩带 ===== */
.confetti-area {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}

.confetti-piece {
  position: absolute;
  top: -20px;
  border: 1px solid rgba(255,255,255,0.2);
  animation: confettiFall linear infinite;
  image-rendering: pixelated;
}

@keyframes confettiFall {
  0% { transform: translateY(-10px) rotate(0deg); opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { transform: translateY(105vh) rotate(720deg); opacity: 0; }
}

/* ===== 底部信息 ===== */
.footer-info {
  position: absolute;
  bottom: 75px;
  display: flex;
  gap: 20px;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
}

.pixel-text {
  font-size: 0.5rem;
  color: rgba(255,255,255,0.5);
  letter-spacing: 2px;
}

/* ===== 底部操作栏 ===== */
.action-bar {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 12px;
  z-index: 10;
}

.pixel-btn {
  font-family: 'Press Start 2P', monospace;
  font-size: 0.5rem;
  padding: 8px 16px;
  background: #2a1a4e;
  color: #fff;
  border: 2px solid #4d96ff;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
  text-shadow: 0 1px 0 rgba(0,0,0,0.5);
  box-shadow: 0 4px 0 #1a0a3e;
  letter-spacing: 1px;
}

.pixel-btn:hover {
  background: #4d96ff;
  transform: translateY(-2px);
  box-shadow: 0 6px 0 #1a0a3e;
}

.pixel-btn:active {
  transform: translateY(4px);
  box-shadow: 0 0 0 #1a0a3e;
}

/* ===== 响应式 ===== */
@media (max-width: 640px) {
  .pixel-title {
    font-size: 1.2rem;
    letter-spacing: 2px;
  }
  .pixel-sub {
    font-size: 0.45rem;
  }
  .cake-section {
    transform: scale(0.8);
  }
  .pixel-btn {
    font-size: 0.4rem;
    padding: 6px 10px;
  }
  .footer-info {
    gap: 10px;
  }
  .pixel-text {
    font-size: 0.35rem;
  }
}
</style>
