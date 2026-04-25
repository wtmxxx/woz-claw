<template>
  <div class="app-wrapper">
    <SnakeGame />
  </div>
</template>

<script setup>
import SnakeGame from './components/SnakeGame.vue'
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  min-height: 100vh;
  background: linear-gradient(135deg, #0b0e1a 0%, #1a2332 100%);
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}
.app-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  padding: 20px;
}
</style>
