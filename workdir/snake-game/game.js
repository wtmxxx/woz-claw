const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('scoreValue');
const startBtn = document.getElementById('startBtn');

const gridSize = 20;
const tileCount = canvas.width / gridSize;

let snake = [{x: 10, y: 10}];
let food = {x: 15, y: 15};
let dx = 0;
let dy = 0;
let score = 0;
let gameInterval;
let gameRunning = false;

function drawGame() {
    // 移动蛇
    if (dx !== 0 || dy !== 0) {
        const head = {x: snake[0].x + dx, y: snake[0].y + dy};
        
        // 检查碰撞
        if (head.x < 0 || head.x >= tileCount || head.y < 0 || head.y >= tileCount || checkSelfCollision(head)) {
            gameOver();
            return;
        }
        
        snake.unshift(head);
        
        // 检查是否吃到食物
        if (head.x === food.x && head.y === food.y) {
            score += 10;
            scoreElement.textContent = score;
            placeFood();
        } else {
            snake.pop();
        }
    }
    
    // 清空画布
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 画食物
    ctx.fillStyle = 'red';
    ctx.fillRect(food.x * gridSize, food.y * gridSize, gridSize - 2, gridSize - 2);
    
    // 画蛇
    ctx.fillStyle = 'green';
    snake.forEach(segment => {
        ctx.fillRect(segment.x * gridSize, segment.y * gridSize, gridSize - 2, gridSize - 2);
    });
}

function placeFood() {
    food = {
        x: Math.floor(Math.random() * tileCount),
        y: Math.floor(Math.random() * tileCount)
    };
    
    // 确保食物不在蛇身上
    snake.forEach(segment => {
        if (segment.x === food.x && segment.y === food.y) {
            placeFood();
        }
    });
}

function checkSelfCollision(head) {
    for (let i = 1; i < snake.length; i++) {
        if (head.x === snake[i].x && head.y === snake[i].y) {
            return true;
        }
    }
    return false;
}

function gameOver() {
    gameRunning = false;
    clearInterval(gameInterval);
    ctx.fillStyle = 'white';
    ctx.font = '30px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('游戏结束!', canvas.width / 2, canvas.height / 2);
    startBtn.textContent = '重新开始';
}

function startGame() {
    if (gameRunning) return;
    
    snake = [{x: 10, y: 10}];
    dx = 1;
    dy = 0;
    score = 0;
    scoreElement.textContent = score;
    placeFood();
    gameRunning = true;
    startBtn.textContent = '游戏中...';
    
    if (gameInterval) clearInterval(gameInterval);
    gameInterval = setInterval(drawGame, 100);
}

// 键盘控制
document.addEventListener('keydown', (e) => {
    if (!gameRunning) return;
    
    switch(e.key) {
        case 'ArrowUp':
            if (dy === 0) { dx = 0; dy = -1; }
            break;
        case 'ArrowDown':
            if (dy === 0) { dx = 0; dy = 1; }
            break;
        case 'ArrowLeft':
            if (dx === 0) { dx = -1; dy = 0; }
            break;
        case 'ArrowRight':
            if (dx === 0) { dx = 1; dy = 0; }
            break;
    }
});

startBtn.addEventListener('click', startGame);

// 初始绘制
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.fillStyle = 'white';
ctx.font = '20px Arial';
ctx.textAlign = 'center';
ctx.fillText('点击开始按钮开始游戏', canvas.width / 2, canvas.height / 2);
