// 俄罗斯方块游戏 - 核心逻辑
document.addEventListener('DOMContentLoaded', function() {
    // 游戏常量
    const COLS = 10;
    const ROWS = 20;
    const BLOCK_SIZE = 30;
    const COLORS = [
        null,
        '#00f0f0', // I - 青色
        '#0000f0', // J - 蓝色
        '#f0a000', // L - 橙色
        '#f0f000', // O - 黄色
        '#00f000', // S - 绿色
        '#a000f0', // T - 紫色
        '#f00000'  // Z - 红色
    ];
    
    // 方块形状定义
    const PIECES = [
        null,
        // I
        [
            [0,0,0,0],
            [1,1,1,1],
            [0,0,0,0],
            [0,0,0,0]
        ],
        // J
        [
            [2,0,0],
            [2,2,2],
            [0,0,0]
        ],
        // L
        [
            [0,0,3],
            [3,3,3],
            [0,0,0]
        ],
        // O
        [
            [4,4],
            [4,4]
        ],
        // S
        [
            [0,5,5],
            [5,5,0],
            [0,0,0]
        ],
        // T
        [
            [0,6,0],
            [6,6,6],
            [0,0,0]
        ],
        // Z
        [
            [7,7,0],
            [0,7,7],
            [0,0,0]
        ]
    ];
    
    // 游戏状态
    let board = createBoard();
    let score = 0;
    let level = 1;
    let lines = 0;
    let gameOver = false;
    let isPaused = false;
    let dropInterval = 1000; // 初始下落间隔(ms)
    let dropCounter = 0;
    let lastTime = 0;
    
    // 当前方块和下一个方块
    let player = {
        pos: {x: 0, y: 0},
        matrix: null,
        score: 0
    };
    
    let nextPiece = null;
    
    // DOM元素
    const canvas = document.getElementById('tetris');
    const nextCanvas = document.getElementById('nextPiece');
    const context = canvas.getContext('2d');
    const nextContext = nextCanvas.getContext('2d');
    const scoreElement = document.getElementById('score');
    const levelElement = document.getElementById('level');
    const linesElement = document.getElementById('lines');
    const startBtn = document.getElementById('startBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    // 初始化游戏
    function init() {
        // 创建游戏板
        board = createBoard();
        
        // 初始化玩家和下一个方块
        player.matrix = createPiece();
        nextPiece = createPiece();
        
        // 重置游戏状态
        score = 0;
        level = 1;
        lines = 0;
        gameOver = false;
        isPaused = false;
        dropInterval = 1000;
        
        // 更新UI
        updateScore();
        
        // 绘制初始状态
        draw();
        drawNextPiece();
        
        // 更新按钮文本
        pauseBtn.innerHTML = '<i class="fas fa-pause"></i> 暂停';
        startBtn.innerHTML = '<i class="fas fa-play"></i> 开始游戏';
    }
    
    // 创建游戏板
    function createBoard() {
        return Array.from({length: ROWS}, () => Array(COLS).fill(0));
    }
    
    // 创建随机方块
    function createPiece() {
        const pieceId = Math.floor(Math.random() * 7) + 1;
        const piece = PIECES[pieceId];
        return {
            matrix: piece,
            id: pieceId
        };
    }
    
    // 碰撞检测
    function collide(board, player) {
        const [m, o] = [player.matrix, player.pos];
        for (let y = 0; y < m.length; ++y) {
            for (let x = 0; x < m[y].length; ++x) {
                if (m[y][x] !== 0 &&
                    (board[y + o.y] &&
                    board[y + o.y][x + o.x]) !== 0) {
                    return true;
                }
            }
        }
        return false;
    }
    
    // 合并方块到游戏板
    function merge(board, player) {
        player.matrix.forEach((row, y) => {
            row.forEach((value, x) => {
                if (value !== 0) {
                    board[y + player.pos.y][x + player.pos.x] = value;
                }
            });
        });
    }
    
    // 玩家移动
    function playerMove(dir) {
        player.pos.x += dir;
        if (collide(board, player)) {
            player.pos.x -= dir;
            return false;
        }
        return true;
    }
    
    // 玩家旋转
    function playerRotate(dir) {
        const pos = player.pos.x;
        let offset = 1;
        rotate(player.matrix, dir);
        
        while (collide(board, player)) {
            player.pos.x += offset;
            offset = -(offset + (offset > 0 ? 1 : -1));
            if (offset > player.matrix[0].length) {
                rotate(player.matrix, -dir);
                player.pos.x = pos;
                return;
            }
        }
    }
    
    // 矩阵旋转
    function rotate(matrix, dir) {
        for (let y = 0; y < matrix.length; ++y) {
            for (let x = 0; x < y; ++x) {
                [
                    matrix[x][y],
                    matrix[y][x]
                ] = [
                    matrix[y][x],
                    matrix[x][y]
                ];
            }
        }
        
        if (dir > 0) {
            matrix.forEach(row => row.reverse());
        } else {
            matrix.reverse();
        }
    }
    
    // 玩家下落
    function playerDrop() {
        player.pos.y++;
        if (collide(board, player)) {
            player.pos.y--;
            merge(board, player);
            playerReset();
            sweep();
            updateScore();
        }
        dropCounter = 0;
    }
    
    // 玩家硬下落（直接落到底部）
    function playerHardDrop() {
        while (!collide(board, player)) {
            player.pos.y++;
        }
        player.pos.y--;
        merge(board, player);
        playerReset();
        sweep();
        updateScore();
        dropCounter = 0;
    }
    
    // 重置玩家（生成新方块）
    function playerReset() {
        player.matrix = nextPiece.matrix;
        player.id = nextPiece.id;
        player.pos.y = 0;
        player.pos.x = Math.floor(COLS / 2) - Math.floor(player.matrix[0].length / 2);
        
        // 生成下一个方块
        nextPiece = createPiece();
        drawNextPiece();
        
        // 检查游戏结束
        if (collide(board, player)) {
            gameOver = true;
            alert(`游戏结束！最终得分: ${score}`);
            init();
        }
    }
    
    // 清除完整的行
    function sweep() {
        let rowCount = 0;
        
        outer: for (let y = board.length - 1; y >= 0; --y) {
            for (let x = 0; x < board[y].length; ++x) {
                if (board[y][x] === 0) {
                    continue outer;
                }
            }
            
            // 移除完整的行
            const row = board.splice(y, 1)[0].fill(0);
            board.unshift(row);
            ++y;
            
            rowCount++;
        }
        
        if (rowCount > 0) {
            // 更新分数
            lines += rowCount;
            
            // 计分规则：1行=100分，2行=300分，3行=500分，4行=800分
            const points = [0, 100, 300, 500, 800];
            score += points[rowCount] * level;
            
            // 每10行升一级
            const newLevel = Math.floor(lines / 10) + 1;
            if (newLevel > level) {
                level = newLevel;
                // 每升一级，下落速度加快
                dropInterval = Math.max(100, 1000 - (level - 1) * 100);
            }
        }
    }
    
    // 更新分数显示
    function updateScore() {
        scoreElement.textContent = score;
        levelElement.textContent = level;
        linesElement.textContent = lines;
    }
    
    // 绘制游戏板
    function draw() {
        context.fillStyle = '#0a0a14';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        // 绘制网格
        context.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        context.lineWidth = 1;
        
        // 垂直线
        for (let x = 0; x <= COLS; x++) {
            context.beginPath();
            context.moveTo(x * BLOCK_SIZE, 0);
            context.lineTo(x * BLOCK_SIZE, ROWS * BLOCK_SIZE);
            context.stroke();
        }
        
        // 水平线
        for (let y = 0; y <= ROWS; y++) {
            context.beginPath();
            context.moveTo(0, y * BLOCK_SIZE);
            context.lineTo(COLS * BLOCK_SIZE, y * BLOCK_SIZE);
            context.stroke();
        }
        
        // 绘制已固定的方块
        drawMatrix
