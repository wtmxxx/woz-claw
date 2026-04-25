#!/usr/bin/env python3
"""
俄罗斯方块 Tetris
使用 tkinter 实现
"""

import tkinter as tk
from tkinter import messagebox
import random
from typing import List, Tuple, Optional

# ===== 游戏常量 =====

# 颜色定义
COLORS = {
    'I': '#00f0f0',  # 青色
    'O': '#f0f000',  # 黄色
    'T': '#a000f0',  # 紫色
    'S': '#00f000',  # 绿色
    'Z': '#f00000',  # 红色
    'J': '#0000f0',  # 蓝色
    'L': '#f0a000',  # 橙色
    'G': '#808080',  # 灰色（预览）
}

# 7种方块形状定义 [旋转状态][行][列]
SHAPES = {
    'I': [
        [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
        [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]],
    ],
    'O': [
        [[1,1], [1,1]],
    ],
    'T': [
        [[0,1,0], [1,1,1], [0,0,0]],
        [[0,1,0], [0,1,1], [0,1,0]],
        [[0,0,0], [1,1,1], [0,1,0]],
        [[0,1,0], [1,1,0], [0,1,0]],
    ],
    'S': [
        [[0,1,1], [1,1,0], [0,0,0]],
        [[0,1,0], [0,1,1], [0,0,1]],
        [[0,0,0], [0,1,1], [1,1,0]],
        [[1,0,0], [1,1,0], [0,1,0]],
    ],
    'Z': [
        [[1,1,0], [0,1,1], [0,0,0]],
        [[0,0,1], [0,1,1], [0,1,0]],
        [[0,0,0], [1,1,0], [0,1,1]],
        [[0,1,0], [1,1,0], [1,0,0]],
    ],
    'J': [
        [[1,0,0], [1,1,1], [0,0,0]],
        [[0,1,1], [0,1,0], [0,1,0]],
        [[0,0,0], [1,1,1], [0,0,1]],
        [[0,1,0], [0,1,0], [1,1,0]],
    ],
    'L': [
        [[0,0,1], [1,1,1], [0,0,0]],
        [[0,1,0], [0,1,0], [0,1,1]],
        [[0,0,0], [1,1,1], [1,0,0]],
        [[1,1,0], [0,1,0], [0,1,0]],
    ],
}

# 计分规则（一次消行数 -> 得分）
SCORE_TABLE = {
    1: 100,
    2: 300,
    3: 500,
    4: 800,
}

# ===== 游戏主类 =====

class Tetris:
    """俄罗斯方块游戏主逻辑"""

    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    def __init__(self):
        self.board: List[List[Optional[str]]] = [
            [None] * self.BOARD_WIDTH for _ in range(self.BOARD_HEIGHT)
        ]
        self.current_piece: Optional[str] = None       # 当前方块类型
        self.current_shape: List[List[int]] = []        # 当前方块的形状矩阵
        self.current_rotation: int = 0                  # 当前旋转状态
        self.current_x: int = 0                         # 当前方块左上角列坐标
        self.current_y: int = 0                         # 当前方块左上角行坐标
        self.next_piece: Optional[str] = None            # 下一个方块
        self.score: int = 0
        self.level: int = 1
        self.lines_cleared: int = 0
        self.game_over: bool = False
        self.paused: bool = False

        self._bag: List[str] = []  # 7-bag 随机生成器
        self._spawn_piece()

    def _generate_bag(self):
        """使用7-bag算法生成一个随机的7块序列"""
        pieces = list(SHAPES.keys())
        random.shuffle(pieces)
        self._bag.extend(pieces)

    def _get_next_from_bag(self) -> str:
        """从bag中取下一个方块"""
        if not self._bag:
            self._generate_bag()
        return self._bag.pop(0)

    def _spawn_piece(self):
        """生成新方块"""
        if self.next_piece is None:
            self.next_piece = self._get_next_from_bag()

        self.current_piece = self.next_piece
        self.next_piece = self._get_next_from_bag()
        self.current_rotation = 0
        self.current_shape = SHAPES[self.current_piece][0]
        # 水平居中
        self.current_x = (self.BOARD_WIDTH - len(self.current_shape[0])) // 2
        self.current_y = 0

        # 检查是否立即碰撞 -> Game Over
        if self._check_collision(self.current_shape, self.current_x, self.current_y):
            self.game_over = True

    def _check_collision(self, shape: List[List[int]], x: int, y: int) -> bool:
        """检测 shape 放在 (x,y) 是否与已固定方块或边界碰撞"""
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    board_x = x + col_idx
                    board_y = y + row_idx
                    # 超出边界
                    if board_x < 0 or board_x >= self.BOARD_WIDTH:
                        return True
                    if board_y >= self.BOARD_HEIGHT:
                        return True
                    # 与已固定的方块重叠
                    if board_y >= 0 and self.board[board_y][board_x] is not None:
                        return True
        return True if y < 0 else False  # 顶部超出也算碰撞

    def _lock_piece(self):
        """固定当前方块到面板"""
        shape = self.current_shape
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    board_x = self.current_x + col_idx
                    board_y = self.current_y + row_idx
                    if 0 <= board_y < self.BOARD_HEIGHT and 0 <= board_x < self.BOARD_WIDTH:
                        self.board[board_y][board_x] = self.current_piece

        # 消除满行
        self._clear_lines()

        # 生成下一个方块
        self._spawn_piece()

    def _clear_lines(self):
        """消除满行并更新分数"""
        lines_to_clear = []
        for row_idx in range(self.BOARD_HEIGHT):
            if all(self.board[row_idx][col] is not None for col in range(self.BOARD_WIDTH)):
                lines_to_clear.append(row_idx)

        if lines_to_clear:
            # 从下往上删除行
            for row_idx in reversed(lines_to_clear):
                del self.board[row_idx]
                self.board.insert(0, [None] * self.BOARD_WIDTH)

            count = len(lines_to_clear)
            self.lines_cleared += count
            self.score += SCORE_TABLE.get(count, 0)
            # 每消10行升级
            self.level = self.lines_cleared // 10 + 1

    def move_left(self) -> bool:
        """左移，成功返回True"""
        if self.game_over or self.paused:
            return False
        if not self._check_collision(self.current_shape, self.current_x - 1, self.current_y):
            self.current_x -= 1
            return True
        return False

    def move_right(self) -> bool:
        """右移，成功返回True"""
        if self.game_over or self.paused:
            return False
        if not self._check_collision(self.current_shape, self.current_x + 1, self.current_y):
            self.current_x += 1
            return True
        return False

    def move_down(self) -> bool:
        """下移一格，成功返回True"""
        if self.game_over or self.paused:
            return False
        if not self._check_collision(self.current_shape, self.current_x, self.current_y + 1):
            self.current_y += 1
            return True
        # 无法下移则固定
        self._lock_piece()
        return False

    def rotate(self) -> bool:
        """旋转当前方块，成功返回True"""
        if self.game_over or self.paused:
            return False
        pieces = SHAPES[self.current_piece]
        new_rotation = (self.current_rotation + 1) % len(pieces)
        new_shape = pieces[new_rotation]

        # 尝试无偏移旋转
        if not self._check_collision(new_shape, self.current_x, self.current_y):
            self.current_shape = new_shape
            self.current_rotation = new_rotation
            return True

        # 墙踢（Wall Kick）：尝试左右偏移
        for kick in [-1, 1, -2, 2]:
            if not self._check_collision(new_shape, self.current_x + kick, self.current_y):
                self.current_shape = new_shape
                self.current_rotation = new_rotation
                self.current_x += kick
                return True

        return False

    def hard_drop(self):
        """硬降：直接落到底"""
        if self.game_over or self.paused:
            return
        while not self._check_collision(self.current_shape, self.current_x, self.current_y + 1):
            self.current_y += 1
        self._lock_piece()

    def get_ghost_y(self) -> int:
        """获取幽灵方块的Y坐标（预览落点）"""
        y = self.current_y
        while not self._check_collision(self.current_shape, self.current_x, y + 1):
            y += 1
        return y


# ===== 图形界面 =====

class TetrisGUI:
    """俄罗斯方块 GUI"""

    CELL_SIZE = 30
    BOARD_PX_W = Tetris.BOARD_WIDTH * CELL_SIZE   # 300
    BOARD_PX_H = Tetris.BOARD_HEIGHT * CELL_SIZE  # 600

    # 右侧信息区宽度
    INFO_WIDTH = 180
    PREVIEW_SIZE = 4 * CELL_SIZE

    # 窗口大小
    WINDOW_WIDTH = BOARD_PX_W + INFO_WIDTH
    WINDOW_HEIGHT = BOARD_PX_H

    # 下落定时器 (毫秒)
    BASE_INTERVAL = 500

    def __init__(self):
        self.game = Tetris()

        self.root = tk.Tk()
        self.root.title("俄罗斯方块 Tetris")
        self.root.resizable(False, False)

        # 主框架
        self.main_frame = tk.Frame(self.root, bg='#1a1a2e')
        self.main_frame.pack(fill='both', expand=True)

        # 画布 - 游戏板
        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.BOARD_PX_W,
            height=self.BOARD_PX_H,
            bg='#0f0f23',
            highlightthickness=2,
            highlightbackground='#4a4a6a'
        )
        self.canvas.pack(side='left', padx=(10, 5), pady=10)

        # 右侧信息面板
        self.info_frame = tk.Frame(self.main_frame, bg='#1a1a2e')
        self.info_frame.pack(side='right', fill='y', padx=(5, 10), pady=10)

        # 下一个方块预览
        tk.Label(self.info_frame, text='下一个', font=('Arial', 12, 'bold'),
                 bg='#1a1a2e', fg='#e0e0e0').pack(pady=(0, 5))
        self.preview_canvas = tk.Canvas(
            self.info_frame,
            width=self.PREVIEW_SIZE,
            height=self.PREVIEW_SIZE,
            bg='#0f0f23',
            highlightthickness=1,
            highlightbackground='#4a4a6a'
        )
        self.preview_canvas.pack(pady=(0, 20))

        # 分数
        self._make_info_label('分数', 'score_label', '0')
        self._make_info_label('等级', 'level_label', '1')
        self._make_info_label('消行', 'lines_label', '0')

        # 操作提示
        tk.Frame(self.info_frame, bg='#1a1a2e', height=20).pack()
        tips = (
            '操作说明',
            '← →  移动',
            '↑     旋转',
            '↓     加速下落',
            '空格  硬降',
            'P     暂停',
            'R     重新开始',
        )
        for tip in tips:
            color = '#e0e0e0' if tip != '操作说明' else '#f0a000'
            font = ('Arial', 9, 'bold' if tip == '操作说明' else 'normal')
            tk.Label(self.info_frame, text=tip, font=font,
                     bg='#1a1a2e', fg=color).pack(anchor='w')

        # 绑定键盘事件
        self.root.bind('<KeyPress>', self._on_key)

        # 启动游戏循环
        self._update_display()
        self._game_loop()

    def _make_info_label(self, title: str, attr: str, initial: str):
        """创建信息标签"""
        tk.Label(self.info_frame, text=title, font=('Arial', 11, 'bold'),
                 bg='#1a1a2e', fg='#a0a0a0').pack(anchor='w', pady=(10, 0))
        label = tk.Label(self.info_frame, text=initial, font=('Arial', 16, 'bold'),
                         bg='#1a1a2e', fg='#ffffff')
        label.pack(anchor='w')
        setattr(self, attr, label)

    def _on_key(self, event):
        """键盘事件处理"""
        g = self.game
        key = event.keysym

        if key == 'r' or key == 'R':
            self._restart()
            return

        if key == 'p' or key == 'P':
            g.paused = not g.paused
            self._update_display()
            return

        if g.game_over or g.paused:
            return

        if key == 'Left':
            g.move_left()
        elif key == 'Right':
            g.move_right()
        elif key == 'Down':
            g.move_down()
        elif key == 'Up':
            g.rotate()
        elif key == 'space':
            g.hard_drop()
        elif key == 'z' or key == 'Z':
            g.rotate()

        self._update_display()

    def _restart(self):
        """重新开始游戏"""
        self.game = Tetris()
        self._update_display()

    def _draw_board(self):
        """绘制游戏板"""
        canvas = self.canvas
        canvas.delete('all')
        g = self.game

        # 绘制网格（细线）
        for row in range(Tetris.BOARD_HEIGHT):
            for col in range(Tetris.BOARD_WIDTH):
                x1 = col * self.CELL_SIZE
                y1 = row * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline='#1a1a3e',
                    fill='#0f0f23',
                    width=1
                )

        # 绘制已固定的方块
        for row_idx in range(Tetris.BOARD_HEIGHT):
            for col_idx in range(Tetris.BOARD_WIDTH):
                piece = g.board[row_idx][col_idx]
                if piece:
                    self._draw_cell(canvas, col_idx, row_idx, COLORS[piece])

        # 绘制幽灵方块（半透明预览）
        if g.current_piece and not g.game_over:
            ghost_y = g.get_ghost_y()
            if ghost_y != g.current_y:
                for row_idx, row in enumerate(g.current_shape):
                    for col_idx, cell in enumerate(row):
                        if cell:
                            x = (g.current_x + col_idx) * self.CELL_SIZE
                            y = (ghost_y + row_idx) * self.CELL_SIZE
                            canvas.create_rectangle(
                                x + 2, y + 2,
                                x + self.CELL_SIZE - 2, y + self.CELL_SIZE - 2,
                                outline=COLORS[g.current_piece],
                                fill='',
                                width=1,
                                stipple='gray25'
                            )

        # 绘制当前活动方块
        if g.current_piece and not g.game_over:
            for row_idx, row in enumerate(g.current_shape):
                for col_idx, cell in enumerate(row):
                    if cell:
                        self._draw_cell(
                            canvas,
                            g.current_x + col_idx,
                            g.current_y + row_idx,
                            COLORS[g.current_piece]
                        )

        # 游戏结束或暂停提示
        if g.game_over:
            canvas.create_rectangle(
                0, self.BOARD_PX_H//2 - 30,
                self.BOARD_PX_W, self.BOARD_PX_H//2 + 30,
                fill='#000000', outline='#ff0000', width=2
            )
            canvas.create_text(
                self.BOARD_PX_W//2, self.BOARD_PX_H//2,
                text='GAME OVER', font=('Arial', 24, 'bold'),
                fill='#ff4444'
            )
            canvas.create_text(
                self.BOARD_PX_W//2, self.BOARD_PX_H//2 + 40,
                text='按 R 重新开始', font=('Arial', 12),
                fill='#aaaaaa'
            )
        elif g.paused:
            canvas.create_rectangle(
                0, self.BOARD_PX_H//2 - 25,
                self.BOARD_PX_W, self.BOARD_PX_H//2 + 25,
                fill='#000000', outline='#ffff00', width=2
            )
            canvas.create_text(
                self.BOARD_PX_W//2, self.BOARD_PX_H//2,
                text='暂停', font=('Arial', 24, 'bold'),
                fill='#ffff00'
            )

    def _draw_cell(self, canvas, col, row, color):
        """绘制一个格子"""
        x1 = col * self.CELL_SIZE
        y1 = row * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE

        # 主方块
        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='#2a2a4e', width=1)
        # 高光效果
        canvas.create_rectangle(x1+2, y1+2, x1+self.CELL_SIZE-2, y1+self.CELL_SIZE-2,
                                 fill=color, outline='', width=0)
        # 左上角亮点
        canvas.create_rectangle(x1+2, y1+2, x1+self.CELL_SIZE-6, y1+self.CELL_SIZE-6,
                                 fill=self._lighten(color, 0.3), outline='', width=0)

    def _lighten(self, color: str, factor: float) -> str:
        """提亮颜色"""
        # 转换 #RRGGBB -> 提亮
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _draw_preview(self):
        """绘制下一个方块预览"""
        canvas = self.preview_canvas
        canvas.delete('all')
        g = self.game

        if g.next_piece:
            shape = SHAPES[g.next_piece][0]
            rows = len(shape)
            cols = len(shape[0])
            color = COLORS[g.next_piece]

            # 居中偏移
            offset_x = (self.PREVIEW_SIZE - cols * self.CELL_SIZE) // 2
            offset_y = (self.PREVIEW_SIZE - rows * self.CELL_SIZE) // 2

            for row_idx, row in enumerate(shape):
                for col_idx, cell in enumerate(row):
                    if cell:
                        x1 = offset_x + col_idx * self.CELL_SIZE
                        y1 = offset_y + row_idx * self.CELL_SIZE
                        x2 = x1 + self.CELL_SIZE
                        y2 = y1 + self.CELL_SIZE
                        canvas.create_rectangle(
                            x1, y1, x2, y2,
                            fill=color,
                            outline='#2a2a4e',
                            width=1
                        )
                        canvas.create_rectangle(
                            x1+2, y1+2, x2-4, y2-4,
                            fill=color,
                            outline=''
                        )

    def _update_info(self):
        """更新信息面板"""
        g = self.game
        self.score_label.config(text=str(g.score))
        self.level_label.config(text=str(g.level))
        self.lines_label.config(text=str(g.lines_cleared))

    def _update_display(self):
        """刷新显示"""
        self._draw_board()
        self._draw_preview()
        self._update_info()

    def _game_loop(self):
        """游戏主循环 - 自动下落"""
        if not self.game.game_over and not self.game.paused:
            self.game.move_down()
            self._update_display()

        # 根据等级调节下落速度
        interval = max(100, self.BASE_INTERVAL - (self.game.level - 1) * 40)
        self.root.after(interval, self._game_loop)

    def run(self):
        """启动游戏"""
        self.root.mainloop()


# ===== 启动入口 =====

if __name__ == '__main__':
    gui = TetrisGUI()
    gui.run()
