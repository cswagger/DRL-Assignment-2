# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import gdown


# Load once at global scope
pattern_names = [
    "0145", "1256", "2367", "4589", "569a", "67ab", "89cd", "9ade", "abef",
    "0123", "4567", "89ab", "cdef", "048c", "159d", "26ae", "37bf",
    "01256", "12569", "04567", "4569a", "567ab", "569ad", "89ade",
    "9abef", "8cdef", "489de", "59aef", "6abef", "89cde"
]
url = "https://drive.google.com/uc?id=1mFj8aOSs-3EiWoVTQnhOV8UDC_tZNt5y"
output = "converted_weights.npz"

# Download the file
gdown.download(url, output, quiet=False, use_cookies=False)

# Now you can load it
weight_table = np.load(output)


def get_isomorphic_indices(pattern):
    """Return all isomorphic variants of a pattern (rotations + mirror)"""
    def rotate_90(p):
        return [ (i % 4) * 4 + (3 - i // 4) for i in p ]
    def mirror_horizontal(p):
        return [ (i // 4) * 4 + (3 - i % 4) for i in p ]
    
    variants = set()
    current = pattern[:]
    for _ in range(4):  # 4 rotations
        variants.add(tuple(current))
        variants.add(tuple(mirror_horizontal(current)))
        current = rotate_90(current)
    return list(variants)

LOG2_TABLE = np.zeros(65536, dtype=int)
for i in range(1, len(LOG2_TABLE)):
    LOG2_TABLE[i] = int(np.log2(i))

# Precompute isomorphics
isomorphic_index_map = {}
for name in pattern_names:
    base_indices = [int(c, 16) for c in name]
    key_prefix = "4-tuple pattern " if len(base_indices) == 4 else "5-tuple pattern "
    key = key_prefix + ''.join(f"{x:x}" for x in base_indices)
    isomorphic_index_map[key] = get_isomorphic_indices(base_indices)

def evaluate_board_fast(board):
    board_1d = board.flatten()
    total = 0

    for key, iso_list in isomorphic_index_map.items():
        arr = weight_table[key]
        for iso_indices in iso_list:
            idx = 0
            for i, board_idx in enumerate(iso_indices):
                val = board_1d[board_idx]
                idx |= (LOG2_TABLE[val] if val > 0 else 0) << (4 * i)
            total += arr[idx]

    return total




class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

import numpy as np



def compress(row):
    new_row = row[row != 0]
    return np.pad(new_row, (0, 4 - len(new_row)), mode='constant')

def merge(row, score_container):
    for i in range(3):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            row[i + 1] = 0
            score_container[0] += row[i]
    return row

def simulate_afterstate(board, score, action):
    new_board = board.copy()
    moved = False
    score_container = [score]

    if action == 0:  # Up
        for j in range(4):
            col = new_board[:, j]
            orig = col.copy()
            col = compress(col)
            col = merge(col, score_container)
            col = compress(col)
            new_board[:, j] = col
            if not np.array_equal(orig, col): moved = True

    elif action == 1:  # Down
        for j in range(4):
            col = new_board[:, j][::-1]
            orig = col.copy()
            col = compress(col)
            col = merge(col, score_container)
            col = compress(col)
            col = col[::-1]
            new_board[:, j] = col
            if not np.array_equal(orig[::-1], col): moved = True

    elif action == 2:  # Left
        for i in range(4):
            row = new_board[i]
            orig = row.copy()
            row = compress(row)
            row = merge(row, score_container)
            row = compress(row)
            new_board[i] = row
            if not np.array_equal(orig, row): moved = True

    elif action == 3:  # Right
        for i in range(4):
            row = new_board[i][::-1]
            orig = row.copy()
            row = compress(row)
            row = merge(row, score_container)
            row = compress(row)
            row = row[::-1]
            new_board[i] = row
            if not np.array_equal(orig[::-1], row): moved = True

    if not moved:
        return None, score

    return new_board, score_container[0]


def get_action(state, score):
    best_action = None
    best_value = -float('inf')

    for action in range(4):
        after_board, _ = simulate_afterstate(state, score, action)
        if after_board is None:
            continue  # illegal move, skip

        value = evaluate_board_fast(after_board)
        if value > best_value:
            best_value = value
            best_action = action

    if best_action is None:
        return random.choice([0, 1, 2, 3])  # fallback

    return best_action

# if __name__ == "__main__":
#     env = Game2048Env()
#     state = env.reset()
#     done = False

#     while not done:
#         # env.render()  # Visualize the board
#         action = get_action(state, env.score)
#         state, a, done, b = env.step(action)
#         print(a)

#     # env.render()
#     print(f"Game over! Final score: {env.score}")
