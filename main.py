import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import random
from scipy.signal import convolve2d

ROWS = 20
COLS = 40
GENERATIONS = 50

INITIAL_PLAYER_A_CELLS = 80
INITIAL_PLAYER_B_CELLS = 80

EMPTY = 0
PLAYER_A = 1
PLAYER_B = 2

def create_board(rows, cols):
    return np.zeros((rows, cols), dtype=int)

def initialize_board(board, player_a_cells, player_b_cells):
    rows, cols = board.shape
    total_cells = rows * cols
    required_cells = 3 + 6 + 6

    if player_a_cells + player_b_cells + required_cells > total_cells:
        raise ValueError("Oyun tahtası için çok fazla başlangıç hücresi.")

    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(all_positions)
    pos_idx = 0

    placed_a = 0
    placed_b = 0

    blinker_center = (rows // 2, cols // 4)
    blinker_positions = [
        (blinker_center[0] - 1, blinker_center[1]),
        (blinker_center[0], blinker_center[1]),
        (blinker_center[0] + 1, blinker_center[1])
    ]

    for r, c in blinker_positions:
        if 0 <= r < rows and 0 <= c < cols and board[r, c] == EMPTY:
            board[r, c] = PLAYER_A
            placed_a += 1


    toad_center = (rows // 2, cols // 4 + 5)
    toad_positions = [
        (toad_center[0], toad_center[1] - 1),
        (toad_center[0], toad_center[1]),
        (toad_center[0], toad_center[1] + 1),
        (toad_center[0] + 1, toad_center[1] - 2),
        (toad_center[0] + 1, toad_center[1] - 1),
        (toad_center[0] + 1, toad_center[1])
    ]

    for r, c in toad_positions:
        if 0 <= r < rows and 0 <= c < cols and board[r, c] == EMPTY:
            board[r, c] = PLAYER_B
            placed_b += 1


    beacon_center = (rows // 2, cols // 4 + 10)
    beacon_positions = [
        (beacon_center[0], beacon_center[1]),
        (beacon_center[0], beacon_center[1] + 1),
        (beacon_center[0] + 1, beacon_center[1]),
        (beacon_center[0] + 1, beacon_center[1] + 1),
        (beacon_center[0] + 2, beacon_center[1] + 2),
        (beacon_center[0] + 2, beacon_center[1] + 3),
        (beacon_center[0] + 3, beacon_center[1] + 2),
        (beacon_center[0] + 3, beacon_center[1] + 3),
    ]

    beacon_cells = [
        (beacon_center[0], beacon_center[1]),
        (beacon_center[0], beacon_center[1] + 1),
        (beacon_center[0] + 1, beacon_center[1]),
        (beacon_center[0] + 1, beacon_center[1] + 1),
        (beacon_center[0] + 2, beacon_center[1] + 2),
        (beacon_center[0] + 2, beacon_center[1] + 3),
        (beacon_center[0] + 3, beacon_center[1] + 2),
        (beacon_center[0] + 3, beacon_center[1] + 3),
    ]

    for idx, (r, c) in enumerate(beacon_cells):
        if 0 <= r < rows and 0 <= c < cols and board[r, c] == EMPTY:
            if idx < 4:
                board[r, c] = PLAYER_A
                placed_a += 1
            else:
                board[r, c] = PLAYER_B
                placed_b += 1

    while placed_a < player_a_cells and pos_idx < len(all_positions):
        r, c = all_positions[pos_idx]
        if board[r, c] == EMPTY:
            board[r, c] = PLAYER_A
            placed_a += 1
        pos_idx += 1

    while placed_b < player_b_cells and pos_idx < len(all_positions):
        r, c = all_positions[pos_idx]
        if board[r, c] == EMPTY:
            board[r, c] = PLAYER_B
            placed_b += 1
        pos_idx += 1

def count_neighbors(board, faction):
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    faction_mask = (board == faction).astype(int)
    neighbors = convolve2d(faction_mask, kernel, mode='same', boundary='wrap')
    return neighbors

def next_generation(board):
    neighbors_a = count_neighbors(board, PLAYER_A)
    neighbors_b = count_neighbors(board, PLAYER_B)

    new_board = np.zeros_like(board)

    birth_a = (board == EMPTY) & (neighbors_a == 3) & (neighbors_b != 3)
    birth_b = (board == EMPTY) & (neighbors_b == 3) & (neighbors_a != 3)

    survive_a = (board == PLAYER_A) & ((neighbors_a == 2) | (neighbors_a == 3))
    survive_b = (board == PLAYER_B) & ((neighbors_b == 2) | (neighbors_b == 3))

    new_board[birth_a] = PLAYER_A
    new_board[birth_b] = PLAYER_B
    new_board[survive_a] = PLAYER_A
    new_board[survive_b] = PLAYER_B

    return new_board

def run_p2life_simulation():
    board = create_board(ROWS, COLS)
    initialize_board(board, INITIAL_PLAYER_A_CELLS, INITIAL_PLAYER_B_CELLS)

    cmap = mcolors.ListedColormap(['white', 'blue', 'red'])
    bounds = [0, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    im = ax.imshow(board, cmap=cmap, norm=norm)
    ax.set_title("P2Life Simülasyonu")
    plt.axis('off')

    def update(frame):
        nonlocal board
        board = next_generation(board)
        im.set_data(board)
        ax.set_title(f"P2Life Simülasyonu")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=GENERATIONS, interval=500, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_p2life_simulation()
