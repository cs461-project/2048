from distutils.util import strtobool
import os
import time
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from gym_game2048.wrappers import RewardByScore
import argparse
import grid_helpers_for_minmax as Helper
import pygame
import warnings
warnings.filterwarnings("ignore")

window = None
window_title = "CS 461 - Term Project (Group 12) - 2048 (w/ MinMax)"
window_width = 400
window_height = 600
clock = None
size = 4

# Global variables for rendering
board_size = 0
block_size = 0
block_x_pos = np.zeros(size)
block_y_pos = np.zeros(size)
left_top_board = (0, 0)
block_color = []
game_color = {}
block_font_color = []
block_font_size = []

render_mode = "human"
metadata = {"render_fps": 3}  # Modify the frame rate as needed

def render(board, score, best_score, max_tile, episode=0, _2048_count=0, step=0, render_mode="human"):
    global window, window_title, window_width, window_height, clock, size, board_size, block_size, block_x_pos, block_y_pos, left_top_board, block_color, game_color, block_font_color, block_font_size, metadata

    def _render_block(board, r, c, canvas: pygame.Surface):
        number = board[r][c]
        pygame.draw.rect(
            canvas,
            block_color[min(11, number)],
            ((block_x_pos[c], block_y_pos[r]), (block_size, block_size))
        )
        # Empty parts do not output a number.
        if board[r][c] == 0:
            return

        # render number
        if number < 7:
            size = block_font_size[0]
        elif number < 10:
            size = block_font_size[1]
        elif number < 13:
            size = block_font_size[2]
        elif number < 20:
            size = block_font_size[3]
        else:
            size = block_font_size[2]
        font = pygame.font.Font(None, size)

        num_str = str(2 ** board[r][c]) if number < 20 else f'2^{number}'

        color = block_font_color[0] if number < 3 else block_font_color[1]
        text = font.render(num_str, True, color)
        text_rect = text.get_rect(center=((block_x_pos[c] + block_size//2, block_y_pos[r] + block_size//2)))
        canvas.blit(text, text_rect)

    def _render_info(canvas, score, best_score, max_tile, episode, _2048_count):
        info_font = pygame.font.Font(None, 35)
        score = info_font.render(f'score: {score}', True, (119, 110, 101))
        best_score = info_font.render(f'best: {best_score}', True, (119, 110, 101))
        max_tile = info_font.render(f'max tile: {max_tile}', True, (119, 110, 101))
        episode = info_font.render(f'episode: {episode}', True, (119, 110, 101))
        _2048_count = info_font.render(f'2048 count: {_2048_count}', True, (119, 110, 101))

        canvas.blit(score, (15, 25))
        canvas.blit(best_score, (15, 65))
        canvas.blit(max_tile, (15, 105))
        canvas.blit(episode, (15, 145))
        canvas.blit(_2048_count, (15, 185))

    pygame.font.init()
    if render_mode == "human" or render_mode == "human_only":
        if window is None:
            pygame.init()

            pygame.display.set_caption(window_title)

            # rendering : Size
            win_mg = 10

            board_size = (window_width - 2 * win_mg)
            block_size = int(board_size / (8 * size + 1) * 7)

            left_top_board = (win_mg, window_height - win_mg - board_size)
            gap = board_size / (1 + 8 * size)

            for i in range(size):
                block_x_pos[i] = int(left_top_board[0] + (8 * i + 1) * gap)
                block_y_pos[i] = int(left_top_board[1] + (8 * i + 1) * gap)

            # rendering: Block Color
            block_color = [
                (205, 193, 180), (238, 228, 218), (237, 224, 200), (242, 177, 121),
                (245, 149, 99), (246, 124, 95), (246, 94, 59), (237, 207, 114),
                (237, 204, 97), (237, 200, 80), (237, 197, 63), (237, 194, 46)
            ]
            game_color['background'] = pygame.Color("#faf8ef")
            game_color['board_background'] = pygame.Color("#bbada0")
            block_font_color = [(119, 110, 101), (249, 246, 242)]

            # rendering: Block Font Size
            block_font_size = [int(block_size * rate) for rate in [0.7, 0.6, 0.5, 0.4]]

            if render_mode == "human" or render_mode == "human_only":
                pygame.display.init()
                # (width, height)
                window = pygame.display.set_mode((window_width, window_height))
            else:
                window = pygame.Surface((window_width, window_height))

        if clock is None:
            clock = pygame.time.Clock()

        canvas = pygame.Surface((window_width, window_height))
        canvas.fill(game_color['background'])
        pygame.draw.rect(
            canvas,
            game_color['board_background'],
            (left_top_board, (board_size, board_size))
        )

        for i in range(size):
            for j in range(size):
                _render_block(board, i, j, canvas)

        _render_info(canvas, score, best_score, max_tile, episode, _2048_count)

        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        clock.tick(metadata["render_fps"])

    if render_mode == "terminal" or render_mode == "human":
        # pretty print the board. 1 -> 2, 2 -> 4, etc.
        print("==============================")
        print("\n".join(["\t".join([str(2 ** x) if 2 ** x != 1 else '-' for x in row]) for row in board]))
        print("==============================")
        print(f"Episode: {episode} | Step: {step} | Score: {score} | Max Tile: {max_tile} | Time Elapsed: {time.time() - start_time:.2f}s\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        help="the mode to render the observation",
        choices=["human", "human_only", "terminal"],
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="the number of episodes to run"
    )
    parser.add_argument(
        "--tree-depth", type=int, default=3, help="the depth of the min-max tree"
    )

    args = parser.parse_args()

    global render_mode
    render_mode = args.render_mode

    return args

class MinMax:
    BRANCH_FAILIURE_PENALTY = -100000
    BIGGEST_TILE_IN_CORNER_BONUS = 1000
    BIGGEST_TILE_IN_CENTER_PENALTY = -1000
    EMPTY_TILE_BONUS = 400
    CORNER_HEURISTIC_MULTIPLIER = 2

    best_action: int

    def __init__(self) -> None:
        self.best_action = None

    @classmethod
    def __convert_direction_format(cls, direction):
        # two models have different direciton formats, this is for converting it
        if direction == 0:
            return 2
        elif direction == 1:
            return 3
        elif direction == 2:
            return 0
        elif direction == 3:
            return 1
        else:
            raise ValueError("Invalid direction value")

    # convert the game grid values to their powers of 2 to fit this model
    def min_max_search(self, grid, depth, is_max):
        new_grid = []
        for row in grid:
            for cell in row:
                if cell == 0:
                    new_grid.append(0)
                else:
                    new_grid.append(2**cell)

        alpha = -np.inf
        beta = np.inf
        self._min_max_search(new_grid, depth, is_max, alpha=alpha, beta=beta)

        return self.__convert_direction_format(self.best_action)

    def _min_max_search(self, grid, depth, is_max, alpha, beta):
        if depth == 0:
            return self._reward(grid)

        if not Helper.canMove(grid):
            return self._reward(grid) + self.BRANCH_FAILIURE_PENALTY

        if is_max:
            v = -np.inf

            (children, moved) = Helper.getAvailableChildren(grid)
            best_node_action = None
            for i, child in enumerate(children):
                child_value = self._min_max_search(child, depth, False, alpha, beta)

                if child_value > v:
                    v = child_value
                    best_node_action = moved[i]
                alpha = max(alpha, child_value)
                if beta <= alpha:
                    break

            self.best_action = best_node_action
            return v
        else:
            v = np.inf
            for cell_index in self.get_available_cells(grid):
                gridcopy = list(grid)
                gridcopy[cell_index] = 2
                v = min(
                    v,
                    self._min_max_search(
                        gridcopy, depth - 1, True, alpha=alpha, beta=beta
                    ),
                )
                beta = min(beta, v)
                if beta <= alpha:
                    break

                gridcopy[cell_index] = 4
                v = min(
                    v,
                    self._min_max_search(
                        gridcopy, depth - 1, True, alpha=alpha, beta=beta
                    ),
                )

                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v

    @classmethod
    def get_available_cells(cls, grid):
        available_cell_indices = []
        for i, cell in enumerate(grid):
            if cell == 0:
                available_cell_indices.append(i)
        return available_cell_indices

    @classmethod
    def _reward(cls, grid):
        total = 0

        total += cls.__corner_heuristic(grid)

        # total = cls.__number_of_empty_cells(grid) * cls.EMPTY_TILE_BONUS

        return total

    @classmethod
    def __get_approximate_score(cls, grid):
        def score_formula(x):
            """
            this formula gives the totale score accumulated whilst generating the tile x
            this includes the score accumulated during the the creation of tiles that were merged
            (It assumes that all newly created tiles are created as 2s but this is negligable)
            """
            if x == 0:
                return 0
            x_log = np.log2(x)
            return (x_log - 1) * 2**x_log

        total = 0
        for cell in grid:
            total += score_formula(cell)
        return total


    @classmethod
    def __number_of_empty_cells(self, grid):
        count = 0
        for cell in grid:
            if cell == 0:
                count += 1
        return count

    @classmethod
    def __corner_heuristic(cls, grid):
        score = 0

        # Define the weight for each corner
        corner_weights = [[5, 3, 2, 1],
                          [3, 2, 1, 0],
                          [2, 1, 0, 0],
                          [1, 0, 0, 0]]

        for i in range(4):
            for j in range(4):
                tile_value = grid[i * 4 + j]
                score += tile_value * corner_weights[i][j]

        return score


def check_tile_achieved(max_tiles, tile):
    count = 0

    for max_tile in max_tiles:
        if max_tile >= tile:
            count += 1
    return count


if __name__ == "__main__":
    args = parse_args()
    rm = args.render_mode

    game = gym.make("gym_game2048/Game2048-v0", render_mode="human", goal=8192)
    episode_scores = []
    episode_max_tiles = []

    best_score = 0
    max_tree_depth = args.tree_depth
    number_of_episodes = args.num_episodes
    _2048_count = 0
    for i in range(1, number_of_episodes + 1):
        print(f"Episode {i} started.\n")
        state = game.reset()
        done = False
        reward = 0

        step = 0
        start_time = time.time()
        score = 0
        max_tile = 0

        current_state = state[0]
        while not done:
            step += 1
            min_max = MinMax()
            action = min_max.min_max_search(current_state, max_tree_depth, True)
            observation, reward, terminated, truncated, info = game.step(action)

            current_state = observation

            score = info["score"]
            max_tile = 2 ** info["max"]

            # Check if 2048 is achieved
            if 2048 in info["score_per_step"]:
                _2048_count += 1

            if score > best_score:
                best_score = score

            render(np.reshape(state[0], (-1, size)), score,  best_score, max_tile, i, _2048_count, step, render_mode=rm)

            if truncated or terminated:
                done = True
                episode_scores.append(score)
                episode_max_tiles.append(max_tile)

    print(f"Total number of episodes: {number_of_episodes}")
    print(f"Minmax tree depth: {max_tree_depth}")
    print("-----------------")
    print(f"Average score: {np.mean(episode_scores)}")
    print(f"Minimum score: {np.min(episode_scores)}")
    print(f"Maximum score: {np.max(episode_scores)}")

    print("-----------------")
    count_512_achieved = check_tile_achieved(episode_max_tiles, 512)
    print(
        f"Tile 512 achieved in {count_512_achieved} / {number_of_episodes} episodes"
        f"({count_512_achieved / number_of_episodes * 100}%)"
    )
    count_1024_achieved = check_tile_achieved(episode_max_tiles, 1024)
    print(
        f"Tile 1024 achieved in {count_1024_achieved} / {number_of_episodes} episodes"
        f"({count_1024_achieved / number_of_episodes * 100}%)"
    )
    count_2048_achieved = check_tile_achieved(episode_max_tiles, 2048)
    print(
        f"Tile 2048 achieved in {count_2048_achieved} / {number_of_episodes} episodes"
        f"({count_2048_achieved / number_of_episodes * 100}%)"
    )
    count_4096_achieved = check_tile_achieved(episode_max_tiles, 4096)
    print(
        f"Tile 4096 achieved in {count_4096_achieved} / {number_of_episodes} episodes"
        f"({count_4096_achieved / number_of_episodes * 100}%)"
    )

    print("Episode scores")
    print(episode_scores)
    print("Episode max tiles")
    print(episode_max_tiles)
