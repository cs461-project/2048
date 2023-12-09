from distutils.util import strtobool
import os
import time
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from gym_game2048.wrappers import RewardByScore
import argparse
import grid_helpers_2048 as Helper


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
        "--tree-depth", type=int, default=2, help="the depth of the min-max tree"
    )

    args = parser.parse_args()

    global render_mode
    render_mode = args.render_mode

    return args


class MinMax:
    BRANCH_FAILIURE_PENALTY = -1000000

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

        self._min_max_search(new_grid, depth, is_max)
        return self.__convert_direction_format(self.best_action)

    def _min_max_search(self, grid, depth, is_max):
        if depth == 0:
            return self._reward(grid)

        if not Helper.canMove(grid):
            return self._reward(grid) + self.BRANCH_FAILIURE_PENALTY

        if is_max:
            v = -np.inf

            (children, moved) = Helper.getAvailableChildren(grid)
            best_node_action = None
            for i, child in enumerate(children):
                child_value = self._min_max_search(child, depth, False)
                if child_value > v:
                    v = child_value
                    best_node_action = moved[i]
            self.best_action = best_node_action
            return v
        else:
            v = np.inf
            for cell_index in self.get_available_cells(grid):
                gridcopy = list(grid)
                gridcopy[cell_index] = 2
                v = min(v, self._min_max_search(gridcopy, depth - 1, True))
                gridcopy[cell_index] = 4
                v = min(v, self._min_max_search(gridcopy, depth - 1, True))
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
        # only heuristic used is the total value of the grid
        total += cls.__get_approximate_score(grid)

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


def check_tile_achieved(max_tiles: list[int], tile: int) -> list[int]:
    count = 0

    for max_tile in max_tiles:
        if max_tile >= tile:
            count += 1
    return count


if __name__ == "__main__":
    args = parse_args()
    rm = args.render_mode

    game = gym.make("gym_game2048/Game2048-v0", render_mode="human")
    episode_scores = []
    episode_max_tiles = []

    best_score = 0
    max_tree_depth = args.tree_depth
    number_of_episodes = args.num_episodes
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

            if score > best_score:
                best_score = score

            game.render()

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

