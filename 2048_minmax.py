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
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py") + f"_{int(time.time())}",
        help="the name of this experiment",
    )
    parser.add_argument(
        "--goal",
        type=int,
        default=int(np.power(2, 32)),
        help="the goal of the game, note that the game will end when the goal is reached",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--env-id",
        type=str,
        default="gym_game2048/Game2048-v0",
        help="the id of the gymnasium environment",
    )
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
        "--tree-depth", type=int, default=4, help="the depth of the min-max tree"
    )

    args = parser.parse_args()

    global render_mode
    render_mode = args.render_mode

    return args


class MinMax:
    BRANCH_FAILIURE_PENALTY = -1000000

    best_action: int

    def __init__(self) -> None:
        best_action = None

    @classmethod
    def __convert_direction_format(cls, direction):
        #two models have different direciton formats, this is for converting it
        if direction == 0:
            return 1
        elif direction == 1:
            return 3    
        elif direction == 2:
            return 0
        elif direction == 3:
            return 2
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
            return self._evaluate(grid)

        if not Helper.canMove(grid):
            return self._evaluate(grid) + self.BRANCH_FAILIURE_PENALTY

        if is_max:
            v = -np.inf

            (children, moved) = Helper.getAvailableChildren(grid)
            best_node_action = None
            for i, child in enumerate(children):
                child_value = self._min_max_search(child, depth - 1, False)
                if child_value > v:
                    v = child_value
                    best_node_action = i
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
    def _evaluate(cls, grid):
        total = 0
        # only heuristic used is the total value of the grid
        total += cls.__get_grid_value_total(grid)

        return total

    @classmethod
    def __get_grid_value_total(cls, grid):
        total = 0
        for cell in grid:
            total += cell
        return total


if __name__ == "__main__":
    args = parse_args()
    rm = args.render_mode

    game = gym.make("gym_game2048/Game2048-v0", render_mode="human")
    episode_scores = []
    episode_max_tiles = []

    best_score = 0
    max_tree_depth = args.tree_depth

    for i in range(1, args.num_episodes + 1):
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
            done = terminated or truncated

            score = info["score"]
            max_tile = 2 ** info["max"]

            if score > best_score:
                best_score = score

            game.render()

        episode_scores.append(score)
        episode_max_tiles.append(max_tile)

        # TODO: Print and save results to a file
