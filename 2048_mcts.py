import numpy as np
import random
import math
from copy import deepcopy
from distutils.util import strtobool
import time
import gymnasium as gym
from gym_game2048.wrappers import PreprocessForTensor, Normalize2048, RewardConverter, RewardByScore
from gymnasium.wrappers import FlattenObservation, TimeLimit, TransformReward
import matplotlib.pyplot as plt
import numpy as np
import pygame
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py") + f"_{int(time.time())}",
        help="the name of this experiment")
    parser.add_argument("--goal", type=int, default=int(np.power(2, 32)), help="the goal of the game, note that the game will end when the goal is reached")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--env-id", type=str, default="gym_game2048/Game2048-v0", help="the id of the gymnasium environment")
    parser.add_argument("--render-mode", type=str, default="human", help="the mode to render the next_state", choices=["human", "human_only", "terminal"])
    parser.add_argument("--num-episodes", type=int, default=10, help="the number of episodes to run")
    parser.add_argument("--n", type=int, default=4, help="the size of the board")
    parser.add_argument("--mcts-simulations", type=int, default=400, help="the number of simulations to run for each step, i.e., the depth of the tree")

    args = parser.parse_args()

    global size, render_mode
    size = args.n
    render_mode = args.render_mode

    return args

window = None
window_title = "CS 461 - Term Project (Group 12) - 2048 (w/ MCTS)"
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
metadata = {"render_fps": 30}  # Modify the frame rate as needed

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

def make_env(env_id, window_title, seed, args):
    def _thunk():
        env = gym.make(env_id, goal=args.goal, render_mode="terminal", window_title=window_title)

        #### Add Custom Wrappers ###
        # env = TimeLimit(env, max_episode_steps=3000)
        # env = RewardConverter(env, goal=6, fail=-5, other=-0.0001)

        env = RewardByScore(env, log=False)

        # env = TransformReward(env, lambda r: r * 0.1)
        # env = Normalize2048(env)

        env = FlattenObservation(env)
        #############################
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return _thunk

class Normalizer:
    def __init__(self):
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, value):
        self.max = max(self.max, value)
        self.min = min(self.min, value)

    def normalize(self, value):
        if self.max > self.min:
            return (value - self.min) / (self.max - self.min)
        return value

class DeepCopyWrapper(gym.Wrapper):
    def __deepcopy__(self, memo={}):
        new_instance = self.__class__
        result = new_instance.__new__(new_instance)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

class DeepCopyEnv():
    def __init__(self, env):
        self.env = DeepCopyWrapper(env) if not isinstance(env, DeepCopyWrapper) else env

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def step(self, action):
        next_state, reward, is_done, _, info = self.env.step(action)
        return next_state, reward, is_done, _, info

    def get_copy(self):
        return DeepCopyEnv(deepcopy(self.env))

    def legal_actions(self):
        return [i for i in range(self.env.action_space.n)]

class Node:
    def __init__(self):
        self.children = {}
        self.number_of_visits = 0
        self.value_sum = 0.0
        self.next_state = None
        self.reward = 0
        self.is_done = False
        self.deep_copy = None

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def mean_value(self) -> float:
        return 0 if self.number_of_visits == 0 or self.is_done else self.value_sum / self.number_of_visits

    def expand_node(self, next_state, reward, is_done, deep_copy, initial_number_of_visits=0):
        self.next_state = next_state
        self.reward = reward
        self.is_done = is_done
        self.deep_copy = deep_copy
        self.number_of_visits = initial_number_of_visits
        for action in deep_copy.legal_actions():
            self.children[action] = Node()

def ucb_score(parent, child, normalizer):
    return ((2 * (math.sqrt(parent.number_of_visits) / (child.number_of_visits + 1))) * (1 / len(parent.children))) + (normalizer.normalize(child.reward + child.mean_value) if child.number_of_visits > 0 else 0)

def backpropagate(path, value, normalizer):
    for node in reversed(path):
        node.number_of_visits += 1
        node.value_sum += value
        normalizer.update(node.reward + node.mean_value)
        value = node.reward + value

def child_select(node, normalizer):
    max_ucb = max(ucb_score(node, child, normalizer) for action, child in node.children.items())
    action = random.choice([action for action, child in node.children.items() if ucb_score(node, child, normalizer) == max_ucb])
    return action, node.children[action]

def action_select(node):
    number_of_visits = np.array([child.number_of_visits for action, child in node.children.items()])
    actions = np.array([action for action, child in node.children.items()])
    return actions[np.argmax(number_of_visits)]

def mcts_run(next_state, reward, is_done, env, num_of_simulations):
        root = Node()
        reward = reward
        root.expand_node(next_state, reward, is_done, env.get_copy(), initial_number_of_visits=0)

        normalizer = Normalizer()

        for i in range(num_of_simulations):
            node = root
            path = [node]

            while node.expanded:
                action, node = child_select(node, normalizer)
                path.append(node)

            parent = path[-2]
            if not parent.is_done:
                deep_copy = parent.deep_copy.get_copy()
                next_state, reward, is_done, _, info = deep_copy.step(action)
                value = reward
                node.expand_node(next_state, reward, is_done, deep_copy, initial_number_of_visits=0)
            else:
                value = 0

            backpropagate(path, value, normalizer)

        return root

def mcts_step(env, next_state, reward, is_done, num_of_simulations):
    result_node = mcts_run(next_state, reward, is_done, env, num_of_simulations)
    return action_select(result_node)

if __name__ == "__main__":
    args = parse_args()
    rm = args.render_mode
    num_of_simulations = args.mcts_simulations

    env = DeepCopyEnv(make_env(args.env_id, window_title, args.seed, args)())
    episode_scores = []
    episode_max_tiles = []

    best_score = 0

    _2048_count = 0

    for i in range (1, args.num_episodes + 1):
        print(f"Episode {i} started.\n")
        state = env.reset()
        is_done = False
        reward = 0

        step = 0
        start_time = time.time()
        scores = []
        maximum_tiles = []

        while not is_done:
            step += 1

            action = mcts_step(env, state, reward, is_done, num_of_simulations)
            state, reward, is_done, _, info = env.step(action)
            score = info["score"]
            max_tile = 2 ** info["max"]

            # check if 2048 is reached
            if 2048 in info["score_per_step"]:
                _2048_count += 1

            scores.append(score)
            maximum_tiles.append(max_tile)

            if score > best_score:
                best_score = score

            render(np.reshape(state, (-1, size)), score,  best_score, max_tile, i, _2048_count, step, render_mode=rm)

        # Add last score to the list
        episode_scores.append(scores[-1])
        # Add maximum of the maximum tiles to the list
        episode_max_tiles.append(max(maximum_tiles))
