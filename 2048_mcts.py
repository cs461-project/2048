from distutils.util import strtobool
import time
import gymnasium as gym
import gym_game2048
from gym_game2048.wrappers import PreprocessForTensor, Normalize2048, RewardConverter, RewardByScore
from gymnasium.wrappers import FlattenObservation, TimeLimit, TransformReward

from mcts_general.agent import MCTSAgent
from mcts_general.config import MCTSAgentConfig
from mcts_general.game import DiscreteGymGame

import numpy as np

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
    parser.add_argument("--render-mode", type=str, default="terminal", help="the mode to render the observation", choices=["terminal"])
    parser.add_argument("--num-episodes", type=int, default=10, help="the number of episodes to run")
    parser.add_argument("--save-logs", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="whether to save logs into the `runs/{run_name}` folder")
    parser.add_argument("--save-models", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="whether to save models into the `runs/{run_name}` folder")
    parser.add_argument("--load-model", type=str, default=None, help="the path to load the model from")

    args = parser.parse_args()

    return args

def make_env(env_id, window_title, seed, args):
    def _thunk():
        # TODO: Check if capturing a video is necessary
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array", goal=args.goal)
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: (x % 500 == 0), disable_logger=True)
        # else:
        env = gym.make(env_id, goal=args.goal, render_mode=args.render_mode, window_title=window_title)

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

if __name__ == "__main__":
    args = parse_args()

    config = MCTSAgentConfig()
    config.num_simulations = 400
    agent = MCTSAgent(config)

    if args.load_model is not None:
        agent = agent.load(args.load_model)

    game = DiscreteGymGame(make_env(args.env_id, "CS 461 - Term Project (Group 12) - 2048 (w/ MCTS)", args.seed, args)(), seed=args.seed)
    episode_scores = []
    episode_max_tiles = []

    for i in range (1, args.num_episodes + 1):
        print(f"Episode {i} started.\n")
        state = game.reset()
        done = False
        reward = 0

        step = 0
        start_time = time.time()
        scores = []
        maximum_tiles = []

        while not done:
            step += 1

            action = agent.step(game, state, reward, done)
            state, reward, done, info = game.step(action)

            score = info["score"]
            max_tile = 2 ** info["max"]

            scores.append(score)
            maximum_tiles.append(max_tile)

            game.render_2048()
            print(f"Episode: {i} | Step: {step} | Score: {score} | Max Tile: {max_tile} | Time Elapsed: {time.time() - start_time:.2f}s\n")

        episode_scores.append(scores)
        episode_max_tiles.append(maximum_tiles)

        os.makedirs(f"runs/{args.exp_name}", exist_ok=True)
        if args.save_models:
            agent.save(f"runs/{args.exp_name}/{os.path.basename(__file__).rstrip('.py')}_episode_{i}.pkl")
