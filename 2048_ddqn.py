import time
import gymnasium as gym
import gym_game2048
from gym_game2048.wrappers import PreprocessForTensor, Normalize2048, RewardConverter, RewardByScore
from gymnasium.wrappers import FlattenObservation, TimeLimit, TransformReward
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import argparse
from distutils.util import strtobool
import pygame
import warnings
warnings.filterwarnings("ignore")

window = None
window_title = "CS 461 - Term Project (Group 12) - 2048 (w/ DDQN)"
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
metadata = {"render_fps": 15}  # Modify the frame rate as needed

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
        print(f"Episode: {episode} | Step: {step} | Score: {score} | Max Tile: {max_tile}\n")

def parse_args():
    # Most important arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py") + f"{int(time.time())}",
        help="the name of this experiment")
    parser.add_argument("--batch-size", type=int, default=512, help="the batch size of the experiment")
    parser.add_argument("--num-steps", type=int, default=5000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--linear-size", type=int, default=128,
        help="size of linear layers")
    parser.add_argument("-lr-decay", "--lr-decay", type=float, default=0.995, help="learning rate decay rate")
    parser.add_argument("--lr-step", type=int, default=100, help="learning rate scheduler step size, decrease learning rate by lr-decay every lr-step steps")
    parser.add_argument("--num-episodes", type=int, default=1000, help="the number of episodes to run, set to 0 for infinite")

    parser.add_argument("--epsilon-start", type=float, default=1.0, help="the starting epsilon value, for epsilon-greedy exploration (i.e., take random actions with probability epsilon)")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="the ending (minimum) epsilon value, for epsilon-greedy exploration (i.e., take random actions with probability epsilon)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="epsilon decay rate, decay epsilon-start by epsilon-decay every episode, until epsilon-end is reached")
    parser.add_argument("--buffer-size", type=int, default=1e4, help="the replay buffer size of the experiment")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--learning-rate", type=float, default=1e-2,
        help="the learning rate of the optimizer") # default=2.5e-4
    parser.add_argument("--goal", type=int, default=2048,
        help="goal")
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--env-id", type=str, default="gym_game2048/Game2048-v0",
        help="the id of the environment")

    args = parser.parse_args()

    return args

def make_env(env_id, window_title, seed, args):
    def _thunk():
        env = gym.make(env_id, goal=args.goal, render_mode=args.render_mode, window_title = window_title)

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

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=buffer_size)
        self.experience = collections.namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DDQN:
    def __init__(self, state_size, action_size, args, seed = None):
        self.state_size = state_size
        self.action_size = action_size
        self.args = args
        self.seed = seed
        if seed is None: self.seed = args.seed

        self.batch_size = args.batch_size

        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units=args.linear_size, fc2_units=args.linear_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units=args.linear_size, fc2_units=args.linear_size).to(device)
        self.transfer_parameters(self.qnetwork_local, self.qnetwork_target)

        # Only train the local network
        for param in self.qnetwork_target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_decay)

        self.memory = ReplayBuffer(buffer_size=int(args.buffer_size), batch_size=args.batch_size, seed=seed)
        self.gamma = args.gamma
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.args.num_steps
        if self.t_step == 0:
            if len(self.memory) > self.args.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.args.gamma)

    def act(self, state, eps=0.):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = self.qnetwork_target(state)

        if random.random() <= eps:
            action = np.random.randint(0, self.action_size)
        else:
            action = np.argmax(values.cpu().numpy())

        return action

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        actions = actions.view(-1, 1) if len(actions.shape) == 1 else actions

        Q_locals = self.qnetwork_local(states)
        Q_locals_next = self.qnetwork_local(next_states)
        Q_targets_next = self.qnetwork_target(next_states)

        Q_value = Q_locals.gather(1, actions).squeeze(1)
        Q_value_next = Q_targets_next.gather(1, torch.max(Q_locals_next, 1)[1].unsqueeze(1)).squeeze(1)
        Q_expected = rewards + gamma * Q_value_next * (1 - dones)

        loss = (Q_value - Q_expected.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.transfer_parameters(self.qnetwork_local, self.qnetwork_target)

    def transfer_parameters(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def check_tile_achieved(max_tiles: list, tile: int) -> int:
    count = 0

    for max_tile in max_tiles:
        if max_tile >= tile:
            count += 1
    return count

def train(args):
    env = make_env(args.env_id, seed=args.seed, args=args, window_title="CS 461 - Term Project (Group 12) - 2048 (w/ DDQN)")()
    rm = args.render_mode
    _2048_count = 0
    best_score = 0

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DDQN(state_size, action_size, args, seed=args.seed)

    eps = args.epsilon_start
    eps_end = args.epsilon_end
    eps_decay = args.epsilon_decay

    episode_scores = []
    episode_max_tiles = []

    total_steps = 0

    if args.num_episodes == 0:
        args.num_episodes = float("inf")

    for i_episode in range(1, args.num_episodes+1):
        state, info = env.reset(seed=args.seed)

        score = 0

        step = 0
        start_time = time.time()

        while True:
            action = agent.act(state, eps)
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_steps += 1
            agent.step(state, action, reward, observation, terminated)
            state = observation
            score += reward
            max_ = 2 ** info["max"]

            if 2048 in info["score_per_step"]:
                _2048_count += 1
            render(np.reshape(state, (-1, size)), info["score"],  best_score, max_, i_episode, _2048_count, total_steps, render_mode=rm)
            if terminated or truncated:
                break

        eps = max(eps_end, eps_decay*eps)

        episode_scores.append(score)
        episode_max_tiles.append(max_)

        if score > best_score:
            best_score = score

# Only for experiments
# def train_new(args):
#     env = make_env(args.env_id, seed=args.seed, args=args, window_title="CS 461 - Term Project (Group 12) - 2048 (w/ DDQN)")()

#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n

#     agent = DDQN(state_size, action_size, args, seed=args.seed)

#     scores = []
#     eps = args.epsilon_start
#     eps_end = args.epsilon_end
#     eps_decay = args.epsilon_decay

#     episode_scores = []
#     episode_max_tiles = []

#     total_steps = 0

#     if args.num_episodes == 0:
#         args.num_episodes = float("inf")

#     for i_episode in range(1, args.num_episodes+1):
#         state, info = env.reset(seed=args.seed)

#         episode_max_tile_so_far = 0
#         score = 0

#         while True:
#             action = agent.act(state, eps)
#             observation, reward, terminated, truncated, info = env.step(action)
#             total_steps += 1
#             agent.step(state, action, reward, observation, terminated)
#             state = observation
#             score += reward
#             if terminated or truncated:
#                 break

#         eps = max(eps_end, eps_decay*eps)

#         score = info["score"]
#         max_ = 2 ** info["max"]

#         episode_scores.append(score)
#         episode_max_tiles.append(max_)

#         print('\rEpisode {}, Step {}, Score {}, Max Tile {}'.format(i_episode, total_steps, score, max_))

#     return episode_scores, episode_max_tiles

if __name__ == "__main__":
    args = parse_args()

    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    train(args)

    # Comment out for the experiments

    # Specify different gamma values
    # gamma_values = [0.1, 0.5, 0.8]

    # gamma_scores_list = []
    # gamma_max_tiles_list = []

    # for gamma in gamma_values:
    #     print(f"================== GAMMA: {gamma} ==================")
    #     args.gamma = gamma
    #     scores, max_tiles = train_new(args)

    #     gamma_scores_list.append(scores)
    #     gamma_max_tiles_list.append(max_tiles)

    # # Plot the results first separately
    # for i, gamma in enumerate(gamma_values):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(gamma_scores_list[i], label=f'Gamma = {gamma}')

    #     plt.title('Scores Over Episodes')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Score')
    #     plt.legend()
    #     plt.savefig(f"ddqn_gamma_{gamma}_max_scores.png")

    # for i, gamma in enumerate(gamma_values):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(gamma_max_tiles_list[i], label=f'Gamma = {gamma}')

    #     plt.title('Max Tiles Over Episodes')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Max Tile')
    #     plt.legend()
    #     plt.savefig(f"ddqn_gamma_{gamma}_max_tiles.png")

    # # Plot the results together
    # # First, max scores
    # plt.figure(figsize=(10, 5))
    # for i, gamma in enumerate(gamma_values):
    #     plt.plot(gamma_scores_list[i], label=f'Gamma = {gamma}')

    # plt.title('Scores Over Episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.savefig(f"ddqn_gamma_max_scores.png")

    # # Then, max tiles
    # plt.figure(figsize=(10, 5))
    # for i, gamma in enumerate(gamma_values):
    #     plt.plot(gamma_max_tiles_list[i], label=f'Gamma = {gamma}')

    # plt.title('Max Tiles Over Episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Max Tile')
    # plt.legend()
    # plt.savefig(f"ddqn_gamma_max_tiles.png")

    # # # Pickle the results
    # import pickle

    # with open('ddqn_gamma_scores.pkl', 'wb') as f:
    #     pickle.dump(gamma_scores_list, f)

    # with open('ddqn_gamma_max_tiles.pkl', 'wb') as f:
    #     pickle.dump(gamma_max_tiles_list, f)

    # learning_rates = [1e-4, 2.5e-4, 5e-4]

    # lr_scores_list = []
    # lr_max_tiles_list = []

    # for lr in learning_rates:
    #     print(f"================== LEARNING RATE: {lr} ==================")
    #     args.learning_rate = lr
    #     scores, max_tiles = train_new(args)

    #     lr_scores_list.append(scores)
    #     lr_max_tiles_list.append(max_tiles)

    # Plot the results first separately
    # for i, lr in enumerate(learning_rates):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(lr_scores_list[i], label=f'Learning Rate = {lr}')

    #     plt.title('Scores Over Episodes')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Score')
    #     plt.legend()
    #     plt.savefig(f"ddqn_lr_{lr}_max_scores.png")

    # for i, lr in enumerate(learning_rates):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(lr_max_tiles_list[i], label=f'Learning Rate = {lr}')

    #     plt.title('Max Tiles Over Episodes')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Max Tile')
    #     plt.legend()
    #     plt.savefig(f"ddqn_lr_{lr}_max_tiles.png")

    # # Plot the results together
    # # First, max scores
    # plt.figure(figsize=(10, 5))
    # for i, lr in enumerate(learning_rates):
    #     plt.plot(lr_scores_list[i], label=f'Learning Rate = {lr}')

    # plt.title('Scores Over Episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.savefig(f"ddqn_lr_max_scores.png")

    # # Then, max tiles
    # plt.figure(figsize=(10, 5))
    # for i, lr in enumerate(learning_rates):
    #     plt.plot(lr_max_tiles_list[i], label=f'Learning Rate = {lr}')

    # plt.title('Max Tiles Over Episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Max Tile')
    # plt.legend()
    # plt.savefig(f"ddqn_lr_max_tiles.png")

    # # Pickle the results
    # import pickle

    # with open('ddqn_lr_scores.pkl', 'wb') as f:
    #     pickle.dump(lr_scores_list, f)

    # with open('ddqn_lr_max_tiles.pkl', 'wb') as f:
    #     pickle.dump(lr_max_tiles_list, f)

    # buffer_sizes = [1e4, 5e4, 1e5]

    # bs_scores_list = []
    # bs_max_tiles_list = []

    # for bs in buffer_sizes:
    #     print(f"================== BUFFER SIZE: {bs} ==================")
    #     args.buffer_size = bs
    #     scores, max_tiles = train_new(args)

    #     bs_scores_list.append(scores)
    #     bs_max_tiles_list.append(max_tiles)

    # # Plot the results first separately
    # for i, bs in enumerate(buffer_sizes):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(bs_scores_list[i], label=f'Buffer Size = {bs}')

    #     plt.title('Scores Over Episodes')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Score')
    #     plt.legend()
    #     plt.savefig(f"ddqn_bs_{bs}_scores.png")

    # for i, bs in enumerate(buffer_sizes):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(bs_max_tiles_list[i], label=f'Buffer Size = {bs}')

    #     plt.title('Max Tiles Over Episodes')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Max Tile')
    #     plt.legend()
    #     plt.savefig(f"ddqn_bs_{bs}_max_tiles.png")

    # # Plot the results together
    # # First, scores
    # plt.figure(figsize=(10, 5))
    # for i, bs in enumerate(buffer_sizes):
    #     plt.plot(bs_scores_list[i], label=f'Buffer Size = {bs}')

    # plt.title('Scores Over Episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Score')
    # plt.legend()

    # plt.savefig(f"ddqn_bs_scores.png")

    # # Then, max tiles
    # plt.figure(figsize=(10, 5))
    # for i, bs in enumerate(buffer_sizes):
    #     plt.plot(bs_max_tiles_list[i], label=f'Buffer Size = {bs}')

    # plt.title('Max Tiles Over Episodes')
    # plt.xlabel('Episodes')
    # plt.ylabel('Max Tile')
    # plt.legend()

    # plt.savefig(f"ddqn_bs_max_tiles.png")

    # # Pickle the results
    # import pickle

    # with open('ddqn_bs_scores.pkl', 'wb') as f:
    #     pickle.dump(bs_scores_list, f)

    # with open('ddqn_bs_max_tiles.pkl', 'wb') as f:
    #     pickle.dump(bs_max_tiles_list, f)

    # Load the results for learning rate

    # import pickle
    # in_file = open("ddqn_lr_scores.pkl", "rb")
    # lr_scores_list = pickle.load(in_file)
    # in_file.close()

    # in_file = open("ddqn_lr_max_tiles.pkl", "rb")
    # lr_max_tiles_list = pickle.load(in_file)
    # in_file.close()

    # # Infer from the results

    # # Average scores
    # lr_scores_means = []
    # for scores in lr_scores_list:
    #     lr_scores_means.append(np.mean(scores))

    # # Pretty print the results
    # print("Average Scores for Different Learning Rates")
    # print("-------------------------------------------")
    # for i, lr in enumerate(learning_rates):
    #     print(f"Learning Rate: {lr}, Average Score: {lr_scores_means[i]}")

    # print()

    # # Pretty print the results
    # print("Tiles Achieved for Different Learning Rates")
    # print("-------------------------------------------")
    # for i, lr in enumerate(learning_rates):
    #     # For each learning rate print the number of tiles achieved for each tile
    #     print(f"Learning Rate: {lr}")
    #     for t in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    #         times_accomplished = lr_max_tiles_list[i].count(t)

    #         if times_accomplished > 0:
    #             print(f"\tTile {t} achieved {times_accomplished} times ({round(times_accomplished / len(lr_max_tiles_list[i]) * 100, 2)}%)")

    # print()

    # # Do the same for gamma

    # # Load the results for learning rate

    # import pickle

    # in_file = open("ddqn_gamma_scores.pkl", "rb")

    # gamma_scores_list = pickle.load(in_file)

    # in_file.close()

    # in_file = open("ddqn_gamma_max_tiles.pkl", "rb")

    # gamma_max_tiles_list = pickle.load(in_file)

    # in_file.close()


    # # Infer from the results

    # # Average scores

    # gamma_scores_means = []

    # for scores in gamma_scores_list:
    #     gamma_scores_means.append(np.mean(scores))

    # # Pretty print the results

    # print("Average Scores for Different Gamma Values")

    # print("-------------------------------------------")

    # for i, gamma in enumerate(gamma_values):
    #     print(f"Gamma: {gamma}, Average Score: {gamma_scores_means[i]}")

    # print()

    # # Pretty print the results

    # print("Tiles Achieved for Different Gamma Values")

    # print("-------------------------------------------")

    # for i, gamma in enumerate(gamma_values):
    #     # For each learning rate print the number of tiles achieved for each tile
    #     print(f"Gamma: {gamma}")
    #     for t in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    #         times_accomplished = gamma_max_tiles_list[i].count(t)

    #         if times_accomplished > 0:
    #             print(f"\tTile {t} achieved {times_accomplished} times ({round(times_accomplished / len(gamma_max_tiles_list[i]) * 100, 2)}%)")

    # print()

    # # Do the same for buffer size

    # # Load the results for learning rate

    # import pickle

    # in_file = open("ddqn_bs_scores.pkl", "rb")

    # bs_scores_list = pickle.load(in_file)

    # in_file.close()

    # in_file = open("ddqn_bs_max_tiles.pkl", "rb")

    # bs_max_tiles_list = pickle.load(in_file)

    # in_file.close()


    # # Infer from the results

    # # Average scores

    # bs_scores_means = []

    # for scores in bs_scores_list:
    #     bs_scores_means.append(np.mean(scores))

    # # Pretty print the results

    # print("Average Scores for Different Buffer Sizes")

    # print("-------------------------------------------")

    # for i, bs in enumerate(buffer_sizes):
    #     print(f"Buffer Size: {int(bs)}, Average Score: {bs_scores_means[i]}")

    # print()

    # # Pretty print the results

    # print("Tiles Achieved for Different Buffer Sizes")

    # print("-------------------------------------------")

    # for i, bs in enumerate(buffer_sizes):
    #     # For each learning rate print the number of tiles achieved for each tile
    #     print(f"Buffer Size: {int(bs)}")
    #     for t in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    #         times_accomplished = bs_max_tiles_list[i].count(t)

    #         if times_accomplished > 0:
    #             print(f"\tTile {t} achieved {times_accomplished} times ({round(times_accomplished / len(bs_max_tiles_list[i]) * 100, 2)}%)")

    # print()
