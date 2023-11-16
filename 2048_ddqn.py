# DDQN Implementation for 2048

from time import sleep
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

import os
import argparse
from distutils.util import strtobool

def parse_args():
    # Most important arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py") + f"_{int(time.time())}",
        help="the name of this experiment")

    parser.add_argument("--batch-size", type=int, default=128, help="the batch size of the experiment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--linear-size", type=int, default=128,
        help="size of linear layers")
    parser.add_argument("-lr-decay", "--lr-decay", type=float, default=0.995, help="learning rate decay rate")
    parser.add_argument("--lr-step", type=int, default=100, help="learning rate scheduler step size, decrease learning rate by lr-decay every lr-step steps")
    parser.add_argument("--num-episodes", type=int, default=1000000, help="the number of episodes to run, set to 0 for infinite")
    # parser.add_argument("--min-episodes", type=int, default=100, help="the minimum number of episodes to run before starting to measure performance")

    parser.add_argument("--epsilon-start", type=float, default=1.0, help="the starting epsilon value, for epsilon-greedy exploration (i.e., take random actions with probability epsilon)")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="the ending (minimum) epsilon value, for epsilon-greedy exploration (i.e., take random actions with probability epsilon)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="epsilon decay rate, decay epsilon-start by epsilon-decay every episode, until epsilon-end is reached")

    parser.add_argument("--buffer-size", type=int, default=3e5, help="the replay buffer size of the experiment")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer") # default=2.5e-4
    parser.add_argument("--goal", type=int, default=2048,
        help="goal")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")

    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--load-model", type=str, default="",
        help="whether to load model `runs/{run_name}` folder")

    # Other arguments
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="gym_game2048/Game2048-v0",
        help="the id of the environment")
    # parser.add_argument("--total-timesteps", type=int, default=500000000,
    #     help="total timesteps of the experiments")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--memo", type=str, default="Linear 128",
        help="memo")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)

    return args

def make_env(env_id, window_title, seed, args):
    def _thunk():
        # TODO: Check if capturing a video is necessary
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array", goal=args.goal)
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: (x % 500 == 0), disable_logger=True)
        # else:
        env = gym.make(env_id, goal=args.goal, render_mode=args.render_mode, window_title = window_title)

        #### Add Custom Wrappers ###
        # env = TimeLimit(env, max_episode_steps=3000)
        # env = RewardConverter(env, goal=6, fail=-5, other=-0.0001)

        env = RewardByScore(env)

        env = TransformReward(env, lambda r: r * 0.1)
        env = Normalize2048(env)

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

    # def act(self, state, eps=0.):
    #     state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    #     self.qnetwork_local.eval()
    #     with torch.no_grad():
    #         values = self.qnetwork_local(state)
    #     self.qnetwork_local.train()

    #     if random.random() <= eps:
    #         action = np.random.randint(0, self.action_size)
    #     else:
    #         action = np.argmax(values.cpu().numpy())
    #     return action

    def act(self, state, eps=0.):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = self.qnetwork_target(state)

        if random.random() <= eps:
            action = np.random.randint(0, self.action_size)
        else:
            action = np.argmax(values.cpu().numpy())

        return action

    # def learn(self, experiences, gamma):
    #     states, actions, rewards, next_states, dones = experiences
    #     Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    #     Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    #     Q_expected = self.qnetwork_local(states).gather(1, actions)
    #     loss = F.mse_loss(Q_expected, Q_targets)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.soft_update(self.qnetwork_local, self.qnetwork_target, 0.001)

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

    # def soft_update(self, local_model, target_model, tau):
    #     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

    def transfer_parameters(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args):
    env = make_env(args.env_id, seed=args.seed, args=args, window_title="CS 461 - Term Project (Group 12) - 2048 (w/ DDQN)")()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DDQN(state_size, action_size, args, seed=args.seed)

    scores = []
    scores_window = collections.deque(maxlen=100)
    eps = args.epsilon_start
    eps_end = args.epsilon_end
    eps_decay = args.epsilon_decay

    if args.num_episodes == 0:
        args.num_episodes = float("inf")

    for i_episode in range(1, args.num_episodes+1):
        state, info = env.reset(seed=args.seed)
        score = 0

        while True:
            action = agent.act(state, eps)
            observation, reward, terminated, truncated, info = env.step(action)
            agent.step(state, action, reward, observation, terminated)
            state = observation
            score += reward
            if terminated or truncated:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)

        if i_episode % 5 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps))
        if np.mean(scores_window)>=args.goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            if args.save_model:
                torch.save(agent.qnetwork_local.state_dict(), f"runs/{args.exp_name}/checkpoint.pth")
            break
    return scores

def debug(args):
    env = make_env("gym_game2048/Game2048-v0", seed=42, rank=0, log_dir=None, args=args)()

    observation, info = env.reset(seed=42)
    for _ in range(10):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"action: {action}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        sleep(5)

        if args.render_mode == "rgb_array":
            env_render = env.render()
            print(env_render)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        debug(args)

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=args.exp_name, config=args)

    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.manual_seed_all(args.seed)

    if args.load_model:
        args.save_model = False

    if args.save_model:
        os.makedirs(f"runs/{args.exp_name}", exist_ok=True)

    scores = train(args)
    if args.track:
        wandb.log({"scores": scores})
