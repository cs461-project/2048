import pickle
from mcts_general.config import MCTSAgentConfig, MCTSContinuousAgentConfig
from mcts_general.game import DeepCopyableGame
from mcts_general.mcts import MCTS, select_action, ContinuousMCTS


class MCTSAgent:

    def __init__(self, config: MCTSAgentConfig):
        self.mcts = MCTS(config)
        self.result_node = None

    @property
    def config(self):
        return self.mcts.config

    def step(self, game_state: DeepCopyableGame, observations, reward, done, output_debug_info=False):
        self.result_node, info = self.mcts.run(
            observation=observations,
            reward=reward,
            game=game_state,
            add_exploration_noise=False,
            override_root_with=self.result_node if self.config.reuse_tree else None,
            done=done
        )
        action = select_action(self.result_node, temperature=self.config.temperature)
        if output_debug_info:
            return action, info
        else:
            return action

    def save(self, path):
        mcts, result_node = self.mcts, self.result_node
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.mcts, self.result_node = mcts, result_node

    def load(self, path):
        with open(path, "rb") as f:
            agent = pickle.load(f)
        self.mcts, self.result_node = agent.mcts, agent.result_node
        return self


class ContinuousMCTSAgent(MCTSAgent):

    def __init__(self, config: MCTSContinuousAgentConfig):
        super(ContinuousMCTSAgent, self).__init__(config)
        self.mcts = ContinuousMCTS(config)
