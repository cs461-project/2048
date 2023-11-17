class MCTSAgentConfig:

    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.999
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.num_simulations = 400
        self.reuse_tree = False
        self.temperature = 0
        self.do_roll_outs = False
        self.number_of_roll_outs = 5
        self.max_roll_out_depth = 20
        self.do_roll_out_steps_with_simulation_true = False


class MCTSContinuousAgentConfig(MCTSAgentConfig):

    def __init__(self):
        super(MCTSContinuousAgentConfig, self).__init__()
        # single progressive widening
        self.C = 1
        self.alpha = 0.25
