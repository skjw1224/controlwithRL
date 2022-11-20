import utils

# Controllers
from dqn import DQN
from ddpg import DDPG
from a2c import A2C
from pid import PID

# Explorers
from explorers import OU_Noise, E_greedy

# Approximators
from nn_create import NeuralNetworks

class Config(object):
    """Save hyperparameters"""
    def __init__(self):
        self.device = None
        self.environment = None
        self.algorithm = {}
        self.hyperparameters = {}
        self.result_save_path = None

        self.alg_key_matching()

    def set_algorithm(self, algo_name):
        self.algorithm = {'controller':
                                  {'function': self.ctrl_key2arg[algo_name],
                                   'name': algo_name,
                                   'action_type': None,
                                   'action_mesh_idx': None,
                                   'model_requirement': None,
                                   'initial_controller': None},
                              'explorer':
                                  {'function': None,
                                   'name': None},
                              'approximator':
                                  {'function': None,
                                   'name': None}}

        # Default (algorithm specific) settings
        self.controller_default_settings()
        self.hyper_default_settings()

    def alg_key_matching(self):
        self.ctrl_key2arg = {
            "DQN": DQN,
            "DDPG": DDPG,
            "A2C": A2C,
            "PID": PID
        }

        self.exp_key2arg = {
            'e_greedy': E_greedy,
            'OU': OU_Noise,
        }

        self.approx_key2arg = {
            'DNN': NeuralNetworks,
        }

    def controller_default_settings(self):
        # Discrete or Continuous
        if self.algorithm['controller']['name'] in ['DQN']:
            self.algorithm['controller']['action_type'] = 'discrete'
        else:
            self.algorithm['controller']['action_type'] = 'continuous'


        # Default initial controller
        self.algorithm['controller']['initial_controller'] = PID

        # Default explorer
        if self.algorithm['controller']['action_type'] == 'continuous':
            self.algorithm['explorer']['name'] = 'OU'
            self.algorithm['explorer']['function'] = self.exp_key2arg['OU']
        else:
            self.algorithm['explorer']['name'] = 'e_greedy'
            self.algorithm['explorer']['function'] = self.exp_key2arg['e_greedy']

        # Default approximator
        self.algorithm['approximator']['name'] = 'DNN'
        self.algorithm['approximator']['function'] = self.approx_key2arg['DNN']


    def hyper_default_settings(self):
        self.hyperparameters['init_ctrl_idx'] = 10
        self.hyperparameters['explore_epi_idx'] = 50
        self.hyperparameters['max_episode'] = 200
        self.hyperparameters['hidden_nodes'] = [64, 64, 32]
        self.hyperparameters['tau'] = 0.05
        self.hyperparameters['buffer_size'] = 600
        self.hyperparameters['minibatch_size'] = 32

        self.hyperparameters['adam_eps'] = 1E-4
        self.hyperparameters['l2_reg'] = 1E-3
        self.hyperparameters['grad_clip_mag'] = 5.0

        self.hyperparameters['save_period'] = 5
        self.hyperparameters['plot_snapshot'] = [0, 5, 10, 15, 20]

        # Algorithm specific settings
        if self.algorithm['controller']['name'] in ['DQN']:
            self.hyperparameters['single_dim_mesh'] = [-1., -.9, -.5, -.2, -.1, -.05, 0., .05, .1, .2, .5, .9, 1.]
            self.algorithm['controller']['action_mesh_idx'] = \
                utils.action_meshgen(self.hyperparameters['single_dim_mesh'], self.environment.a_dim)
            self.hyperparameters['learning_rate'] = 2E-4
        elif self.algorithm['controller']['name'] == 'DDPG':
            self.hyperparameters['critic_learning_rate'] = 1E-2
            self.hyperparameters['actor_learning_rate'] = 1E-3
        elif self.algorithm['controller']['name'] == 'A2C':
            self.hyperparameters['n_step_TD'] = 10
            self.hyperparameters['critic_learning_rate'] =2E-4
            self.hyperparameters['actor_learning_rate'] = 1E-4
            self.hyperparameters['eps_decay_rate'] = 0.99
