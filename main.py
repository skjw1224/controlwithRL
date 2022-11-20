import torch
import sys
import os
from env import CstrEnv


from config import Config
from train import Train

config = Config()

config.environment = CstrEnv()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'results/' + config.environment.envname
try:
    os.mkdir('results')
    os.mkdir(path)
except FileExistsError:
    pass
config.result_save_path = path + '/'

config.save_model = False

# algo_name = "DQN"
# algo_name = "DDPG"
algo_name = "A2C"

config.set_algorithm(algo_name)
trainer = Train(config)
trainer.env_rollout()
trainer.plot()