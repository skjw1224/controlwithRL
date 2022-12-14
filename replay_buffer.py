import torch
import numpy as np
from collections import deque
from itertools import islice
import random

class ReplayBuffer(object):
    def __init__(self, env, device, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.env = env
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.device = device

    def add(self, *args):
        experience = list(args)
        self.memory.append(experience)

    def clear(self):
        self.memory.clear()

    def sample(self):
        """Random sample with batch size: shuffle indices. For off-policy methods"""
        batch = random.sample(self.memory, k=min(len(self.memory), self.batch_size))

        # Pytorch replay buffer - squeeze 3rd dim (B, x, 1) -> (B, x)
        x_batch = torch.from_numpy(np.array([e[0] for e in batch]).squeeze(-1)).float().to(self.device)
        u_batch = torch.from_numpy(np.array([e[1] for e in batch]).squeeze(-1)).float().to(self.device)
        r_batch = torch.from_numpy(np.array([e[2] for e in batch]).squeeze(-1)).float().to(self.device)
        x2_batch = torch.from_numpy(np.array([e[3] for e in batch]).squeeze(-1)).float().to(self.device)
        term_batch = torch.from_numpy(np.expand_dims(np.array([e[4] for e in batch]), axis=1)).float().to(self.device)

        return x_batch, u_batch, r_batch, x2_batch, term_batch


    def sample_sequence(self):
        """Ordered sequence replay with batch size: Do not shuffle indices. For on-policy methods"""

        min_start = max(len(self.memory) - self.batch_size, 1)  # If batch_size = episode length
        start_idx = np.random.randint(0, min_start)

        batch = deque(islice(self.memory, start_idx, start_idx + self.batch_size))

        # Pytorch replay buffer - squeeze 3rd dim (B, x, 1) -> (B, x)
        x_batch = torch.from_numpy(np.array([e[0] for e in batch]).squeeze(-1)).float().to(self.device)
        u_batch = torch.from_numpy(np.array([e[1] for e in batch]).squeeze(-1)).float().to(self.device)
        r_batch = torch.from_numpy(np.array([e[2] for e in batch]).squeeze(-1)).float().to(self.device)
        x2_batch = torch.from_numpy(np.array([e[3] for e in batch]).squeeze(-1)).float().to(self.device)
        term_batch = torch.from_numpy(np.expand_dims(np.array([e[4] for e in batch]), axis=1)).float().to(self.device)

        return x_batch, u_batch, r_batch, x2_batch, term_batch


    def __len__(self):
        return len(self.memory)