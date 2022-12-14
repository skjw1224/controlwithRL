import numpy as np
import random
import copy

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.a_dim = self.env.a_dim

        self.mu0 = 0.
        self.theta = 0.15
        self.sigma = 0.2

        self.eps0 = 0.1
        self.epi_denom = 1

        self.mu = self.mu0 * np.ones([self.a_dim, 1])

        random.seed(123)

    def exp_schedule(self, epi):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu
        self.epsilon = self.eps0 / (1. + (epi / self.epi_denom))

    def sample(self, epi, step, u_nom):
        if step == 0:
            self.exp_schedule(epi)

        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.a_dim, 1)
        self.state += dx
        noise = self.state * self.epsilon
        u_exp = noise + u_nom

        return u_exp

class E_greedy(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.a_dim = self.env.a_dim

        self.eps0 = 0.1
        self.epi_denom = 1

    def exp_schedule(self, epi):
        self.eps = self.eps0 / (1. + (epi / self.epi_denom))

    def sample(self, epi, step, u_nom):
        if step == 0:
            self.exp_schedule(epi)

        if np.random.random() <= self.eps:
            u_exp = np.random.randint(low=0, high=self.a_dim, size=[1, 1])
        else:
            u_exp = u_nom

        return u_exp