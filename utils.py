import numpy as np
import torch

def scale(var, min, max, shift=True):  # [min, max] --> [-1, 1]
    shifting_factor = max + min if shift else torch.zeros_like(max)
    scaled_var = (2. * var - shifting_factor) / (max - min)
    return scaled_var

def descale(scaled_var, min, max):  # [-1, 1] --> [min, max]
    var = (max - min) / 2. * scaled_var + (max + min) / 2.
    return var


def action_meshgen(single_dim_mesh, env_a_dim):
    # single_dim_mesh = [-1., -.9, -.5, -.2, -.1, -.05, 0., .05, .1, .2, .5, .9, 1.]  # M
    n_grid = len(single_dim_mesh)
    single_dim_mesh = np.array(single_dim_mesh)
    a_dim = n_grid ** env_a_dim  # M ** A
    a_mesh = np.stack(np.meshgrid(*[single_dim_mesh for _ in range(env_a_dim)]))  # (A, M, M, .., M)
    a_mesh_idx = np.arange(a_dim).reshape(*[n_grid for _ in range(env_a_dim)])  # (M, M, .., M)

    return a_mesh, a_mesh_idx, a_dim

def action_idx2mesh(vec_idx, a_mesh, a_mesh_idx, a_dim):
    env_a_dim = len(a_mesh)

    mesh_idx = (a_mesh_idx == vec_idx).nonzero()
    a_nom = np.array([a_mesh[i, :][tuple(mesh_idx)] for i in range(env_a_dim)])
    return a_nom