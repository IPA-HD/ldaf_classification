import torch
import torch.nn as nn

from components import mean_free
import odeint

class LAF(nn.Module):
    """
    Linearized S-Assignment Flow.
    """
    def __init__(self, fitness, t=3.0, krylov_dim=5):
        super(LAF, self).__init__()
        self.fitness = fitness
        # A = fitness @ R_s0
        self.phi = odeint.phi(fitness, krylov_dim, t)
        self.t_end = t
    
    def forward(self, s0):
        v0 = mean_free(self.fitness(s0))
        v_end = mean_free(self.phi(s0, v0))
        return self.t_end*v_end