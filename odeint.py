"""
Naive forward integration methods.
"""
import torch
import torch.nn as nn
import numpy as np
from arnoldi import arnoldi_iteration
from components import replicator

class phi(nn.Module):
    """Exponential integrator based on Arnoldi iteration."""
    def __init__(self, fitness, krylov_dim, t):
        super(phi, self).__init__()
        self.fitness = fitness
        self.krylov_dim = krylov_dim
        self.t_end = t

    def forward(self, s0, v):
        """
        Compute the action phi(Omega @ R_{s0})[v]
        """
        A = lambda x: self.fitness(replicator(s0, x))

        # phi(A) = phi(V @ T @ V.t) = V @ phi(T) @ V.t
        # phi(A)b = V @ phi(T) @ (V.t @ b) = V @ (phi(T) @ e_1) norm(b)
        # target: phi(T)e_1
        #      (T e_1)   (expm(T) phi(T)e_1)
        # expm (0  0 ) = (0            1   )

        krylov_vectors, T = arnoldi_iteration(A, v, self.krylov_dim)
        extended_T = torch.zeros(T.shape[0], T.shape[1]+1, T.shape[2]+1, device=T.device)
        # this use of t_end is equivalent to phi(tA)
        extended_T[:,:-1,:-1] += self.t_end*T
        extended_T[:,0,-1] += self.t_end

        expm_extended = torch.matrix_exp(extended_T)
        krylov_coords = expm_extended[:,:-1,-1:]

        v_end = torch.zeros_like(v)
        for j in range(self.krylov_dim):
            v_end += krylov_coords[:,j,0].reshape((-1,1,1,1))*krylov_vectors[j,...]

        return v_end*torch.linalg.vector_norm(v, dim=(1,2,3)).reshape((-1,1,1,1))