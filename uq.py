"""
Routines related to Uncertainty Quantification of LDAF models.
"""
import torch
import numpy as np
from components import *
from arnoldi import arnoldi_iteration

def pushforward_class_covariance(Omega, s0, t_end, distr, krylov_dim=5):
    """
    Compute classification pixel marginal covariance of pushforward distribution under LDAF dynamics.
    WARNING: Omega needs to be symmetric.
    
    If v(0) ~ N(0, Sigma_0) then v(T) ~ N(mu, Sigma) and any marginal still has normal distribution
    with a subset of these parameters. Here, we compute the marginal distribution of v(T) for the single
    classification pixel. According to the UQ paper:
        Sigma = expm(tA) @ Sigma_0 @ expm(tA^T)
        mu = t*varphi(tA) @ b
    where 
        dot v(t) = A @ v(t) + b
        A = Pi_0 @ Omega @ R_{s_0}
        b = Pi_0 @ Omega @ s_0
    Computing Sigma requires computing
    the action of c matrix exponentials.
    """
    batch_size, c = s0.shape[:2]
    n = np.prod(s0.shape[2:])
    device = s0.device

    # this is why Omega needs to be symmetric: otherwise we would use Omega.t here
    At = lambda v: t_end*replicator(s0, Omega(mean_free(v)))

    # U will hold expm(tA^T) @ e_i
    U = torch.zeros(batch_size, c*n, c, device=device)
    for i in range(c):
        v = torch.zeros_like(s0)
        v[:,i,-1,:] += 1
        V, T = arnoldi_iteration(At, v, krylov_dim)

        # if Arnoldi iteration is started at b then
        # expm(A)b = V @ expm(T) @ (V.t @ b) = V @ (expm(T) @ e_1) norm(b)
        expT = torch.matrix_exp(T)
        krylov_coords = expT[:,:,0] # multiply by e_1
        
        # compute V @ krylov_coords
        v_end = torch.einsum("ib...,bi->b...", V, krylov_coords)
        v_end *= torch.linalg.vector_norm(v, dim=(1,2,3), keepdim=True)
        U[:,:,i] += v_end.reshape((batch_size, c*n))

    # project columns of U to tangent space
    U = mean_free(U.reshape((batch_size, c, n, c)))
    U = U.reshape((batch_size, c*n, c))

    Sigma = torch.bmm(U.transpose(1,2), distr.cov_action(U))
    return Sigma

def dense_pushforward_class_covariance(Omega, s0, t_end, distr):
    """
    Compute classification pixel marginal covariance of pushforward distribution under LDAF dynamics.
    Variant for dense Omega matrix.
    WARNING: Omega needs to be symmetric.
    
    If v(0) ~ N(0, Sigma_0) then v(T) ~ N(mu, Sigma) and any marginal still has normal distribution
    with a subset of these parameters. Here, we compute the marginal distribution of v(T) for the single
    classification pixel. According to the UQ paper:
        Sigma = expm(tA) @ Sigma_0 @ expm(tA^T)
        mu = t*varphi(tA) @ b
    where 
        dot v(t) = A @ v(t) + b
        A = Pi_0 @ Omega @ R_{s_0}
        b = Pi_0 @ Omega @ s_0
    Computing Sigma requires computing
    the action of c matrix exponentials.
    """
    batch_size, c = s0.shape[:2]
    n = np.prod(s0.shape[2:])
    device = s0.device

    Pi_0 = dense_projection(torch.ones_like(s0).reshape((batch_size, c, n)))
    R_s0 = dense_replicator(s0.reshape((batch_size, c, n)))
    # this is why Omega needs to be symmetric: otherwise we would use Omega.t here
    tA = t_end*torch.bmm(R_s0, torch.matmul(Omega.unsqueeze(0), Pi_0))

    U = torch.matrix_exp(tA)
    U = torch.bmm(Pi_0, U[:,:,-c:])

    Sigma = torch.bmm(U.transpose(1,2), distr.cov_action(U))
    return Sigma

def dense_matrix_exp_action(Omega, s0, t_end, v):
    """
    Compute expm(tA)v.
    """
    vshape = s0.shape
    batch_size, c = vshape[:2]
    n = np.prod(vshape[2:])
    device = s0.device

    Pi_0 = dense_projection(torch.ones_like(s0).reshape((batch_size, c, n)))
    R_s0 = dense_replicator(s0.reshape((batch_size, c, n)))
    tA = t_end*torch.bmm(Pi_0, torch.matmul(Omega.unsqueeze(0), R_s0))
    #tA = t_end*torch.bmm(R_s0, torch.matmul(Omega.unsqueeze(0), Pi_0))
    U = torch.matrix_exp(tA)
    
    return torch.bmm(U, v.reshape((batch_size, c*n, 1))).reshape(vshape)