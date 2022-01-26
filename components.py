"""
"""
import torch
import math
import numpy as np
from torch.nn.functional import softmax

def lift(W, V):
    r"""
    Lifting map on the assignment manifold.

    Parameters
    ----------
    W : (batch, c, ...) torch.tensor
        Assignment matrix in :math:`\mathcal{W}`.
    V : (batch, c, ...) torch.tensor
        Tangent vector.

    Returns
    -------
    L : (batch, c, ...) torch.tensor
        Lifted assignment.
    """
    assert W.shape == V.shape
    return softmax(V + torch.log(W), dim=1)

def replicator(s, x):
    """
    Apply replicator to vector x at s.
    """
    sx = s*x
    s_sum = sx.sum(dim=1, keepdim=True)
    return sx - s_sum*s

def mean_free(x):
	"""
	Project vector to the tangent space T_0W.
	"""
	return x - x.mean(dim=1, keepdim=True)

def dense_replicator(s):
    """
    Construct dense replicator operator
    s (batch, c, n)
    This is not block diagonal because the channel dimension
    comes first in pytorch.
    """
    batch_size, c, n = s.shape
    D = torch.diag_embed(s.reshape((batch_size, c*n)))
    B = torch.zeros(batch_size, c, n, c, n, device=s.device)
    for i in range(n):
        B[:,:,i,:,i] += s[:,:,i].unsqueeze(1)
    B = s.reshape((batch_size, c*n, 1))*B.reshape((batch_size, c*n, c*n))
    return D - B

def tangent_basis(z):
    """
    Map (m-) coordinates z (batch_size, c-1, ...) to tangent space vector v (batch_size, c, ...).
    """
    v_shape = list(z.shape)
    v_shape[1] += 1
    v = torch.zeros(*v_shape, device=z.device)
    v[:,:-1,...] += z
    v[:,-1,...] -= z.sum(dim=1)
    return v

def tangent_basis_transposed(v):
    """
    """
    return v[:,:-1,...] - v[:,-1:,...]

def tangent_basis_operator(A):
    """
    Expand operator A ((c-1)*n, (c-1)*n) in tangent basis to (c*n, c*n).
    """
    expanded_shape = list(A.shape)
    expanded_shape[0] += 1
    expanded_shape[2] += 1
    A_expanded = torch.zeros(*expanded_shape, device=A.device)
    A_expanded[:-1,:,:-1,:] += A
    A_expanded[-1,:,:-1,:] -= A.sum(dim=0)
    A_expanded[:-1,:,-1,:] -= A.sum(dim=2)
    A_expanded[-1,:,-1,:] -= A.sum(dim=(0,2))
    return A_expanded

def dense_projection(s):
    """
    Construct dense projection operator
    s (batch, c, ...)
    The entries of s are not used, only its shape and device.
    Result is not block diagonal because the channel dimension
    comes first in pytorch.
    """
    batch_size, c = s.shape[:2]
    n = np.prod(s.shape[2:])
    Pi0 = dense_replicator(torch.ones_like(s)/math.sqrt(c))
    Pi0 += torch.diag_embed((1-1/math.sqrt(c))*torch.ones(batch_size, c*n, device=s.device))
    return Pi0

def project_operator(A, c, n):
    """
    Modify (cn) x (cn) operator A by tangent space projection from the left
    i.e. return Pi_0 @ A
    """
    A = A.reshape((c, n, c*n))
    return (A - A.mean(dim=0, keepdim=True)).reshape((c*n, c*n))