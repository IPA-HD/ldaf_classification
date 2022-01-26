import torch
import torch.nn as nn
from components import *
from diag_rank_update import DiagRankUpdate

class UpdatedNormal(nn.Module):
    """
    Multivariate normal distribution on tangent space coordinates with covariance matrix
    built by Sigma = B @ B.t
    where B is a diagonal matrix modified by rank-1 updates.
    The corresponding covariance in ambient coordinates is P @ B @ B.t @ P.t
    """
    def __init__(self, n, c, n_updates=1):
        """
        Mean free normal distribution with covariance matrix
        P @ B @ B.t @ P.t
        """
        super(UpdatedNormal, self).__init__()
        self.shape = (c-1, n)
        self.param_dim = (c-1)*n
        self.n_updates = n_updates
        #self.diag = nn.Parameter(0.1*torch.ones(self.param_dim))
        self.diag = nn.Parameter(0.1*(torch.ones(self.param_dim)+0.1*torch.randn(self.param_dim)))
        updates = 0.1*(torch.ones(n_updates, 2, self.param_dim)+0.1*torch.randn(n_updates, 2, self.param_dim))
        self.updates = nn.Parameter(updates)
        #self.updates = nn.Parameter(torch.zeros(n_updates, 2, self.param_dim))
        self.B = DiagRankUpdate(self.diag, self.updates)

    def init_from(self, other):
        with torch.no_grad():
            self.diag.fill_(0.0)
            self.diag += other.diag
            self.updates.fill_(0.0)
            self.updates += other.updates[...]

    def draw(self, num_samples=1):
        """
        Samples with shape (num_samples, c, n, 1)
        """
        v = torch.randn(num_samples, self.param_dim, 1, device=self.B.device())
        return tangent_basis(self.B.batchDot(v).reshape((num_samples, *self.shape, 1)))

    def cov_action(self, U):
        """
        Action of covariance matrix on U (batch, c*n, k)

        Warning: the covariance does not have shape Pi_0 @ Sigma @ Pi_0 as 
        below and thus projections may be expected to be handled outside 
        of this function. However, P @ Pi_0 = P and thus this covariance action
        does not change if the input is projected beforehand.
        """
        batch_size = U.shape[0]
        k = U.shape[-1]
        c = self.shape[0]+1
        n = self.shape[1]
        U = tangent_basis_transposed(U.reshape(batch_size, c, n, k))
        X = self.B.batchDot(self.B.batchDotTransposed(U.reshape(batch_size, (c-1)*n, k)))
        return tangent_basis(X.reshape(batch_size, c-1, n, k)).reshape(batch_size, c*n, k)

    def cov_dense(self):
        """
        Dense covariance matrix in ambient coordinates.
        """
        c = self.shape[0]+1
        n = self.shape[1]
        #op = self.B.matmul(self.B.t()).tensor().reshape((c-1,n,c-1,n))
        #return tangent_basis_operator(op).reshape(n*c, n*c)
        op = tangent_basis(self.B.tensor().reshape((1,c-1,n*(c-1)*n))).reshape((c*n, (c-1)*n))
        return op @ op.t()

    def cov_coords(self):
        """
        Covariance matrix in m-coordinates.
        """
        return self.B.matmul(self.B.t())

    def kl_to(self, pi):
        """
        Relative entropy to distribution pi.
        """
        Sigma_1 = self.cov_coords()
        Sigma_0 = pi.cov_coords()
        return Sigma_1.kl_divergence(Sigma_0)

class MultivariateNormal(nn.Module):
    """
    Multivariate normal distribution on tangent space.
    """
    def __init__(self, n, c):
        """
        Mean free normal distribution with covariance matrix
        Pi_0 @ cov @ Pi_0
        where cov is constructed from a dense matrix B by
        cov = B @ B.t
        """
        super(MultivariateNormal, self).__init__()
        self.shape = (c, n)
        self.param_dim = c*n
        self.coord_param_dim = (c-1)*n
        self.B = nn.Parameter(torch.eye(c*n)+0.01*torch.randn(c*n, c*n))

    def draw(self, num_samples=1):
        """
        Samples with shape (num_samples, c, n, 1)
        """
        v = torch.randn(num_samples, self.param_dim, 1, device=self.B.device)
        return mean_free(torch.matmul(self.B.unsqueeze(0), v).reshape((num_samples, *self.shape, 1)))

    def cov_action(self, U):
        """
        Action of covariance matrix on U (batch, c*n, k)

        Warning: although we assume all normal distributions to be supported on the
        tangent space and thus covariances to have shape Pi_0 @ Sigma @ Pi_0, the
        projections are to be handled outside of this function, i.e. this function
        only computes the action of Sigma.
        """
        X = torch.matmul(self.B.t().unsqueeze(0), U)
        return torch.matmul(self.B.unsqueeze(0), X)

    def cov_dense(self):
        """
        Dense covariance matrix.
        This includes tangent space projection.
        """
        B0 = project_operator(self.B, *self.shape)
        return torch.matmul(B0, B0.t())

    def cov_coords(self):
        """
        Dense covariance matrix of m-coordinates.
        This includes tangent space projection.
        """
        Sigma = self.cov_dense().reshape((*self.shape, *self.shape))
        return Sigma[:-1,:,:-1,:].reshape((self.coord_param_dim, self.coord_param_dim))

    def kl_to(self, pi):
        """
        Relative entropy to distribution pi.
        """
        assert isinstance(pi, MultivariateNormal)
        Sigma_1 = pi.cov_coords()
        Sigma_0 = self.cov_coords()

        # pinv(Sigma_1) @ Sigma_0 = lstsq(Sigma_1, Sigma_0)
        # and the r.h.s. is cheaper to compute and more stable
        # but the backward pass is not implemented as of pytorch 1.10
        #trace = torch.trace(torch.linalg.lstsq(Sigma_1, Sigma_0).solution)
        trace = torch.trace(Sigma_1.pinverse() @ Sigma_0)

        log_det_prior = torch.logdet(Sigma_1)
        log_det_posterior = torch.logdet(Sigma_0)
        relative_entropy = 0.5*(trace - self.coord_param_dim + log_det_prior - log_det_posterior)
        return relative_entropy