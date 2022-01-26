"""
Diagonal Matrix with rank-1 updates.
"""
import itertools
import torch
from torch.functional import Tensor

class DiagRankUpdate(object):
    """Diagonal Matrix with rank-1 updates"""

    def __init__(self, diag, rankUpdates):
        super(DiagRankUpdate, self).__init__()
        self.diag = diag
        self.rankUpdates = rankUpdates

        assert rankUpdates.ndim == 3
        assert rankUpdates.shape[1] == 2
        assert rankUpdates.shape[2] == diag.shape[0]
        assert rankUpdates.device == diag.device
        assert rankUpdates.dtype == diag.dtype

    @property 
    def dtype(self):
        return self.diag.dtype

    @property 
    def ndim(self):
        return 2

    def __repr__(self) -> str:
       return "{0}×{0} DiagonalMatrix with {1} Rank-1 Update".format(
           self.diag.size()[0],
           len(self.rankUpdates)
       ) + ("s" if len(self.rankUpdates)!=1 else "")

    def tensor(self):
        return torch.diag(self.diag) + torch.matmul(self.rankUpdates[:,0,:].t(), self.rankUpdates[:,1,:])

    def device(self):
        return self.diag.device

    def dim(self):
        return 2

    def size(self):
        return torch.Size([
            self.diag.size()[0],
            self.diag.size()[0]
        ])

    def t(self):
        return DiagRankUpdate(self.diag.clone(), torch.flip(self.rankUpdates, (1,)))

    def add(self, other):
        if type(other) != DiagRankUpdate:
            return torch.add(self.tensor(), other)

        return DiagRankUpdate(
            self.diag + other.diag,
            torch.cat((self.rankUpdates, other.rankUpdates))
        )

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return other.add(self)

    def negative(self):
        return DiagRankUpdate(
            -self.diag,
            self.rankUpdates * torch.tensor([[-1, 1]])
        )

    def __sub__(self, other):
        return self.add(other.negative())

    def __rsub__(self, other):
        return other.add(self.negative())

    def matmul(self, other):
        if type(other) != DiagRankUpdate:
            return torch.mul(self.tensor(), other)

        return DiagRankUpdate(
            self.diag * other.diag,

            torch.cat((
                torch.cat(
                    (
                        self.diag[None, None, :] * other.rankUpdates[:, (0,), :],
                        other.rankUpdates[:, (1,), :]
                    ),
                    dim = 1
                ),
                torch.cat(
                    (
                        self.rankUpdates[:, (0,), :],
                        other.diag[None, None, :] * self.rankUpdates[:, (1,), :]
                    ),
                    dim=1
                ),
                torch.stack([
                    torch.stack((s[1].dot(o[0]) * s[0], o[1])) for s, o in itertools.product(
                        self.rankUpdates,
                        other.rankUpdates
                    )]
                )
            ))
        )

    def batchDot(self, v):
        """
        Batched multiplication self @ v
        with batch of matrices v (batch_size, n, k)
        """
        assert v.ndim == 3
        assert v.shape[1] == self.rankUpdates.shape[2]
        n = v.shape[1]

        diag_bmm = self.diag.reshape((1, n, 1))*v
        inner_prod = torch.matmul(self.rankUpdates[:,1,:].unsqueeze(0), v)
        # inner_prod now has shape (batch_size, n_updates, k)
        outer_prod = torch.matmul(
            self.rankUpdates[:,0,:].t().unsqueeze(0),
            inner_prod
        )
        # outer_prod now has shape (batch_size, n, k)
        return diag_bmm + outer_prod

    def batchDotTransposed(self, v):
        """
        Batched multiplication self.t() @ v
        with batch of matrices v (batch_size, n, k)
        """
        assert v.ndim == 3
        assert v.shape[1] == self.rankUpdates.shape[2]
        n = v.shape[1]

        diag_bmm = self.diag.reshape((1, n, 1))*v
        inner_prod = torch.matmul(self.rankUpdates[:,0,:].unsqueeze(0), v)
        # inner_prod now has shape (batch_size, n_updates, k)
        outer_prod = torch.matmul(
            self.rankUpdates[:,1,:].t().unsqueeze(0),
            inner_prod
        )
        # outer_prod now has shape (batch_size, n, k)
        return diag_bmm + outer_prod

    def dotRight(self, other):
        """
        Multiply self @ other
        """
        return self.diag * other + torch.matmul(
            torch.matmul( self.rankUpdates[:,1,:] , other ),
            self.rankUpdates[:,0,:]
        )

    def dotLeft(self, other):
        """
        Multiply other @ self
        """
        return self.diag * other + torch.matmul(
            torch.matmul( self.rankUpdates[:,0,:] , other ),
            self.rankUpdates[:,1,:]
        )

    def dotBoth(self, v, w):
        """
        Let A be self and v, w ∈ ℝⁿ. Then `dotBoth` computes
            vᵀ A w
        """
        return (self.diag * v * w).sum() + torch.dot(
            torch.matmul(self.rankUpdates[:, 0, :], v),
            torch.matmul(self.rankUpdates[:, 1, :], w)
        )    

    def trace(self):
        return self.diag.sum() + sum([torch.dot(r[0], r[1]) for r in self.rankUpdates])

    def appendUpdate(self, other):
        return DiagRankUpdate(
            self.diag.clone(),
            torch.cat((self.rankUpdates, other[None, :, :]))
        )

    def inverse(self):
        if self.rankUpdates.shape[0] == 0:
            return DiagRankUpdate(
                1 / self.diag,
                torch.empty((0,2,self.size()[0]), device=self.device())
            )
        
        else:
            inv = DiagRankUpdate(self.diag, self.rankUpdates[0:-1, :, :]).inverse()
            v = self.rankUpdates[-1,0,:]
            w = self.rankUpdates[-1,1,:]

            return inv.appendUpdate(
                    torch.stack((
                        inv.dotRight(v).negative(), 
                        inv.dotLeft(w)  / (
                            1 + inv.dotBoth(w, v)
                        )
                    ))
                )

    def det(self):
        if self.rankUpdates.shape[0] == 0:
            return self.diag.prod()
        else:
            reduced = DiagRankUpdate(
                self.diag,
                self.rankUpdates[0:-1, :, :]
            )
            v = self.rankUpdates[-1, 0, :]
            w = self.rankUpdates[-1, 1, :]

            return (1 + reduced.inverse().dotBoth(w, v)) * reduced.det()

    def log_det(self):
        if self.rankUpdates.shape[0] == 0:
            return self.diag.log().sum()
        else:
            reduced = DiagRankUpdate(
                self.diag,
                self.rankUpdates[0:-1, :, :]
            )
            v = self.rankUpdates[-1, 0, :]
            w = self.rankUpdates[-1, 1, :]

            return torch.log(1 + reduced.inverse().dotBoth(w, v)) + reduced.log_det()

    def kl_divergence(self, other, mu0=None, mu1=None):
        inv = other.inverse()
        if not mu0 is None:
            mu1mu0 = mu1 - mu0
            return (
                inv.matmul(self).trace()
                +
                inv.dotBoth(mu1mu0, mu1mu0)
                -
                self.size()[0]
                +
                other.log_det() - self.log_det()
            ) / 2
        
        kl = (
            inv.matmul(self).trace()
            -
            self.size()[0]
            +
            other.log_det() - self.log_det()
        ) / 2
        if kl < 0:
            print("Warning, KL was < 0.", kl)
        return kl


    def projectionBoth(self):
        n = self.size()[0]
        ones = -torch.ones(n) / n

        a = self.rankUpdates[:,0,:]
        b = self.rankUpdates[:,1,:]

        a_sum = a.sum(dim=1)
        b_sum = b.sum(dim=1)

        return self.appendUpdate(
            torch.stack((
                ones,
                self.diag + a_sum @ b
            ))
        ).appendUpdate(
            torch.stack((
                self.diag + b_sum @ a,
                ones
            ))
        ).appendUpdate(
            torch.stack((
                (self.diag.sum() + (a_sum * b_sum).sum()) * ones,
                ones,
            ))
        )