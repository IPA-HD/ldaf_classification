import torch
import torch.nn as nn

class DenseOmega(nn.Module):
    """
    Dense (+symmetric) Omega matrix 
    which applies to vectorized state with shape (batch, c, n, 1).
    """
    def __init__(self, n, c):
        super(DenseOmega, self).__init__()
        self.n = n
        self.c = c
        # self.fc should have bias=False
        # but this does not have an effect due to the custom forward pass below
        # the bug was not fixed to maintain compatibility with older model checkpoints
        self.fc = nn.Linear(n*c, n*c)
    
    def forward(self, v):
        batch = v.shape[0]
        x = v.reshape((batch, self.c*self.n, 1))
        y1 = torch.matmul(self.fc.weight, x)
        y2 = torch.matmul(self.fc.weight.t(), x)
        return 0.5*(y1+y2).reshape((batch, self.c, self.n, 1))

    def dense_matrix(self):
        return 0.5*(self.fc.weight + self.fc.weight.t())