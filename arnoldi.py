import torch

def arnoldi_iteration(A, b, krylov_dim=5):
    """Computes a basis of the (krylov_dim + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^krylov_dim b}.

    Arguments
      A: linear operator
      b: initial vector
      krylov_dim: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: (krylov_dim + 1) x b.shape the last dimensions columns are an orthonormal 
        basis of the Krylov subspace. 
    """
    batch_size = b.shape[0]
    img_shape = b.shape[1:]
    h = torch.zeros((batch_size, krylov_dim+1, krylov_dim), device=b.device)
    Q = torch.zeros((batch_size, krylov_dim+1, *img_shape), device=b.device)
    q = b / torch.linalg.vector_norm(b, dim=(1,2,3), keepdim=True)
    Q[:,0,...] += q

    Q = Q.reshape((batch_size, krylov_dim+1, -1))
    for i in range(krylov_dim):
        v = A(q).reshape((batch_size, -1))

        h[:,:i+1,i] += torch.matmul(Q[:,:i+1,:].clone(), v.unsqueeze(-1).clone()).squeeze(-1)
        v -= torch.matmul(h[:,:i+1,i].clone().unsqueeze(1), Q[:,:i+1,:].clone()).squeeze(1)
        h[:,i+1,i] += torch.linalg.vector_norm(v, dim=1)

        if torch.all(h[:,i+1,i] > 1e-10):
            q = v / h[:,i+1,i].clone().reshape((-1,1))
            Q[:,i+1,:] += q
            q = q.reshape((batch_size, *img_shape))
        else:
            Q = Q.reshape((batch_size, krylov_dim+1, *img_shape))
            return Q[:,:-1,...].transpose(0, 1), h[:,:-1,:]
        v = v.reshape((batch_size, *img_shape))

    Q = Q.reshape((batch_size, krylov_dim+1, *img_shape))
    return Q[:,:-1,...].transpose(0, 1), h[:,:-1,:]