import numpy as np
import scipy.sparse as sp

def finite_difference_2D(Nx, Ny):
    """
    Constructs the 2D finite difference matrix for a grid of size Nx x Ny
    using the standard 5-point stencil.
    """
    def finite_difference_1D(N):
        diagonals = [
            np.full(N, -2),    # Main diagonal
            np.full(N-1, 1),   # Upper diagonal
            np.full(N-1, 1)    # Lower diagonal
        ]
        return sp.diags(diagonals, offsets=[0, -1, 1], format='csr')

    Dx = finite_difference_1D(Nx)
    Dy = finite_difference_1D(Ny)

    I_Nx = sp.identity(Nx)
    I_Ny = sp.identity(Ny)

    L2D = sp.kron(I_Ny, Dx) + sp.kron(Dy, I_Nx)

    return L2D

# Example usage
Nx, Ny = 5, 5
L2D = finite_difference_2D(Nx, Ny)
print(L2D.toarray().shape)