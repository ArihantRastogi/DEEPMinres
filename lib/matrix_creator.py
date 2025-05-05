import numpy as np
import scipy.sparse as sparse
import struct
import os
import argparse
from pathlib import Path


def create_diag_dominant_matrix(dim, density=0.05, symmetric=True, positive_definite=True):
    """
    Creates a diagonally dominant sparse matrix.
    
    Parameters:
    -----------
    dim : int
        Size of the matrix (dim x dim)
    density : float
        Density of the sparse matrix (between 0 and 1)
    symmetric : bool
        Whether to create a symmetric matrix
    positive_definite : bool
        Whether to ensure the matrix is positive definite
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        The created sparse matrix
    """
    A = sparse.random(dim, dim, density=density, format='csr')
    
    if symmetric:
        A = A + A.T
    
    # Make diagonally dominant
    diag_values = np.abs(A).sum(axis=1).A1 * 1.1  # Ensure diagonal dominance
    
    if positive_definite and symmetric:
        # Add a small positive value to ensure positive definiteness
        diag_values += 0.1
    
    A.setdiag(diag_values)
    
    return A


def create_poisson_matrix(n, dim=1):
    """
    Creates a matrix for the Poisson equation using finite differences.
    
    Parameters:
    -----------
    n : int
        Number of grid points in each dimension
    dim : int
        Number of dimensions (1, 2, or 3)
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        The created sparse matrix
    """
    if dim == 1:
        # 1D Poisson
        size = n
        diagonals = [2 * np.ones(size), -1 * np.ones(size-1), -1 * np.ones(size-1)]
        return sparse.diags(diagonals, [0, -1, 1], format='csr')
    
    elif dim == 2:
        # 2D Poisson
        size = n * n
        main_diag = 4 * np.ones(size)
        off_diag1 = -1 * np.ones(size-1)
        off_diag2 = -1 * np.ones(size-n)
        
        # Adjust boundary connections
        for i in range(n-1):
            off_diag1[(i+1)*n - 1] = 0  # No wrap-around at borders
        
        offsets = [0, -1, 1, -n, n]
        diagonals = [main_diag, off_diag1, off_diag1, off_diag2, off_diag2]
        return sparse.diags(diagonals, offsets, format='csr')
    
    elif dim == 3:
        # 3D Poisson
        size = n * n * n
        main_diag = 6 * np.ones(size)
        off_diag1 = -1 * np.ones(size-1)  # x-direction
        off_diag2 = -1 * np.ones(size-n)  # y-direction
        off_diag3 = -1 * np.ones(size-n*n)  # z-direction
        
        # Adjust boundary connections in x-direction
        for i in range(n*n-1):
            off_diag1[(i+1)*n - 1] = 0  # No wrap-around at x borders
        
        # Adjust boundary connections in y-direction
        for i in range(n-1):
            for j in range(n):
                off_diag2[(i+1)*n*n - j - 1] = 0  # No wrap-around at y borders
        
        offsets = [0, -1, 1, -n, n, -n*n, n*n]
        diagonals = [main_diag, off_diag1, off_diag1, off_diag2, off_diag2, off_diag3, off_diag3]
        return sparse.diags(diagonals, offsets, format='csr')
    
    else:
        raise ValueError("Dimension must be 1, 2, or 3")


def create_random_rhs(dim, normalize=True):
    """
    Creates a random right-hand side vector for testing.
    
    Parameters:
    -----------
    dim : int
        Size of the vector
    normalize : bool
        Whether to normalize the vector to unit length
        
    Returns:
    --------
    numpy.ndarray
        The created vector
    """
    b = np.random.randn(dim)
    if normalize:
        b = b / np.linalg.norm(b)
    return b


def save_matrix_to_bin(A, filename, dtype='f'):
    """
    Saves a sparse matrix to binary file in the format required by the project.
    
    Parameters:
    -----------
    A : scipy.sparse.csr_matrix
        The sparse matrix to save
    filename : str
        Output filename
    dtype : str
        Data type ('d' for double, 'f' for float)
    """
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    
    len_data = 8 if dtype == 'd' else 4
    
    with open(filename, 'wb') as f:
        # Write header info
        num_rows, num_cols = A.shape
        nnz = A.nnz
        outS = len(A.indptr) - 1  # Size of outer index pointer array
        innS = nnz  # Size of inner index array
        
        # Write dimensions and sizes
        f.write(struct.pack('i', num_rows))
        f.write(struct.pack('i', num_cols))
        f.write(struct.pack('i', nnz))
        f.write(struct.pack('i', outS))
        f.write(struct.pack('i', innS))
        
        # Write data values
        for val in A.data:
            f.write(struct.pack(dtype, val))
        
        # Write row pointers
        for idx in A.indptr[:-1]:  # Skip the last element
            f.write(struct.pack('i', idx))
        
        # Write column indices
        for idx in A.indices:
            f.write(struct.pack('i', idx))


def save_vector_to_bin(v, filename, dtype='d'):
    """
    Saves a vector to binary file in the format required by the project.
    
    Parameters:
    -----------
    v : numpy.ndarray
        The vector to save
    filename : str
        Output filename
    dtype : str
        Data type ('d' for double, 'f' for float)
    """
    # Add a dummy first element (size) as per project format
    size = len(v)
    v_with_size = np.insert(v, 0, size)
    
    # Save to file
    v_with_size.astype(dtype).tofile(filename)


def test_saved_matrix(matrix_filename, vector_filename, N, dtype='f'):
    """
    Tests if the saved matrix and vector can be properly loaded.
    
    Parameters:
    -----------
    matrix_filename : str
        Path to the saved matrix file
    vector_filename : str
        Path to the saved vector file
    N : int
        Dimension of the matrix
    dtype : str
        Data type ('d' for double, 'f' for float)
    """
    import helper_functions as hf
    
    # Load matrix
    print("Testing matrix load...")
    A_loaded = hf.readA_sparse(N, matrix_filename, dtype)
    print(f"Matrix loaded successfully. Shape: {A_loaded.shape}, nnz: {A_loaded.nnz}")
    
    # Load vector
    print("Testing vector load...")
    b_loaded = hf.get_vec(vector_filename, normalize=False, d_type=dtype)
    print(f"Vector loaded successfully. Shape: {b_loaded.shape}")


def main():
    parser = argparse.ArgumentParser(description='Create and save sparse matrices for DEEPMinres')
    parser.add_argument('--dim', type=int, required=True, help='Dimension of the matrix')
    parser.add_argument('--N', type=int, help='Grid size for Poisson matrices (if dim is 1, 2, or 3^3)', default=64)
    parser.add_argument('--matrix_type', type=str, choices=['diag_dominant', 'poisson'], 
                        default='diag_dominant', help='Type of matrix to create')
    parser.add_argument('--poisson_dim', type=int, choices=[1, 2, 3], default=3, 
                        help='Dimension for Poisson matrix')
    parser.add_argument('--symmetric', action='store_true', help='Make matrix symmetric')
    parser.add_argument('--positive_definite', action='store_true', help='Make matrix positive definite')
    parser.add_argument('--density', type=float, default=0.01, help='Density for random sparse matrix')
    parser.add_argument('--dtype', type=str, choices=['f', 'd'], default='f', 
                        help='Data type (f=float, d=double)')
    parser.add_argument('--output_dir', type=str, default='../generated_matrices', 
                        help='Directory to save output files')
    parser.add_argument('--create_rhs', action='store_true', help='Create and save right-hand side vector')
    parser.add_argument('--test_load', action='store_true', help='Test loading the saved files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate base filename
    base_name = f"{args.matrix_type}_dim{args.dim}"
    if args.matrix_type == 'poisson':
        base_name += f"_pdim{args.poisson_dim}"
    if args.symmetric:
        base_name += "_sym"
    if args.positive_definite:
        base_name += "_pd"
    
    # Create matrix
    if args.matrix_type == 'diag_dominant':
        A = create_diag_dominant_matrix(args.dim, args.density, args.symmetric, args.positive_definite)
    elif args.matrix_type == 'poisson':
        if args.poisson_dim == 3 and args.N**3 != args.dim:
            print(f"Warning: For 3D Poisson with N={args.N}, expected dim={args.N**3}, got dim={args.dim}")
            print(f"Using dim={args.N**3} instead")
            args.dim = args.N**3
        elif args.poisson_dim == 2 and args.N**2 != args.dim:
            print(f"Warning: For 2D Poisson with N={args.N}, expected dim={args.N**2}, got dim={args.dim}")
            print(f"Using dim={args.N**2} instead")
            args.dim = args.N**2
        A = create_poisson_matrix(args.N, args.poisson_dim)
    
    # Save matrix
    matrix_filename = os.path.join(args.output_dir, f"{base_name}.bin")
    save_matrix_to_bin(A, matrix_filename, args.dtype)
    print(f"Matrix saved to {matrix_filename}")
    
    # Create and save right-hand side vector if requested
    if args.create_rhs:
        b = create_random_rhs(args.dim)
        vector_filename = os.path.join(args.output_dir, f"{base_name}_rhs.bin")
        save_vector_to_bin(b, vector_filename, args.dtype)
        print(f"Vector saved to {vector_filename}")
    
    # Test loading if requested
    if args.test_load:
        try:
            # Add parent directory to sys.path to import helper_functions
            import sys
            sys.path.insert(1, str(Path(__file__).resolve().parent))
            
            if args.create_rhs:
                test_saved_matrix(matrix_filename, vector_filename, args.N, args.dtype)
            else:
                print("Test load requires --create_rhs option")
        except ImportError:
            print("Could not import helper_functions for testing. Is the module available?")


if __name__ == "__main__":
    main()
