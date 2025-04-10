import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time

# Import our modules
from lib.minres import MinimumResidual
from lib.generator import finite_difference_2D

def test_minres_with_2d_laplacian():
    """
    Test the MINRES solver with a 2D finite difference Laplacian matrix
    """
    print("Testing MINRES solver with 2D Laplacian matrix...")
    
    # Create a 2D finite difference matrix
    print("Generating 2D finite difference matrix...")
    N = 50  # Grid size (N x N)
    A = finite_difference_2D(N, N)
    
    # Create a random exact solution
    print(f"Problem size: {A.shape[0]} x {A.shape[1]}")
    x_exact = np.random.rand(A.shape[0])
    
    # Compute right-hand side
    b = A.dot(x_exact)
    
    # Create the MINRES solver
    solver = MinimumResidual(A)
    
    # Initial guess (zero vector)
    x_init = np.zeros(A.shape[0])
    
    # Solve using MINRES
    print("\nSolving with MINRES...")
    start_time = time.time()
    x_minres, residuals = solver.solve(b, x_init, max_it=500, tol=1.0e-8, verbose=True)
    solve_time = time.time() - start_time
    print(f"MINRES took {solve_time:.4f} seconds")
    
    # Check error
    print("\nError Analysis:")
    error_results = solver.check_error(x_minres, b, x_exact)
    for metric, value in error_results.items():
        print(f"{metric}: {value:.6e}")
    
    # Compare with direct solver
    print("\nComparing with direct solver...")
    start_time = time.time()
    x_direct = sparse.linalg.spsolve(A, b)
    direct_time = time.time() - start_time
    print(f"Direct solver took {direct_time:.4f} seconds")
    
    direct_error = np.linalg.norm(x_direct - x_exact) / np.linalg.norm(x_exact)
    print(f"Direct solver relative error: {direct_error:.6e}")
    
    # Compare MINRES with direct solver
    minres_direct_diff = np.linalg.norm(x_minres - x_direct) / np.linalg.norm(x_direct)
    print(f"Difference between MINRES and direct solver: {minres_direct_diff:.6e}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(residuals, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Residual (log scale)')
    plt.title('MINRES Convergence for 2D Laplacian')
    plt.tight_layout()
    plt.savefig('minres_convergence.png')
    print("Convergence plot saved as 'minres_convergence.png'")
    plt.show()
    
    return x_minres, residuals

def test_minres_conditioning():
    """
    Test how MINRES performs with different condition numbers
    """
    print("\nTesting MINRES with different condition numbers...")
    
    # Grid sizes to test (resulting in different condition numbers)
    grid_sizes = [10, 20, 40, 80]
    iterations = []
    condition_numbers = []
    
    for N in grid_sizes:
        print(f"\nTesting with {N}x{N} grid")
        A = finite_difference_2D(N, N)
        
        # Estimate condition number (expensive for large matrices)
        if N <= 40:  # Only compute for smaller matrices
            eigenvalues = sparse.linalg.eigsh(A, k=2, which='BE', return_eigenvectors=False)
            cond = abs(eigenvalues[0] / eigenvalues[1])
            print(f"Estimated condition number: {cond:.2e}")
            condition_numbers.append(cond)
        else:
            condition_numbers.append(None)
            print("Condition number estimation skipped (too large)")
        
        # Create a random exact solution
        x_exact = np.random.rand(A.shape[0])
        
        # Compute right-hand side
        b = A.dot(x_exact)
        
        # Create the MINRES solver
        solver = MinimumResidual(A)
        
        # Initial guess
        x_init = np.zeros(A.shape[0])
        
        # Solve using MINRES
        x_minres, residuals = solver.solve(b, x_init, max_it=500, tol=1.0e-8, verbose=False)
        iterations.append(len(residuals) - 1)
        
        print(f"MINRES iterations required: {iterations[-1]}")
        print(f"Final residual: {residuals[-1]:.6e}")
        
        # Check error
        error_results = solver.check_error(x_minres, b, x_exact)
        print(f"Relative error: {error_results['relative_error']:.6e}")
    
    # Plot iterations vs. grid size
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, iterations, 'bo-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Iterations Required')
    plt.title('MINRES Iterations vs. Grid Size')
    plt.tight_layout()
    plt.savefig('minres_iterations_vs_grid.png')
    print("Grid size plot saved as 'minres_iterations_vs_grid.png'")
    plt.show()
    
    return grid_sizes, iterations, condition_numbers

if __name__ == "__main__":
    # Test MINRES with 2D Laplacian
    x_minres, residuals = test_minres_with_2d_laplacian()
    
    # Test MINRES with different condition numbers
    grid_sizes, iterations, condition_numbers = test_minres_conditioning()
    
    print("\nAll tests completed!")
