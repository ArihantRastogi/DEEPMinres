import numpy as np
import scipy.sparse as sparse
import time

class MinimumResidual:
    """
    Minimum Residual method (MINRES) for solving sparse linear systems.
    This method is especially effective for symmetric indefinite matrices
    but can also be used for general cases.
    """
    
    def __init__(self, A_sparse):
        """
        Initialize the MINRES solver with a sparse matrix.
        
        Parameters:
        -----------
        A_sparse : scipy.sparse matrix
            The coefficient matrix of the linear system
        """
        if A_sparse.shape[0] != A_sparse.shape[1]:
            print("A is not a square matrix!")
        self.n = A_sparse.shape[0]
        self.machine_tol = np.finfo(np.float64).eps
        # Use double precision for better accuracy
        self.A_sparse = A_sparse.copy().astype(np.float64)
    
    def multiply_A(self, x):
        """Multiply the matrix A by vector x"""
        return self.A_sparse.dot(x)
    
    def norm(self, x):
        """Compute the 2-norm of vector x"""
        return np.linalg.norm(x)
    
    def dot(self, x, y):
        """Compute the dot product of vectors x and y"""
        return np.dot(x, y)
    
    def apply_precond(self, r, M=None):
        """
        Apply a preconditioner to the residual vector.
        
        Parameters:
        -----------
        r : numpy array
            Residual vector
        M : callable, optional
            Preconditioner function that takes a vector and returns the preconditioned vector
            
        Returns:
        --------
        z : numpy array
            Preconditioned vector
        """
        if M is None:
            return r.copy()
        else:
            return M(r)
    
    def solve(self, b, x_init=None, M=None, max_it=1000, tol=1.0e-12, 
              atol=1.0e-15, stagnation_check=True, restart_every=None, verbose=True):
        """
        Solve the linear system Ax = b using the MINRES method.
        
        Parameters:
        -----------
        b : numpy array
            Right-hand side vector
        x_init : numpy array, optional
            Initial guess for the solution (default: zero vector)
        M : callable, optional
            Preconditioner function
        max_it : int, optional
            Maximum number of iterations
        tol : float, optional
            Relative convergence tolerance
        atol : float, optional
            Absolute convergence tolerance
        stagnation_check : bool, optional
            Whether to check for stagnation in convergence
        restart_every : int, optional
            Restart the algorithm every N iterations (None means no restart)
        verbose : bool, optional
            Whether to print progress information
            
        Returns:
        --------
        x : numpy array
            Solution vector
        res_arr : list
            Residual history
        """
        # Set up initial vectors and values
        if x_init is None:
            x = np.zeros(self.n, dtype=np.float64)
        else:
            x = x_init.copy().astype(np.float64)
        
        b = b.astype(np.float64)  # Ensure right-hand side is double precision
        b_norm = self.norm(b)
        stopping_tol = max(tol * b_norm, atol)  # Combined stopping criterion
        
        # Initial residual
        r = b - self.multiply_A(x)
        r_norm = self.norm(r)
        res_arr = [r_norm]
        
        if verbose:
            print(f"Initial MINRES residual = {r_norm}, target = {stopping_tol}")
        
        if r_norm < stopping_tol:
            if verbose:
                print(f"MINRES converged in 0 iterations. Final residual is {r_norm}")
            return x, res_arr
        
        # Track stagnation
        stagnation_counter = 0
        best_res = r_norm
        best_x = x.copy()
        
        # Main iteration loop with restart capability
        total_iter = 0
        while total_iter < max_it:
            # Determine number of iterations for this round
            if restart_every is not None:
                local_max_it = min(restart_every, max_it - total_iter)
            else:
                local_max_it = max_it - total_iter
                
            # Setup for Lanczos process
            r = b - self.multiply_A(x)
            beta1 = self.norm(r)
            
            if beta1 < stopping_tol:
                if verbose:
                    print(f"MINRES converged in {total_iter} iterations. Final residual is {beta1}")
                return x, res_arr
                
            # Initialize Lanczos vectors    
            v_old = np.zeros(self.n)
            v = r / beta1
            
            # Initialize recurrence QR values
            c = -1.0  # First Givens rotation
            s = 0.0
            
            # Initialize direction vectors
            w = np.zeros(self.n)
            w_old = np.zeros(self.n)
            
            # Initialize remaining values
            beta = beta1
            eta = beta  # First component of the right-hand side
            
            for i in range(local_max_it):
                total_iter += 1
                
                # Lanczos process: Generate a new basis vector
                z = self.apply_precond(v, M)  # Apply preconditioner
                p = self.multiply_A(z)
                
                # Orthogonalize p against previous Lanczos vectors
                alpha = self.dot(v, p)
                p = p - alpha * v - beta * v_old
                
                # Reorthogonalization to maintain orthogonality
                for j in range(min(3, i)):  # Limited reorthogonalization for efficiency
                    p = p - self.dot(v, p) * v
                
                # Get next Lanczos vector
                beta_old = beta
                beta = self.norm(p)
                if beta < self.machine_tol:
                    # Breakdown in Lanczos process, indicates exact solution found
                    beta = 0.0
                else:
                    v_old = v.copy()
                    v = p / beta
                
                # Apply previous Givens rotations to the new column of the tridiagonal matrix
                if i > 0:
                    h1 = c * alpha - s * beta_old
                    h2 = s * alpha + c * beta_old
                    alpha = h1
                
                # Construct and apply new Givens rotation
                gamma = np.sqrt(alpha**2 + beta**2)
                if gamma < self.machine_tol:
                    # Avoid division by zero
                    gamma = self.machine_tol
                    
                c_new = alpha / gamma
                s_new = beta / gamma
                
                # Update solution
                w_next = (z - beta_old * w_old - alpha * w) / gamma
                x = x + c_new * eta * w_next
                
                # Prepare for next iteration
                w_old = w.copy()
                w = w_next.copy()
                
                # Update rotation and residual estimate
                eta = -s_new * eta  # Next component of the right-hand side
                c = c_new
                s = s_new
                
                # Compute residual for convergence check (without explicitly forming r = b - Ax)
                r_norm = abs(eta)
                res_arr.append(r_norm)
                
                # Output progress
                if verbose and (total_iter % 10 == 0 or total_iter == 1):
                    print(f"Iteration {total_iter}, residual = {r_norm:.6e}")
                
                # Check for stagnation
                if stagnation_check:
                    if r_norm < best_res * 0.99:  # Meaningful improvement
                        best_res = r_norm
                        best_x = x.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                        if stagnation_counter > 20:  # 20 iterations with minimal progress
                            if verbose:
                                print(f"Stagnation detected at iteration {total_iter}.")
                            x = best_x.copy()
                            break
                
                # Check convergence
                if r_norm < stopping_tol:
                    if verbose:
                        print(f"MINRES converged in {total_iter} iterations to residual {r_norm:.6e}")
                    return x, res_arr
                
                # Check for Lanczos breakdown (exact solution)
                if beta == 0.0:
                    if verbose:
                        print(f"MINRES found exact solution at iteration {total_iter}")
                    return x, res_arr
            
            # After each restart, compute true residual to avoid accumulation of errors
            if restart_every is not None:
                r = b - self.multiply_A(x)
                r_norm = self.norm(r)
                if verbose:
                    print(f"Restart at iteration {total_iter}, true residual = {r_norm:.6e}")
                
                if r_norm < stopping_tol:
                    if verbose:
                        print(f"MINRES converged after restart to residual {r_norm:.6e}")
                    return x, res_arr
        
        # If we get here, we've reached maximum iterations
        r = b - self.multiply_A(x)  # Compute final true residual
        r_norm = self.norm(r)
        if verbose:
            print(f"MINRES reached max iterations ({max_it}). Final residual is {r_norm:.6e}")
        
        # Return the best solution found if using stagnation check
        if stagnation_check and best_res < r_norm:
            if verbose:
                print(f"Returning best solution with residual {best_res:.6e}")
            return best_x, res_arr
        
        return x, res_arr
    
    def check_error(self, x, b, exact_solution=None):
        """
        Check the error of the computed solution.
        
        Parameters:
        -----------
        x : numpy array
            Computed solution
        b : numpy array
            Right-hand side vector
        exact_solution : numpy array, optional
            Exact solution (if available)
            
        Returns:
        --------
        dict : Dictionary with error metrics
        """
        # Compute residual (b - Ax)
        residual = b - self.multiply_A(x)
        residual_norm = self.norm(residual)
        relative_residual = residual_norm / self.norm(b) if self.norm(b) > self.machine_tol else residual_norm
        
        results = {
            "residual_norm": residual_norm,
            "relative_residual": relative_residual
        }
        
        # If exact solution is available, compute error
        if exact_solution is not None:
            error = x - exact_solution
            error_norm = self.norm(error)
            relative_error = error_norm / self.norm(exact_solution) if self.norm(exact_solution) > self.machine_tol else error_norm
            
            results.update({
                "error_norm": error_norm,
                "relative_error": relative_error
            })
        
        return results

# Example usage
if __name__ == "__main__":
    # Create a test case
    n = 100
    A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).tocsr()
    
    # Create a random exact solution
    x_exact = np.random.rand(n)
    
    # Compute right-hand side
    b = A.dot(x_exact)
    
    # Create the MINRES solver
    solver = MinimumResidual(A)
    
    # Initial guess
    x_init = np.zeros(n)
    
    # Simple diagonal preconditioner
    def diagonal_preconditioner(r):
        diag = A.diagonal()
        return r / np.maximum(np.abs(diag), 1e-10)
    
    # Solve the system
    print("Solving with MINRES...")
    x_minres, residuals = solver.solve(b, x_init, M=diagonal_preconditioner, 
                                        restart_every=100, stagnation_check=True, verbose=True)
    
    # Check error
    print("\nError Analysis:")
    error_results = solver.check_error(x_minres, b, x_exact)
    for metric, value in error_results.items():
        print(f"{metric}: {value:.6e}")
    
    # Compare with direct solver
    print("\nComparing with direct solver...")
    x_direct = sparse.linalg.spsolve(A, b)
    direct_error = np.linalg.norm(x_direct - x_exact) / np.linalg.norm(x_exact)
    print(f"Direct solver relative error: {direct_error:.6e}")
    
    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.semilogy(residuals, 'b-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Residual (log scale)')
        plt.title('MINRES Convergence')
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot")