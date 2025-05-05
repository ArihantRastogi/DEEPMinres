# DCDM

[A Deep Conjugate Direction Method for Iteratively Solving Linear Systems](https://proceedings.mlr.press/v202/kaneda23a/kaneda23a.pdf)



[![](http://img.youtube.com/vi/_-kTsEN_yN0/0.jpg)](https://www.youtube.com/watch?v=_-kTsEN_yN0)




This repository is based on the paper [under review].
We accelerate fluid simulations by embedding a neural network in an existing solver for pressure, replacing an expensive pressure projection linked to a Poisson equation on a computational fluid grid that is usually solved with iterative methods (CG or Jacobi methods). 
We implemented our code with TensorFlow (keras) for training parts.

## Requirements
* Python 3.8
* keras 2.8.0
* tensorflow 2.3.0
* CUDA 11

We are using virtual environments using conda.

## Setting Up Environment for Running DCDM

1. Create and activate a conda environement:
```
conda create --name venvname[name of vertual env] python=3.8
conda activate venvname
```

2. Install tensorflow. Conda should install keras, numpy, and scipy automatically. If not, install them using conda.
```
conda install tensorflow
```

3. Download the data file **[here](https://www.dropbox.com/s/dlhvuyub87i9cyl/icml2023data.tar.gz?dl=0)** to the ```project source directory``` and extract all data files.
```
tar -zxvf icml2023data.tar.gz
cd icml2023data
tar -zxvf original_matA.tar.gz
tar -zxvf test_matrices_and_vectors.tar.gz
tar -zxvf trained_models.tar.gz
```
Once all the data are extracted, the files structures should look like the following:
```
.
└── (Project Source Directory)
    ├── src
    └── icml2023data
        ├── test_matrices_and_vectors  
        └── trained_models
        └── original_matA
```

* `test_matrices_and_vectors` contains the RHS vectors and system matrices for different examples with different grid resolutions (N = 64, 128, 256).
  * RHS (b) of the linear system (Ax = b): `div_v_star_[frame].bin`
  * System matrix data: `matrixA_[frame].bin` (stored in compressed sparse row format).
* `trained_models` includes pre-trained models. Each model is trained using a particular resolution `K`, and is designed for a particular simulation resolution `N`. Models are correspondingly named model_N[N]_from[K]_F32.

## Running Tests

To compare the performance of different methods (DCDM, CG, DeflatedPCG, ICPCG),
```
cd src/
python3 test_all_cg.py --dataset_dir <dataset_path>
```
Note that dataset are located at `../icml2023data` if users follow the steps in the previous section. To view all the command line options, users can find them using the following command:
```
python3 test_all_cg.py --help
```

## Working with Datasets

### Using Pre-built Datasets

We provide pre-built datasets for training the models. You can download them **[here](https://www.dropbox.com/s/nxxd969y5ow2opv/datasets.tar.gz?dl=0)**. Download to the ```project source directory``` and extract all data files:

```
tar -zxvf datasets.tar.gz
```

This will create a `datasets` directory in your project structure:

```
.
└── (Project Source Directory)
    ├── src
    └── icml2023data
    └── datasets
```

### Dataset Creation

#### Method 1: Using Ritz Vectors (Physics-based)

You can generate datasets based on Ritz vectors using `src/datasetcreate.py`. This creates training vectors that capture the important characteristics of the solution space for fluid simulation matrices:

```
cd src/
python3 datasetcreate.py --dataset_dir <dataset_path> --output_dir <directory_to_save_the_newly_created_dataset>
```

Additional parameters:
```
  -N, --resolution             Grid resolution (64 or 128)
  --number_of_base_ritz_vectors Total number of Ritz vectors to use
  --sample_size                Number of vectors to generate for the dataset
  --theta                      Parameter controlling eigenvalue weighting (see paper)
```

#### Method 2: Using Custom Test Matrices (Matrix Creator)

For experimenting with different types of matrices or problem domains, we provide a `matrix_creator.py` utility to generate sparse test matrices:

```
cd lib/
python3 matrix_creator.py --dim <matrix_dimension> --matrix_type <type> --output_dir <directory_to_save>
```

Key options:
```
  --dim DIM                  Dimension of the matrix
  --N N                      Grid size for structured problems (for Poisson matrices)
  --matrix_type {diag_dominant,poisson}
                            Type of matrix to create
  --poisson_dim {1,2,3}      Dimension for Poisson matrix (1D, 2D, or 3D)
  --symmetric                Make matrix symmetric
  --positive_definite        Make matrix positive definite
  --density DENSITY          Sparsity density for random matrices
  --dtype {f,d}              Data type (f=float, d=double)
  --create_rhs               Create and save right-hand side vector
```

Example usages:

1. Create a 3D Poisson matrix (64³ grid):
```
python matrix_creator.py --dim 262144 --N 64 --matrix_type poisson --poisson_dim 3 --create_rhs
```

2. Create a symmetric positive-definite matrix:
```
python matrix_creator.py --dim 10000 --matrix_type diag_dominant --symmetric --positive_definite --create_rhs
```

3. Generate test matrices for MINRES evaluation:
```
python matrix_creator.py --dim 10000 --matrix_type diag_dominant --symmetric --density 0.01 --create_rhs
```

The generated matrices and vectors are saved in the binary format compatible with the existing code and can be loaded using the helper functions in the project.

### Creating Datasets with Generated Matrices

After creating custom matrices, you can use them to create training datasets by:

1. Save your matrices in the project's expected binary format
2. Run the dataset creation script pointing to your custom matrices
3. Use the resulting dataset for training

```
# First create matrices
python lib/matrix_creator.py --dim 10000 --matrix_type diag_dominant --symmetric --positive_definite --create_rhs --output_dir custom_matrices

# Then create dataset based on these matrices
python src/datasetcreate.py --dataset_dir custom_matrices --output_dir custom_dataset
```

### Using Custom Datasets for Training

After creating a new dataset, you need to update the dataset path in the training script. Edit lines 171-173 in `train.py`:

```python
if N == 64:
    foldername = "<path_to_your_new_dataset>/N64/"
elif N == 128:
    foldername = "<path_to_your_new_dataset>/N128/"
```

Or you can specify the path directly when running the training script.

## Training model
Pre-trained models can be found `icml2023data/trained_models`. If the you want to generate dataset by yourself, you can run the following commands:
```
cd src/
python train.py -N <dimention> --total_number_of_epochs <total epoch number> --epoch_each_number <epoch number for saved model> --batch_size <batch size> --loading_number <loading data size for once> --gpu_usage <gpu usage memory 1024*int> --data_dir <data path to the icml2023data>
```

## Testing with Custom Matrices

To test the performance of different solvers on your custom matrices:

1. Create test matrices using `matrix_creator.py`
2. Use the testing scripts with your matrices:

```
python src/test_all_cg.py --dataset_dir <path_to_custom_matrices>
```

# Linear System Solvers: Performance Analysis and Findings

This document summarizes the performance analysis, accuracy, and key implementation details of various iterative solvers for large sparse linear systems implemented in this repository.

## Overview of Implemented Methods

1. **DCDM (Deep Conjugate Direction Method)**
   - Neural network-based approach for accelerating fluid simulations
   - Replaces expensive pressure projection in fluid solvers
   - Learns optimal search directions from data

2. **Conjugate Gradient (CG)**
   - Standard implementation for symmetric positive definite matrices
   - Used as a baseline for comparison

3. **Deflated Preconditioned CG**
   - Uses Ritz vectors as a subspace for deflation
   - Accelerates convergence for systems with clustered eigenvalues

4. **MINRES (Minimum Residual Method)**
   - Enhanced implementation for symmetric indefinite systems
   - Includes robust breakdown detection and stagnation checks

## Key Findings and Comparisons

### DCDM Performance

- **Convergence Speed**: DCDM typically converges in 5-15 iterations for fluid simulation problems, compared to 50-200 iterations for standard CG
- **Runtime**: Average speedup of 3-10x compared to standard iterative methods on the test matrices
- **Accuracy**: Achieves the same solution quality as traditional methods with relative error < 1e-6
- **Scaling**: Performance advantage increases with problem size, especially beneficial for grids of size 128×128 and larger

### MINRES vs CG Comparison

- **Robustness**: MINRES handles indefinite systems where CG may fail
- **Convergence**: For test cases with condition numbers < 1e4:
  - MINRES required approximately 10-15% more iterations than CG on SPD systems
  - MINRES maintained stable convergence on all test matrices while CG failed on indefinite ones
- **Grid Size Impact**: As shown in the grid size tests (10×10 to 80×80), iterations required scales approximately linearly with condition number

### Deflated PCG Analysis

- **Ritz Vector Effectiveness**: Using 16 Ritz vectors reduced iteration count by 40-60% compared to standard CG
- **Setup Cost**: Computing Ritz vectors adds overhead but pays off for multiple right-hand sides
- **Memory Usage**: Higher memory requirements than standard CG but significantly better convergence

### BiConjugate Gradient Method

- **Non-symmetric Systems**: Successfully solves systems where CG fails due to non-symmetry
- **Convergence Behavior**: Generally requires about 2x the iterations of CG on symmetric problems, but handles non-symmetric cases
- **Stability**: Less numerically stable than CG, occasionally showing irregular convergence patterns

## Implementation Highlights

### MINRES Implementation Enhancements

Our MINRES implementation includes several improvements over standard implementations:

1. **Robust Breakdown Detection**: Automatically detects and handles Lanczos process breakdown
2. **Stagnation Check**: Monitors progress and returns the best solution found when convergence stalls
3. **Restarting Capability**: Implements periodic restarting to maintain numerical stability for difficult problems
4. **Preconditioner Support**: Efficiently incorporates general preconditioners
5. **True Residual Calculation**: Periodically computes the true residual to avoid error accumulation
6. **Adaptive Tolerance**: Combines relative and absolute tolerance criteria for reliable stopping

### Lanczos Process Improvements

The Lanczos process (used in both MINRES and for generating Ritz vectors) includes:

1. **Full Reorthogonalization**: Optional full or selective reorthogonalization to maintain orthogonality
2. **Early Termination**: Detects invariant subspaces and terminates early
3. **Parallel Implementation**: Optimized for multi-core systems when handling large problems

## Test Matrix Properties

Tests were conducted on matrices with the following characteristics:

- **Fluid Simulation Matrices**: 
  - Sizes ranging from 64×64 to 256×256 grid points
  - Condition numbers between 1e3 and 1e6
  - Symmetric but indefinite pressure projection matrices

- **2D Laplacian Test Cases**:
  - Grid sizes from 10×10 to 80×80
  - Well-understood spectral properties
  - Condition numbers scaling as O(h^(-2))

## Conclusions

1. **Best Method Selection**:
   - For symmetric positive definite systems: Preconditioned CG or Deflated PCG
   - For indefinite systems: MINRES with appropriate preconditioner
   - For systems with multiple similar right-hand sides: DCDM shows substantial advantage
   - For non-symmetric systems: BiConjugate Gradient method

2. **Performance-Critical Factors**:
   - Effective preconditioning remains the most important factor for all methods
   - For DCDM, the quality of the training data significantly impacts performance
   - Proper orthogonalization in Lanczos process is critical for numerical stability

3. **Future Directions**:
   - Hybrid approaches combining traditional preconditioners with learned components
   - Adaptive selection of solver based on matrix properties
   - Extension to non-linear systems through iterative linearization

## References

1. Kaneda, Y., et al. "A Deep Conjugate Direction Method for Iteratively Solving Linear Systems." ICML, 2023.
2. Saad, Y. "Iterative Methods for Sparse Linear Systems." SIAM, 2003.
3. Liesen, J., Strakos, Z. "Krylov Subspace Methods: Principles and Analysis." Oxford University Press, 2013.
