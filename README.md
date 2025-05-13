# DeepMINRES: Neural Network Enhanced MINRES Algorithm

## Overview

DeepMINRES is a novel approach to accelerate the Minimum Residual (MINRES) iterative method for solving large symmetric linear systems. By leveraging machine learning techniques, specifically a neural network, DeepMINRES learns optimal search directions that can significantly improve convergence rates compared to the standard MINRES algorithm.

This implementation focuses on solving systems of the form **Ax = b**, where **A** is a symmetric positive definite matrix, by training a neural network to predict optimal coefficients for constructing search directions.

## Key Features

- **Neural Network Enhancement**: Uses deep learning to predict optimal search directions
- **Improved Convergence**: Achieves faster convergence rates compared to standard MINRES
- **Robustness**: Performs well across a wide range of condition numbers
- **Comprehensive Benchmarking**: Includes tools to compare performance against standard MINRES
- **Visualization Tools**: Provides detailed visualization of convergence behavior and performance metrics

## Dependencies

Python libraries required:

```python
numpy
tensorflow
matplotlib
sklearn (scikit-learn)
tqdm
pandas
seaborn
scipy
```

Install dependencies with:

```bash
pip install numpy tensorflow matplotlib tqdm scikit-learn pandas seaborn scipy
```

## Code Structure

The ipynb consists of two main cells:

1. **Training**: Core implementation of the DeepMINRES algorithm, including:
   - Data generation functions
   - Neural network model architecture
   - DeepMINRES and standard MINRES solvers
   - Basic benchmarking tools

2. **Analysis**: Comprehensive performance analysis framework, including:
   - Advanced benchmarking across different matrix properties
   - Statistical analysis of performance improvements
   - Visualization tools for performance metrics

## Algorithm Overview

### Basic Workflow

1. **Data Generation**: Create training data by solving linear systems with varying difficulty
2. **Neural Network Training**: Train a model to predict optimal search direction coefficients
3. **DeepMINRES Solver**: Use the trained model to accelerate MINRES iterations

### Key Components

- **Data Generation**: Generates matrices with controlled condition numbers and collects optimal coefficients
- **Neural Network**: Takes recent Lanczos vectors and residuals as input, outputs coefficients for search direction
- **DeepMINRES Algorithm**: Applies the neural network within the MINRES iteration framework

## Usage Guide

### Basic Example

```python
# Generate training data
X_train, y_train = generate_training_data(num_train_samples=2000, n=64)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Build and train the neural network
model = build_neural_network()
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Solve a linear system using DeepMINRES
A = generate_spd_matrix(64)
b = A @ np.random.randn(64)
x0 = np.zeros(64)
x, residuals = deepminres(A, b, x0, model)
```

### Running Benchmarks

```python
# Generate test matrices
test_matrices = [generate_spd_matrix(64) for _ in range(5)]
test_solutions = [np.random.randn(64) for _ in range(5)]

# Compare DeepMINRES against standard MINRES
deep_results = evaluate_solver(deepminres, test_matrices, test_solutions, "DeepMINRES")
std_results = evaluate_solver(standard_minres, test_matrices, test_solutions, "Standard MINRES")

# Run comprehensive benchmark
results_df, summary = run_experiment(num_matrices=1000, max_iterations=200)
```

## Detailed Performance Analysis

The code includes tools for:
- Comparing iteration counts across matrices with different condition numbers
- Analyzing the relationship between condition number and convergence improvement
- Visualizing convergence behavior for sample matrices
- Computing statistical summaries of performance metrics

## Technical Details

### Neural Network Architecture

The neural network consists of:
- Input layer: 6n features (3 Lanczos vectors and 3 residuals)
- Hidden layers: 512 → 256 → 128 neurons with ReLU activation
- Regularization: Dropout and BatchNormalization for improved generalization
- Output layer: 3 coefficients with tanh activation

### Search Direction Computation

DeepMINRES constructs search directions as:

p = α₁q_k + α₂q_{k-1} + α₃q_{k-2}

where:
- q_k, q_{k-1}, q_{k-2} are the three most recent Lanczos vectors
- α₁, α₂, α₃ are coefficients predicted by the neural network

## Experimental Results

DeepMINRES typically achieves:
- **Faster Convergence**: Requires significantly fewer iterations than standard MINRES
- **Better Scaling**: Greater improvements for ill-conditioned matrices
- **Consistent Performance**: Maintains accuracy while accelerating convergence


