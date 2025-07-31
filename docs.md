# rs-sci: Rust Scientific Computing Library

A comprehensive mathematical library for Rust, providing tools for algebraic structures, calculus, linear algebra, complex numbers, statistics, and number theory.

## Table of Contents

1. [Overview](#overview)
2. [Number Systems](#number-systems)
3. [Algebraic Structures](#algebraic-structures)
4. [Complex Numbers](#complex-numbers)
5. [Linear Algebra](#linear-algebra)
6. [Calculus](#calculus)
7. [Statistics](#statistics)

## Overview

RS-SCI is designed to provide mathematically rigorous implementations of fundamental mathematical concepts in Rust. The library emphasizes:

- **Type Safety**: Leveraging Rust's type system for mathematical correctness
- **Performance**: Efficient numerical algorithms
- **Completeness**: Comprehensive coverage of mathematical domains
- **Ergonomics**: Clean, intuitive APIs with helpful macros


## Number Systems

The library provides rigorous implementations of fundamental number systems:

### Natural Numbers (ℕ)

```rust
use rs_sci::numbers::Natural;

let n1 = Natural::new(42);
let n2 = Natural::from(17);
println!("{}", n1); // 42
```

### Integers (ℤ)

```rust
use rs_sci::numbers::Integer;

let z1 = Integer::new(-15);
let z2 = Integer::from(Natural::new(10));
```

### Rational Numbers (ℚ)

Exact fraction arithmetic with automatic simplification:

```rust
use rs_sci::numbers::Rational;

let q1 = Rational::new(3, 4);      // 3/4
let q2 = Rational::new(6, 8);      // Simplified to 3/4
let q3 = Rational::new(-15, 25);   // Simplified to -3/5

assert_eq!(q1, q2); // true - automatic simplification
```

**Mathematical Properties:**
- Automatic GCD-based simplification: $\frac{a}{b} = \frac{a/\gcd(a,b)}{b/\gcd(a,b)}$
- Sign normalization (denominator always positive)
- Exact arithmetic (no floating-point errors)

### Real Numbers (ℝ)

```rust
use rs_sci::numbers::Real;

let r1 = Real::new(3.14159);
let r2 = Real::from_rational(Rational::new(22, 7)); // π approximation
```

## Algebraic Structures

The algebra module provides abstract algebraic structures with verifiable axioms.

### Basic Structure

```rust
use rs_sci::algebra::AlgebraicStructure;

let a = AlgebraicStructure(5.0);
let b = AlgebraicStructure(3.0);
let c = AlgebraicStructure(2.0);

// Basic operations
let sum = a + b;        // 8.0
let product = a * b;    // 15.0
```

### Axiom Verification

The library allows verification of fundamental algebraic axioms:

#### Associativity
Verify $(a \circ b) \circ c = a \circ (b \circ c)$ for operations:

```rust
use rs_sci::algebra::{AlgebraicStructure, Associativity};

let a = AlgebraicStructure(2.0);
let b = AlgebraicStructure(3.0);
let c = AlgebraicStructure(4.0);

// Verify additive associativity
assert!(a.verify_associativity_add(&b, &c));

// Verify multiplicative associativity  
assert!(a.verify_associativity_mul(&b, &c));
```

#### Commutativity
Verify $a \circ b = b \circ a$:

```rust
use rs_sci::algebra::Commutativity;

assert!(a.verify_commutativity_add(&b));
assert!(a.verify_commutativity_mul(&b));
```

#### Identity Elements
Verify identity properties:

```rust
use rs_sci::algebra::Identity;

let a = AlgebraicStructure(5.0);
let additive_id = 0.0;
let multiplicative_id = 1.0;

assert!(a.verify_additive_identity(&additive_id));
assert!(a.verify_multiplicative_identity(&multiplicative_id));
```

#### Distributivity
Verify $a \cdot (b + c) = (a \cdot b) + (a \cdot c)$:

```rust
use rs_sci::algebra::Distributivity;

assert!(a.verify_distributivity(&b, &c));
```

### Macro-Generated Structures

Create custom algebraic structures with automatic implementations:

```rust
use rs_sci::{algstruct, impl_axioms};

// Define a Ring structure
algstruct!(Ring, +, *);
impl_axioms!(Ring, Associativity, Commutativity, Identity(0.0, 1.0), Distributivity);

// Define a Field structure  
algstruct!(Field, +, *);
impl_axioms!(Field, Associativity, Commutativity, Identity(0.0, 1.0), Distributivity);

let r1 = Ring::from(3.0);
let r2 = Ring::from(4.0);
let sum = r1 + r2; // Ring addition
```

## Complex Numbers

Comprehensive complex number implementation with full mathematical operations.

### Construction

```rust
use rs_sci::{complex, im};
use rs_sci::complex::Complex;

// Various construction methods
let z1 = Complex::new(3.0, 4.0);           // 3 + 4i
let z2 = complex!(1.0, -2.0);              // 1 - 2i  
let z3 = complex!(5.0);                    // 5 + 0i
let z4 = im!(3.0);                         // 0 + 3i

// From polar coordinates
let z5 = Complex::from_polar(2.0, std::f64::consts::PI/4.0); // 2∠π/4
```

### Basic Operations

```rust
let z1 = Complex::new(1.0, 2.0);
let z2 = Complex::new(3.0, -1.0);

let sum = z1 + z2;        // (4, 1)
let diff = z1 - z2;       // (-2, 3)  
let product = z1 * z2;    // (5, 5)
let quotient = z1 / z2;   // (0.1, 0.7)
```

**Mathematical Foundation:**
- Addition: $(a + bi) + (c + di) = (a + c) + (b + d)i$
- Multiplication: $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$
- Division: $\frac{a + bi}{c + di} = \frac{(a + bi)(c - di)}{c^2 + d^2}$

### Complex Analysis Functions

```rust
let z = Complex::new(1.0, 1.0);

// Modulus and argument
let modulus = z.modulus();        // |z| = √(1² + 1²) = √2
let argument = z.argument();      // arg(z) = arctan(1/1) = π/4

// Complex conjugate
let conj = z.conjugate();         // 1 - i

// Exponential and logarithmic functions
let exp_z = z.exp();              // e^(1+i) = e·(cos(1) + i·sin(1))
let ln_z = z.ln();                // ln|z| + i·arg(z)

// Power functions
let z_squared = z.pow(2.0);       // z²
let sqrt_z = z.sqrt();            // √z (principal branch)
```

### Trigonometric Functions

Complete set of complex trigonometric functions:

```rust
let z = Complex::new(0.5, 0.5);

// Circular functions
let sin_z = z.sin();    // sin(z) = sin(x)cosh(y) + i·cos(x)sinh(y)
let cos_z = z.cos();    // cos(z) = cos(x)cosh(y) - i·sin(x)sinh(y)  
let tan_z = z.tan();    // tan(z) = sin(z)/cos(z)

// Hyperbolic functions
let sinh_z = z.sinh();  // sinh(z) = sinh(x)cos(y) + i·cosh(x)sin(y)
let cosh_z = z.cosh();  // cosh(z) = cosh(x)cos(y) + i·sinh(x)sin(y)
let tanh_z = z.tanh();  // tanh(z) = sinh(z)/cosh(z)
```

### Constants

```rust
use rs_sci::complex::Complex;

let i = Complex::I;        // 0 + 1i
let one = Complex::ONE;    // 1 + 0i  
let zero = Complex::ZERO;  // 0 + 0i
```

## Linear Algebra

Comprehensive vector and matrix operations with numerical linear algebra algorithms.

### Vectors

#### Construction

```rust
use rs_sci::{vector, linear::Vector};

// Various construction methods
let v1 = vector!([1.0, 2.0, 3.0]);
let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
let zeros = Vector::zeros(5);        // [0, 0, 0, 0, 0]
let ones = Vector::ones(3);          // [1, 1, 1]
```

#### Vector Operations

```rust
let v1 = vector!([1.0, 2.0, 3.0]);
let v2 = vector!([4.0, 5.0, 6.0]);

// Basic operations
let sum = (v1.clone() + v2.clone()).unwrap();     // [5, 7, 9]
let diff = (v1.clone() - v2.clone()).unwrap();    // [-3, -3, -3]
let scaled = v1.clone() * 2.0;                    // [2, 4, 6]

// Vector products
let dot_product = v1.dot(&v2).unwrap();           // 1×4 + 2×5 + 3×6 = 32
let cross_product = v1.cross(&v2).unwrap();       // Only for 3D vectors

// Properties
let magnitude = v1.magnitude();                    // √(1² + 2² + 3²) = √14
let normalized = v1.normalize();                   // Unit vector
let dimension = v1.dimension();                    // 3
```

**Mathematical Definitions:**
- Dot product: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i$
- Cross product: $\mathbf{u} \times \mathbf{v} = (u_2v_3 - u_3v_2, u_3v_1 - u_1v_3, u_1v_2 - u_2v_1)$
- Magnitude: $\|\mathbf{v}\| = \sqrt{\sum_{i=1}^n v_i^2}$

### Matrices

#### Construction

```rust
use rs_sci::{matrix, linear::Matrix};

// Various construction methods
let m1 = matrix!([1.0, 2.0], [3.0, 4.0]);
let m2 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
let identity = Matrix::identity(3);
let zeros = Matrix::zeros(2, 3);
```

#### Matrix Operations

```rust
let a = matrix!([1.0, 2.0], [3.0, 4.0]);
let b = matrix!([5.0, 6.0], [7.0, 8.0]);

// Basic operations
let sum = (a.clone() + b.clone()).unwrap();           // Element-wise addition
let product = (a.clone() * b.clone()).unwrap();       // Matrix multiplication
let transpose = a.transpose();                        // A^T
let scaled = a.clone() * 2.0;                         // Scalar multiplication

// Matrix properties
let det = a.determinant().unwrap();                   // det(A) = ad - bc
let trace = a.trace().unwrap();                       // tr(A) = sum of diagonal
let frobenius_norm = a.frobenius_norm();              // ||A||_F

// Matrix-vector multiplication
let v = vector!([1.0, 2.0]);
let result = (a * &v).unwrap();
```

#### Advanced Matrix Operations

##### Matrix Decompositions

```rust
let a = matrix!([4.0, 2.0], [2.0, 3.0]);

// LU Decomposition: A = LU
let (l, u) = a.lu_decomposition().unwrap();

// QR Decomposition: A = QR  
let (q, r) = a.qr_decomposition().unwrap();

// Cholesky Decomposition: A = LL^T (for positive definite matrices)
let l_chol = a.cholesky().unwrap();

// Singular Value Decomposition: A = UΣV^T
let (u_svd, sigma, v_t) = a.svd(1000, 1e-10).unwrap();
```

##### Eigenvalue Problems

```rust
// Eigendecomposition: A = PDP^(-1)
let (eigenvectors, eigenvalues) = a.eigendecomposition(1000, 1e-10).unwrap();

// Power iteration for dominant eigenvalue
let (lambda, eigenvec) = a.power_iteration(1000, 1e-10).unwrap();
```

##### Matrix Properties

```rust
let symmetric = a.is_symmetric();              // Check if A = A^T
let pos_def = a.is_positive_definite();        // Check positive definiteness
let rank = a.rank().unwrap();                  // Matrix rank
```

**Mathematical Background:**

**LU Decomposition:** $A = LU$ where $L$ is lower triangular and $U$ is upper triangular.

**QR Decomposition:** $A = QR$ where $Q$ is orthogonal and $R$ is upper triangular.

**SVD:** $A = U\Sigma V^T$ where $U$ and $V$ are orthogonal matrices and $\Sigma$ is diagonal.

## Calculus

Numerical methods for differentiation, integration, and differential equations.

### Differentiation

#### Numerical Derivatives

```rust
use rs_sci::calculus::Calculus;

let f = |x: f64| x.powi(2);  // f(x) = x²

// First derivative using central difference
let derivative = Calculus::derivative(f, 2.0, 0.0001);  // f'(2) ≈ 4.0

// Second derivative
let second_deriv = Calculus::second_derivative(f, 2.0, 0.0001);  // f''(2) ≈ 2.0
```

**Mathematical Formula:**
- Central difference: $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$
- Second derivative: $f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$

#### Partial Derivatives

```rust
let f = |x: f64, y: f64| x*x + y*y;  // f(x,y) = x² + y²

// Partial derivatives
let df_dx = Calculus::partial_x(f, 1.0, 2.0, 0.0001);  // ∂f/∂x = 2x ≈ 2.0
let df_dy = Calculus::partial_y(f, 1.0, 2.0, 0.0001);  // ∂f/∂y = 2y ≈ 4.0
```

### Integration

#### Numerical Integration Methods

```rust
let f = |x: f64| x.powi(2);  // ∫₀¹ x² dx = 1/3

// Rectangle method
let integral_rect = Calculus::integrate_rectangle(f, 0.0, 1.0, 1000);

// Trapezoidal rule  
let integral_trap = Calculus::integrate_trapezoid(f, 0.0, 1.0, 1000);

// Simpson's rule (most accurate)
let integral_simp = Calculus::integrate_simpson(f, 0.0, 1.0, 1000);
```

**Mathematical Formulas:**
- **Rectangle:** $\int_a^b f(x)dx \approx \sum_{i=0}^{n-1} f(x_i) \Delta x$
- **Trapezoidal:** $\int_a^b f(x)dx \approx \frac{\Delta x}{2}\left[f(a) + 2\sum_{i=1}^{n-1}f(x_i) + f(b)\right]$
- **Simpson's:** $\int_a^b f(x)dx \approx \frac{\Delta x}{3}\left[f(a) + 4\sum_{i=1,3,5...}f(x_i) + 2\sum_{i=2,4,6...}f(x_i) + f(b)\right]$

### Differential Equations

#### Ordinary Differential Equations (ODEs)

```rust
// Solve dy/dx = f(x, y) with initial condition y(x₀) = y₀

let f = |x: f64, y: f64| x + y;  // dy/dx = x + y

// Euler's method
let solution_euler = Calculus::euler(f, 0.0, 1.0, 0.1, 10);

// 4th-order Runge-Kutta (more accurate)
let solution_rk4 = Calculus::runge_kutta4(f, 0.0, 1.0, 0.1, 10);
```

**Runge-Kutta 4th Order:**
$$\begin{align}
k_1 &= hf(x_n, y_n) \\
k_2 &= hf(x_n + h/2, y_n + k_1/2) \\
k_3 &= hf(x_n + h/2, y_n + k_2/2) \\
k_4 &= hf(x_n + h, y_n + k_3) \\
y_{n+1} &= y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{align}$$

### Vector Calculus

#### Gradient, Divergence, and Curl

```rust
let scalar_field = |x: f64, y: f64| x*x + y*y;
let vector_field = |x: f64, y: f64| (x*y, x + y);

// Gradient of scalar field: ∇f = (∂f/∂x, ∂f/∂y)
let grad = Calculus::gradient(scalar_field, 1.0, 1.0, 0.0001);  // (2, 2)

// Divergence of vector field: ∇·F = ∂P/∂x + ∂Q/∂y  
let div = Calculus::divergence(vector_field, 1.0, 1.0, 0.0001);

// Curl (z-component): ∇×F = ∂Q/∂x - ∂P/∂y
let curl_z = Calculus::curl_z(vector_field, 1.0, 1.0, 0.0001);
```

### Series and Approximations

#### Taylor Series

```rust
let f = |x: f64| x.exp();           // f(x) = eˣ
let df = vec![|x: f64| x.exp()];    // All derivatives of eˣ are eˣ

// Taylor series approximation around x = 0
let approximation = Calculus::taylor_series(f, &df, 0.0, 0.1, 5);
```

**Taylor Series:** $f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$

### Optimization

#### Gradient Descent

```rust
let f = |x: f64, y: f64| x*x + y*y;  // Minimize f(x,y) = x² + y²

let (min_x, min_y) = Calculus::gradient_descent(
    f, 
    1.0, 1.0,    // Initial point
    0.1,         // Learning rate
    1e-6,        // Tolerance
    1000         // Max iterations
);
```

## Statistics

Comprehensive statistical analysis and probability distributions.

### Descriptive Statistics

#### Central Tendency

```rust
use rs_sci::stats::Stats;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0];

// Measures of central tendency
let mean = Stats::mean(&data).unwrap();           // 3.33
let median = Stats::median(&data).unwrap();       // 3.5  
let mode = Stats::mode(&data).unwrap();           // [5.0]
```

#### Dispersion

```rust
// Measures of spread
let variance = Stats::variance(&data).unwrap();    // Sample variance
let std_dev = Stats::std_dev(&data).unwrap();     // Standard deviation  
let range = Stats::range(&data).unwrap();         // Max - Min
```

**Mathematical Definitions:**
- **Sample Variance:** $s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$
- **Standard Deviation:** $s = \sqrt{s^2}$

#### Quartiles and Percentiles

```rust
// Quartiles: Q1, Q2 (median), Q3
let (q1, q2, q3) = Stats::quartiles(&data).unwrap();

// Any percentile
let p90 = Stats::percentile(&data, 90.0).unwrap();  // 90th percentile
```

### Distribution Characteristics

```rust
// Shape of distribution
let skewness = Stats::skewness(&data).unwrap();    // Measure of asymmetry
let kurtosis = Stats::kurtosis(&data).unwrap();    // Measure of tail heaviness
```

**Formulas:**
- **Skewness:** $\gamma_1 = \frac{m_3}{s^3}$ where $m_3 = \frac{1}{n}\sum(x_i - \bar{x})^3$
- **Kurtosis (excess):** $\gamma_2 = \frac{m_4}{s^4} - 3$ where $m_4 = \frac{1}{n}\sum(x_i - \bar{x})^4$

### Bivariate Statistics

```rust
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

// Measures of association
let covariance = Stats::covariance(&x, &y).unwrap();
let correlation = Stats::correlation(&x, &y).unwrap();  // Pearson correlation

// Linear regression: y = mx + b
let (slope, intercept) = Stats::linear_regression(&x, &y).unwrap();
```

**Correlation:** $r = \frac{\text{cov}(X,Y)}{s_X s_Y}$

### Probability Distributions

#### Normal Distribution

```rust
// Probability density function
let pdf = Stats::normal_pdf(1.5, 0.0, 1.0);  // N(0,1) at x=1.5

// Binomial distribution
let pmf = Stats::binomial_pmf(3, 10, 0.3);    // P(X=3) where X~Bin(10,0.3)
```

### Advanced Statistics

#### Standardization

```rust
// Convert to z-scores
let z_scores = Stats::z_scores(&data).unwrap();
```

**Z-score:** $z = \frac{x - \mu}{\sigma}$

#### Robust Statistics

```rust
// Median Absolute Deviation (robust measure of spread)  
let mad = Stats::mad(&data).unwrap();

// Winsorized mean (trim extreme values)
let trimmed_mean = Stats::winsorized_mean(&data, 0.1).unwrap();  // Trim 10%
```

#### Time Series Analysis

```rust
// Moving averages
let ma = Stats::moving_average(&data, 3).unwrap();       // 3-period moving average

// Exponential smoothing
let smoothed = Stats::exponential_smoothing(&data, 0.3).unwrap();

// Autocorrelation
let autocorr = Stats::autocorrelation(&data, 1).unwrap();  // Lag-1 autocorrelation
```

#### Multiple Testing Corrections

```rust
let p_values = vec![0.01, 0.03, 0.05, 0.10];

// Bonferroni correction
let bonf_corrected = Stats::bonferroni_correction(&p_values).unwrap();

// Benjamini-Hochberg (False Discovery Rate)
let bh_corrected = Stats::benjamini_hochberg(&p_values).unwrap();
```

#### Information Theory

```rust
let probabilities = vec![0.5, 0.3, 0.2];

// Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
let entropy = Stats::entropy(&probabilities).unwrap();

// Kullback-Leibler divergence
let p = vec![0.5, 0.3, 0.2];
let q = vec![0.4, 0.4, 0.2];
let kl_div = Stats::kullback_leibler_divergence(&p, &q).unwrap();
```

#### Summary Statistics

```rust
use rs_sci::stats::summary_statistics;

// Get comprehensive summary
let summary = summary_statistics(&data).unwrap();
println!("Mean: {:.3}", summary.mean);
println!("Std Dev: {:.3}", summary.std_dev);
println!("Skewness: {:.3}", summary.skewness);
// ... all statistics at once
```
