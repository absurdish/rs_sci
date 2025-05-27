use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
//
//
//  Vectors
//
//

/// 1 dimensional array
#[derive(Debug, Clone)]
pub struct Vector(Vec<f64>);


/// easiest way to create a vector
#[macro_export]
macro_rules! vector {
    ([ $($x:expr),* $(,)* ]) => {{
        rs_sci::linear::Vector::new(vec![ $($x as f64),* ])
    }};
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self(data)
    }

    /// initialize zero vector
    pub fn zeros(size: usize) -> Self {
        Self(vec![0.0; size])
    }

    /// initialize identity vector
    pub fn ones(size: usize) -> Self {
        Self(vec![1.0; size])
    }

    /// return vectors length
    pub fn dimension(&self) -> usize {
        self.0.len()
    }

    /// calculate vectors magnitude: \sqrt{x_1^2+x_2^2+...+x_n^2}
    pub fn magnitude(&self) -> f64 {
        self.0.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// normalize the vector: \frac{\ket{x}}{\braket{x\mid x}}
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        Self::new(self.0.iter().map(|x| x / mag).collect())
    }

    /// dot product of two vectors
    pub fn dot(&self, rhs: &Vector) -> Result<f64, &'static str> {
        if self.dimension() != rhs.dimension() {
            return Err("Vectors must have same dimension for dot product");
        }
        Ok(self.0.iter().zip(rhs.0.iter()).map(|(a, b)| a * b).sum())
    }

    /// cross product of two vectors
    pub fn cross(&self, other: &Vector) -> Result<Vector, &'static str> {
        if self.dimension() != 3 || other.dimension() != 3 {
            return Err("Cross product is only defined for 3D vectors");
        }
        Ok(Vector::new(vec![
            self.0[1] * other.0[2] - self.0[2] * other.0[1],
            self.0[2] * other.0[0] - self.0[0] * other.0[2],
            self.0[0] * other.0[1] - self.0[1] * other.0[0],
        ]))
    }

    pub fn transpose(&self) -> Matrix {
        Matrix::new(vec![self.0.clone()]).unwrap()
    }

    pub fn as_column_matrix(&self) -> Matrix {
        Matrix::new(self.0.iter().map(|&x| vec![x]).collect()).unwrap()
    }
}

impl Add for &Vector {
    type Output = Result<Vector, &'static str>;

    fn add(self, other: &Vector) -> Self::Output {
        if self.dimension() != other.dimension() {
            return Err("Vectors must have same dimension for addition");
        }
        Ok(Vector::new(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| a + b)
                .collect(),
        ))
    }
}

impl Sub for Vector {
    type Output = Result<Self, &'static str>;

    fn sub(self, other: Self) -> Self::Output {
        if self.dimension() != other.dimension() {
            return Err("Vectors must have same dimension for subtraction");
        }
        Ok(Self::new(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| a - b)
                .collect(),
        ))
    }
}

impl Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self::new(self.0.iter().map(|x| x * scalar).collect())
    }
}

impl Div<f64> for Vector {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        if scalar == 0.0 {
            panic!("Division by zero");
        }
        Self::new(self.0.iter().map(|x| x / scalar).collect())
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "[")?;
        for (i, val) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.6}", val)?;
        }
        write!(f, "]")
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

#[macro_export]
macro_rules! matrix {
    ([ $( [ $($x:expr),* $(,)* ] ),* $(,)* ]) => {{
        let data = vec![ $( vec![ $($x as f64),* ] ),* ];
        Matrix::new(data).unwrap()
    }};

    ([ $($x:expr),* $(,)* ]) => {{
        let data = vec![vec![ $($x as f64),* ]];
        Matrix::new(data).unwrap()
    }};
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Matrix cannot be empty");
        }
        let rows = data.len();
        let cols = data[0].len();
        if !data.iter().all(|row| row.len() == cols) {
            return Err("All rows must have same length");
        }
        Ok(Self { data, rows, cols })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }

    pub fn identity(size: usize) -> Self {
        let mut data = vec![vec![0.0; size]; size];
        for i in 0..size {
            data[i][i] = 1.0;
        }
        Self {
            data,
            rows: size,
            cols: size,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut result = vec![vec![0.0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j][i] = self.data[i][j];
            }
        }
        Self {
            data: result,
            rows: self.cols,
            cols: self.rows,
        }
    }

    pub fn determinant(&self) -> Result<f64, &'static str> {
        if self.rows != self.cols {
            return Err("Determinant only defined for square matrices");
        }
        match self.rows {
            1 => Ok(self.data[0][0]),
            2 => Ok(self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]),
            _ => {
                let mut det = 0.0;
                for j in 0..self.cols {
                    det += self.data[0][j] * self.cofactor(0, j)?;
                }
                Ok(det)
            }
        }
    }

    fn cofactor(&self, row: usize, col: usize) -> Result<f64, &'static str> {
        let minor = self.minor(row, col)?;
        Ok(minor * if (row + col) % 2 == 0 { 1.0 } else { -1.0 })
    }

    fn minor(&self, row: usize, col: usize) -> Result<f64, &'static str> {
        let submatrix = self.submatrix(row, col)?;
        submatrix.determinant()
    }

    fn submatrix(&self, row: usize, col: usize) -> Result<Matrix, &'static str> {
        let mut result = Vec::new();
        for i in 0..self.rows {
            if i == row {
                continue;
            }
            let mut new_row = Vec::new();
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
                new_row.push(self.data[i][j]);
            }
            result.push(new_row);
        }
        Matrix::new(result)
    }

    pub fn inverse(&self) -> Result<Matrix, &'static str> {
        let det = self.determinant()?;
        if det == 0.0 {
            return Err("Matrix is not invertible");
        }

        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j][i] = self.cofactor(i, j)? / det;
            }
        }

        Ok(Self {
            data: result,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn svd(
        &self,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<(Matrix, Vector, Matrix), &'static str> {
        // SVD: A = U Σ V^T
        let ata = self.transpose() * self.clone();
        let (v, s) = ata?.eigendecomposition(max_iter, tolerance)?;

        // Calculate singular values
        let singular_values = s.iter().map(|&x| x.sqrt()).collect();
        let sigma = Vector::new(singular_values);

        // Calculate U
        let mut u_cols = Vec::new();
        for i in 0..v.cols {
            let v_i = v.get_column(i);
            let u_i = (self * &v_i)? / sigma[i];
            u_cols.push(u_i.0);
        }

        let u = Matrix::new(u_cols.into_iter().map(|col| col).collect())?;
        Ok((u, sigma, v))
    }

    pub fn eigendecomposition(
        &self,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<(Matrix, Vec<f64>), &'static str> {
        if self.rows != self.cols {
            return Err("Matrix must be square for eigendecomposition");
        }

        let n = self.rows;
        let mut eigenvectors = Vec::new();
        let mut eigenvalues = Vec::new();
        let mut working_matrix = self.clone();

        for _ in 0..n {
            let (eigenval, eigenvec) = working_matrix.power_iteration(max_iter, tolerance)?;
            eigenvalues.push(eigenval);
            eigenvectors.push(eigenvec.clone().0);

            // Deflate the matrix
            working_matrix = self.deflate(&eigenvec, eigenval)?;
        }

        Ok((Matrix::new(eigenvectors)?, eigenvalues))
    }

    fn deflate(&self, eigenvector: &Vector, eigenvalue: f64) -> Result<Matrix, &'static str> {
        let n = self.rows;
        let mut result = self.clone();

        for i in 0..n {
            for j in 0..n {
                result.data[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }

        Ok(result)
    }

    pub fn cholesky(&self) -> Result<Matrix, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix must be square for Cholesky decomposition");
        }

        let n = self.rows;
        let mut l = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if j == i {
                    for k in 0..j {
                        sum += l.data[j][k] * l.data[j][k];
                    }
                    let val = self.data[j][j] - sum;
                    if val <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    l.data[j][j] = val.sqrt();
                } else {
                    for k in 0..j {
                        sum += l.data[i][k] * l.data[j][k];
                    }
                    l.data[i][j] = (self.data[i][j] - sum) / l.data[j][j];
                }
            }
        }

        Ok(l)
    }
    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..i {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_positive_definite(&self) -> bool {
        self.cholesky().is_ok()
    }

    pub fn trace(&self) -> Result<f64, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix must be square to compute trace");
        }

        Ok((0..self.rows).map(|i| self.data[i][i]).sum())
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt()
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl Index<usize> for Matrix {
    type Output = Vec<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Add for &Matrix {
    type Output = Result<Matrix, &'static str>;

    fn add(self, other: &Matrix) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices must have same dimensions for addition");
        }
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Matrix::new(result)
    }
}

impl Sub for Matrix {
    type Output = Result<Self, &'static str>;

    fn sub(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices must have same dimensions for subtraction");
        }
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        Matrix::new(result)
    }
}

impl Mul for Matrix {
    type Output = Result<Self, &'static str>;

    fn mul(self, other: Self) -> Self::Output {
        if self.cols != other.rows {
            return Err("Invalid matrix dimensions for multiplication");
        }
        let mut result = vec![vec![0.0; other.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Matrix::new(result)
    }
}

impl Mul<f64> for Matrix {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            data: self
                .data
                .iter()
                .map(|row| row.iter().map(|&x| x * scalar).collect())
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Div<f64> for Matrix {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        if scalar == 0.0 {
            panic!("Division by zero");
        }
        Self {
            data: self
                .data
                .iter()
                .map(|row| row.iter().map(|&x| x / scalar).collect())
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Result<Vector, &'static str>;

    fn mul(self, vector: &Vector) -> Self::Output {
        if self.cols != vector.dimension() {
            return Err("Matrix columns must match vector dimension");
        }
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i][j] * vector.0[j];
            }
        }
        Ok(Vector::new(result))
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:.6}", self.data[i][j])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl Matrix {
    // LU Decomposition
    pub fn lu_decomposition(&self) -> Result<(Matrix, Matrix), &'static str> {
        if self.rows != self.cols {
            return Err("Matrix must be square for LU decomposition");
        }

        let n = self.rows;
        let mut l = vec![vec![0.0; n]; n];
        let mut u = vec![vec![0.0; n]; n];

        for i in 0..n {
            // Upper triangular
            for k in i..n {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += l[i][j] * u[j][k];
                }
                u[i][k] = self.data[i][k] - sum;
            }

            // Lower triangular
            for k in i..n {
                if i == k {
                    l[i][i] = 1.0;
                } else {
                    let mut sum = 0.0;
                    for j in 0..i {
                        sum += l[k][j] * u[j][i];
                    }
                    l[k][i] = (self.data[k][i] - sum) / u[i][i];
                }
            }
        }

        Ok((Matrix::new(l)?, Matrix::new(u)?))
    }

    // QR Decomposition using Gram-Schmidt process
    pub fn qr_decomposition(&self) -> Result<(Matrix, Matrix), &'static str> {
        if self.rows < self.cols {
            return Err("Matrix must have at least as many rows as columns");
        }

        let n = self.cols;
        let mut q = vec![vec![0.0; n]; self.rows];
        let mut r = vec![vec![0.0; n]; n];

        for j in 0..n {
            let mut v = self.get_column(j);

            for i in 0..j {
                let q_col = Matrix::from_column(&q, i);
                r[i][j] = q_col.dot(&self.get_column(j))?;
                let m = v - q_col * r[i][j];
                v = m?
            }

            r[j][j] = v.magnitude();
            if r[j][j] != 0.0 {
                let q_col = v / r[j][j];
                for i in 0..self.rows {
                    q[i][j] = q_col[i];
                }
            }
        }

        Ok((Matrix::new(q)?, Matrix::new(r)?))
    }

    // Calculate matrix rank
    pub fn rank(&self) -> Result<usize, &'static str> {
        let tolerance = 1e-10;
        let (_, r) = self.qr_decomposition()?;

        let mut rank = 0;
        for i in 0..std::cmp::min(r.rows, r.cols) {
            if r.data[i][i].abs() > tolerance {
                rank += 1;
            }
        }

        Ok(rank)
    }

    pub fn power_iteration(
        &self,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<(f64, Vector), &'static str> {
        if self.rows != self.cols {
            return Err("Matrix must be square for eigenvalue calculation");
        }

        let n = self.rows;
        // Use Vector's direct implementation
        let mut v = Vector::ones(n).normalize();
        let mut lambda_old = 0.0;

        for _ in 0..max_iter {
            // Use explicit matrix-vector multiplication
            let av = self.multiply_vector(&v)?;

            // Use Vector's direct dot product implementation
            let lambda = v.dot(&av)?;

            if (lambda - lambda_old).abs() < tolerance {
                return Ok((lambda, v));
            }

            lambda_old = lambda;
            v = av.normalize();
        }

        Err("Power iteration did not converge")
    }

    pub fn multiply_vector(&self, v: &Vector) -> Result<Vector, &'static str> {
        if self.cols != v.dimension() {
            return Err("Matrix columns must match vector dimension");
        }

        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i][j] * v[j];
            }
        }
        Ok(Vector::new(result))
    }

    // Helper methods
    fn get_column(&self, j: usize) -> Vector {
        Vector::new((0..self.rows).map(|i| self.data[i][j]).collect())
    }

    fn from_column(data: &[Vec<f64>], j: usize) -> Vector {
        Vector::new((0..data.len()).map(|i| data[i][j]).collect())
    }
}

#[allow(unused)]
trait VectorOps {
    fn dot(&self, other: &Self) -> Result<f64, &'static str>;
    fn magnitude(&self) -> f64;
    fn normalize(&self) -> Self;
}

impl VectorOps for Vector {
    fn dot(&self, other: &Self) -> Result<f64, &'static str> {
        if self.0.len() != other.0.len() {
            return Err("Vectors must have same dimension for dot product");
        }
        Ok(self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum())
    }

    fn magnitude(&self) -> f64 {
        self.0.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn normalize(&self) -> Self {
        let mag = self.magnitude();
        Self::new(self.0.iter().map(|x| x / mag).collect())
    }
}
