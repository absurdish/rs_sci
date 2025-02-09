use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
    re: T,
    im: T,
}

impl<T> Complex<T> {
    /// creates new complex number from real and imaginary parts
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(3.0, 4.0); // 3 + 4i
    /// ```
    /// ---
    /// basic constructor for complex numbers
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: Copy> Complex<T> {
    pub fn re(&self) -> T {
        self.re
    }

    pub fn im(&self) -> T {
        self.im
    }
}

impl Complex<f64> {
    /// calculates absolute value (modulus) of complex number
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(3.0, 4.0);
    /// assert_eq!(z.modulus(), 5.0);
    /// ```
    /// ---
    /// computes √(re² + im²)
    pub fn modulus(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// calculates argument (phase) of complex number
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(1.0, 1.0);
    /// assert_eq!(z.argument(), std::f64::consts::PI/4.0);
    /// ```
    /// ---
    /// returns angle in radians from positive real axis

    pub fn argument(&self) -> f64 {
        self.im.atan2(self.re)
    }
    /// returns complex conjugate
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(3.0, 4.0);
    /// let conj = z.conjugate(); // 3 - 4i
    /// ```
    /// ---
    /// negates imaginary part while keeping real part

    pub fn conjugate(&self) -> Self {
        Self::new(self.re, -self.im)
    }
    /// computes exponential of complex number
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::I * std::f64::consts::PI;
    /// let exp_z = z.exp(); // ≈ -1 + 0i
    /// ```
    /// ---
    /// uses euler's formula: e^(a+bi) = e^a(cos(b) + i*sin(b))
    pub fn exp(&self) -> Self {
        let r = self.re.exp();
        Self::new(r * self.im.cos(), r * self.im.sin())
    }
    /// computes natural logarithm of complex number
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(1.0, 0.0);
    /// let ln_z = z.ln(); // 0 + 0i
    /// ```
    /// ---
    /// returns ln|z| + i*arg(z)
    pub fn ln(&self) -> Self {
        Complex::new(self.modulus().ln(), self.argument())
    }
    /// raises complex number to real power
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(1.0, 1.0);
    /// let z_squared = z.pow(2.0);
    /// ```
    /// ---
    /// uses polar form for computation
    pub fn pow(&self, n: f64) -> Self {
        let r = self.modulus().powf(n);
        let theta = self.argument() * n;
        Self::new(r * theta.cos(), r * theta.sin())
    }
    /// calculates square root of complex number
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(-1.0, 0.0);
    /// let sqrt_z = z.sqrt(); // 0 + 1i
    /// ```
    /// ---
    /// returns principal square root
    pub fn sqrt(&self) -> Self {
        let r = self.modulus().sqrt();
        let theta = self.argument() / 2.0;
        Self::new(r * theta.cos(), r * theta.sin())
    }

    pub fn sin(&self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    pub fn cos(&self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    pub fn tan(&self) -> Self {
        self.sin() / self.cos()
    }

    pub fn sinh(&self) -> Self {
        Self::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }

    pub fn cosh(&self) -> Self {
        Self::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }

    pub fn tanh(&self) -> Self {
        self.sinh() / self.cosh()
    }
}
impl<T: Copy + Add<Output = T>> Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Copy + Sub<Output = T>> Sub for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>> Div
    for Complex<T>
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl<T: Display> Display for Complex<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}+{}i", self.re, self.im)
    }
}

impl From<f64> for Complex<f64> {
    fn from(x: f64) -> Self {
        Self::new(x, 0.0)
    }
}

impl From<i32> for Complex<f64> {
    fn from(x: i32) -> Self {
        Self::new(x as f64, 0.0)
    }
}

impl Complex<f64> {
    pub const I: Complex<f64> = Complex { re: 0.0, im: 1.0 };
    pub const ONE: Complex<f64> = Complex { re: 1.0, im: 0.0 };
    pub const ZERO: Complex<f64> = Complex { re: 0.0, im: 0.0 };
}

impl Complex<f64> {
    /// creates complex number from polar coordinates
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::from_polar(2.0, std::f64::consts::PI/4.0);
    /// ```
    /// ---
    /// converts (r,θ) to x + yi form
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }
    
    /// converts to polar coordinates
    ///
    /// #### Example
    /// ```txt
    /// let z = Complex::new(1.0, 1.0);
    /// let (r, theta) = z.to_polar();
    /// ```
    /// ---
    /// returns tuple of (modulus, argument)
    pub fn to_polar(&self) -> (f64, f64) {
        (self.modulus(), self.argument())
    }
}
