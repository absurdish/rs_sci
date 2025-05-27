use std::ops::{Add, Sub, Mul, Div};
use std::fmt;
use std::cmp::Ordering;


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Natural(u64);

impl Natural {
    pub fn new(value: u64) -> Self {
        Natural(value)
    }
    
    pub fn value(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Integer(i64);

impl Integer {
    pub fn new(value: i64) -> Self {
        Integer(value)
    }
    
    pub fn value(&self) -> i64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    pub fn new(numerator: i64, denominator: i64) -> Self {
        if denominator == 0 {
            panic!("Denominator cannot be zero");
        }
        
        let gcd = gcd(numerator.abs() as u64, denominator.abs() as u64) as i64;
        let sign = if denominator < 0 { -1 } else { 1 };
        
        Rational {
            numerator: sign * numerator / gcd,
            denominator: sign * denominator.abs() / gcd,
        }
    }
    
    pub fn from_integer(value: Integer) -> Self {
        Rational::new(value.0, 1)
    }
    
    pub fn numerator(&self) -> i64 {
        self.numerator
    }
    
    pub fn denominator(&self) -> i64 {
        self.denominator
    }
    
    pub fn to_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Real(f64);

impl Real {
    pub fn new(value: f64) -> Self {
        Real(value)
    }
    
    pub fn from_rational(rational: Rational) -> Self {
        Real(rational.to_f64())
    }
    
    pub fn value(&self) -> f64 {
        self.0
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b = b - a;
        if b == 0 {
            break;
        }
    }
    
    a << shift
}

impl From<u64> for Natural {
    fn from(value: u64) -> Self {
        Natural(value)
    }
}

impl From<i64> for Integer {
    fn from(value: i64) -> Self {
        Integer(value)
    }
}

impl From<Natural> for Integer {
    fn from(value: Natural) -> Self {
        Integer(value.0 as i64)
    }
}

impl fmt::Display for Natural {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
