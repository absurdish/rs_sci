use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlgebraicStructure<T>(pub T);

impl<T: Add<Output = T> + Copy> Add for AlgebraicStructure<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        AlgebraicStructure(self.0 + other.0)
    }
}

impl<T: Sub<Output = T> + Copy> Sub for AlgebraicStructure<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        AlgebraicStructure(self.0 - other.0)
    }
}

impl<T: Mul<Output = T> + Copy> Mul for AlgebraicStructure<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        AlgebraicStructure(self.0 * other.0)
    }
}

impl<T: Div<Output = T> + Copy> Div for AlgebraicStructure<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        AlgebraicStructure(self.0 / other.0)
    }
}

pub trait Distributivity<T> {
    fn verify_distributivity(&self, b: &Self, c: &Self) -> bool;
}

#[macro_export]
macro_rules! algstruct {
    ($name:ident, +, *) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $name<T>(T);

        impl<T: std::ops::Add<Output = T> + Copy> std::ops::Add for $name<T> {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                $name(self.0 + other.0)
            }
        }

        impl<T: std::ops::Sub<Output = T> + Copy> std::ops::Sub for $name<T> {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                $name(self.0 - other.0)
            }
        }

        impl<T: std::ops::Mul<Output = T> + Copy> std::ops::Mul for $name<T> {
            type Output = Self;
            fn mul(self, other: Self) -> Self {
                $name(self.0 * other.0)
            }
        }

        impl<T: std::ops::Div<Output = T> + Copy> std::ops::Div for $name<T> {
            type Output = Self;
            fn div(self, other: Self) -> Self {
                $name(self.0 / other.0)
            }
        }

        impl<T> From<T> for $name<T> {
            fn from(value: T) -> Self {
                $name(value)
            }
        }
    };

    ($name:ident, +) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct $name<T>(T);

        impl<T: std::ops::Add<Output = T> + Copy> std::ops::Add for $name<T> {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                $name(self.0 + other.0)
            }
        }

        impl<T: std::ops::Sub<Output = T> + Copy> std::ops::Sub for $name<T> {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                $name(self.0 - other.0)
            }
        }

        impl<T> From<T> for $name<T> {
            fn from(value: T) -> Self {
                $name(value)
            }
        }
    };

    ($name:ident, *) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct $name<T>(T);

        impl<T: std::ops::Mul<Output = T> + Copy> std::ops::Mul for $name<T> {
            type Output = Self;
            fn mul(self, other: Self) -> Self {
                $name(self.0 * other.0)
            }
        }

        impl<T: std::ops::Div<Output = T> + Copy> std::ops::Div for $name<T> {
            type Output = Self;
            fn div(self, other: Self) -> Self {
                $name(self.0 / other.0)
            }
        }

        impl<T> From<T> for $name<T> {
            fn from(value: T) -> Self {
                $name(value)
            }
        }
    };
}

pub trait AdditiveOperation<T> {
    fn add_op(&self, other: &Self) -> Self;
}

pub trait MultiplicativeOperation<T> {
    fn mul_op(&self, other: &Self) -> Self;
}

impl<T: Add<Output = T> + Copy> AdditiveOperation<T> for AlgebraicStructure<T> {
    fn add_op(&self, other: &Self) -> Self {
        *self + *other
    }
}

impl<T: Mul<Output = T> + Copy> MultiplicativeOperation<T> for AlgebraicStructure<T> {
    fn mul_op(&self, other: &Self) -> Self {
        *self * *other
    }
}

pub trait ClosureUnderOperation<T> {
    fn verify_closure_add(&self, other: &Self) -> bool
    where
        Self: AdditiveOperation<T>;
    fn verify_closure_mul(&self, other: &Self) -> bool
    where
        Self: MultiplicativeOperation<T>;
}

impl<T: Add<Output = T> + Mul<Output = T> + Copy + PartialEq> ClosureUnderOperation<T>
    for AlgebraicStructure<T>
{
    fn verify_closure_add(&self, other: &Self) -> bool
    where
        Self: AdditiveOperation<T>,
    {
        let _result = self.add_op(other);
        true
    }

    fn verify_closure_mul(&self, other: &Self) -> bool
    where
        Self: MultiplicativeOperation<T>,
    {
        let _result = self.mul_op(other);
        true
    }
}

pub trait Associativity<T> {
    fn verify_associativity_add(&self, b: &Self, c: &Self) -> bool
    where
        Self: AdditiveOperation<T>;
    fn verify_associativity_mul(&self, b: &Self, c: &Self) -> bool
    where
        Self: MultiplicativeOperation<T>;
}

impl<T: Add<Output = T> + Mul<Output = T> + Copy + PartialEq> Associativity<T>
    for AlgebraicStructure<T>
{
    fn verify_associativity_add(&self, b: &Self, c: &Self) -> bool
    where
        Self: AdditiveOperation<T>,
    {
        let left = self.add_op(b).add_op(c);
        let right = self.add_op(&b.add_op(c));
        left == right
    }

    fn verify_associativity_mul(&self, b: &Self, c: &Self) -> bool
    where
        Self: MultiplicativeOperation<T>,
    {
        let left = self.mul_op(b).mul_op(c);
        let right = self.mul_op(&b.mul_op(c));
        left == right
    }
}

pub trait Commutativity<T> {
    fn verify_commutativity_add(&self, other: &Self) -> bool
    where
        Self: AdditiveOperation<T>;
    fn verify_commutativity_mul(&self, other: &Self) -> bool
    where
        Self: MultiplicativeOperation<T>;
}

impl<T: Add<Output = T> + Mul<Output = T> + Copy + PartialEq> Commutativity<T>
    for AlgebraicStructure<T>
{
    fn verify_commutativity_add(&self, other: &Self) -> bool
    where
        Self: AdditiveOperation<T>,
    {
        let left = self.add_op(other);
        let right = other.add_op(self);
        left == right
    }

    fn verify_commutativity_mul(&self, other: &Self) -> bool
    where
        Self: MultiplicativeOperation<T>,
    {
        let left = self.mul_op(other);
        let right = other.mul_op(self);
        left == right
    }
}

// Update the Identity trait and implementation
pub trait Identity<T> {
    fn verify_additive_identity(&self, identity: &T) -> bool;
    fn verify_multiplicative_identity(&self, identity: &T) -> bool;
    fn get_additive_identity() -> T;
    fn get_multiplicative_identity() -> T;
}

impl Identity<f32> for AlgebraicStructure<f32> {
    fn verify_additive_identity(&self, identity: &f32) -> bool {
        let id = AlgebraicStructure(*identity);
        *self + id == *self && id + *self == *self
    }

    fn verify_multiplicative_identity(&self, identity: &f32) -> bool {
        let id = AlgebraicStructure(*identity);
        *self * id == *self && id * *self == *self
    }

    fn get_additive_identity() -> f32 {
        0.0
    }

    fn get_multiplicative_identity() -> f32 {
        1.0
    }
}

// Update the Inverse trait and implementation
pub trait Inverse<T> {
    fn has_additive_inverse(&self) -> bool;
    fn has_multiplicative_inverse(&self) -> bool;
    fn get_additive_inverse(&self) -> Option<Self>
    where
        Self: Sized;
    fn get_multiplicative_inverse(&self) -> Option<Self>
    where
        Self: Sized;
}

impl Inverse<f32> for AlgebraicStructure<f32> {
    fn has_additive_inverse(&self) -> bool {
        true
    }

    fn has_multiplicative_inverse(&self) -> bool {
        self.0 != 0.0
    }

    fn get_additive_inverse(&self) -> Option<Self> {
        Some(AlgebraicStructure(-self.0))
    }

    fn get_multiplicative_inverse(&self) -> Option<Self> {
        if self.has_multiplicative_inverse() {
            Some(AlgebraicStructure(1.0 / self.0))
        } else {
            None
        }
    }
}

impl Distributivity<f32> for AlgebraicStructure<f32> {
    fn verify_distributivity(&self, b: &Self, c: &Self) -> bool {
        let left = *self * (*b + *c);
        let right = (*self * *b) + (*self * *c);
        left == right
    }
}

// Macro for implementing axioms
#[macro_export]
macro_rules! impl_axioms {
    ($type:ident, $($axiom:ident $(($($arg:expr),*))? ),*) => {
        $(
            $crate::impl_axiom!($type, $axiom $(($($arg),*))?);
        )*
    };
}
#[macro_export]
macro_rules! impl_axiom {
    ($type:ident, Associativity) => {
        impl<T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + PartialEq>
            $crate::algebra::Associativity<T> for $type<T>
        where
            $type<T>:
                $crate::algebra::AdditiveOperation<T> + $crate::algebra::MultiplicativeOperation<T>,
        {
            fn verify_associativity_add(&self, b: &Self, c: &Self) -> bool {
                let left = (*self + *b) + *c;
                let right = *self + (*b + *c);
                left == right
            }

            fn verify_associativity_mul(&self, b: &Self, c: &Self) -> bool {
                let left = (*self * *b) * *c;
                let right = *self * (*b * *c);
                left == right
            }
        }
    };

    ($type:ident, Commutativity) => {
        impl<T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + PartialEq>
            $crate::algebra::Commutativity<T> for $type<T>
        where
            $type<T>:
                $crate::algebra::AdditiveOperation<T> + $crate::algebra::MultiplicativeOperation<T>,
        {
            fn verify_commutativity_add(&self, other: &Self) -> bool {
                let left = *self + *other;
                let right = *other + *self;
                left == right
            }

            fn verify_commutativity_mul(&self, other: &Self) -> bool {
                let left = *self * *other;
                let right = *other * *self;
                left == right
            }
        }
    };

    ($type:ident, Identity($($identity:expr),*)) => {
        impl $crate::algebra::Identity<f32> for $type<f32> {
            fn verify_additive_identity(&self, identity: &f32) -> bool {
                let id = $type(*identity);
                *self + id == *self && id + *self == *self
            }

            fn verify_multiplicative_identity(&self, identity: &f32) -> bool {
                let id = $type(*identity);
                *self * id == *self && id * *self == *self
            }

            fn get_additive_identity() -> f32 {
                0.0
            }

            fn get_multiplicative_identity() -> f32 {
                1.0
            }
        }
    };

    ($type:ident, Distributivity) => {
        impl<T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + PartialEq>
            $crate::algebra::Distributivity<T> for $type<T>
        {
            fn verify_distributivity(&self, b: &Self, c: &Self) -> bool {
                let left = *self * (*b + *c);
                let right = (*self * *b) + (*self * *c);
                left == right
            }
        }
    };
}

algstruct!(Ring, +, *);

impl_axioms!(
    Ring,
    Associativity,
    Commutativity,
    Identity(0.0, 1.0),
    Distributivity
);

algstruct!(Field, +, *);
impl_axioms!(
    Field,
    Associativity,
    Commutativity,
    Identity(0.0, 1.0),
    Distributivity
);

#[allow(unused)]
impl<T: Div<Output = T> + Copy + PartialEq> Field<T> {
    fn has_multiplicative_inverse(&self) -> bool
    where
        T: Default,
    {
        self.0 != T::default()
    }
}
