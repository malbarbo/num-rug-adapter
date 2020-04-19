use num_bigint::{BigInt, ParseBigIntError};
use num_rational::BigRational;
use num_traits::{FromPrimitive, Num, Signed, ToPrimitive};

use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::ops::*;

use std::str::FromStr;

// TODO: Write more tests.

/// This macro allows the implementation of the trait `From` for the type `U`
/// from_trait_impl!(U, type_name, from_id, T0 T1 ... Tn);
//#[macro_export]
macro_rules! from_trait_impl {
    ($ty_for:ty, $ty_output_expr:expr, $ty_id:ident, $($t:ty)*) => ($(
        impl From<$t> for $ty_for {
            #[inline]
            fn from(s: $t) -> Self {
                $ty_output_expr($ty_id::from(s))
            }
        }
    )*)
}

/// partial_eq_trait_impl!(for_who, type_name, T0 T1 .. Tn);
macro_rules! partial_eq_trait_impl {
    ($ty_for:ty, $ty_id:ident, $($t:ty)*) => ($(
        impl PartialEq<$t> for $ty_for {
            #[inline]
            fn eq(&self, other: &$t) -> bool {
                self.0 == $ty_id::from(*other)
            }
        }

        impl PartialEq<$ty_for> for $t {
            #[inline]
            fn eq(&self, other: &$ty_for) -> bool {
                $ty_id::from(*self) == other.0
            }
        }
    )*)
}

/// partial_ord_trait_impl!(for_who, type_name, T0 T1 .. Tn);
macro_rules! partial_ord_trait_impl {
    ($ty_for:ty, $ty_id:ident, $($t:ty)*) => ($(
        impl PartialOrd<$t> for $ty_for {
            #[inline]
            fn partial_cmp(&self, other: &$t) -> Option<Ordering> {
                self.0.partial_cmp(&$ty_id::from(*other))
            }
        }

        // impl PartialEq<$ty_for> for $t {
        //     #[inline]
        //     fn eq(&self, other: &$ty_for) -> bool {
        //         $ty_id::from(*self) == other.0
        //     }
        // }
    )*)
}

// FIXME: Find a better name.
/// ops_trait_impl_by_by!(Trait, Rhs, Lhs, Output, func, ops, struct);
/// ops_trait_impl_by_int!(Trait, Rhs, Lhs, Output, func, ops, struct);
macro_rules! ops_trait_impl_ref_ref {
    ($trait_name:ident, $ty_input_right:ty, $ty_input_left:ty, $ty_output:ty, $func_name:ident, $op:tt, $ty_output_expr:expr) => {
        impl $trait_name<$ty_input_right> for $ty_input_left {
            type Output = $ty_output;

            #[inline]
            fn $func_name(self, other: $ty_input_right) -> Self::Output {
                $ty_output_expr(&self.0 $op &other.0)
            }
        }
    }
}

macro_rules! ops_trait_impl_move_ref {
    ($trait_name:ident, $ty_input_right:ty, $ty_input_left:ty, $ty_output:ty, $func_name:ident, $op:tt, $ty_output_expr:expr) => {
        impl $trait_name<$ty_input_right> for $ty_input_left {
            type Output = $ty_output;

            #[inline]
            fn $func_name(self, other: $ty_input_right) -> Self::Output {
                $ty_output_expr(self.0 $op &other.0)
            }
        }
    }
}

macro_rules! ops_trait_impl_move_move {
    ($trait_name:ident, $ty_input_right:ty, $ty_input_left:ty, $ty_output:ty, $func_name:ident, $op:tt, $ty_output_expr:expr) => {
        impl $trait_name<$ty_input_right> for $ty_input_left {
            type Output = $ty_output;

            #[inline]
            fn $func_name(self, other: $ty_input_right) -> Self::Output {
                $ty_output_expr(self.0 $op other.0)
            }
        }
    }
}

macro_rules! ops_trait_impl_move_int {
    ($trait_name:ident, $ty_input_right:ty, $ty_input_left:ty, $ty_output:ty, $func_name:ident, $op:tt, $ty_output_expr:expr) => {
        impl $trait_name<$ty_input_right> for $ty_input_left {
            type Output = $ty_output;

            #[inline]
            fn $func_name(self, other: $ty_input_right) -> Self::Output {
                $ty_output_expr(self.0 $op other)
            }
        }
    }
}

// FIXME: Find a better name.
/// opsa_trait_impl_by!(Trait, Rhs, Lhs, func, ops);
/// opsa_trait_impl_int!(Trait, Rhs, Lhs, func, ops);
macro_rules! opsa_trait_impl_ref {
    ($trait_name:ident, $ty_input_right:ty, $ty_input_left:ty, $func_name:ident, $op:tt) => {
        impl $trait_name<$ty_input_right> for $ty_input_left {

            #[inline]
            fn $func_name(&mut self, other: $ty_input_right) {
                self.0 $op &other.0;
            }
        }
    }
}

macro_rules! opsa_trait_impl_move {
    ($trait_name:ident, $ty_input_right:ty, $ty_input_left:ty, $func_name:ident, $op:tt) => {
        impl $trait_name<$ty_input_right> for $ty_input_left {

            #[inline]
            fn $func_name(&mut self, other: $ty_input_right) {
                self.0 $op other.0;
            }
        }
    }
}

macro_rules! opsa_trait_impl_int {
    ($trait_name:ident, $ty_input_right:ty, $ty_input_left:ty, $func_name:ident, $op:tt) => {
        impl $trait_name<$ty_input_right> for $ty_input_left {

            #[inline]
            fn $func_name(&mut self, other: $ty_input_right) {
                self.0 $op other
            }
        }
    }
}

// Integer

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Integer(BigInt);

impl Integer {
    #[inline]
    pub fn new() -> Self {
        Integer(BigInt::default())
    }

    #[inline]
    pub fn from_str_radix(
        s: &str,
        radix: u32,
    ) -> Result<Self, ParseBigIntError> {
        BigInt::from_str_radix(s, radix).map(Integer)
    }

    #[inline]
    pub fn to_u8(&self) -> Option<u8> {
        self.0.to_u8()
    }

    #[inline]
    pub fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }

    #[inline]
    pub fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }

    #[inline]
    pub fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }

    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().unwrap()
    }

    #[inline]
    pub fn abs(&self) -> Self {
        Integer(self.0.abs())
    }

    // FIXME: Better, fast implementation required.
    // This is a hacky solution.
    #[inline]
    pub fn abs_ref(&self) -> Self {
        Integer(self.0.abs())
    }

    #[inline]
    pub fn div_rem(&self, other: Self) -> (Self, Self) {
        let (a, b) = num_integer::Integer::div_rem(&self.0, &other.0);
        (Integer(a), Integer(b))
    }

    // FIXME: Better, fast implementation required.
    // This is a hacky solution.
    #[inline]
    pub fn div_rem_ref(&self, other: &Self) -> (Self, Self) {
        let (a, b) = num_integer::Integer::div_rem(&self.0, &other.0);
        (Integer(a), Integer(b))
    }

    #[inline]
    pub fn div_rem_floor(&self, other: Self) -> (Self, Self) {
        let (a, b) = num_integer::Integer::div_mod_floor(&self.0, &other.0);
        (Integer(a), Integer(b))
    }

    // FIXME: Better, fast implementation required.
    // This is a hacky solution.
    #[inline]
    pub fn div_rem_floor_ref(&self, other: &Self) -> (Self, Self) {
        let (a, b) = num_integer::Integer::div_mod_floor(&self.0, &other.0);
        (Integer(a), Integer(b))
    }

    #[inline]
    pub fn mod_u(&self, modulo: u32) -> u32 {
        (self.0.abs() % modulo).to_u32().unwrap()
    }

    #[inline]
    pub fn is_odd(&self) -> bool {
        num_integer::Integer::is_odd(&self.0)
    }

    #[inline]
    pub fn from_f64(v: f64) -> Option<Self> {
        BigInt::from_f64(v).map(Integer)
    }

    #[inline]
    pub fn gcd(&self, other: &Self) -> Self {
        Integer(num_integer::Integer::gcd(&self.0, &other.0))
    }

    // FIXME: Better, fast implementation required.
    // This is a hacky solution.
    #[inline]
    pub fn gcd_ref(&self, other: &Self) -> Self {
        Integer(num_integer::Integer::gcd(&self.0, &other.0))
    }
}

// From.
from_trait_impl!(Integer, Integer, BigInt, i32 isize u8 u32 usize);

impl From<&Integer> for Integer {
    #[inline]
    fn from(s: &Integer) -> Self {
        Integer(s.0.clone())
    }
}

// Operation.
ops_trait_impl_move_move!(Add, Integer, Integer, Integer, add, +, Integer);
ops_trait_impl_move_ref!(Add, &Integer, Integer, Integer, add, +, Integer);
ops_trait_impl_ref_ref!(Add, &Integer, &Integer, Integer, add, +, Integer);
ops_trait_impl_ref_ref!(Add, Integer, &Integer, Integer, add, +, Integer);

ops_trait_impl_move_move!(Mul, Integer, Integer, Integer, mul, *, Integer);
ops_trait_impl_move_ref!(Mul, &Integer, Integer, Integer, mul, *, Integer);
ops_trait_impl_ref_ref!(Mul, &Integer, &Integer, Integer, mul, *, Integer);
ops_trait_impl_move_int!(Mul, u32, Integer, Integer, mul, *, Integer);

ops_trait_impl_move_move!(Div, Integer, Integer, Integer, div, /, Integer);
ops_trait_impl_move_ref!(Div, &Integer, Integer, Integer, div, /, Integer);
ops_trait_impl_ref_ref!(Div, Integer, &Integer, Integer, div, /, Integer);

ops_trait_impl_move_move!(Rem, Integer, Integer, Integer, rem, %, Integer);
ops_trait_impl_move_ref!(Rem, &Integer, Integer, Integer, rem, %, Integer);
ops_trait_impl_ref_ref!(Rem, &Integer, &Integer, Integer, rem, %, Integer);
ops_trait_impl_ref_ref!(Rem, Integer, &Integer, Integer, rem, %, Integer);

ops_trait_impl_move_move!(BitAnd, Integer, Integer, Integer, bitand, &, Integer);
ops_trait_impl_move_ref!(BitAnd, &Integer, Integer, Integer, bitand, &, Integer);
ops_trait_impl_ref_ref!(BitAnd, &Integer, &Integer, Integer, bitand, &, Integer);
ops_trait_impl_ref_ref!(BitAnd, Integer, &Integer, Integer, bitand, &, Integer);

ops_trait_impl_move_move!(BitOr, Integer, Integer, Integer, bitor, |, Integer);
ops_trait_impl_move_ref!(BitOr, &Integer, Integer, Integer, bitor, |, Integer);
ops_trait_impl_ref_ref!(BitOr, &Integer, &Integer, Integer, bitor, |, Integer);
ops_trait_impl_ref_ref!(BitOr, Integer, &Integer, Integer, bitor, |, Integer);

ops_trait_impl_move_move!(BitXor, Integer, Integer, Integer, bitxor, ^, Integer);
ops_trait_impl_move_ref!(BitXor, &Integer, Integer, Integer, bitxor, ^, Integer);
ops_trait_impl_ref_ref!(BitXor, &Integer, &Integer, Integer, bitxor, ^, Integer);
ops_trait_impl_ref_ref!(BitXor, Integer, &Integer, Integer, bitxor, ^, Integer);

// Operation and Assignment.
opsa_trait_impl_ref!(AddAssign, &Integer, Integer, add_assign, *=);
opsa_trait_impl_move!(AddAssign, Integer, Integer, add_assign, *=);
opsa_trait_impl_int!(AddAssign, i64, Integer, add_assign, *=);

opsa_trait_impl_ref!(MulAssign, &Integer, Integer, mul_assign, *=);
opsa_trait_impl_move!(MulAssign, Integer, Integer, mul_assign, *=);
opsa_trait_impl_int!(MulAssign, u32, Integer, mul_assign, *=);

// Bit manipulation.

macro_rules! bit_ops_trait_impl {
    ($ty_for:ty, $ty_output_expr:expr, $($t:ty)*) => ($(
        impl Shr<$t> for $ty_for {
            type Output = $ty_for;

            #[inline]
            fn shr(self, rhs: $t) -> Self::Output {
                $ty_output_expr(self.0 >> rhs as usize)
            }
        }

        impl Shr<$t> for &$ty_for {
            type Output = $ty_for;

            #[inline]
            fn shr(self, rhs: $t) -> Self::Output {
                $ty_output_expr(&self.0 >> rhs as usize)
            }
        }

        impl ShrAssign<$t> for $ty_for {
            #[inline]
            fn shr_assign(&mut self, rhs: $t) {
                self.0 >>= rhs as usize;
            }
        }

        // TODO: Try to simplify/factorize by Shl/Shr like for Add, Mul, ...
        impl Shl<$t> for $ty_for {
            type Output = $ty_for;

            #[inline]
            fn shl(self, rhs: $t) -> Self::Output {
                $ty_output_expr(self.0 >> rhs as usize)
            }
        }

        impl Shl<$t> for &$ty_for {
            type Output = $ty_for;

            #[inline]
            fn shl(self, rhs: $t) -> Self::Output {
                $ty_output_expr(&self.0 << rhs as usize)
            }
        }

        impl ShlAssign<$t> for $ty_for {
            #[inline]
            fn shl_assign(&mut self, rhs: $t) {
                self.0 <<= rhs as usize;
            }
        }
    )*)
}

bit_ops_trait_impl!(Integer, Integer, u32);

impl Not for Integer {
    type Output = Integer;

    #[inline]
    fn not(self) -> Self::Output {
        Integer(!self.0)
    }
}

impl Not for &Integer {
    type Output = Integer;

    #[inline]
    fn not(self) -> Self::Output {
        Integer(!&self.0)
    }
}

partial_eq_trait_impl!(Integer, BigInt, i64 isize i32);

partial_ord_trait_impl!(Integer, BigInt, i64 isize i32);

impl FromStr for Integer {
    type Err = <BigInt as FromStr>::Err;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Integer(s.parse()?))
    }
}

impl Neg for Integer {
    type Output = Integer;

    #[inline]
    fn neg(self) -> Self {
        Integer(-self.0)
    }
}

impl Display for Integer {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Rational

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rational(BigRational);

impl Rational {
    #[inline]
    pub fn new() -> Self {
        Rational(BigRational::from(BigInt::default()))
    }

    // #[inline]
    // pub fn from(i: Integer) -> Self {
    //     Rational(BigRational::from(i.0))
    // }

    #[inline]
    pub fn from_f64(v: f64) -> Option<Self> {
        BigRational::from_f64(v).map(Rational)
    }

    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0.numer().to_f64().unwrap() / self.0.denom().to_f64().unwrap()
    }

    #[inline]
    pub fn numer(&self) -> &Integer {
        unsafe { ::std::mem::transmute(self.0.numer()) }
    }

    #[inline]
    pub fn denom(&self) -> &Integer {
        unsafe { ::std::mem::transmute(self.0.denom()) }
    }

    #[inline]
    pub fn abs(self) -> Self {
        Rational(self.0.abs())
    }

    #[inline]
    pub fn abs_ref(&self) -> Self {
        Rational(self.0.abs())
    }

    // TODO: Get this to work.
    // This is a hacky solution not possible.
    #[inline]
    pub fn fract_floor_ref(&self) -> &Self {
        panic!()
    }
}

/// from_trait_impl!(U, type_name, from_id, T0 T1 ... Tn);
macro_rules! rational_from_trait_impl {
    ($ty_for:ty, $ty_output_expr:expr, $ty_id:ident, $($t:ty)*) => ($(
        impl From<$t> for $ty_for {
            #[inline]
            fn from(s: $t) -> Self {
                $ty_output_expr($ty_id::from(BigInt::from(s)))
            }
        }
    )*)
}

/*
// The goal is to unify `rational_from_trait_impl!` and `from_trait_impl!`.
// TODO: Research and Test.
macro_rules! test_from_trait_impl {
    ($ty_for:ty, $ty_output_expr:expr, $($funcs:tt)*, $($t:ty)*) => ($(
        impl From<$t> for $ty_for {
            #[inline]
            fn from(s: $t) -> Self {
                //$ty_output_expr($ty_id::from(BigInt::from(s)))
                compose!(s; $($funcs)*)
            }
        }
    )*)
}

// Source of macro compose.
// https://play.rust-lang.org/?gist=931dd68424cf4201ea9a0655a34b49cf&version=stable&backtrace=0
// https://users.rust-lang.org/t/implementing-function-composition/8255/5
macro_rules! compose {
    [$id:ident; $func:ident] => {{
        $func($id)
    }};

    [$id:ident; $func:ident $($rest:path)*] => {{
        $func( compose!($id; $($rest)*) )
    }};
}

test_from_trait_impl!(Rational, Rational, BigRational::from BigInt::from, i32 isize u8 u32 usize);
// */

// From.
rational_from_trait_impl!(Rational, Rational, BigRational, i32 isize u8 u32 usize);

impl From<&Integer> for Rational {
    #[inline]
    fn from(s: &Integer) -> Self {
        Rational(BigRational::from(s.0.clone()))
    }
}

impl From<&Rational> for Rational {
    #[inline]
    fn from(s: &Rational) -> Self {
        Rational(s.0.clone())
    }
}

/// rational_partial_eq_trait_impl!(for_who, type_name, T0 T1 .. Tn);
macro_rules! rational_partial_eq_trait_impl {
    ($ty_for:ty, $ty_id:ident, $($t:ty)*) => ($(
        impl PartialEq<$t> for $ty_for {
            #[inline]
            fn eq(&self, other: &$t) -> bool {
                self.0 == $ty_id::from(BigInt::from(*other))
            }
        }

        impl PartialEq<$ty_for> for $t {
            #[inline]
            fn eq(&self, other: &$ty_for) -> bool {
                $ty_id::from(BigInt::from(*self)) == other.0
            }
        }
    )*)
}

// PartialEq.
rational_partial_eq_trait_impl!(Rational, BigRational, isize i32 i64);

// Operation.
ops_trait_impl_move_move!(Add, Rational, Rational, Rational, add, +, Rational);
ops_trait_impl_move_ref!(Add, &Rational, Rational, Rational, add, +, Rational);

ops_trait_impl_move_move!(Mul, Rational, Rational, Rational, mul, *, Rational);
ops_trait_impl_move_ref!(Mul, &Rational, Rational, Rational, mul, *, Rational);

ops_trait_impl_move_move!(Div, Rational, Rational, Rational, div, /, Rational);
ops_trait_impl_ref_ref!(Div, &Rational, &Rational, Rational, div, /, Rational);

impl PartialOrd<i64> for Rational {
    #[inline]
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.0.partial_cmp(&BigRational::from(BigInt::from(*other)))
    }
}

impl Neg for Rational {
    type Output = Rational;

    #[inline]
    fn neg(self) -> Self {
        Rational(-self.0)
    }
}

impl Display for Rational {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub trait Assign<Src = Self> {
    fn assign(&mut self, src: Src);
}

impl Assign<&Rational> for (&mut Rational, &mut Integer) {
    fn assign(&mut self, _src: &Rational) {
        panic!()
    }
}

pub mod ops {
    use super::{Integer, Rational};

    pub trait Pow<Rhs> {
        type Output;
        fn pow(self, rhs: Rhs) -> Self::Output;
    }

    impl Pow<u32> for Integer {
        type Output = Integer;

        fn pow(self, rhs: u32) -> Self::Output {
            Integer(num_traits::Pow::pow(&self.0, rhs))
        }
    }

    pub trait PowAssign<Rhs> {
        fn pow_assign(&mut self, rhs: Rhs);
    }

    impl PowAssign<u32> for Integer {
        fn pow_assign(&mut self, rhs: u32) {
            // FIXME: make it efficient
            self.0 = num_traits::Pow::pow(&self.0, rhs);
        }
    }

    pub trait NegAssign {
        fn neg_assign(&mut self);
    }

    impl NegAssign for Integer {
        fn neg_assign(&mut self) {
            self.0 = -std::mem::replace(self, Integer::new()).0;
        }
    }

    impl NegAssign for Rational {
        #[inline]
        fn neg_assign(&mut self) {
            self.0 = -std::mem::replace(self, Rational::new()).0;
        }
    }
}

pub mod rand {
    use super::Integer;
    use std::marker::PhantomData;

    pub struct RandState<'a> {
        _marker: PhantomData<&'a ()>,
    }

    impl<'a> RandState<'a> {
        pub fn new() -> Self {
            unsafe { libc::srand(libc::time(std::ptr::null_mut()) as _) };
            RandState {
                _marker: PhantomData,
            }
        }

        pub fn borrow_mut(&self) -> &Self {
            self
        }

        pub fn bits(&mut self, bits: u32) -> u32 {
            assert!(bits <= 32);
            (unsafe { libc::rand() } as u32) & (u32::max_value() >> (32 - bits))
        }

        pub fn seed(&mut self, seed: &Integer) {
            unsafe { libc::srand(seed.to_f64() as _) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ops::NegAssign;
    use super::*;

    #[test]
    fn bits() {
        let mut rand = rand::RandState::new();
        for bits in 1..32 {
            for _ in 0..100 {
                let r = rand.bits(bits);
                let max = 1 << bits;
                assert!(max > r, "{} > {}", max, r);
            }
        }
    }

    #[test]
    fn neg_rational() {
        let mut x = Rational::from_f64(5.0).unwrap();
        let x_neg = Rational::from_f64(-5.0).unwrap();
        x.neg_assign();
        assert_eq!(x, x_neg);
    }

    #[test]
    fn neg_integer() {
        let mut x = Integer::from(5);
        let x_neg = Integer::from(-5);
        x.neg_assign();
        assert_eq!(x, x_neg);
    }
}
