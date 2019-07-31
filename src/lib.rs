use num_bigint::{BigInt, ParseBigIntError};
use num_rational::BigRational;
use num_traits::{FromPrimitive, Num, Signed, ToPrimitive};

use std::cmp::Ordering;
use std::fmt::{self, Display, Formatter};
use std::ops::*;

use std::str::FromStr;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Integer(BigInt);

impl Integer {
    #[inline]
    pub fn new() -> Self {
        Integer(BigInt::default())
    }

    #[inline]
    pub fn from_str_radix(s: &str, radix: u32) -> Result<Self, ParseBigIntError> {
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

    #[inline]
    pub fn div_rem(&self, other: Self) -> (Self, Self) {
        let (a, b) = num_integer::Integer::div_rem(&self.0, &other.0);
        (Integer(a), Integer(b))
    }

    #[inline]
    pub fn div_rem_floor(&self, other: Self) -> (Self, Self) {
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
}

impl From<i32> for Integer {
    #[inline]
    fn from(s: i32) -> Self {
        Integer(BigInt::from(s))
    }
}

impl From<isize> for Integer {
    #[inline]
    fn from(s: isize) -> Self {
        Integer(BigInt::from(s))
    }
}

impl From<u32> for Integer {
    #[inline]
    fn from(s: u32) -> Self {
        Integer(BigInt::from(s))
    }
}

impl From<usize> for Integer {
    #[inline]
    fn from(s: usize) -> Self {
        Integer(BigInt::from(s))
    }
}

impl Mul for Integer {
    type Output = Integer;

    #[inline]
    fn mul(self, other: Integer) -> Self::Output {
        Integer(self.0 * other.0)
    }
}

impl Mul<u32> for Integer {
    type Output = Integer;

    #[inline]
    fn mul(self, other: u32) -> Self::Output {
        Integer(self.0 * other)
    }
}

impl MulAssign<&Integer> for Integer {
    #[inline]
    fn mul_assign(&mut self, other: &Integer) {
        self.0 *= &other.0;
    }
}

impl Add for Integer {
    type Output = Integer;

    #[inline]
    fn add(self, other: Integer) -> Self::Output {
        Integer(self.0 + other.0)
    }
}

impl Add<&Integer> for &Integer {
    type Output = Integer;

    #[inline]
    fn add(self, other: &Integer) -> Self::Output {
        Integer(&self.0 + &other.0)
    }
}

impl AddAssign<i64> for Integer {
    #[inline]
    fn add_assign(&mut self, other: i64) {
        self.0 += other;
    }
}

impl AddAssign<&Integer> for Integer {
    #[inline]
    fn add_assign(&mut self, other: &Integer) {
        self.0 += &other.0;
    }
}

impl Shr<u32> for Integer {
    type Output = Integer;

    #[inline]
    fn shr(self, rhs: u32) -> Self::Output {
        Integer(self.0 >> rhs as usize)
    }
}

impl ShrAssign<u32> for Integer {
    #[inline]
    fn shr_assign(&mut self, rhs: u32) {
        self.0 >>= rhs as usize;
    }
}

impl Shl<u32> for Integer {
    type Output = Integer;

    #[inline]
    fn shl(self, rhs: u32) -> Self::Output {
        Integer(self.0 << rhs as usize)
    }
}

impl Not for Integer {
    type Output = Integer;

    #[inline]
    fn not(self) -> Self::Output {
        Integer(!self.0)
    }
}

impl Rem for Integer {
    type Output = Integer;

    #[inline]
    fn rem(self, other: Integer) -> Self::Output {
        Integer(self.0 % other.0)
    }
}

impl BitAnd for Integer {
    type Output = Integer;

    #[inline]
    fn bitand(self, other: Integer) -> Self::Output {
        Integer(self.0 & other.0)
    }
}

impl BitOr for Integer {
    type Output = Integer;

    #[inline]
    fn bitor(self, other: Integer) -> Self::Output {
        Integer(self.0 | other.0)
    }
}

impl BitXor for Integer {
    type Output = Integer;

    #[inline]
    fn bitxor(self, other: Integer) -> Self::Output {
        Integer(self.0 ^ other.0)
    }
}

impl PartialEq<i64> for Integer {
    #[inline]
    fn eq(&self, other: &i64) -> bool {
        self.0 == BigInt::from(*other)
    }
}

impl PartialOrd<i64> for Integer {
    #[inline]
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.0.partial_cmp(&BigInt::from(*other))
    }
}

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

    #[inline]
    pub fn from(i: Integer) -> Self {
        Rational(BigRational::from(i.0))
    }

    #[inline]
    pub fn from_f64(v: f64) -> Option<Self> {
        BigRational::from_f64(v).map(Rational)
    }

    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.0.numer().to_f64().unwrap() / self.0.denom().to_f64().unwrap()
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
    pub fn fract_floor_ref(&self) -> &Self {
        panic!()
    }
}

impl Add for Rational {
    type Output = Rational;

    #[inline]
    fn add(self, other: Rational) -> Self::Output {
        Rational(self.0 + other.0)
    }
}

impl PartialEq<i64> for Rational {
    #[inline]
    fn eq(&self, other: &i64) -> bool {
        self.0 == BigRational::from(BigInt::from(*other))
    }
}

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

impl Mul for Rational {
    type Output = Rational;

    #[inline]
    fn mul(self, other: Rational) -> Self::Output {
        Rational(self.0 * other.0)
    }
}

impl Div for Rational {
    type Output = Rational;

    #[inline]
    fn div(self, other: Rational) -> Self::Output {
        Rational(self.0 / other.0)
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
    use super::Integer;

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
}
