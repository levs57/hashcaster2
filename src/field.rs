//! Hashcaster field `F2^128`.
//!
//! This is the GHASH/POLYVAL-style binary field used by the original
//! Hashcaster code.  The implementation is intentionally small here: scalar
//! API, carry-less multiply backend, and a portable fallback.

use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
compile_error!("hashcaster2 requires AArch64 crypto/PMULL support; build with -C target-cpu=native or -C target-feature=+aes");

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct F128 {
    raw: u128,
}

impl F128 {
    pub const ZERO: Self = Self { raw: 0 };
    pub const ONE: Self = Self {
        raw: 257870231182273679343338569694386847745,
    };

    #[inline(always)]
    pub const fn from_raw(raw: u128) -> Self {
        Self { raw }
    }

    #[inline(always)]
    pub const fn raw(self) -> u128 {
        self.raw
    }

    #[inline(always)]
    pub fn basis(bit: usize) -> Self {
        assert!(bit < 128);
        Self::from_raw(1u128 << bit)
    }

    pub fn inverse(self) -> Self {
        assert!(self != Self::ZERO);
        let mut x = self;
        let mut out = Self::ONE;
        for _ in 1..128 {
            x *= x;
            out *= x;
        }
        out
    }
}

impl Add for F128 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::from_raw(self.raw ^ rhs.raw)
    }
}

impl AddAssign for F128 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.raw ^= rhs.raw;
    }
}

impl Sub for F128 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + rhs
    }
}

impl SubAssign for F128 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self += rhs;
    }
}

impl Mul for F128 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_raw(mul_dispatch(self.raw, rhs.raw))
    }
}

impl MulAssign for F128 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "pclmulqdq"))]
#[inline(always)]
fn mul_dispatch(a: u128, b: u128) -> u128 {
    unsafe { x86::mul_128(a, b) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
fn mul_dispatch(a: u128, b: u128) -> u128 {
    unsafe { aarch64::mul_128(a, b) }
}

#[cfg(not(any(
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
    target_arch = "aarch64"
)))]
#[inline(always)]
fn mul_dispatch(a: u128, b: u128) -> u128 {
    software::mul_128(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(dead_code, unsafe_op_in_unsafe_fn)]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(enable = "sse2,pclmulqdq")]
    pub unsafe fn mul_128(a: u128, b: u128) -> u128 {
        let a0 = _mm_set_epi64x((a >> 64) as i64, a as i64);
        let b0 = _mm_set_epi64x((b >> 64) as i64, b as i64);

        let a1 = _mm_shuffle_epi32(a0, 0x0e);
        let b1 = _mm_shuffle_epi32(b0, 0x0e);
        let a2 = _mm_xor_si128(a0, a1);
        let b2 = _mm_xor_si128(b0, b1);

        let t0 = _mm_clmulepi64_si128(a0, b0, 0x00);
        let t1 = _mm_clmulepi64_si128(a0, b0, 0x11);
        let t2 = _mm_xor_si128(
            _mm_clmulepi64_si128(a2, b2, 0x00),
            _mm_xor_si128(t0, t1),
        );

        let v0 = t0;
        let v1 = _mm_xor_si128(_mm_shuffle_epi32(t0, 0x0e), t2);
        let v2 = _mm_xor_si128(t1, _mm_shuffle_epi32(t2, 0x0e));
        let v3 = _mm_shuffle_epi32(t1, 0x0e);

        let v2 = xor5(
            v2,
            v0,
            _mm_srli_epi64(v0, 1),
            _mm_srli_epi64(v0, 2),
            _mm_srli_epi64(v0, 7),
        );
        let v1 = xor4(
            v1,
            _mm_slli_epi64(v0, 63),
            _mm_slli_epi64(v0, 62),
            _mm_slli_epi64(v0, 57),
        );
        let v3 = xor5(
            v3,
            v1,
            _mm_srli_epi64(v1, 1),
            _mm_srli_epi64(v1, 2),
            _mm_srli_epi64(v1, 7),
        );
        let v2 = xor4(
            v2,
            _mm_slli_epi64(v1, 63),
            _mm_slli_epi64(v1, 62),
            _mm_slli_epi64(v1, 57),
        );

        core::mem::transmute(_mm_unpacklo_epi64(v2, v3))
    }

    #[inline(always)]
    unsafe fn xor4(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m128i {
        _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d))
    }

    #[inline(always)]
    unsafe fn xor5(a: __m128i, b: __m128i, c: __m128i, d: __m128i, e: __m128i) -> __m128i {
        _mm_xor_si128(a, _mm_xor_si128(_mm_xor_si128(b, c), _mm_xor_si128(d, e)))
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[allow(dead_code, unsafe_op_in_unsafe_fn)]
mod aarch64 {
    use core::arch::aarch64::*;

    #[target_feature(enable = "aes")]
    pub unsafe fn mul_128(a: u128, b: u128) -> u128 {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        let z0 = core::mem::transmute::<_, uint64x2_t>(vmull_p64(a_lo, b_lo));
        let z1 = core::mem::transmute::<_, uint64x2_t>(vmull_p64(a_hi, b_hi));
        let mid = core::mem::transmute::<_, uint64x2_t>(vmull_p64(a_lo ^ a_hi, b_lo ^ b_hi));
        let z2 = veorq_u64(mid, veorq_u64(z0, z1));

        let v0 = z0;
        let mut v1 = veorq_u64(vextq_u64::<1>(z0, z0), z2);
        let mut v2 = veorq_u64(z1, vextq_u64::<1>(z2, z2));
        let mut v3 = vextq_u64::<1>(z1, z1);

        v2 = veorq_u64(
            v2,
            veorq_u64(
                veorq_u64(v0, vshrq_n_u64::<1>(v0)),
                veorq_u64(vshrq_n_u64::<2>(v0), vshrq_n_u64::<7>(v0)),
            ),
        );
        v1 = veorq_u64(
            v1,
            veorq_u64(
                vshlq_n_u64::<63>(v0),
                veorq_u64(vshlq_n_u64::<62>(v0), vshlq_n_u64::<57>(v0)),
            ),
        );
        v3 = veorq_u64(
            v3,
            veorq_u64(
                veorq_u64(v1, vshrq_n_u64::<1>(v1)),
                veorq_u64(vshrq_n_u64::<2>(v1), vshrq_n_u64::<7>(v1)),
            ),
        );
        v2 = veorq_u64(
            v2,
            veorq_u64(
                vshlq_n_u64::<63>(v1),
                veorq_u64(vshlq_n_u64::<62>(v1), vshlq_n_u64::<57>(v1)),
            ),
        );

        (vgetq_lane_u64::<0>(v2) as u128) | ((vgetq_lane_u64::<0>(v3) as u128) << 64)
    }
}

mod software {
    #![allow(dead_code)]

    pub fn mul_128(a: u128, b: u128) -> u128 {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        let z0 = clmul64(a_lo, b_lo);
        let z1 = clmul64(a_hi, b_hi);
        let z2 = clmul64(a_lo ^ a_hi, b_lo ^ b_hi) ^ z0 ^ z1;

        reduce_karatsuba(z0, z1, z2)
    }

    pub(super) fn reduce_karatsuba(z0: u128, z1: u128, z2: u128) -> u128 {
        let v0 = z0;
        let mut v1 = swap64(z0) ^ z2;
        let mut v2 = z1 ^ swap64(z2);
        let mut v3 = swap64(z1);

        v2 ^= v0 ^ shr64_lanes(v0, 1) ^ shr64_lanes(v0, 2) ^ shr64_lanes(v0, 7);
        v1 ^= shl64_lanes(v0, 63) ^ shl64_lanes(v0, 62) ^ shl64_lanes(v0, 57);
        v3 ^= v1 ^ shr64_lanes(v1, 1) ^ shr64_lanes(v1, 2) ^ shr64_lanes(v1, 7);
        v2 ^= shl64_lanes(v1, 63) ^ shl64_lanes(v1, 62) ^ shl64_lanes(v1, 57);

        (v2 as u64 as u128) | ((v3 as u64 as u128) << 64)
    }

    #[inline(always)]
    fn clmul64(a: u64, b: u64) -> u128 {
        let lo = bmul64(a, b);
        let hi = rev64(bmul64(rev64(a), rev64(b))) >> 1;
        (lo as u128) | ((hi as u128) << 64)
    }

    #[cfg(test)]
    pub(super) fn mul_128_bitserial(a: u128, b: u128) -> u128 {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        let z0 = clmul64_bitserial(a_lo, b_lo);
        let z1 = clmul64_bitserial(a_hi, b_hi);
        let z2 = clmul64_bitserial(a_lo ^ a_hi, b_lo ^ b_hi) ^ z0 ^ z1;

        let v0 = z0;
        let mut v1 = swap64(z0) ^ z2;
        let mut v2 = z1 ^ swap64(z2);
        let mut v3 = swap64(z1);

        v2 ^= v0 ^ shr64_lanes(v0, 1) ^ shr64_lanes(v0, 2) ^ shr64_lanes(v0, 7);
        v1 ^= shl64_lanes(v0, 63) ^ shl64_lanes(v0, 62) ^ shl64_lanes(v0, 57);
        v3 ^= v1 ^ shr64_lanes(v1, 1) ^ shr64_lanes(v1, 2) ^ shr64_lanes(v1, 7);
        v2 ^= shl64_lanes(v1, 63) ^ shl64_lanes(v1, 62) ^ shl64_lanes(v1, 57);

        (v2 as u64 as u128) | ((v3 as u64 as u128) << 64)
    }

    #[cfg(test)]
    fn clmul64_bitserial(a: u64, b: u64) -> u128 {
        let mut out = 0u128;
        for bit in 0..64 {
            if ((b >> bit) & 1) != 0 {
                out ^= (a as u128) << bit;
            }
        }
        out
    }

    #[inline(always)]
    fn bmul64(x: u64, y: u64) -> u64 {
        let x0 = x & 0x1111_1111_1111_1111;
        let x1 = x & 0x2222_2222_2222_2222;
        let x2 = x & 0x4444_4444_4444_4444;
        let x3 = x & 0x8888_8888_8888_8888;
        let y0 = y & 0x1111_1111_1111_1111;
        let y1 = y & 0x2222_2222_2222_2222;
        let y2 = y & 0x4444_4444_4444_4444;
        let y3 = y & 0x8888_8888_8888_8888;

        let z0 = (x0.wrapping_mul(y0)
            ^ x1.wrapping_mul(y3)
            ^ x2.wrapping_mul(y2)
            ^ x3.wrapping_mul(y1))
            & 0x1111_1111_1111_1111;
        let z1 = (x0.wrapping_mul(y1)
            ^ x1.wrapping_mul(y0)
            ^ x2.wrapping_mul(y3)
            ^ x3.wrapping_mul(y2))
            & 0x2222_2222_2222_2222;
        let z2 = (x0.wrapping_mul(y2)
            ^ x1.wrapping_mul(y1)
            ^ x2.wrapping_mul(y0)
            ^ x3.wrapping_mul(y3))
            & 0x4444_4444_4444_4444;
        let z3 = (x0.wrapping_mul(y3)
            ^ x1.wrapping_mul(y2)
            ^ x2.wrapping_mul(y1)
            ^ x3.wrapping_mul(y0))
            & 0x8888_8888_8888_8888;

        z0 | z1 | z2 | z3
    }

    #[inline(always)]
    fn rev64(mut x: u64) -> u64 {
        x = ((x & 0x5555_5555_5555_5555) << 1) | ((x >> 1) & 0x5555_5555_5555_5555);
        x = ((x & 0x3333_3333_3333_3333) << 2) | ((x >> 2) & 0x3333_3333_3333_3333);
        x = ((x & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((x >> 4) & 0x0f0f_0f0f_0f0f_0f0f);
        x = ((x & 0x00ff_00ff_00ff_00ff) << 8) | ((x >> 8) & 0x00ff_00ff_00ff_00ff);
        x = ((x & 0x0000_ffff_0000_ffff) << 16) | ((x >> 16) & 0x0000_ffff_0000_ffff);
        x.rotate_left(32)
    }

    #[inline(always)]
    fn swap64(x: u128) -> u128 {
        (x >> 64) | (x << 64)
    }

    #[inline(always)]
    fn shr64_lanes(x: u128, shift: u32) -> u128 {
        ((x as u64 >> shift) as u128) | (((x >> 64) as u64 >> shift) as u128) << 64
    }

    #[inline(always)]
    fn shl64_lanes(x: u128, shift: u32) -> u128 {
        ((x as u64).wrapping_shl(shift) as u128)
            | (((x >> 64) as u64).wrapping_shl(shift) as u128) << 64
    }
}

#[cfg(test)]
pub(crate) fn mul_reference_bitserial(a: u128, b: u128) -> u128 {
    software::mul_128_bitserial(a, b)
}
