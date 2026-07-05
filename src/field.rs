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

    /// Field squaring, `self * self`.
    ///
    /// In GF(2^128) squaring is cheaper than a general multiply: the cross
    /// term `2*a_lo*a_hi` vanishes, so on the PMULL backend this is two PMULLs
    /// plus one reduction instead of the three PMULLs a general multiply needs.
    #[inline(always)]
    pub fn square(self) -> Self {
        Self::from_raw(ext::square(self.raw))
    }

    /// Repeated squaring: returns `self^(2^n)` (the `n`-fold Frobenius).
    #[inline(always)]
    fn frobenius(self, n: u32) -> Self {
        let mut x = self;
        for _ in 0..n {
            x = x.square();
        }
        x
    }

    /// Multiplicative inverse.
    ///
    /// `a^-1 = a^(2^128 - 2)`.  Computed with an Itoh–Tsujii addition chain for
    /// `2^127 - 1` (`beta_k = a^(2^k - 1)`), then one final squaring — 12
    /// multiplies + 127 squarings, versus the naive 127 multiplies + 127
    /// squarings.  Since a squaring is itself cheaper than a multiply on the
    /// PMULL backend, this is a large win on the whole inverse.
    pub fn inverse(self) -> Self {
        assert!(self != Self::ZERO);
        let a = self;
        // beta_k := a^(2^k - 1).  Build up 127 = 1111111b via the recurrences
        //   beta_{i+j} = (beta_i)^(2^j) * beta_j   (Frobenius then multiply)
        //   beta_{i+1} = (beta_i)^2 * a.
        let b1 = a; // 2^1 - 1
        let b2 = b1.frobenius(1) * b1; // 2^2 - 1
        let b3 = b2.square() * a; // 2^3 - 1
        let b6 = b3.frobenius(3) * b3; // 2^6 - 1
        let b7 = b6.square() * a; // 2^7 - 1
        let b14 = b7.frobenius(7) * b7; // 2^14 - 1
        let b15 = b14.square() * a; // 2^15 - 1
        let b30 = b15.frobenius(15) * b15; // 2^30 - 1
        let b31 = b30.square() * a; // 2^31 - 1
        let b62 = b31.frobenius(31) * b31; // 2^62 - 1
        let b63 = b62.square() * a; // 2^63 - 1
        let b126 = b63.frobenius(63) * b63; // 2^126 - 1
        let b127 = b126.square() * a; // 2^127 - 1
        b127.square() // (2^127 - 1) * 2 = 2^128 - 2
    }

    /// Deferred-reduction dot product: `sum_i a[i] * b[i]` over the field, with
    /// a single final reduction.  Much faster than a fold of independent
    /// multiplies for dot-product-shaped loops.  `a` and `b` must be equal
    /// length.
    #[inline]
    pub fn dot_product(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(a.len(), b.len());
        // `F128` is `#[repr(transparent)]` over `u128`, so a `&[F128]` is a
        // `&[u128]` with the same layout.
        let ar: &[u128] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const u128, a.len()) };
        let br: &[u128] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const u128, b.len()) };
        Self::from_raw(ext::dot(ar, br))
    }
}

/// Deferred-reduction multiply-accumulator for GF(2^128).
///
/// Accumulates a sum of products `sum_i a_i * b_i` as an *unreduced* 256-bit
/// value (two 128-bit halves) and applies the expensive field reduction only
/// once, in [`F128Acc::finalize`].  This is the fastest way to evaluate
/// dot-product / inner-product shaped expressions.
///
/// On hardware carryless-multiply backends the two halves are the low/high
/// words of the carryless product; on software fallback targets `lo` holds the
/// running reduced sum and `hi` is unused.  Either way the observable result is
/// identical.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct F128Acc {
    lo: u128,
    hi: u128,
}

impl F128Acc {
    pub const ZERO: Self = Self { lo: 0, hi: 0 };

    #[inline(always)]
    pub fn new() -> Self {
        Self::ZERO
    }

    /// Fold `a * b` into the accumulator (no reduction yet).
    #[inline(always)]
    pub fn accumulate(&mut self, a: F128, b: F128) {
        let (lo, hi) = ext::acc_mul(a.raw, b.raw);
        self.lo ^= lo;
        self.hi ^= hi;
    }

    /// Merge another accumulator into this one (for parallel reduction: sum
    /// partial accumulators, then finalize once).
    #[inline(always)]
    pub fn combine(&mut self, other: &Self) {
        self.lo ^= other.lo;
        self.hi ^= other.hi;
    }

    /// Apply the final field reduction and return the field element.
    #[inline(always)]
    pub fn finalize(self) -> F128 {
        F128::from_raw(ext::acc_reduce(self.lo, self.hi))
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

// ---------------------------------------------------------------------------
// Dispatch for the extended field API (square / batched multiply / deferred-
// reduction accumulator).  The AArch64+PMULL backend gets bespoke SIMD paths;
// every other target uses portable fallbacks expressed in terms of
// `mul_dispatch`, so results are identical everywhere (only speed differs).
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod ext {
    use super::aarch64;

    #[inline(always)]
    pub(super) fn square(a: u128) -> u128 {
        unsafe { aarch64::square_128(a) }
    }
    #[inline(always)]
    pub(super) fn dot(a: &[u128], b: &[u128]) -> u128 {
        unsafe { aarch64::dot_product(a, b) }
    }
    #[inline(always)]
    pub(super) fn acc_mul(a: u128, b: u128) -> (u128, u128) {
        unsafe { aarch64::acc_mul(a, b) }
    }
    #[inline(always)]
    pub(super) fn acc_reduce(lo: u128, hi: u128) -> u128 {
        unsafe { aarch64::acc_reduce(lo, hi) }
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "pclmulqdq"))]
mod ext {
    use super::x86;

    #[inline(always)]
    pub(super) fn square(a: u128) -> u128 {
        unsafe { x86::square_128(a) }
    }
    #[inline(always)]
    pub(super) fn dot(a: &[u128], b: &[u128]) -> u128 {
        unsafe { x86::dot_product(a, b) }
    }
    #[inline(always)]
    pub(super) fn acc_mul(a: u128, b: u128) -> (u128, u128) {
        unsafe { x86::acc_mul(a, b) }
    }
    #[inline(always)]
    pub(super) fn acc_reduce(lo: u128, hi: u128) -> u128 {
        unsafe { x86::acc_reduce(lo, hi) }
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "aes"),
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "pclmulqdq")
)))]
mod ext {
    use super::mul_dispatch;

    #[inline(always)]
    pub(super) fn square(a: u128) -> u128 {
        mul_dispatch(a, a)
    }
    #[inline(always)]
    pub(super) fn dot(a: &[u128], b: &[u128]) -> u128 {
        let mut acc = 0u128;
        for i in 0..a.len() {
            acc ^= mul_dispatch(a[i], b[i]);
        }
        acc
    }
    // Portable accumulator: `lo` holds the running *reduced* sum, `hi` unused.
    #[inline(always)]
    pub(super) fn acc_mul(a: u128, b: u128) -> (u128, u128) {
        (mul_dispatch(a, b), 0)
    }
    #[inline(always)]
    pub(super) fn acc_reduce(lo: u128, _hi: u128) -> u128 {
        lo
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(dead_code, unsafe_op_in_unsafe_fn)]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use core::mem::transmute;

    const C: u64 = 0xC200_0000_0000_0000;

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

    #[target_feature(enable = "sse2,pclmulqdq")]
    pub unsafe fn square_128(a: u128) -> u128 {
        let z0 = clmul64(a as u64, a as u64);
        let z1 = clmul64((a >> 64) as u64, (a >> 64) as u64);
        acc_reduce(z0, z1)
    }

    #[target_feature(enable = "sse2,pclmulqdq")]
    pub unsafe fn dot_product(a: &[u128], b: &[u128]) -> u128 {
        let n = a.len();
        let mut lo0 = 0u128;
        let mut hi0 = 0u128;
        let mut lo1 = 0u128;
        let mut hi1 = 0u128;

        let mut i = 0;
        while i + 2 <= n {
            let (l0, h0) = acc_mul(a[i], b[i]);
            let (l1, h1) = acc_mul(a[i + 1], b[i + 1]);
            lo0 ^= l0;
            hi0 ^= h0;
            lo1 ^= l1;
            hi1 ^= h1;
            i += 2;
        }
        if i < n {
            let (l0, h0) = acc_mul(a[i], b[i]);
            lo0 ^= l0;
            hi0 ^= h0;
        }

        acc_reduce(lo0 ^ lo1, hi0 ^ hi1)
    }

    #[target_feature(enable = "sse2,pclmulqdq")]
    pub unsafe fn acc_mul(a: u128, b: u128) -> (u128, u128) {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        let z0 = clmul64(a_lo, b_lo);
        let z1 = clmul64(a_hi, b_hi);
        let z2 = clmul64(a_lo ^ a_hi, b_lo ^ b_hi) ^ z0 ^ z1;

        let lo = (z0 as u64 as u128) | (((z0 >> 64) ^ (z2 as u64 as u128)) << 64);
        let hi = ((z1 as u64 as u128) ^ (z2 >> 64)) | ((z1 >> 64) << 64);
        (lo, hi)
    }

    #[target_feature(enable = "sse2,pclmulqdq")]
    pub unsafe fn acc_reduce(lo: u128, hi: u128) -> u128 {
        let w0 = lo as u64;
        let w1 = (lo >> 64) as u64;
        let w2 = hi as u64;
        let w3 = (hi >> 64) as u64;

        // Fold word 0, then fold the updated word 1.  This is the scalar
        // counterpart of the AArch64 PMULL reducer.
        let p0 = clmul64(w0, C);
        let w1 = w1 ^ (p0 as u64);
        let mut h0 = w2 ^ ((p0 >> 64) as u64) ^ w0;

        let p1 = clmul64(w1, C);
        h0 ^= p1 as u64;
        let h1 = w3 ^ ((p1 >> 64) as u64) ^ w1;

        (h0 as u128) | ((h1 as u128) << 64)
    }

    #[inline(always)]
    unsafe fn clmul64(a: u64, b: u64) -> u128 {
        let av = _mm_set_epi64x(0, a as i64);
        let bv = _mm_set_epi64x(0, b as i64);
        transmute(_mm_clmulepi64_si128(av, bv, 0x00))
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
    use core::mem::transmute;

    /// POLYVAL reduction constant in reversed (POLYVAL) bit order.  The
    /// *forward*-order 0x87 is GHASH's constant and a different field.
    const C: u64 = 0xC200_0000_0000_0000;

    /// Three-way XOR.  Uses the FEAT_SHA3 `EOR3` fused instruction when the
    /// `sha3` target feature is available (it is under `-C target-cpu=native`
    /// on Apple M-series), shortening the reduction's XOR tree by one op each.
    #[cfg(target_feature = "sha3")]
    #[inline(always)]
    unsafe fn xor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
        veor3q_u64(a, b, c)
    }
    #[cfg(not(target_feature = "sha3"))]
    #[inline(always)]
    unsafe fn xor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
        veorq_u64(a, veorq_u64(b, c))
    }

    /// 128x128 -> 256-bit carryless product via 3-PMULL Karatsuba, returned as
    /// unreduced halves `(lo, hi)` where the value is `lo + hi * x^128` and each
    /// half is a little-endian 64-bit-word pair `(L0,L1)` / `(L2,L3)`.
    ///
    /// The forward PMULLs read the operands from GPRs (that is where the u128
    /// arguments arrive); reduction stays entirely in the NEON domain.
    #[inline(always)]
    unsafe fn clmul_wide(a_lo: u64, a_hi: u64, b_lo: u64, b_hi: u64) -> (uint64x2_t, uint64x2_t) {
        // Karatsuba: z0 = a_lo*b_lo, z1 = a_hi*b_hi, mid term folds into z2.
        let z0 = transmute::<_, uint64x2_t>(vmull_p64(a_lo, b_lo));
        let z1 = transmute::<_, uint64x2_t>(vmull_p64(a_hi, b_hi));
        let mid = transmute::<_, uint64x2_t>(vmull_p64(a_lo ^ a_hi, b_lo ^ b_hi));
        let z2 = xor3(mid, z0, z1);

        let zero = vdupq_n_u64(0);
        let lo = veorq_u64(z0, vextq_u64::<1>(zero, z2));
        let hi = veorq_u64(z1, vextq_u64::<1>(z2, zero));
        (lo, hi)
    }

    /// Squaring specialisation: in GF(2^128) the cross term `2*a_lo*a_hi`
    /// vanishes, so the middle Karatsuba PMULL disappears entirely — only two
    /// PMULLs remain, `lo = a_lo^2`, `hi = a_hi^2`.
    #[inline(always)]
    unsafe fn sq_wide(a_lo: u64, a_hi: u64) -> (uint64x2_t, uint64x2_t) {
        let lo = transmute::<_, uint64x2_t>(vmull_p64(a_lo, a_lo));
        let hi = transmute::<_, uint64x2_t>(vmull_p64(a_hi, a_hi));
        (lo, hi)
    }

    /// Reduce a 256-bit carryless product `lo + hi * x^128` modulo the POLYVAL
    /// polynomial, entirely in the NEON domain.  Two-step fold, bit-for-bit
    /// identical to `software::reduce_karatsuba`.
    ///
    /// For a 64-bit word `x` at word position `p`, its reduction contribution
    /// is `clmul(x, C)`: `.lo` lands at word `p+1`, `.hi ^ x` at word `p+2`.
    #[inline(always)]
    unsafe fn reduce(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
        let zero = vdupq_n_u64(0);
        let cdup = transmute::<uint64x2_t, poly64x2_t>(vdupq_n_u64(C));

        // Fold L0 (lo.lane0): read it with PMULL2 off a lane-swapped copy so the
        // whole fold stays in NEON (no GPR round-trip on the latency path).
        let lo_sw = vextq_u64::<1>(lo, lo); // (L1, L0)
        let p0 = transmute::<_, uint64x2_t>(vmull_high_p64(transmute(lo_sw), cdup));
        let lo = veorq_u64(lo, vextq_u64::<1>(zero, p0)); // lo.lane1 ^= p0.lo
        let hi = xor3(hi, vextq_u64::<1>(p0, zero), vzip1q_u64(lo, zero));

        // Fold w1 (lo.lane1), reading lane 1 directly via PMULL2.
        let p1 = transmute::<_, uint64x2_t>(vmull_high_p64(transmute(lo), cdup));
        let hi = xor3(hi, p1, vzip2q_u64(zero, lo));
        hi
    }

    #[inline]
    #[target_feature(enable = "aes")]
    pub unsafe fn mul_128(a: u128, b: u128) -> u128 {
        let (lo, hi) = clmul_wide(a as u64, (a >> 64) as u64, b as u64, (b >> 64) as u64);
        transmute::<uint64x2_t, u128>(reduce(lo, hi))
    }

    /// `a * a` using the two-PMULL squaring specialisation.
    #[inline]
    #[target_feature(enable = "aes")]
    pub unsafe fn square_128(a: u128) -> u128 {
        let (lo, hi) = sq_wide(a as u64, (a >> 64) as u64);
        transmute::<uint64x2_t, u128>(reduce(lo, hi))
    }

    /// Deferred-reduction dot product: `reduce(sum_i a_i * b_i)`.  The 256-bit
    /// accumulator lives in two NEON registers across the whole loop and the
    /// expensive reduction runs exactly once at the end.  Two independent
    /// accumulator lanes hide PMULL latency.  `a` and `b` must be equal length.
    #[inline]
    #[target_feature(enable = "aes")]
    pub unsafe fn dot_product(a: &[u128], b: &[u128]) -> u128 {
        let n = a.len();
        let mut lo0 = vdupq_n_u64(0);
        let mut hi0 = vdupq_n_u64(0);
        let mut lo1 = vdupq_n_u64(0);
        let mut hi1 = vdupq_n_u64(0);

        let mut i = 0;
        while i + 2 <= n {
            let (l0, h0) =
                clmul_wide(a[i] as u64, (a[i] >> 64) as u64, b[i] as u64, (b[i] >> 64) as u64);
            let (l1, h1) = clmul_wide(
                a[i + 1] as u64,
                (a[i + 1] >> 64) as u64,
                b[i + 1] as u64,
                (b[i + 1] >> 64) as u64,
            );
            lo0 = veorq_u64(lo0, l0);
            hi0 = veorq_u64(hi0, h0);
            lo1 = veorq_u64(lo1, l1);
            hi1 = veorq_u64(hi1, h1);
            i += 2;
        }
        if i < n {
            let (l0, h0) =
                clmul_wide(a[i] as u64, (a[i] >> 64) as u64, b[i] as u64, (b[i] >> 64) as u64);
            lo0 = veorq_u64(lo0, l0);
            hi0 = veorq_u64(hi0, h0);
        }
        let lo = veorq_u64(lo0, lo1);
        let hi = veorq_u64(hi0, hi1);
        transmute::<uint64x2_t, u128>(reduce(lo, hi))
    }

    /// Single unreduced multiply-accumulate step for `F128Acc`: returns the
    /// product's two 256-bit halves as u128s to be XORed into a caller-held
    /// accumulator.
    #[inline]
    #[target_feature(enable = "aes")]
    pub unsafe fn acc_mul(a: u128, b: u128) -> (u128, u128) {
        let (lo, hi) = clmul_wide(a as u64, (a >> 64) as u64, b as u64, (b >> 64) as u64);
        (transmute::<uint64x2_t, u128>(lo), transmute::<uint64x2_t, u128>(hi))
    }

    /// Final reduction of an accumulated 256-bit `(lo, hi)` value.
    #[inline]
    #[target_feature(enable = "aes")]
    pub unsafe fn acc_reduce(lo: u128, hi: u128) -> u128 {
        transmute::<uint64x2_t, u128>(reduce(transmute(lo), transmute(hi)))
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

#[cfg(test)]
mod ext_tests {
    use super::*;

    /// Cheap deterministic PRNG (splitmix64) producing full 128-bit values.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_u128(&mut self) -> u128 {
            (self.next_u64() as u128) | ((self.next_u64() as u128) << 64)
        }
        fn next_f128(&mut self) -> F128 {
            F128::from_raw(self.next_u128())
        }
    }

    /// Reference field multiply via the bit-serial oracle.
    fn ref_mul(a: F128, b: F128) -> F128 {
        F128::from_raw(mul_reference_bitserial(a.raw(), b.raw()))
    }

    /// Naive inverse (127 squarings + 127 multiplies), the previous algorithm,
    /// used as an independent oracle for the addition-chain inverse.
    fn ref_inverse(a: F128) -> F128 {
        let mut x = a;
        let mut out = F128::ONE;
        for _ in 1..128 {
            x = x * x;
            out = out * x;
        }
        out
    }

    #[test]
    fn square_matches_oracle() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..20_000 {
            let a = rng.next_f128();
            assert_eq!(a.square(), ref_mul(a, a));
            assert_eq!(a.square(), a * a);
        }
        assert_eq!(F128::ZERO.square(), F128::ZERO);
        assert_eq!(F128::ONE.square(), F128::ONE);
    }

    #[test]
    fn inverse_matches_oracle_and_is_inverse() {
        let mut rng = Rng(0xDEAD_BEEF);
        for _ in 0..20_000 {
            let a = rng.next_f128();
            if a == F128::ZERO {
                continue;
            }
            let inv = a.inverse();
            assert_eq!(inv, ref_inverse(a));
            assert_eq!(a * inv, F128::ONE);
        }
        assert_eq!(F128::ONE.inverse(), F128::ONE);
    }

    #[test]
    fn dot_product_and_accumulator_match_oracle() {
        let mut rng = Rng(0x0F0F_5A5A);
        for len in [0usize, 1, 2, 3, 4, 5, 8, 9, 16, 31, 64, 129] {
            for _ in 0..64 {
                let a: Vec<F128> = (0..len).map(|_| rng.next_f128()).collect();
                let b: Vec<F128> = (0..len).map(|_| rng.next_f128()).collect();

                // Oracle: fold of independent reference multiplies.
                let mut want = F128::ZERO;
                for i in 0..len {
                    want += ref_mul(a[i], b[i]);
                }

                assert_eq!(F128::dot_product(&a, &b), want, "dot len={len}");

                // F128Acc via per-element accumulate.
                let mut acc2 = F128Acc::ZERO;
                for i in 0..len {
                    acc2.accumulate(a[i], b[i]);
                }
                assert_eq!(acc2.finalize(), want, "accumulate len={len}");
            }
        }
    }

    #[test]
    fn accumulator_combine_is_linear() {
        let mut rng = Rng(0x7777_3333);
        for _ in 0..2_000 {
            let n = 10;
            let a: Vec<F128> = (0..n).map(|_| rng.next_f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.next_f128()).collect();
            let mut want = F128::ZERO;
            for i in 0..n {
                want += ref_mul(a[i], b[i]);
            }
            // Split the work across two accumulators, then combine.
            let mut acc_a = F128Acc::new();
            let mut acc_b = F128Acc::new();
            for i in 0..n {
                if i % 2 == 0 {
                    acc_a.accumulate(a[i], b[i]);
                } else {
                    acc_b.accumulate(a[i], b[i]);
                }
            }
            acc_a.combine(&acc_b);
            assert_eq!(acc_a.finalize(), want);
        }
    }
}
