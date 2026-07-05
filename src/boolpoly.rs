//! Low-level Boolean polynomial arithmetic.
//!
//! Hot kernels work with degree-192 Boolean polynomials as three `u64` limbs.
//! The product of two such words is a degree-384 word, represented by six
//! limbs.  A fourth limb is kept in `BoolPoly` so product-ring moduli with a
//! degree-192 top bit can use the same small helper routines.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BoolPoly {
    limbs: [u64; 4],
}

impl BoolPoly {
    pub const ZERO: Self = Self { limbs: [0; 4] };
    pub const ONE: Self = Self {
        limbs: [1, 0, 0, 0],
    };

    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self { limbs }
    }

    pub const fn from_u16(value: u16) -> Self {
        Self {
            limbs: [value as u64, 0, 0, 0],
        }
    }

    #[inline]
    pub const fn limbs(self) -> [u64; 4] {
        self.limbs
    }

    #[inline]
    pub fn bit(self, bit: usize) -> bool {
        debug_assert!(bit < 256);
        ((self.limbs[bit / 64] >> (bit % 64)) & 1) != 0
    }

    #[inline]
    pub fn degree(self) -> Option<usize> {
        degree(self)
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.limbs.iter().all(|&limb| limb == 0)
    }

    #[inline]
    pub fn xor(self, rhs: Self) -> Self {
        Self {
            limbs: [
                self.limbs[0] ^ rhs.limbs[0],
                self.limbs[1] ^ rhs.limbs[1],
                self.limbs[2] ^ rhs.limbs[2],
                self.limbs[3] ^ rhs.limbs[3],
            ],
        }
    }
}

impl std::ops::BitXor for BoolPoly {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

impl std::ops::BitXorAssign for BoolPoly {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        for idx in 0..4 {
            self.limbs[idx] ^= rhs.limbs[idx];
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WideBoolPoly {
    limbs: [u64; 6],
}

impl WideBoolPoly {
    pub const ZERO: Self = Self { limbs: [0; 6] };

    pub const fn from_limbs(limbs: [u64; 6]) -> Self {
        Self { limbs }
    }

    #[inline]
    pub const fn limbs(self) -> [u64; 6] {
        self.limbs
    }

    #[inline]
    pub fn degree(self) -> Option<usize> {
        wide_degree(self)
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.limbs.iter().all(|&limb| limb == 0)
    }
}

impl std::ops::BitXor for WideBoolPoly {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        let lhs = self.limbs;
        let rhs = rhs.limbs;
        Self {
            limbs: [
                lhs[0] ^ rhs[0],
                lhs[1] ^ rhs[1],
                lhs[2] ^ rhs[2],
                lhs[3] ^ rhs[3],
                lhs[4] ^ rhs[4],
                lhs[5] ^ rhs[5],
            ],
        }
    }
}

impl std::ops::BitXorAssign for WideBoolPoly {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        for idx in 0..6 {
            self.limbs[idx] ^= rhs.limbs[idx];
        }
    }
}

pub fn clmul_192(lhs: BoolPoly, rhs: BoolPoly) -> WideBoolPoly {
    let a = lhs.limbs();
    let b = rhs.limbs();
    let (a0, a1, a2) = (a[0], a[1], a[2]);
    let (b0, b1, b2) = (b[0], b[1], b[2]);

    // Three-limb Karatsuba over GF(2): six carryless 64x64 products instead of
    // the nine of the schoolbook form.  The `embed_word_poly` inputs are dense
    // 192-bit polynomials, so the old zero-skip branches almost never fired and
    // only cost mispredictions; this variant is branch-free.
    let p0 = clmul_64(a0, b0);
    let p1 = clmul_64(a1, b1);
    let p2 = clmul_64(a2, b2);
    let p01 = clmul_64(a0 ^ a1, b0 ^ b1);
    let p02 = clmul_64(a0 ^ a2, b0 ^ b2);
    let p12 = clmul_64(a1 ^ a2, b1 ^ b2);

    // Coefficient blocks (each a 128-bit carryless product), placed at X^k.
    let c0 = p0;
    let c1 = p01 ^ p0 ^ p1;
    let c2 = p02 ^ p0 ^ p2 ^ p1;
    let c3 = p12 ^ p1 ^ p2;
    let c4 = p2;

    WideBoolPoly::from_limbs([
        c0 as u64,
        ((c0 >> 64) as u64) ^ (c1 as u64),
        ((c1 >> 64) as u64) ^ (c2 as u64),
        ((c2 >> 64) as u64) ^ (c3 as u64),
        ((c3 >> 64) as u64) ^ (c4 as u64),
        (c4 >> 64) as u64,
    ])
}

pub fn square_192(value: BoolPoly) -> WideBoolPoly {
    let value = value.limbs();
    let mut out = [0u64; 6];
    for i in 0..3 {
        let product = clmul_64(value[i], value[i]);
        out[2 * i] = product as u64;
        out[2 * i + 1] = (product >> 64) as u64;
    }
    WideBoolPoly::from_limbs(out)
}

pub(crate) fn mul_small(lhs: BoolPoly, rhs: u16) -> BoolPoly {
    let mut out = BoolPoly::ZERO;
    let mut bits = rhs;
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        xor_shifted(&mut out, lhs, bit);
        bits &= bits - 1;
    }
    out
}

pub(crate) fn mul_mod_small(lhs: BoolPoly, rhs: u16, modulus: BoolPoly) -> BoolPoly {
    let mut product = WideBoolPoly::ZERO;
    let mut bits = rhs;
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        xor_shifted_wide_poly(&mut product, lhs, bit);
        bits &= bits - 1;
    }
    mod_poly(product, modulus)
}

pub(crate) fn div_exact_u16(mut value: BoolPoly, divisor: u16) -> BoolPoly {
    let divisor_degree = poly_degree_u16(divisor) as usize;
    let mut out = BoolPoly::ZERO;
    while let Some(value_degree) = value.degree() {
        if value_degree < divisor_degree {
            break;
        }
        let shift = value_degree - divisor_degree;
        out.limbs[shift / 64] ^= 1u64 << (shift % 64);
        xor_shifted(&mut value, BoolPoly::from_u16(divisor), shift);
    }
    assert_eq!(value, BoolPoly::ZERO);
    out
}

pub(crate) fn mod_u16(mut value: BoolPoly, modulus: u16) -> u16 {
    let modulus_degree = poly_degree_u16(modulus) as usize;
    while let Some(value_degree) = value.degree() {
        if value_degree < modulus_degree {
            break;
        }
        xor_shifted(&mut value, BoolPoly::from_u16(modulus), value_degree - modulus_degree);
    }
    debug_assert_eq!(value.limbs[1], 0);
    debug_assert_eq!(value.limbs[2], 0);
    debug_assert_eq!(value.limbs[3], 0);
    value.limbs[0] as u16
}

pub(crate) fn mod_poly(mut value: WideBoolPoly, modulus: BoolPoly) -> BoolPoly {
    let modulus_degree = modulus.degree().expect("modulus is nonzero");
    while let Some(value_degree) = value.degree() {
        if value_degree < modulus_degree {
            break;
        }
        xor_shifted_wide_poly(&mut value, modulus, value_degree - modulus_degree);
    }
    debug_assert_eq!(value.limbs[3], 0);
    debug_assert_eq!(value.limbs[4], 0);
    debug_assert_eq!(value.limbs[5], 0);
    BoolPoly::from_limbs([value.limbs[0], value.limbs[1], value.limbs[2], 0])
}

pub(crate) fn gf_mul(lhs: u16, rhs: u16, modulus: u16) -> u16 {
    poly_mod_u32(poly_mul_u16(lhs, rhs), modulus)
}

pub(crate) fn gf_inv(value: u16, modulus: u16) -> Option<u16> {
    assert_ne!(value, 0);
    let degree = poly_degree_u16(modulus) as usize;
    (1..(1u16 << degree)).find(|&candidate| gf_mul(value, candidate, modulus) == 1)
}

pub(crate) fn poly_degree_u16(value: u16) -> u32 {
    debug_assert_ne!(value, 0);
    15 - value.leading_zeros()
}

fn degree(value: BoolPoly) -> Option<usize> {
    let limbs = value.limbs();
    for idx in (0..4).rev() {
        let limb = limbs[idx];
        if limb != 0 {
            return Some(64 * idx + 63 - limb.leading_zeros() as usize);
        }
    }
    None
}

fn wide_degree(value: WideBoolPoly) -> Option<usize> {
    let limbs = value.limbs();
    for idx in (0..6).rev() {
        let limb = limbs[idx];
        if limb != 0 {
            return Some(64 * idx + 63 - limb.leading_zeros() as usize);
        }
    }
    None
}

fn xor_shifted(out: &mut BoolPoly, value: BoolPoly, shift: usize) {
    let mut out_limbs = out.limbs;
    let limbs = value.limbs();
    let limb_shift = shift / 64;
    let bit_shift = shift % 64;
    for idx in 0..4 {
        let limb = limbs[idx];
        if limb == 0 || idx + limb_shift >= 4 {
            continue;
        }
        out_limbs[idx + limb_shift] ^= limb << bit_shift;
        if bit_shift != 0 && idx + limb_shift + 1 < 4 {
            out_limbs[idx + limb_shift + 1] ^= limb >> (64 - bit_shift);
        }
    }
    *out = BoolPoly::from_limbs(out_limbs);
}

fn xor_shifted_wide_poly(out: &mut WideBoolPoly, value: BoolPoly, shift: usize) {
    let limbs = value.limbs();
    let limb_shift = shift / 64;
    let bit_shift = shift % 64;
    for idx in 0..4 {
        let limb = limbs[idx];
        if limb == 0 || idx + limb_shift >= 6 {
            continue;
        }
        out.limbs[idx + limb_shift] ^= limb << bit_shift;
        if bit_shift != 0 && idx + limb_shift + 1 < 6 {
            out.limbs[idx + limb_shift + 1] ^= limb >> (64 - bit_shift);
        }
    }
}

fn clmul_64(lhs: u64, rhs: u64) -> u128 {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "pclmulqdq"
    ))]
    unsafe {
        return clmul_64_x86(lhs, rhs);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        return clmul_64_aarch64(lhs, rhs);
    }

    #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
    compile_error!("hashcaster2 requires AArch64 crypto/PMULL support; build with -C target-cpu=native or -C target-feature=+aes");

    #[allow(unreachable_code)]
    {
    let mut out = 0u128;
    let mut bits = rhs;
    while bits != 0 {
        let bit = bits.trailing_zeros();
        out ^= (lhs as u128) << bit;
        bits &= bits - 1;
    }
    out
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "pclmulqdq"
))]
#[inline(always)]
unsafe fn clmul_64_x86(lhs: u64, rhs: u64) -> u128 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_mm_clmulepi64_si128, _mm_set_epi64x};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_mm_clmulepi64_si128, _mm_set_epi64x};

    let product = unsafe {
        _mm_clmulepi64_si128(
            _mm_set_epi64x(0, lhs as i64),
            _mm_set_epi64x(0, rhs as i64),
            0x00,
        )
    };
    let limbs: [u64; 2] = unsafe { core::mem::transmute(product) };
    (limbs[0] as u128) | ((limbs[1] as u128) << 64)
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline(always)]
unsafe fn clmul_64_aarch64(lhs: u64, rhs: u64) -> u128 {
    use core::arch::aarch64::vmull_p64;

    unsafe { core::mem::transmute(vmull_p64(lhs, rhs)) }
}

fn poly_mul_u16(lhs: u16, rhs: u16) -> u32 {
    let mut out = 0u32;
    let mut bits = rhs;
    while bits != 0 {
        let bit = bits.trailing_zeros();
        out ^= (lhs as u32) << bit;
        bits &= bits - 1;
    }
    out
}

fn poly_mod_u32(mut value: u32, modulus: u16) -> u16 {
    let modulus_degree = poly_degree_u16(modulus);
    while value != 0 && poly_degree_u32(value) >= modulus_degree {
        value ^= (modulus as u32) << (poly_degree_u32(value) - modulus_degree);
    }
    value as u16
}

fn poly_degree_u32(value: u32) -> u32 {
    debug_assert_ne!(value, 0);
    31 - value.leading_zeros()
}

#[cfg(test)]
mod perf_tests {
    use super::*;

    fn schoolbook_clmul_192(lhs: BoolPoly, rhs: BoolPoly) -> WideBoolPoly {
        let lhs = lhs.limbs();
        let rhs = rhs.limbs();
        let mut out = [0u64; 6];
        for i in 0..3 {
            for j in 0..3 {
                let product = clmul_64(lhs[i], rhs[j]);
                out[i + j] ^= product as u64;
                out[i + j + 1] ^= (product >> 64) as u64;
            }
        }
        WideBoolPoly::from_limbs(out)
    }

    struct Rng(u128);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb)
                .rotate_left(37);
            self.0 as u64
        }
        fn poly192(&mut self) -> BoolPoly {
            BoolPoly::from_limbs([self.next(), self.next(), self.next() & 0xffff_ffff_ffff_ffff, 0])
        }
    }

    #[test]
    fn karatsuba_clmul_192_matches_schoolbook() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15_1234_5678_9abc_def0);
        for _ in 0..100_000 {
            let a = rng.poly192();
            let b = rng.poly192();
            assert_eq!(clmul_192(a, b).limbs(), schoolbook_clmul_192(a, b).limbs());
        }
    }

    #[test]
    #[ignore = "timing microbenchmark; run with --ignored --nocapture"]
    fn bench_clmul_192() {
        use std::time::Instant;
        let mut rng = Rng(0x1234_5678_9abc_def0_dead_beef_cafe_babe);
        let inputs: Vec<(BoolPoly, BoolPoly, BoolPoly)> =
            (0..4096).map(|_| (rng.poly192(), rng.poly192(), rng.poly192())).collect();

        let iters = 2000usize;
        // Warmup.
        let mut acc = WideBoolPoly::ZERO;
        for &(a, b, c) in &inputs {
            acc ^= clmul_192(a, b) ^ square_192(c);
        }

        let mut best_new = f64::INFINITY;
        let mut best_old = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            for &(a, b, c) in &inputs {
                acc ^= clmul_192(a, b) ^ square_192(c);
            }
            best_new = best_new.min(t.elapsed().as_secs_f64());

            let t = Instant::now();
            for &(a, b, c) in &inputs {
                acc ^= schoolbook_clmul_192(a, b) ^ square_192(c);
            }
            best_old = best_old.min(t.elapsed().as_secs_f64());
        }
        std::hint::black_box(acc);
        let n = inputs.len() as f64;
        println!(
            "clmul_192^square_192  karatsuba: {:.2} ns/op   schoolbook: {:.2} ns/op   speedup {:.2}x",
            best_new / n * 1e9,
            best_old / n * 1e9,
            best_old / best_new,
        );
    }
}
