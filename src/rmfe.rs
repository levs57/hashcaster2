//! RMFE subspace for the packed Boolean payload.
//!
//! The embedding target is the product ring
//! `F2[x] / (m_0(x) ... m_29(x))`, where the selected irreducible factors have
//! total degree 192.  Each local factor carries a small RMFE basis, and CRT
//! lifting gives a 99-coordinate embedding.  The protocol uses the first 96
//! coordinates so packed rows stay aligned to three `u32` words.

use std::sync::OnceLock;

pub const RMFE_BITS: usize = 96;
pub const PRODUCT_DEGREE: usize = 192;

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
        poly_degree(self)
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmfeValidationError {
    WrongTotalDegree,
    WrongBasisLength,
    InvalidLocalBasis { factor: usize },
    BadCrtResidue { factor: usize, basis: usize },
    DependentGlobalBasis,
}

pub struct RmfeSubspace {
    product_modulus: BoolPoly,
    basis: [BoolPoly; RMFE_BITS],
}

pub fn product_modulus() -> BoolPoly {
    subspace().product_modulus
}

pub fn basis_elements() -> &'static [BoolPoly; RMFE_BITS] {
    &subspace().basis
}

pub fn embed_word(word: u128) -> BoolPoly {
    debug_assert!(word >> RMFE_BITS == 0);
    let basis = basis_elements();
    let mut out = BoolPoly::ZERO;
    let mut bits = word;
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        out ^= basis[bit];
        bits &= bits - 1;
    }
    out
}

pub fn validate_rmfe() -> Result<(), RmfeValidationError> {
    build_subspace().map(|_| ())
}

fn subspace() -> &'static RmfeSubspace {
    static SUBSPACE: OnceLock<RmfeSubspace> = OnceLock::new();
    SUBSPACE.get_or_init(|| build_subspace().expect("hardcoded RMFE constants must validate"))
}

#[derive(Clone, Copy)]
struct FactorSpec {
    degree: usize,
    modulus: u16,
    rmfe_bits: usize,
    basis: [u16; 4],
}

const FACTORS: [FactorSpec; 30] = [
    FactorSpec { degree: 4, modulus: 0b10011, rmfe_bits: 2, basis: [0x01, 0x02, 0, 0] },
    FactorSpec { degree: 4, modulus: 0b11001, rmfe_bits: 2, basis: [0x01, 0x02, 0, 0] },
    FactorSpec { degree: 4, modulus: 0b11111, rmfe_bits: 2, basis: [0x01, 0x02, 0, 0] },
    FactorSpec { degree: 5, modulus: 0b100101, rmfe_bits: 3, basis: [0x01, 0x02, 0x1d, 0] },
    FactorSpec { degree: 5, modulus: 0b101001, rmfe_bits: 3, basis: [0x01, 0x02, 0x19, 0] },
    FactorSpec { degree: 5, modulus: 0b101111, rmfe_bits: 3, basis: [0x01, 0x02, 0x1b, 0] },
    FactorSpec { degree: 5, modulus: 0b110111, rmfe_bits: 3, basis: [0x01, 0x02, 0x13, 0] },
    FactorSpec { degree: 5, modulus: 0b111011, rmfe_bits: 3, basis: [0x01, 0x02, 0x17, 0] },
    FactorSpec { degree: 5, modulus: 0b111101, rmfe_bits: 3, basis: [0x01, 0x02, 0x15, 0] },
    FactorSpec { degree: 6, modulus: 0b1000011, rmfe_bits: 3, basis: [0x01, 0x02, 0x10, 0] },
    FactorSpec { degree: 6, modulus: 0b1001001, rmfe_bits: 3, basis: [0x01, 0x02, 0x14, 0] },
    FactorSpec { degree: 6, modulus: 0b1010111, rmfe_bits: 3, basis: [0x01, 0x02, 0x10, 0] },
    FactorSpec { degree: 6, modulus: 0b1011011, rmfe_bits: 3, basis: [0x01, 0x02, 0x14, 0] },
    FactorSpec { degree: 6, modulus: 0b1100001, rmfe_bits: 3, basis: [0x01, 0x02, 0x08, 0] },
    FactorSpec { degree: 6, modulus: 0b1100111, rmfe_bits: 3, basis: [0x01, 0x02, 0x08, 0] },
    FactorSpec { degree: 6, modulus: 0b1101101, rmfe_bits: 3, basis: [0x01, 0x02, 0x08, 0] },
    FactorSpec { degree: 6, modulus: 0b1110011, rmfe_bits: 3, basis: [0x01, 0x02, 0x08, 0] },
    FactorSpec { degree: 6, modulus: 0b1110101, rmfe_bits: 3, basis: [0x01, 0x02, 0x08, 0] },
    FactorSpec { degree: 8, modulus: 0b100011011, rmfe_bits: 4, basis: [0xff, 0x0c, 0x0b, 0xc3] },
    FactorSpec { degree: 8, modulus: 0b100011101, rmfe_bits: 4, basis: [0x2c, 0xdb, 0x40, 0x13] },
    FactorSpec { degree: 8, modulus: 0b100101011, rmfe_bits: 4, basis: [0xf3, 0xf5, 0x26, 0x75] },
    FactorSpec { degree: 8, modulus: 0b100101101, rmfe_bits: 4, basis: [0x85, 0x43, 0xf0, 0x0a] },
    FactorSpec { degree: 8, modulus: 0b100111001, rmfe_bits: 4, basis: [0x1a, 0xd3, 0xc3, 0x55] },
    FactorSpec { degree: 8, modulus: 0b100111111, rmfe_bits: 4, basis: [0xdd, 0x27, 0x38, 0x0a] },
    FactorSpec { degree: 8, modulus: 0b101001101, rmfe_bits: 4, basis: [0xaf, 0x38, 0x29, 0x77] },
    FactorSpec { degree: 8, modulus: 0b101011111, rmfe_bits: 4, basis: [0xde, 0x47, 0x59, 0x5f] },
    FactorSpec { degree: 8, modulus: 0b101100011, rmfe_bits: 4, basis: [0xb1, 0x6b, 0xd3, 0xb9] },
    FactorSpec { degree: 8, modulus: 0b101100101, rmfe_bits: 4, basis: [0x66, 0xe8, 0x12, 0x97] },
    FactorSpec { degree: 8, modulus: 0b101101001, rmfe_bits: 4, basis: [0x16, 0x2f, 0x95, 0xb3] },
    FactorSpec { degree: 8, modulus: 0b101110001, rmfe_bits: 4, basis: [0xd2, 0x4b, 0xb0, 0x5b] },
];

fn build_subspace() -> Result<RmfeSubspace, RmfeValidationError> {
    let total_degree: usize = FACTORS.iter().map(|factor| factor.degree).sum();
    if total_degree != PRODUCT_DEGREE {
        return Err(RmfeValidationError::WrongTotalDegree);
    }

    for (factor_idx, factor) in FACTORS.iter().enumerate() {
        if !valid_local_rmfe_basis(factor.modulus, factor.degree, &factor.basis[..factor.rmfe_bits]) {
            return Err(RmfeValidationError::InvalidLocalBasis { factor: factor_idx });
        }
    }

    let mut product = BoolPoly::ONE;
    for factor in FACTORS {
        product = poly_mul_small(product, factor.modulus);
    }
    if product.degree() != Some(PRODUCT_DEGREE) {
        return Err(RmfeValidationError::WrongTotalDegree);
    }

    let mut basis = Vec::with_capacity(RMFE_BITS);
    let mut sources = Vec::with_capacity(RMFE_BITS);
    for (factor_idx, factor) in FACTORS.iter().enumerate() {
        let quotient = poly_div_exact_u16(product, factor.modulus);
        let quotient_mod = poly_mod_u16(quotient, factor.modulus);
        let quotient_inv = gf_inv(quotient_mod, factor.modulus)
            .expect("CRT quotient is invertible modulo a coprime factor");

        for &local in &factor.basis[..factor.rmfe_bits] {
            if basis.len() == RMFE_BITS {
                break;
            }
            let scaled = gf_mul(local, quotient_inv, factor.modulus);
            basis.push(poly_mul_mod_small(quotient, scaled, product));
            sources.push((factor_idx, local));
        }
    }
    if basis.len() != RMFE_BITS {
        return Err(RmfeValidationError::WrongBasisLength);
    }

    for (basis_idx, (&poly, &(source_factor, local))) in basis.iter().zip(&sources).enumerate() {
        for (factor_idx, factor) in FACTORS.iter().enumerate() {
            let residue = poly_mod_u16(poly, factor.modulus);
            let expected = if factor_idx == source_factor { local } else { 0 };
            if residue != expected {
                return Err(RmfeValidationError::BadCrtResidue {
                    factor: factor_idx,
                    basis: basis_idx,
                });
            }
        }
    }

    if !independent_global_basis(&basis) {
        return Err(RmfeValidationError::DependentGlobalBasis);
    }

    let basis: [BoolPoly; RMFE_BITS] = basis
        .try_into()
        .map_err(|_| RmfeValidationError::WrongBasisLength)?;
    Ok(RmfeSubspace {
        product_modulus: product,
        basis,
    })
}

fn valid_local_rmfe_basis(modulus: u16, degree: usize, basis: &[u16]) -> bool {
    let mut pivots = vec![0u16; degree];
    let mut targets = vec![0u16; degree];

    for i in 0..basis.len() {
        for j in i..basis.len() {
            let product = gf_mul(basis[i], basis[j], modulus);
            let target = if i == j { basis[i] } else { 0 };
            if !insert_linear_constraint(product, target, &mut pivots, &mut targets) {
                return false;
            }
        }
    }
    true
}

fn insert_linear_constraint(
    mut vector: u16,
    mut target: u16,
    pivots: &mut [u16],
    targets: &mut [u16],
) -> bool {
    while vector != 0 {
        let bit = poly_degree_u16(vector) as usize;
        if pivots[bit] == 0 {
            pivots[bit] = vector;
            targets[bit] = target;
            return true;
        }
        vector ^= pivots[bit];
        target ^= targets[bit];
    }
    target == 0
}

fn independent_global_basis(values: &[BoolPoly]) -> bool {
    let mut pivots = [BoolPoly::ZERO; PRODUCT_DEGREE];
    for &value in values {
        if !insert_global_vector(value, &mut pivots) {
            return false;
        }
    }
    true
}

fn insert_global_vector(mut vector: BoolPoly, pivots: &mut [BoolPoly; PRODUCT_DEGREE]) -> bool {
    while let Some(bit) = vector.degree() {
        if pivots[bit] == BoolPoly::ZERO {
            pivots[bit] = vector;
            return true;
        }
        vector ^= pivots[bit];
    }
    false
}

fn poly_mul_small(lhs: BoolPoly, rhs: u16) -> BoolPoly {
    let mut out = BoolPoly::ZERO;
    let mut bits = rhs;
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        poly_xor_shifted(&mut out, lhs, bit);
        bits &= bits - 1;
    }
    out
}

fn poly_mul_mod_small(lhs: BoolPoly, rhs: u16, modulus: BoolPoly) -> BoolPoly {
    let mut product = Product384::default();
    let mut bits = rhs;
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        product_xor_shifted_poly(&mut product, lhs, bit);
        bits &= bits - 1;
    }
    product_mod_poly(product, modulus)
}

fn poly_div_exact_u16(mut value: BoolPoly, divisor: u16) -> BoolPoly {
    let divisor_degree = poly_degree_u16(divisor) as usize;
    let mut out = BoolPoly::ZERO;
    while let Some(value_degree) = value.degree() {
        if value_degree < divisor_degree {
            break;
        }
        let shift = value_degree - divisor_degree;
        out.limbs[shift / 64] ^= 1u64 << (shift % 64);
        poly_xor_shifted(&mut value, BoolPoly::from_u16(divisor), shift);
    }
    assert_eq!(value, BoolPoly::ZERO);
    out
}

fn poly_mod_u16(mut value: BoolPoly, modulus: u16) -> u16 {
    let modulus_degree = poly_degree_u16(modulus) as usize;
    while let Some(value_degree) = value.degree() {
        if value_degree < modulus_degree {
            break;
        }
        poly_xor_shifted(&mut value, BoolPoly::from_u16(modulus), value_degree - modulus_degree);
    }
    debug_assert_eq!(value.limbs[1], 0);
    debug_assert_eq!(value.limbs[2], 0);
    debug_assert_eq!(value.limbs[3], 0);
    value.limbs[0] as u16
}

fn product_mod_poly(mut value: Product384, modulus: BoolPoly) -> BoolPoly {
    let modulus_degree = modulus.degree().expect("modulus is nonzero");
    while let Some(value_degree) = product_degree(value) {
        if value_degree < modulus_degree {
            break;
        }
        product_xor_shifted_poly(&mut value, modulus, value_degree - modulus_degree);
    }
    debug_assert_eq!(value.limbs[3], 0);
    debug_assert_eq!(value.limbs[4], 0);
    debug_assert_eq!(value.limbs[5], 0);
    BoolPoly {
        limbs: [value.limbs[0], value.limbs[1], value.limbs[2], 0],
    }
}

fn poly_xor_shifted(out: &mut BoolPoly, value: BoolPoly, shift: usize) {
    let limb_shift = shift / 64;
    let bit_shift = shift % 64;
    for idx in 0..4 {
        let limb = value.limbs[idx];
        if limb == 0 || idx + limb_shift >= 4 {
            continue;
        }
        out.limbs[idx + limb_shift] ^= limb << bit_shift;
        if bit_shift != 0 && idx + limb_shift + 1 < 4 {
            out.limbs[idx + limb_shift + 1] ^= limb >> (64 - bit_shift);
        }
    }
}

fn product_xor_shifted_poly(out: &mut Product384, value: BoolPoly, shift: usize) {
    let limb_shift = shift / 64;
    let bit_shift = shift % 64;
    for idx in 0..4 {
        let limb = value.limbs[idx];
        if limb == 0 || idx + limb_shift >= 6 {
            continue;
        }
        out.limbs[idx + limb_shift] ^= limb << bit_shift;
        if bit_shift != 0 && idx + limb_shift + 1 < 6 {
            out.limbs[idx + limb_shift + 1] ^= limb >> (64 - bit_shift);
        }
    }
}

fn poly_degree(value: BoolPoly) -> Option<usize> {
    for idx in (0..4).rev() {
        let limb = value.limbs[idx];
        if limb != 0 {
            return Some(64 * idx + 63 - limb.leading_zeros() as usize);
        }
    }
    None
}

#[derive(Clone, Copy, Default)]
struct Product384 {
    limbs: [u64; 6],
}

fn product_degree(value: Product384) -> Option<usize> {
    for idx in (0..6).rev() {
        let limb = value.limbs[idx];
        if limb != 0 {
            return Some(64 * idx + 63 - limb.leading_zeros() as usize);
        }
    }
    None
}

fn gf_mul(lhs: u16, rhs: u16, modulus: u16) -> u16 {
    poly_mod_u32(poly_mul_u16(lhs, rhs), modulus)
}

fn gf_inv(value: u16, modulus: u16) -> Option<u16> {
    assert_ne!(value, 0);
    let degree = poly_degree_u16(modulus) as usize;
    (1..(1u16 << degree)).find(|&candidate| gf_mul(value, candidate, modulus) == 1)
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

fn poly_degree_u16(value: u16) -> u32 {
    debug_assert_ne!(value, 0);
    15 - value.leading_zeros()
}

fn poly_degree_u32(value: u32) -> u32 {
    debug_assert_ne!(value, 0);
    31 - value.leading_zeros()
}
