//! RMFE embedding matrix for the packed Boolean payload.
//!
//! The target ring is `F2[x] / (m_0(x) ... m_29(x))`, with total degree 192.
//! Each factor carries a small local RMFE basis; CRT lifting produces an
//! embedding from 96 Boolean coordinates to 192 Boolean polynomial
//! coefficients.  The public protocol-facing object is just that Boolean
//! matrix.

use std::sync::OnceLock;

use crate::boolpoly::{self, BoolPoly};
use crate::matrix::BooleanMatrix;

pub const RMFE_BITS: usize = 96;
pub const PRODUCT_DEGREE: usize = 192;
pub const PRODUCT_BITS: usize = 2 * PRODUCT_DEGREE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmfeValidationError {
    WrongTotalDegree,
    WrongBasisLength,
    InvalidLocalBasis { factor: usize },
    BadCrtResidue { factor: usize, basis: usize },
    DependentGlobalBasis,
}

pub fn embedding_matrix() -> &'static BooleanMatrix<PRODUCT_DEGREE> {
    &subspace().embedding_matrix
}

pub fn projection_matrix() -> &'static BooleanMatrix<PRODUCT_DEGREE> {
    &subspace().projection_matrix
}

pub fn validate_rmfe() -> Result<(), RmfeValidationError> {
    build_subspace().map(|_| ())
}

struct RmfeSubspace {
    embedding_matrix: BooleanMatrix<PRODUCT_DEGREE>,
    projection_matrix: BooleanMatrix<PRODUCT_DEGREE>,
}

struct RmfeSubspaceBuilder {
    embedding_matrix: BooleanMatrix<PRODUCT_DEGREE>,
    projection_matrix: BooleanMatrix<PRODUCT_DEGREE>,
    #[cfg(test)]
    product_modulus: BoolPoly,
    #[cfg(test)]
    basis: [BoolPoly; RMFE_BITS],
}

fn subspace() -> &'static RmfeSubspace {
    static SUBSPACE: OnceLock<RmfeSubspace> = OnceLock::new();
    SUBSPACE.get_or_init(|| {
        let built = build_subspace().expect("hardcoded RMFE constants must validate");
        RmfeSubspace {
            embedding_matrix: built.embedding_matrix,
            projection_matrix: built.projection_matrix,
        }
    })
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

fn build_subspace() -> Result<RmfeSubspaceBuilder, RmfeValidationError> {
    let total_degree: usize = FACTORS.iter().map(|factor| factor.degree).sum();
    if total_degree != PRODUCT_DEGREE {
        return Err(RmfeValidationError::WrongTotalDegree);
    }

    let mut local_projections = Vec::with_capacity(FACTORS.len());
    for (factor_idx, factor) in FACTORS.iter().enumerate() {
        if let Some(projection) = build_local_projection(factor.modulus, factor.degree, &factor.basis[..factor.rmfe_bits]) {
            local_projections.push(projection);
        } else {
            return Err(RmfeValidationError::InvalidLocalBasis { factor: factor_idx });
        }
    }

    let mut product = BoolPoly::ONE;
    for factor in FACTORS {
        product = boolpoly::mul_small(product, factor.modulus);
    }
    if product.degree() != Some(PRODUCT_DEGREE) {
        return Err(RmfeValidationError::WrongTotalDegree);
    }

    let mut basis = Vec::with_capacity(RMFE_BITS);
    let mut sources = Vec::with_capacity(RMFE_BITS);
    for (factor_idx, factor) in FACTORS.iter().enumerate() {
        let quotient = boolpoly::div_exact_u16(product, factor.modulus);
        let quotient_mod = boolpoly::mod_u16(quotient, factor.modulus);
        let quotient_inv = boolpoly::gf_inv(quotient_mod, factor.modulus)
            .expect("CRT quotient is invertible modulo a coprime factor");

        for &local in &factor.basis[..factor.rmfe_bits] {
            if basis.len() == RMFE_BITS {
                break;
            }
            let scaled = boolpoly::gf_mul(local, quotient_inv, factor.modulus);
            basis.push(boolpoly::mul_mod_small(quotient, scaled, product));
            sources.push((factor_idx, local));
        }
    }
    if basis.len() != RMFE_BITS {
        return Err(RmfeValidationError::WrongBasisLength);
    }

    for (basis_idx, (&poly, &(source_factor, local))) in basis.iter().zip(&sources).enumerate() {
        for (factor_idx, factor) in FACTORS.iter().enumerate() {
            let residue = boolpoly::mod_u16(poly, factor.modulus);
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
    Ok(RmfeSubspaceBuilder {
        embedding_matrix: build_embedding_matrix(&basis),
        projection_matrix: build_projection_matrix(product, &local_projections),
        #[cfg(test)]
        product_modulus: product,
        #[cfg(test)]
        basis,
    })
}

fn build_embedding_matrix(basis: &[BoolPoly; RMFE_BITS]) -> BooleanMatrix<PRODUCT_DEGREE> {
    let mut matrix = BooleanMatrix::zero(RMFE_BITS);
    for (input_bit, &poly) in basis.iter().enumerate() {
        for coeff in 0..PRODUCT_DEGREE {
            if poly.bit(coeff) {
                matrix.set(coeff, input_bit);
            }
        }
    }
    matrix
}

fn build_projection_matrix(
    product: BoolPoly,
    local_projections: &[LocalProjection],
) -> BooleanMatrix<PRODUCT_DEGREE> {
    let mut lift_data = Vec::with_capacity(FACTORS.len());
    for factor in FACTORS {
        let quotient = boolpoly::div_exact_u16(product, factor.modulus);
        let quotient_mod = boolpoly::mod_u16(quotient, factor.modulus);
        let quotient_inv = boolpoly::gf_inv(quotient_mod, factor.modulus)
            .expect("CRT quotient is invertible modulo a coprime factor");
        lift_data.push((factor.modulus, quotient, quotient_inv));
    }

    let mut matrix = BooleanMatrix::zero(PRODUCT_BITS);
    for input_bit in 0..PRODUCT_BITS {
        let mut output = BoolPoly::ZERO;
        for (factor_idx, &(modulus, quotient, quotient_inv)) in lift_data.iter().enumerate() {
            let residue = monomial_mod_u16(input_bit, modulus);
            let projected = local_projections[factor_idx].apply(residue);
            if projected != 0 {
                let scaled = boolpoly::gf_mul(projected, quotient_inv, modulus);
                output ^= boolpoly::mul_mod_small(quotient, scaled, product);
            }
        }
        for coeff in 0..PRODUCT_DEGREE {
            if output.bit(coeff) {
                matrix.set(coeff, input_bit);
            }
        }
    }
    matrix
}

#[derive(Clone, Copy)]
struct LocalProjection {
    pivots: [u16; 16],
    targets: [u16; 16],
}

impl LocalProjection {
    fn apply(self, mut vector: u16) -> u16 {
        let mut target = 0u16;
        while vector != 0 {
            let bit = boolpoly::poly_degree_u16(vector) as usize;
            if self.pivots[bit] == 0 {
                vector ^= 1u16 << bit;
            } else {
                vector ^= self.pivots[bit];
                target ^= self.targets[bit];
            }
        }
        target
    }
}

fn build_local_projection(modulus: u16, degree: usize, basis: &[u16]) -> Option<LocalProjection> {
    let mut projection = LocalProjection {
        pivots: [0; 16],
        targets: [0; 16],
    };

    for i in 0..basis.len() {
        for j in i..basis.len() {
            let product = boolpoly::gf_mul(basis[i], basis[j], modulus);
            let target = if i == j { basis[i] } else { 0 };
            if !insert_linear_constraint(
                product,
                target,
                &mut projection.pivots[..degree],
                &mut projection.targets[..degree],
            ) {
                return None;
            }
        }
    }
    Some(projection)
}

fn monomial_mod_u16(degree: usize, modulus: u16) -> u16 {
    let mut out = 1u16;
    for _ in 0..degree {
        out = boolpoly::gf_mul(out, 0b10, modulus);
    }
    out
}

fn insert_linear_constraint(
    mut vector: u16,
    mut target: u16,
    pivots: &mut [u16],
    targets: &mut [u16],
) -> bool {
    while vector != 0 {
        let bit = boolpoly::poly_degree_u16(vector) as usize;
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

#[cfg(test)]
pub(crate) fn test_build_subspace() -> Result<(BoolPoly, [BoolPoly; RMFE_BITS], BooleanMatrix<PRODUCT_DEGREE>, BooleanMatrix<PRODUCT_DEGREE>), RmfeValidationError> {
    build_subspace().map(|built| (built.product_modulus, built.basis, built.embedding_matrix, built.projection_matrix))
}
