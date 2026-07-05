use crate::{field::F128, matrix::BooleanMatrix, rmfe};

pub const RMFE_WORD_BYTES: usize = rmfe::RMFE_BITS / 8;

pub fn evaluate_univar(poly: &[F128], point: F128) -> F128 {
    let mut acc = F128::ZERO;
    for &coeff in poly.iter().rev() {
        acc *= point;
        acc += coeff;
    }
    acc
}

pub fn rmfe_one_eval(point: F128) -> F128 {
    let ones = [F128::ONE; rmfe::RMFE_BITS];
    let coeffs = apply_boolean_matrix(rmfe::embedding_matrix(), &ones);
    evaluate_univar(&coeffs, point)
}

pub fn rmfe_byte_eval_tables(point: F128) -> [[F128; 256]; RMFE_WORD_BYTES] {
    let mut basis_evals = [F128::ZERO; rmfe::RMFE_BITS];
    let matrix = rmfe::embedding_matrix();
    for input_bit in 0..rmfe::RMFE_BITS {
        let mut acc = F128::ZERO;
        for coeff in (0..rmfe::PRODUCT_DEGREE).rev() {
            acc *= point;
            if matrix.row(coeff)[input_bit / 64] & (1u64 << (input_bit % 64)) != 0 {
                acc += F128::ONE;
            }
        }
        basis_evals[input_bit] = acc;
    }

    core::array::from_fn(|byte_idx| {
        core::array::from_fn(|value| {
            let mut acc = F128::ZERO;
            for bit in 0..8 {
                let input_bit = 8 * byte_idx + bit;
                if ((value >> bit) & 1) != 0 && input_bit < rmfe::RMFE_BITS {
                    acc += basis_evals[input_bit];
                }
            }
            acc
        })
    })
}

pub fn eval_packed_word(word: u128, tables: &[[F128; 256]; RMFE_WORD_BYTES]) -> F128 {
    let mut acc = F128::ZERO;
    for byte_idx in 0..RMFE_WORD_BYTES {
        acc += tables[byte_idx][((word >> (8 * byte_idx)) & 0xff) as usize];
    }
    acc
}

pub fn apply_boolean_matrix<const OUT: usize>(
    matrix: &BooleanMatrix<OUT>,
    input: &[F128],
) -> [F128; OUT] {
    assert_eq!(input.len(), matrix.input_len());
    let mut out = [F128::ZERO; OUT];
    for row in 0..OUT {
        let matrix_row = matrix.row(row);
        let mut acc = F128::ZERO;
        for (word_idx, mut word) in matrix_row.iter().copied().enumerate() {
            let base = word_idx * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                if base + bit < input.len() {
                    acc += input[base + bit];
                }
                word &= word - 1;
            }
        }
        out[row] = acc;
    }
    out
}

pub fn eq_poly_v(point: &[F128]) -> Vec<F128> {
    let mut eq = vec![F128::ONE];
    for &r in point {
        let half = eq.len();
        eq.resize(2 * half, F128::ZERO);
        for idx in (0..half).rev() {
            let base = eq[idx];
            let high = base * r;
            eq[idx] = base + high;
            eq[idx + half] = high;
        }
    }
    eq
}

pub fn fill_eq_poly_v(point: &[F128], out: &mut [F128]) {
    assert_eq!(out.len(), 1usize << point.len());
    out[0] = F128::ONE;
    let mut len = 1usize;
    for &r in point {
        for idx in (0..len).rev() {
            let base = out[idx];
            let high = base * r;
            out[idx] = base + high;
            out[idx + len] = high;
        }
        len *= 2;
    }
}

pub fn eq_eval_v(point: &[F128], at: &[F128]) -> F128 {
    assert_eq!(point.len(), at.len());
    let mut acc = F128::ONE;
    for (&r, &s) in point.iter().zip(at) {
        acc *= F128::ONE + r + s;
    }
    acc
}
