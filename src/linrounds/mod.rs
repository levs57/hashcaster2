pub mod prover;
pub mod verifier;

use std::sync::OnceLock;

use crate::{
    field::F128,
    protocol_state::{self, KECCAK_BITS, LANE_BITS},
    util::eq_poly_v,
};

const RHO_OFFSETS: [[usize; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linround {
    Theta,
    RhoPi,
    ThetaRhoPi,
}

pub(crate) fn matrix_eval(
    round: Linround,
    in_x: &[F128],
    in_y: &[F128],
    in_z: &[F128],
    out_x: &[F128],
    out_y: &[F128],
    out_z: &[F128],
) -> F128 {
    let input_eq = physical_eq(in_x, in_y, in_z);
    let output_eq = physical_eq(out_x, out_y, out_z);
    let mut image = [F128::ZERO; KECCAK_BITS];
    apply(round, &input_eq, &mut image);

    let mut acc = F128::ZERO;
    for i in 0..KECCAK_BITS {
        acc += image[i] * output_eq[i];
    }
    acc
}

pub(crate) fn physical_eq(r_x: &[F128], r_y: &[F128], r_z: &[F128]) -> [F128; KECCAK_BITS] {
    debug_assert_eq!(r_x.len(), 3);
    debug_assert_eq!(r_y.len(), 3);
    debug_assert_eq!(r_z.len(), 6);

    let eq_x = eq_poly_v(r_x);
    let eq_y = eq_poly_v(r_y);
    let eq_z = eq_poly_v(r_z);
    let mut out = [F128::ZERO; KECCAK_BITS];
    for x in 0..5 {
        for y in 0..5 {
            let xy = eq_x[x] * eq_y[y];
            for z in 0..LANE_BITS {
                out[protocol_state::state_idx(x, y, z)] = xy * eq_z[z];
            }
        }
    }
    out
}

pub(crate) fn apply(round: Linround, input: &[F128; KECCAK_BITS], output: &mut [F128; KECCAK_BITS]) {
    match round {
        Linround::Theta => theta_apply(input, output),
        Linround::RhoPi => rho_pi_apply(input, output),
        Linround::ThetaRhoPi => {
            let mut tmp = [F128::ZERO; KECCAK_BITS];
            theta_apply(input, &mut tmp);
            rho_pi_apply(&tmp, output);
        }
    }
}

pub(crate) fn apply_transposed(
    round: Linround,
    input: &[F128; KECCAK_BITS],
    output: &mut [F128; KECCAK_BITS],
) {
    match round {
        Linround::Theta => theta_apply_transposed(input, output),
        Linround::RhoPi => rho_pi_apply_transposed(input, output),
        Linround::ThetaRhoPi => {
            let mut tmp = [F128::ZERO; KECCAK_BITS];
            rho_pi_apply_transposed(input, &mut tmp);
            theta_apply_transposed(&tmp, output);
        }
    }
}

pub fn apply_inverse(
    round: Linround,
    input: &[F128; KECCAK_BITS],
    output: &mut [F128; KECCAK_BITS],
) {
    match round {
        Linround::Theta => theta_apply_inverse(input, output),
        Linround::RhoPi => rho_pi_apply_inverse(input, output),
        Linround::ThetaRhoPi => {
            let mut tmp = [F128::ZERO; KECCAK_BITS];
            rho_pi_apply_inverse(input, &mut tmp);
            theta_apply_inverse(&tmp, output);
        }
    }
}

fn theta_apply(input: &[F128; KECCAK_BITS], output: &mut [F128; KECCAK_BITS]) {
    let mut c = [[F128::ZERO; LANE_BITS]; 5];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            c[x][z] = input[protocol_state::state_idx(x, 0, z)]
                + input[protocol_state::state_idx(x, 1, z)]
                + input[protocol_state::state_idx(x, 2, z)]
                + input[protocol_state::state_idx(x, 3, z)]
                + input[protocol_state::state_idx(x, 4, z)];
        }
    }

    let mut d = [[F128::ZERO; LANE_BITS]; 5];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            d[x][z] = c[(x + 4) % 5][z] + c[(x + 1) % 5][(z + 63) % 64];
        }
    }

    for x in 0..5 {
        for y in 0..5 {
            for z in 0..LANE_BITS {
                output[protocol_state::state_idx(x, y, z)] =
                    input[protocol_state::state_idx(x, y, z)] + d[x][z];
            }
        }
    }
}

fn theta_apply_transposed(input: &[F128; KECCAK_BITS], output: &mut [F128; KECCAK_BITS]) {
    output.copy_from_slice(input);

    let mut d = [[F128::ZERO; LANE_BITS]; 5];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            for y in 0..5 {
                d[x][z] += input[protocol_state::state_idx(x, y, z)];
            }
        }
    }

    let mut c = [[F128::ZERO; LANE_BITS]; 5];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            c[x][z] = d[(x + 1) % 5][z] + d[(x + 4) % 5][(z + 1) % 64];
        }
    }

    for x in 0..5 {
        for y in 0..5 {
            for z in 0..LANE_BITS {
                output[protocol_state::state_idx(x, y, z)] += c[x][z];
            }
        }
    }
}

fn theta_apply_inverse(input: &[F128; KECCAK_BITS], output: &mut [F128; KECCAK_BITS]) {
    let mut c_out = [F128::ZERO; 5 * LANE_BITS];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            for y in 0..5 {
                c_out[x * LANE_BITS + z] += input[protocol_state::state_idx(x, y, z)];
            }
        }
    }

    let mut c_in = [F128::ZERO; 5 * LANE_BITS];
    for (row, mask) in theta_parity_inverse().iter().enumerate() {
        let mut acc = F128::ZERO;
        for (word_idx, mut word) in mask.iter().copied().enumerate() {
            let base = word_idx * 64;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                acc += c_out[base + bit];
                word &= word - 1;
            }
        }
        c_in[row] = acc;
    }

    let mut d = [[F128::ZERO; LANE_BITS]; 5];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            d[x][z] = c_in[((x + 4) % 5) * LANE_BITS + z]
                + c_in[((x + 1) % 5) * LANE_BITS + (z + 63) % 64];
        }
    }

    for x in 0..5 {
        for y in 0..5 {
            for z in 0..LANE_BITS {
                output[protocol_state::state_idx(x, y, z)] =
                    input[protocol_state::state_idx(x, y, z)] + d[x][z];
            }
        }
    }
}

fn rho_pi_apply(input: &[F128; KECCAK_BITS], output: &mut [F128; KECCAK_BITS]) {
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = RHO_OFFSETS[a][b];
            for z in 0..LANE_BITS {
                output[protocol_state::state_idx(x, y, z)] =
                    input[protocol_state::state_idx(a, b, (z + 64 - r) % 64)];
            }
        }
    }
}

fn rho_pi_apply_transposed(input: &[F128; KECCAK_BITS], output: &mut [F128; KECCAK_BITS]) {
    for x in 0..5 {
        for y in 0..5 {
            let out_x = y;
            let out_y = (2 * x + 3 * y) % 5;
            let r = RHO_OFFSETS[x][y];
            for z in 0..LANE_BITS {
                output[protocol_state::state_idx(x, y, (z + 64 - r) % 64)] +=
                    input[protocol_state::state_idx(out_x, out_y, z)];
            }
        }
    }
}

fn rho_pi_apply_inverse(input: &[F128; KECCAK_BITS], output: &mut [F128; KECCAK_BITS]) {
    for x in 0..5 {
        for y in 0..5 {
            let out_x = y;
            let out_y = (2 * x + 3 * y) % 5;
            let r = RHO_OFFSETS[x][y];
            for z in 0..LANE_BITS {
                output[protocol_state::state_idx(x, y, z)] =
                    input[protocol_state::state_idx(out_x, out_y, (z + r) % 64)];
            }
        }
    }
}

fn theta_parity_inverse() -> &'static [[u64; 5]; 5 * LANE_BITS] {
    static INVERSE: OnceLock<[[u64; 5]; 5 * LANE_BITS]> = OnceLock::new();
    INVERSE.get_or_init(build_theta_parity_inverse)
}

fn build_theta_parity_inverse() -> [[u64; 5]; 5 * LANE_BITS] {
    const N: usize = 5 * LANE_BITS;
    let mut rows = [[0u64; 10]; N];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            let row = x * LANE_BITS + z;
            set_bit(&mut rows[row], x * LANE_BITS + z);
            set_bit(&mut rows[row], ((x + 4) % 5) * LANE_BITS + z);
            set_bit(&mut rows[row], ((x + 1) % 5) * LANE_BITS + (z + 63) % 64);
            set_bit(&mut rows[row], N + row);
        }
    }

    for col in 0..N {
        let pivot = (col..N)
            .find(|&row| bit(&rows[row], col))
            .expect("theta parity transform is invertible");
        rows.swap(col, pivot);
        for row in 0..N {
            if row != col && bit(&rows[row], col) {
                let pivot_row = rows[col];
                for word in 0..10 {
                    rows[row][word] ^= pivot_row[word];
                }
            }
        }
    }

    let mut inverse = [[0u64; 5]; N];
    for row in 0..N {
        inverse[row].copy_from_slice(&rows[row][5..10]);
    }
    inverse
}

fn set_bit(row: &mut [u64; 10], bit: usize) {
    row[bit / 64] ^= 1u64 << (bit % 64);
}

fn bit(row: &[u64; 10], bit: usize) -> bool {
    ((row[bit / 64] >> (bit % 64)) & 1) != 0
}
