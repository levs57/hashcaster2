use crate::boolpoly::{self, BoolPoly};
use crate::challenger::{Challenger, FsChallenger, ProofReader, ProofTranscript, ProofWriter};
use crate::field::{self, F128};
use crate::matrix::{BooleanMatrix, FourRussians128, FourRussians256, PACKED_U64S};
use crate::rmfe;

#[test]
fn field_basics() {
    let a = F128::from_raw(0b1011);
    let b = F128::from_raw(0b1100);
    assert_eq!((a + b).raw(), 0b0111);
    assert_eq!(a + a, F128::ZERO);

    let mut state = 0x1234_5678_9abc_def0_0fed_cba9_8765_4321u128;
    for _ in 0..128 {
        state = state.wrapping_mul(0xda94_2042_e4dd_58b5).rotate_left(37);
        let x = F128::from_raw(state);
        assert_eq!(x * F128::ONE, x);
        assert_eq!(F128::ONE * x, x);
    }
}

#[test]
fn field_mul_matches_bitserial_reference() {
    let mut state = 0x5e8b_39d7_f4a7_c15d_1234_5678_9abc_def0u128;
    for _ in 0..256 {
        state = state
            .wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb)
            .rotate_left(37);
        let a = state;
        state = state
            .wrapping_mul(0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03)
            .rotate_left(19);
        let b = state;

        assert_eq!(
            (F128::from_raw(a) * F128::from_raw(b)).raw(),
            field::mul_reference_bitserial(a, b)
        );
    }
}

#[test]
fn transcript_observes_only_safe_messages() {
    let domain = b"transcript-discipline";
    let hint = F128::from_raw(0x1234);
    let message = F128::from_raw(0x5678);

    let mut baseline = FsChallenger::new(domain);
    let expected_without_message = baseline.sample_f128();

    let mut writer = ProofWriter::new(FsChallenger::new(domain));
    writer.write_unsafe_hint(hint);
    assert_eq!(writer.sample_f128(), expected_without_message);

    let mut writer = ProofWriter::new(FsChallenger::new(domain));
    writer.write_f128(message);
    assert_ne!(writer.sample_f128(), expected_without_message);

    let proof = ProofTranscript::new(vec![hint]);
    let mut reader = ProofReader::new(FsChallenger::new(domain), proof);
    assert_eq!(reader.read_unsafe_hint().unwrap(), hint);
    assert_eq!(reader.sample_f128(), expected_without_message);
}

#[test]
fn four_russians_96x128_matches_naive() {
    let rows: [u128; 128] = core::array::from_fn(|idx| row_mask(idx as u128));
    let normal = BooleanMatrix::from_rows_u128(96, &rows);
    let matrix = FourRussians128::from_boolean_matrix(&normal);
    let input = packed_input_96();
    let mut fast = vec![0u64; 128 * PACKED_U64S];
    let mut slow = vec![0u64; 128 * PACKED_U64S];
    matrix.apply(&input, &mut fast);
    naive_apply(&normal, &input, &mut slow);
    assert_eq!(fast, slow);
}

#[test]
fn four_russians_96x256_matches_naive() {
    let rows: [u128; 256] = core::array::from_fn(|idx| row_mask((idx * 17) as u128));
    let normal = BooleanMatrix::from_rows_u128(96, &rows);
    let matrix = FourRussians256::from_boolean_matrix(&normal);
    let input = packed_input_96();
    let mut fast = vec![0u64; 256 * PACKED_U64S];
    let mut slow = vec![0u64; 256 * PACKED_U64S];
    matrix.apply(&input, &mut fast);
    naive_apply(&normal, &input, &mut slow);
    assert_eq!(fast, slow);
}

#[test]
fn rmfe_subspace_validates() {
    assert_eq!(rmfe::validate_rmfe(), Ok(()));
    let (product, basis, embedding, projection) = rmfe::test_build_subspace().unwrap();
    assert_eq!(product.degree(), Some(rmfe::PRODUCT_DEGREE));
    assert!(basis.iter().all(|poly| poly.degree().unwrap() < rmfe::PRODUCT_DEGREE));
    assert_eq!(&embedding, rmfe::embedding_matrix());
    assert_eq!(&projection, rmfe::projection_matrix());
}

#[test]
fn rmfe_embedding_matrix_is_linear() {
    let a = 0x1234_5678_9abc_def0_1357_2468u128;
    let b = 0xfedc_ba98_7654_3210_aaaa_5555u128;
    assert_eq!(
        apply_embedding_matrix(a ^ b),
        apply_embedding_matrix(a) ^ apply_embedding_matrix(b),
    );
    assert_eq!(apply_embedding_matrix(0), BoolPoly::ZERO);
}

#[test]
fn rmfe_embedding_matrix_matches_basis() {
    let (_, basis, _, _) = rmfe::test_build_subspace().unwrap();
    for bit in [0usize, 1, 17, 63, 64, 95] {
        assert_eq!(apply_embedding_matrix(1u128 << bit), basis[bit]);
    }
}

#[test]
fn rmfe_embedding_matrix_works_with_effective_kernel() {
    let matrix = crate::matrix::FourRussians192::from_boolean_matrix(rmfe::embedding_matrix());
    let input_words = [0x1234_5678_9abc_def0_1357_2468u128, 0x5a5a, 0, (1u128 << 95) | 7];
    let mut input = vec![0u64; 96 * PACKED_U64S];
    for bit in 0..96 {
        let mut lanes = [0u128; 4];
        for lane in 0..4 {
            lanes[lane] = (input_words[lane] >> bit) & 1;
        }
        write_packed_block(&mut input[bit * PACKED_U64S..][..PACKED_U64S], lanes);
    }

    let mut out = vec![0u64; 192 * PACKED_U64S];
    matrix.apply(&input, &mut out);

    for lane in 0..4 {
        let expected = apply_embedding_matrix(input_words[lane]);
        let mut actual = BoolPoly::ZERO;
        for coeff in 0..192 {
            let block = read_packed_block(&out[coeff * PACKED_U64S..][..PACKED_U64S]);
            if block[lane] != 0 {
                actual ^= BoolPoly::from_limbs(bit_limb(coeff));
            }
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn rmfe_projection_has_multiplicative_friendly_property() {
    let (_, basis, _, _) = rmfe::test_build_subspace().unwrap();
    for i in 0..rmfe::RMFE_BITS {
        for j in 0..rmfe::RMFE_BITS {
            let product = boolpoly::clmul_192(basis[i], basis[j]);
            let projected = apply_projection_matrix(product);
            let expected = if i == j { basis[i] } else { BoolPoly::ZERO };
            assert_eq!(projected, expected, "failed at ({i}, {j})");
        }
    }
}

#[test]
fn boolpoly_clmul_192_matches_naive() {
    let a = BoolPoly::from_limbs([0x1234_5678_9abc_def0, 0x1111_2222_3333_4444, 0x8000_0000_0000_0001, 0]);
    let b = BoolPoly::from_limbs([0xfedc_ba98_7654_3210, 0x5555_aaaa_7777_9999, 0x0000_ffff_0000_ffff, 0]);
    assert_eq!(boolpoly::clmul_192(a, b).limbs(), naive_clmul_192(a, b).limbs());
}

fn row_mask(seed: u128) -> u128 {
    let mut x = seed ^ 0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03;
    let mut mask = 0u128;
    for _ in 0..24 {
        x = x.wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb);
        mask ^= 1u128 << ((x as usize) % 96);
    }
    mask
}

fn packed_input_96() -> Vec<u64> {
    let mut out = vec![0u64; 96 * PACKED_U64S];
    for idx in 0..96 {
        let a = (idx as u128).wrapping_mul(0x1000_0000_01) & ((1u128 << 96) - 1);
        write_packed_block(&mut out[idx * PACKED_U64S..][..PACKED_U64S], [
            a,
            (a.rotate_left(17) >> 32) & ((1u128 << 96) - 1),
            (a ^ 0xabc) & ((1u128 << 96) - 1),
            a.wrapping_mul(7) & ((1u128 << 96) - 1),
        ]);
    }
    out
}

fn naive_apply<const OUT: usize>(
    matrix: &BooleanMatrix<OUT>,
    input: &[u64],
    out: &mut [u64],
) {
    out.fill(0);
    for row_idx in 0..OUT {
        for bit in 0..matrix.input_len() {
            if matrix.get(row_idx, bit) {
                xor_packed_block(
                    &mut out[row_idx * PACKED_U64S..][..PACKED_U64S],
                    &input[bit * PACKED_U64S..][..PACKED_U64S],
                );
            }
        }
    }
}

fn write_packed_block(out: &mut [u64], values: [u128; 4]) {
    let words = [
        values[0] | (values[1] << 96),
        (values[1] >> 32) | (values[2] << 64),
        (values[2] >> 64) | (values[3] << 32),
    ];
    out[0] = words[0] as u64;
    out[1] = (words[0] >> 64) as u64;
    out[2] = words[1] as u64;
    out[3] = (words[1] >> 64) as u64;
    out[4] = words[2] as u64;
    out[5] = (words[2] >> 64) as u64;
}

fn read_packed_block(input: &[u64]) -> [u128; 4] {
    let words = [
        (input[0] as u128) | ((input[1] as u128) << 64),
        (input[2] as u128) | ((input[3] as u128) << 64),
        (input[4] as u128) | ((input[5] as u128) << 64),
    ];
    let mask = (1u128 << 96) - 1;
    [
        words[0] & mask,
        ((words[0] >> 96) | (words[1] << 32)) & mask,
        ((words[1] >> 64) | (words[2] << 64)) & mask,
        words[2] >> 32,
    ]
}

fn xor_packed_block(out: &mut [u64], rhs: &[u64]) {
    for idx in 0..PACKED_U64S {
        out[idx] ^= rhs[idx];
    }
}

fn apply_embedding_matrix(word: u128) -> BoolPoly {
    apply_matrix_to_poly(rmfe::embedding_matrix(), &[word as u64, (word >> 64) as u64])
}

fn apply_projection_matrix(product: boolpoly::WideBoolPoly) -> BoolPoly {
    apply_matrix_to_poly(rmfe::projection_matrix(), &product.limbs())
}

fn apply_matrix_to_poly<const OUT: usize>(matrix: &BooleanMatrix<OUT>, input: &[u64]) -> BoolPoly {
    assert_eq!(OUT, rmfe::PRODUCT_DEGREE);
    assert_eq!(input.len(), matrix.words_per_row());
    let mut limbs = [0u64; 4];
    for row in 0..OUT {
        let parity = matrix
            .row(row)
            .iter()
            .zip(input)
            .fold(0u32, |acc, (&a, &b)| acc ^ ((a & b).count_ones() & 1));
        if parity != 0 {
            limbs[row / 64] ^= 1u64 << (row % 64);
        }
    }
    BoolPoly::from_limbs(limbs)
}

fn bit_limb(bit: usize) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    limbs[bit / 64] = 1u64 << (bit % 64);
    limbs
}

fn naive_clmul_192(lhs: BoolPoly, rhs: BoolPoly) -> boolpoly::WideBoolPoly {
    let lhs = lhs.limbs();
    let rhs = rhs.limbs();
    let mut out = [0u64; 6];
    for i in 0..192 {
        if ((lhs[i / 64] >> (i % 64)) & 1) == 0 {
            continue;
        }
        for j in 0..192 {
            if ((rhs[j / 64] >> (j % 64)) & 1) != 0 {
                let bit = i + j;
                out[bit / 64] ^= 1u64 << (bit % 64);
            }
        }
    }
    boolpoly::WideBoolPoly::from_limbs(out)
}
