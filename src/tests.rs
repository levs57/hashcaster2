use crate::boolpoly::{self, BoolPoly};
use crate::challenger::{Challenger, FsChallenger, ProofReader, ProofTranscript, ProofWriter};
use crate::chi_round::{
    prover::{ProverCfg, ProverScratch},
    verifier::{HybridClaim, VerifierCfg},
};
use crate::field::{self, F128};
use crate::linrounds::{
    self,
    prover::ProverCfg as LinroundProverCfg,
    verifier::VerifierCfg as LinroundVerifierCfg,
    Linround,
};
use crate::matrix::{BooleanMatrix, FourRussians128, FourRussians256, PACKED_U64S};
use crate::protocol_state::{self, KeccakWitness, ProtocolState};
use crate::{rmfe, util};

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
    rmfe::validate_rmfe();
    let (product, basis, embedding, projection) = rmfe::test_build_subspace();
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
    let (_, basis, _, _) = rmfe::test_build_subspace();
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
    let (_, basis, _, _) = rmfe::test_build_subspace();
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

#[test]
fn packed_keccak_witness_matches_scalar_keccak() {
    let mut scalar_states = [[0u64; protocol_state::KECCAK_LANES]; protocol_state::PACKED_KECCAKS];
    let mut initial = [0u128; protocol_state::KECCAK_BITS];

    let mut rng = 0x1234_5678_9abc_def0_1357_2468_ace0_bdf1u128;
    for instance in 0..protocol_state::PACKED_KECCAKS {
        for lane in 0..protocol_state::KECCAK_LANES {
            rng = rng
                .wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb)
                .rotate_left(37);
            scalar_states[instance][lane] = rng as u64;
        }
    }

    for instance in 0..protocol_state::PACKED_KECCAKS {
        for y in 0..5 {
            for x in 0..5 {
                let lane = scalar_states[instance][x + 5 * y];
                for z in 0..64 {
                    if ((lane >> z) & 1) != 0 {
                        initial[protocol_state::state_idx(x, y, z)] |= 1u128 << instance;
                    }
                }
            }
        }
    }

    let mut protocol = ProtocolState::new();
    protocol.witness.input_mut().copy_from_slice(&initial);
    protocol.generate_keccak();
    assert_eq!(protocol.witness.input(), &initial);

    let mut pre_chi_trace =
        [[[0u64; protocol_state::KECCAK_LANES]; protocol_state::KECCAK_ROUNDS];
            protocol_state::PACKED_KECCAKS];
    for (instance, state) in scalar_states.iter_mut().enumerate() {
        protocol_state::keccak_f_lanes_with_pre_chi(state, &mut pre_chi_trace[instance]);
    }

    for instance in 0..protocol_state::PACKED_KECCAKS {
        for round in 0..protocol_state::KECCAK_ROUNDS {
            for y in 0..5 {
                for x in 0..5 {
                    let lane = pre_chi_trace[instance][round][x + 5 * y];
                    for z in 0..64 {
                        let packed = protocol.witness.pre_chi_word(0, round, x, y, z);
                        assert_eq!(
                            ((packed >> instance) & 1) as u64,
                            (lane >> z) & 1,
                            "pre-chi round {round}, instance {instance}, lane ({x}, {y}), bit {z}",
                        );
                    }
                }
            }
        }
    }

    let final_state = protocol.witness.output();
    for instance in 0..protocol_state::PACKED_KECCAKS {
        for y in 0..5 {
            for x in 0..5 {
                let lane = scalar_states[instance][x + 5 * y];
                for z in 0..64 {
                    let packed = final_state[protocol_state::state_idx(x, y, z)];
                    assert_eq!(
                        ((packed >> instance) & 1) as u64,
                        (lane >> z) & 1,
                        "instance {instance}, lane ({x}, {y}), bit {z}",
                    );
                }
            }
        }
    }
}

#[test]
fn chi_round_prover_passes_verifier_on_zero_witness() {
    let witness = ProtocolState::new_blocks(1, 1).witness;
    let claim = HybridClaim {
        t: F128::from_raw(17),
        r_x: vec![F128::from_raw(3), F128::from_raw(5), F128::from_raw(7)],
        r_y: vec![F128::from_raw(11), F128::from_raw(13), F128::from_raw(19)],
        r_z: vec![
            F128::from_raw(23),
            F128::from_raw(29),
            F128::from_raw(31),
            F128::from_raw(37),
            F128::from_raw(41),
            F128::from_raw(43),
        ],
        r_out: Vec::new(),
        ev: F128::ZERO,
    };
    let prover_cfg = ProverCfg {
        log_packed_instances: 0,
        round: 0,
    };
    let verifier_cfg = VerifierCfg {
        log_packed_instances: 0,
    };

    let mut scratch = ProverScratch::new(0);
    let mut writer = ProofWriter::new(FsChallenger::new(b"chi-zero"));
    let expected = prover_cfg.prove(&mut writer, &witness, claim.clone(), &mut scratch);
    let proof = writer.into_proof();

    let mut reader = ProofReader::new(FsChallenger::new(b"chi-zero"), proof);
    let actual = verifier_cfg.verify(&mut reader, claim).unwrap();
    reader.finish().unwrap();
    assert_eq!(actual.t, expected.claim.t);
    assert_eq!(actual.ev, expected.claim.ev);
    assert_eq!(actual.r_x, expected.claim.r_x);
    assert_eq!(actual.r_y, expected.claim.r_y);
    assert_eq!(actual.r_z, expected.claim.r_z);
    assert_eq!(actual.r_out, expected.claim.r_out);
}

#[test]
fn chi_round_prover_passes_verifier_with_virtual_padding() {
    let mut protocol = ProtocolState::new_blocks(1, 1);
    protocol.witness.set_pre_chi_word(0, 0, 1, 0, 0, 1);
    protocol.witness.set_pre_chi_word(0, 0, 2, 0, 0, 1);

    let mut claim = HybridClaim {
        t: F128::from_raw(17),
        r_x: vec![F128::from_raw(3), F128::from_raw(5), F128::from_raw(7)],
        r_y: vec![F128::from_raw(11), F128::from_raw(13), F128::from_raw(19)],
        r_z: vec![
            F128::from_raw(23),
            F128::from_raw(29),
            F128::from_raw(31),
            F128::from_raw(37),
            F128::from_raw(41),
            F128::from_raw(43),
        ],
        r_out: vec![F128::from_raw(47)],
        ev: F128::ZERO,
    };
    let prover_cfg = ProverCfg {
        log_packed_instances: 1,
        round: 0,
    };
    let verifier_cfg = VerifierCfg {
        log_packed_instances: 1,
    };

    let mut scratch = ProverScratch::new(1);
    claim.ev = slow_chi_initial_claim(&protocol.witness, 0, 1, &claim);
    let mut writer = ProofWriter::new(FsChallenger::new(b"chi-padding"));
    let expected = prover_cfg.prove(&mut writer, &protocol.witness, claim.clone(), &mut scratch);
    let proof = writer.into_proof();

    let mut reader = ProofReader::new(FsChallenger::new(b"chi-padding"), proof);
    let actual = verifier_cfg.verify(&mut reader, claim).unwrap();
    reader.finish().unwrap();
    assert_eq!(actual.t, expected.claim.t);
    assert_eq!(actual.ev, expected.claim.ev);
    assert_eq!(actual.r_x, expected.claim.r_x);
    assert_eq!(actual.r_y, expected.claim.r_y);
    assert_eq!(actual.r_z, expected.claim.r_z);
    assert_eq!(actual.r_out, expected.claim.r_out);
}

#[test]
fn linround_prover_passes_verifier() {
    for round in [Linround::Theta, Linround::RhoPi, Linround::ThetaRhoPi] {
        let mut input = [F128::ZERO; protocol_state::KECCAK_BITS];
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    let idx = protocol_state::state_idx(x, y, z);
                    input[idx] = F128::from_raw((idx as u128 + 1).wrapping_mul(17));
                }
            }
        }

        let mut output = [F128::ZERO; protocol_state::KECCAK_BITS];
        linrounds::apply(round, &input, &mut output);

        let mut claim = HybridClaim {
            t: F128::from_raw(17),
            r_x: vec![F128::from_raw(3), F128::from_raw(5), F128::from_raw(7)],
            r_y: vec![F128::from_raw(11), F128::from_raw(13), F128::from_raw(19)],
            r_z: vec![
                F128::from_raw(23),
                F128::from_raw(29),
                F128::from_raw(31),
                F128::from_raw(37),
                F128::from_raw(41),
                F128::from_raw(43),
            ],
            r_out: vec![F128::from_raw(47), F128::from_raw(53)],
            ev: F128::ZERO,
        };
        claim.ev = physical_state_eval(&output, &claim.r_x, &claim.r_y, &claim.r_z);

        let mut writer = ProofWriter::new(FsChallenger::new(b"linround-test"));
        let expected = LinroundProverCfg { round }.prove(&mut writer, claim.clone(), &input);
        let proof = writer.into_proof();
        let mut reader = ProofReader::new(FsChallenger::new(b"linround-test"), proof);
        let actual = LinroundVerifierCfg { round }
            .verify(&mut reader, claim)
            .expect("linround verifier");
        reader.finish().expect("proof consumed");

        assert_eq!(actual.ev, expected.ev);
        assert_eq!(actual.r_x, expected.r_x);
        assert_eq!(actual.r_y, expected.r_y);
        assert_eq!(actual.r_z, expected.r_z);
    }
}

#[test]
fn linround_inverse_roundtrip() {
    for round in [Linround::Theta, Linround::RhoPi, Linround::ThetaRhoPi] {
        let mut input = [F128::ZERO; protocol_state::KECCAK_BITS];
        for idx in 0..protocol_state::KECCAK_BITS {
            input[idx] = F128::from_raw((idx as u128 + 13).wrapping_mul(0x101));
        }

        let mut image = [F128::ZERO; protocol_state::KECCAK_BITS];
        linrounds::apply(round, &input, &mut image);

        let mut recovered = [F128::ZERO; protocol_state::KECCAK_BITS];
        linrounds::apply_inverse(round, &image, &mut recovered);
        assert_eq!(recovered, input);
    }
}

fn slow_chi_initial_claim(
    witness: &KeccakWitness,
    round: usize,
    log_packed_instances: usize,
    claim: &HybridClaim,
) -> F128 {
    let eq_x = util::eq_poly_v(&claim.r_x);
    let eq_y = util::eq_poly_v(&claim.r_y);
    let eq_z = util::eq_poly_v(&claim.r_z);
    let eq_out = util::eq_poly_v(&claim.r_out);
    let mut u = vec![F128::ZERO; rmfe::PRODUCT_BITS];

    for out in 0..(1usize << log_packed_instances) {
        for y in 0..5 {
            for z in 0..64 {
                let eq_yzout = eq_out[out] * eq_y[y] * eq_z[z];
                for x in 0..5 {
                    let scalar = eq_yzout * eq_x[x];
                    let left = slow_embed_word(slow_state_word(
                        witness,
                        round,
                        out,
                        (x + 1) % 5,
                        y,
                        z,
                    ));
                    let right = slow_embed_word(slow_state_word(
                        witness,
                        round,
                        out,
                        (x + 2) % 5,
                        y,
                        z,
                    ));
                    let center = slow_embed_word(slow_state_word(witness, round, out, x, y, z));
                    let c: [F128; rmfe::PRODUCT_DEGREE] =
                        core::array::from_fn(|idx| center[idx] + right[idx]);
                    slow_add_product(&mut u, &left, &right, scalar);
                    slow_add_product(&mut u, &c, &c, scalar);
                }
            }
        }
    }

    let projected = util::apply_boolean_matrix(rmfe::projection_matrix(), &u);
    util::evaluate_univar(&projected, claim.t)
}

fn slow_state_word(
    witness: &KeccakWitness,
    round: usize,
    out: usize,
    x: usize,
    y: usize,
    z: usize,
) -> u128 {
    if out < witness.blocks() {
        witness.pre_chi_word(out, round, x, y, z)
    } else {
        protocol_state::zero_pre_chi_word(round, x, y, z)
    }
}

fn slow_embed_word(word: u128) -> [F128; rmfe::PRODUCT_DEGREE] {
    let bits: [F128; rmfe::RMFE_BITS] = core::array::from_fn(|bit| {
        if ((word >> bit) & 1) == 0 {
            F128::ZERO
        } else {
            F128::ONE
        }
    });
    util::apply_boolean_matrix(rmfe::embedding_matrix(), &bits)
}

fn slow_add_product(
    out: &mut [F128],
    lhs: &[F128; rmfe::PRODUCT_DEGREE],
    rhs: &[F128; rmfe::PRODUCT_DEGREE],
    scalar: F128,
) {
    if scalar == F128::ZERO {
        return;
    }
    for i in 0..rmfe::PRODUCT_DEGREE {
        if lhs[i] == F128::ZERO {
            continue;
        }
        for j in 0..rmfe::PRODUCT_DEGREE {
            out[i + j] += lhs[i] * rhs[j] * scalar;
        }
    }
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

fn physical_state_eval(state: &[F128], r_x: &[F128], r_y: &[F128], r_z: &[F128]) -> F128 {
    let eq_x = util::eq_poly_v(r_x);
    let eq_y = util::eq_poly_v(r_y);
    let eq_z = util::eq_poly_v(r_z);
    let mut out = F128::ZERO;
    for x in 0..5 {
        for y in 0..5 {
            let xy = eq_x[x] * eq_y[y];
            for z in 0..64 {
                out += xy * eq_z[z] * state[protocol_state::state_idx(x, y, z)];
            }
        }
    }
    out
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
