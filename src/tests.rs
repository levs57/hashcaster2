use crate::challenger::{Challenger, FsChallenger, ProofReader, ProofTranscript, ProofWriter};
use crate::field::{self, F128};
use crate::matrix::{FourRussians128, FourRussians256, Packed4x96};
use crate::rmfe::{self, BoolPoly};

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
    let matrix = FourRussians128::from_rows_96x128(&rows);
    let input = packed_input_96();
    let mut fast = [Packed4x96::default(); 128];
    let mut slow = [Packed4x96::default(); 128];
    matrix.apply(&input, &mut fast);
    naive_apply(&rows, &input, &mut slow);
    assert_eq!(fast, slow);
}

#[test]
fn four_russians_96x256_matches_naive() {
    let rows: [u128; 256] = core::array::from_fn(|idx| row_mask((idx * 17) as u128));
    let matrix = FourRussians256::from_rows_96x256(&rows);
    let input = packed_input_96();
    let mut fast = [Packed4x96::default(); 256];
    let mut slow = [Packed4x96::default(); 256];
    matrix.apply(&input, &mut fast);
    naive_apply(&rows, &input, &mut slow);
    assert_eq!(fast, slow);
}

#[test]
fn rmfe_subspace_validates() {
    assert_eq!(rmfe::validate_rmfe(), Ok(()));
    assert_eq!(rmfe::basis_elements().len(), rmfe::RMFE_BITS);
    assert_eq!(rmfe::product_modulus().degree(), Some(rmfe::PRODUCT_DEGREE));
    assert!(rmfe::basis_elements()
        .iter()
        .all(|poly| poly.degree().unwrap() < rmfe::PRODUCT_DEGREE));
}

#[test]
fn rmfe_embedding_is_linear() {
    let a = 0x1234_5678_9abc_def0_1357_2468u128;
    let b = 0xfedc_ba98_7654_3210_aaaa_5555u128;
    assert_eq!(rmfe::embed_word(a ^ b), rmfe::embed_word(a) ^ rmfe::embed_word(b));
    assert_eq!(rmfe::embed_word(0), BoolPoly::ZERO);
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

fn packed_input_96() -> [Packed4x96; 96] {
    core::array::from_fn(|idx| {
        let a = (idx as u128).wrapping_mul(0x1000_0000_01) & ((1u128 << 96) - 1);
        Packed4x96::pack([
            a,
            (a.rotate_left(17) >> 32) & ((1u128 << 96) - 1),
            (a ^ 0xabc) & ((1u128 << 96) - 1),
            a.wrapping_mul(7) & ((1u128 << 96) - 1),
        ])
    })
}

fn naive_apply<const OUT: usize>(
    rows: &[u128; OUT],
    input: &[Packed4x96],
    out: &mut [Packed4x96; OUT],
) {
    for row_idx in 0..OUT {
        let mut mask = rows[row_idx];
        let mut acc = Packed4x96::default();
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            acc.xor_assign(input[bit]);
            mask &= mask - 1;
        }
        out[row_idx] = acc;
    }
}
