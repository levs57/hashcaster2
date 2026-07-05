use std::hint::black_box;
use std::time::{Duration, Instant};

use hashcaster2::{
    challenger::{FsChallenger, ProofReader, ProofWriter},
    chi_round::{
        prover::{ProverCfg as ChiProverCfg, ProverScratch as ChiScratch},
        verifier::HybridClaim,
    },
    field::F128,
    iota,
    linrounds::{
        self,
        prover::ProverCfg as LinroundProverCfg,
        Linround,
    },
    protocol_state::{self, PACKED_KECCAKS, PACKED_MASK, ProtocolState},
    util,
    verifier::GkrVerifierCfg,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let instances = arg_or(&args, 1, 16 * PACKED_KECCAKS).max(1);
    let runs = arg_or(&args, 2, 3).max(1);
    let workers = arg_or(
        &args,
        3,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
    )
    .max(1);
    let bucket_bits = arg_or(&args, 4, 7);

    let blocks = instances.div_ceil(PACKED_KECCAKS).max(1);
    let log_packed_instances = ceil_log2(blocks);
    let virtual_blocks = 1usize << log_packed_instances;
    let capacity = blocks * PACKED_KECCAKS;

    println!("Hashcaster2 main GKR benchmark");
    println!("instances:    {instances}");
    println!("blocks:       {blocks} x {PACKED_KECCAKS}");
    println!("capacity:     {capacity}");
    println!("virtual:      {virtual_blocks} blocks");
    println!("runs:         {runs}");
    println!("workers:      {workers}");
    println!("bucket bits:  {bucket_bits}");
    println!();

    let mut protocol = ProtocolState::new_blocks(blocks, workers);
    fill_inputs(protocol.witness.input_mut());
    let started = Instant::now();
    protocol.generate_keccak();
    let witness_gen = started.elapsed();

    let claim = output_claim(&protocol, log_packed_instances);
    let mut chi_scratch =
        ChiScratch::new_with_workers_and_bucket_bits(log_packed_instances, workers, bucket_bits);

    let mut sanity_writer = ProofWriter::new(FsChallenger::new(b"hashcaster2-main"));
    let expected = prove_gkr(
        &mut sanity_writer,
        &protocol,
        claim.clone(),
        &mut chi_scratch,
        log_packed_instances,
        None,
    );
    let proof = sanity_writer.into_proof();
    let mut reader = ProofReader::new(FsChallenger::new(b"hashcaster2-main"), proof);
    let actual = GkrVerifierCfg { log_packed_instances }
        .verify(&mut reader, claim.clone())
        .expect("main verifier");
    reader.finish().expect("proof consumed");
    assert_eq!(actual.ev, expected.ev);
    assert_eq!(
        actual.ev,
        state_claim_eval(
            protocol.witness.input(),
            protocol.witness.blocks(),
            false,
            &actual,
        ),
    );

    let mut prove_times = Vec::with_capacity(runs);
    let mut verify_times = Vec::with_capacity(runs);
    let mut profiles = Vec::with_capacity(runs);
    let mut proof_elems = 0usize;
    let mut guard = 0u128;
    for run in 0..runs {
        let mut writer = ProofWriter::new(FsChallenger::new(b"hashcaster2-main"));
        let mut profile = MainProfile::default();
        let started = Instant::now();
        let out = prove_gkr(
            &mut writer,
            &protocol,
            claim.clone(),
            &mut chi_scratch,
            log_packed_instances,
            Some(&mut profile),
        );
        let prove_time = started.elapsed();
        let proof = writer.into_proof();
        proof_elems = proof.len();

        let mut reader = ProofReader::new(FsChallenger::new(b"hashcaster2-main"), proof);
        let started = Instant::now();
        let checked = GkrVerifierCfg { log_packed_instances }
            .verify(&mut reader, claim.clone())
            .expect("main verifier");
        reader.finish().expect("proof consumed");
        let verify_time = started.elapsed();

        assert_eq!(checked.ev, out.ev);
        guard ^= out.ev.raw().rotate_left((run % 127) as u32);
        prove_times.push(prove_time);
        verify_times.push(verify_time);
        profiles.push(profile);
    }

    prove_times.sort_unstable();
    verify_times.sort_unstable();
    let prove_median = median(&prove_times);
    let verify_median = median(&verify_times);
    let total = witness_gen + prove_median;
    let rate = instances as f64 / total.as_secs_f64();

    println!("witness gen:   {}", fmt_duration(witness_gen));
    println!("prove median:  {}", fmt_duration(prove_median));
    print_phase("  chi", &profiles, |p| p.chi);
    print_phase("  linrounds", &profiles, |p| p.linrounds);
    println!("verify median: {}", fmt_duration(verify_median));
    println!("proof elems:   {proof_elems}");
    println!("total median:  {}", fmt_duration(total));
    println!("rate:          {:.2} Keccak/s", rate);
    println!("projected f:   {:.2} Keccak-f[1600]/s", rate / 24.0);
    println!("guard:         {:032x}", black_box(guard));
}

fn prove_gkr(
    ctx: &mut ProofWriter<FsChallenger>,
    protocol: &ProtocolState,
    mut claim: HybridClaim,
    chi_scratch: &mut ChiScratch,
    log_packed_instances: usize,
    mut profile: Option<&mut MainProfile>,
) -> HybridClaim {
    for round in (0..protocol_state::KECCAK_ROUNDS).rev() {
        claim = iota::VerifierCfg {
            log_packed_instances,
            round,
        }
        .verify(claim);

        let started = Instant::now();
        let chi = ChiProverCfg {
            log_packed_instances,
            round,
        }
        .prove(ctx, &protocol.witness, claim, chi_scratch);
        if let Some(profile) = profile.as_deref_mut() {
            profile.chi += started.elapsed();
        }

        let started = Instant::now();
        let mut theta_state = [F128::ZERO; protocol_state::KECCAK_BITS];
        let mut prev_state = [F128::ZERO; protocol_state::KECCAK_BITS];
        linrounds::apply_inverse(Linround::RhoPi, &chi.pre_chi_state, &mut theta_state);
        linrounds::apply_inverse(Linround::Theta, &theta_state, &mut prev_state);

        claim = LinroundProverCfg { round: Linround::RhoPi }
            .prove(ctx, chi.claim, &theta_state);
        claim = LinroundProverCfg { round: Linround::Theta }
            .prove(ctx, claim, &prev_state);
        if let Some(profile) = profile.as_deref_mut() {
            profile.linrounds += started.elapsed();
        }
    }
    claim
}

#[derive(Clone, Copy, Default)]
struct MainProfile {
    chi: Duration,
    linrounds: Duration,
}

fn output_claim(protocol: &ProtocolState, log_packed_instances: usize) -> HybridClaim {
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
        r_out: (0..log_packed_instances)
            .map(|idx| F128::from_raw(47 + 2 * idx as u128))
            .collect(),
        ev: F128::ZERO,
    };
    claim.ev = state_claim_eval(
        protocol.witness.output(),
        protocol.witness.blocks(),
        true,
        &claim,
    );
    claim
}

fn state_claim_eval(
    state: &[u128],
    blocks: usize,
    use_zero_output: bool,
    claim: &HybridClaim,
) -> F128 {
    let eq_x = util::eq_poly_v(&claim.r_x);
    let eq_y = util::eq_poly_v(&claim.r_y);
    let eq_z = util::eq_poly_v(&claim.r_z);
    let eq_out = util::eq_poly_v(&claim.r_out);
    let tables = util::rmfe_byte_eval_tables(claim.t);
    let mut out = F128::ZERO;
    for block in 0..eq_out.len() {
        let out_scale = eq_out[block];
        if out_scale == F128::ZERO {
            continue;
        }
        for x in 0..5 {
            for y in 0..5 {
                let xy = out_scale * eq_x[x] * eq_y[y];
                for z in 0..64 {
                    let word = if block < blocks {
                        state[block * protocol_state::KECCAK_BITS + protocol_state::state_idx(x, y, z)]
                    } else if use_zero_output {
                        protocol_state::zero_output_word(x, y, z)
                    } else {
                        0
                    };
                    if word != 0 {
                        out += xy * eq_z[z] * util::eval_packed_word(word, &tables);
                    }
                }
            }
        }
    }
    out
}

fn fill_inputs(input: &mut [u128]) {
    let mut x = 0x1234_5678_9abc_def0_1357_2468_ace0_bdf1u128;
    for (idx, value) in input.iter_mut().enumerate() {
        x = x
            .wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb)
            .rotate_left(37);
        *value = (x ^ (idx as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15)) & PACKED_MASK;
    }
}

fn arg_or(args: &[String], idx: usize, default: usize) -> usize {
    args.get(idx)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize
    }
}

fn median(values: &[Duration]) -> Duration {
    values[values.len() / 2]
}

fn print_phase(name: &str, profiles: &[MainProfile], get: impl Fn(&MainProfile) -> Duration) {
    let mut values: Vec<Duration> = profiles.iter().map(get).collect();
    values.sort_unstable();
    println!("{name:<14} {}", fmt_duration(median(&values)));
}

fn fmt_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.3} s")
    } else if secs >= 1e-3 {
        format!("{:.3} ms", secs * 1e3)
    } else if secs >= 1e-6 {
        format!("{:.3} us", secs * 1e6)
    } else {
        format!("{:.3} ns", secs * 1e9)
    }
}
