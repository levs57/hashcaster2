use std::hint::black_box;
use std::time::{Duration, Instant};

use hashcaster2::challenger::{FsChallenger, ProofReader, ProofWriter};
use hashcaster2::chi_round::{
    prover::{ProverCfg, ProverProfile, ProverScratch},
    verifier::{HybridClaim, VerifierCfg},
};
use hashcaster2::field::F128;
use hashcaster2::protocol_state::{PACKED_KECCAKS, PACKED_MASK, ProtocolState};
use hashcaster2::{rmfe, util};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let instances = arg_or_env_usize(
        &args,
        1,
        "CHI_PROVER_INSTANCES",
        16 * PACKED_KECCAKS,
    )
    .max(1);
    let runs = arg_or_env_usize(&args, 2, "CHI_PROVER_RUNS", 5).max(1);
    let round = arg_or_env_usize(&args, 3, "CHI_PROVER_ROUND", 0);
    let default_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let workers = arg_or_env_usize(&args, 4, "CHI_PROVER_WORKERS", default_workers).max(1);
    let bucket_bits = bucket_bits_arg(&args);

    println!("Chi-round prover sweep");
    println!("instances:    {instances}");
    println!("runs:         {runs}");
    println!("round:        {round}");
    println!("workers:      {workers}");
    println!("buckets:      {:?}", bucket_bits);
    println!();

    for bucket_bits in bucket_bits {
        run_case(instances, runs, round, workers, bucket_bits);
    }
}

fn run_case(instances: usize, runs: usize, round: usize, workers: usize, bucket_bits: usize) {
    let blocks = instances.div_ceil(PACKED_KECCAKS);
    let log_packed_instances = ceil_log2(blocks);
    let virtual_blocks = 1usize << log_packed_instances;
    let capacity = blocks * PACKED_KECCAKS;
    let virtual_capacity = virtual_blocks * PACKED_KECCAKS;
    let mut protocol = ProtocolState::new_blocks(blocks, 1);
    fill_inputs(protocol.witness.input_mut());
    protocol.generate_keccak();

    let prover_cfg = ProverCfg {
        log_packed_instances,
        round,
    };
    let verifier_cfg = VerifierCfg {
        log_packed_instances,
    };
    let claim = initial_claim(log_packed_instances);
    let mut scratch =
        ProverScratch::new_with_workers_and_bucket_bits(log_packed_instances, workers, bucket_bits);

    let mut sanity_writer = ProofWriter::with_capacity(
        FsChallenger::new(b"chi-prover-sweep"),
        proof_capacity(log_packed_instances),
    );
    let expected = prover_cfg.prove(
        &mut sanity_writer,
        &protocol.witness,
        claim.clone(),
        &mut scratch,
    );
    let proof = sanity_writer.into_proof();
    let mut verifier_claim = claim.clone();
    let pi_u = util::apply_boolean_matrix(
        rmfe::projection_matrix(),
        &proof.elements()[..rmfe::PRODUCT_BITS],
    );
    verifier_claim.ev = util::evaluate_univar(&pi_u, verifier_claim.t);
    let mut reader = ProofReader::new(FsChallenger::new(b"chi-prover-sweep"), proof);
    let actual = verifier_cfg.verify(&mut reader, verifier_claim).expect("sanity verifier");
    reader.finish().expect("sanity proof consumed");
    assert_eq!(actual.ev, expected.claim.ev);

    let mut timings = Vec::with_capacity(runs);
    let mut profiles = Vec::with_capacity(runs);
    let mut guard = 0u128;
    for run in 0..runs {
        let mut writer = ProofWriter::with_capacity(
            FsChallenger::new(b"chi-prover-sweep"),
            proof_capacity(log_packed_instances),
        );
        let mut profile = ProverProfile::default();
        let started = Instant::now();
        let out = prover_cfg.prove_profiled(
            &mut writer,
            &protocol.witness,
            claim.clone(),
            &mut scratch,
            &mut profile,
        );
        let elapsed = started.elapsed();
        let proof = writer.into_proof();
        timings.push(elapsed);
        profiles.push(profile);
        guard ^= checksum(proof.elements(), out.claim.ev).rotate_left((run % 127) as u32);
    }
    let mut sorted_timings = timings.clone();
    sorted_timings.sort_unstable();
    let best = sorted_timings[0];
    let median = median_duration(&sorted_timings);
    let average = Duration::from_secs_f64(
        timings.iter().map(|timing| timing.as_secs_f64()).sum::<f64>() / timings.len() as f64,
    );

    println!("requested:");
    println!("  bucket bits:  {bucket_bits}");
    println!("  instances:    {instances}");
    println!("  blocks:       {blocks} x {PACKED_KECCAKS}");
    println!("  capacity:     {capacity} instances");
    println!("  virtual:      {virtual_blocks} blocks ({virtual_capacity} instances)");
    println!(
        "  virtual/real: {:.3}x blocks",
        virtual_blocks as f64 / blocks as f64,
    );
    println!(
        "  padding:      {} real + {} packed + {} virtual instances",
        instances,
        capacity - instances,
        virtual_capacity - capacity,
    );
    println!("  log_out:      {log_packed_instances}");
    println!("  workers:      {}", scratch.workers());
    println!("  scratch bits: {}", scratch.bucket_bits());
    println!("  proof elems:  {}", proof_capacity(log_packed_instances));
    println!("  best:         {}", fmt_duration(best));
    println!("  median:       {}", fmt_duration(median));
    println!("  average:      {}", fmt_duration(average));
    print_phase("build U", &profiles, |p| p.build_u);
    print_phase("  clear U", &profiles, |p| p.build_u_clear);
    print_phase("  acc U", &profiles, |p| p.build_u_accumulate);
    print_phase("  merge U", &profiles, |p| p.build_u_merge);
    print_phase("  recover U", &profiles, |p| p.build_u_recover);
    print_phase("setup", &profiles, |p| p.setup);
    print_phase("out messages", &profiles, |p| p.out_messages);
    print_phase("out folds", &profiles, |p| p.out_folds);
    print_phase("  first msg", &profiles, |p| p.out_first_message);
    print_phase("  first fold", &profiles, |p| p.out_first_fold);
    print_phase("  later msgs", &profiles, |p| p.out_later_messages);
    print_phase("  later folds", &profiles, |p| p.out_later_folds);
    print_phase("expand", &profiles, |p| p.expand);
    print_phase("tail messages", &profiles, |p| p.tail_messages);
    print_phase("tail folds", &profiles, |p| p.tail_folds);
    print_phase("final", &profiles, |p| p.final_claim);
    let median_rate = instances as f64 / median.as_secs_f64();
    println!(
        "  median rate:  {:.2} Keccak/s",
        median_rate,
    );
    println!(
        "  projected f:  {:.2} Keccak-f[1600]/s",
        median_rate / 24.0,
    );
    println!(
        "  capacity rate:{:>9.2} Keccak/s",
        capacity as f64 / median.as_secs_f64(),
    );
    println!("  guard:        {:032x}", black_box(guard));
    println!();
}

fn print_phase(name: &str, profiles: &[ProverProfile], get: impl Fn(&ProverProfile) -> Duration) {
    let mut values: Vec<Duration> = profiles.iter().map(get).collect();
    values.sort_unstable();
    println!(
        "  {name:<13} best {:>10} | median {:>10}",
        fmt_duration(values[0]),
        fmt_duration(median_duration(&values)),
    );
}

fn initial_claim(log_packed_instances: usize) -> HybridClaim {
    HybridClaim {
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
    }
}

fn proof_capacity(log_packed_instances: usize) -> usize {
    rmfe::PRODUCT_BITS + 2 * (log_packed_instances + 3 + 6) + 5
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

fn checksum(proof: &[F128], ev: F128) -> u128 {
    let mut out = ev.raw();
    for (idx, value) in proof.iter().enumerate().step_by(17) {
        out ^= value.raw().rotate_left((idx % 127) as u32);
    }
    out
}

fn ceil_log2(value: usize) -> usize {
    assert!(value > 0);
    usize::BITS as usize - (value - 1).leading_zeros() as usize
}

fn median_duration(sorted: &[Duration]) -> Duration {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        Duration::from_secs_f64((sorted[mid - 1].as_secs_f64() + sorted[mid].as_secs_f64()) * 0.5)
    }
}

fn fmt_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 1e-6 {
        format!("{:.3} ns", secs * 1e9)
    } else if secs < 1e-3 {
        format!("{:.3} us", secs * 1e6)
    } else if secs < 1.0 {
        format!("{:.3} ms", secs * 1e3)
    } else {
        format!("{:.3} s", secs)
    }
}

fn arg_or_env_usize(args: &[String], index: usize, env: &str, default: usize) -> usize {
    args.get(index)
        .cloned()
        .or_else(|| std::env::var(env).ok())
        .map(|value| value.parse().expect("expected usize"))
        .unwrap_or(default)
}

fn bucket_bits_arg(args: &[String]) -> Vec<usize> {
    let raw = args
        .get(5)
        .cloned()
        .or_else(|| std::env::var("CHI_PROVER_BUCKET_BITS").ok())
        .unwrap_or_else(|| "8,7,6".to_string());
    raw.split(',')
        .map(|part| {
            let bits = part.trim().parse::<usize>().expect("bucket bits");
            assert!((6..=8).contains(&bits));
            bits
        })
        .collect()
}
