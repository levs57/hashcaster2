use std::hint::black_box;
use std::time::{Duration, Instant};

use hashcaster2::protocol_state::{KECCAK_BITS, PACKED_KECCAKS, PACKED_MASK, ProtocolState};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_keccaks = arg_or_env_usize(&args, 1, "KECCAK_WITNESS_TARGET", 200_000);
    let runs = arg_or_env_usize(&args, 2, "KECCAK_WITNESS_RUNS", 5).max(1);
    let all_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let thread_counts = thread_counts(all_threads);
    let blocks = target_keccaks.div_ceil(PACKED_KECCAKS).max(1);
    let capacity = blocks * PACKED_KECCAKS;

    println!("Packed Keccak witness generation sweep");
    println!("target keccaks: {target_keccaks}");
    println!("packed blocks:  {blocks} ({PACKED_KECCAKS} keccaks/block)");
    println!("capacity:       {capacity} keccaks");
    println!("runs:           {runs}");
    println!("witness memory: {:.2} MiB", witness_bytes(blocks) as f64 / mib());
    println!();

    for workers in thread_counts {
        let mut protocol = ProtocolState::new_for_keccaks(target_keccaks, workers);
        fill_inputs(protocol.witness.input_mut());

        let mut timings = Vec::with_capacity(runs);
        let mut guard = 0u128;
        for run in 0..runs {
            let started = Instant::now();
            protocol.generate_keccak();
            let elapsed = started.elapsed();
            timings.push(elapsed);
            guard ^= checksum(protocol.witness.output()).rotate_left((run % 127) as u32);
        }

        timings.sort_unstable();
        let best = timings[0];
        let median = median_duration(&timings);
        let avg = Duration::from_secs_f64(
            timings.iter().map(|t| t.as_secs_f64()).sum::<f64>() / timings.len() as f64,
        );
        println!("workers:        {workers}");
        println!("  scratch:      {:.2} KiB", scratch_bytes(workers) as f64 / 1024.0);
        println!("  best:         {}", fmt_duration(best));
        println!("  median:       {}", fmt_duration(median));
        println!("  average:      {}", fmt_duration(avg));
        println!(
            "  median rate:  {:.2} Kkeccak/s",
            capacity as f64 / median.as_secs_f64() / 1e3,
        );
        println!(
            "  payload:      {:.2} Gbit/s",
            capacity as f64 * 1600.0 / median.as_secs_f64() / 1e9,
        );
        println!("  guard:        {:032x}", black_box(guard));
        println!();
    }
}

fn fill_inputs(input: &mut [u128]) {
    let mut x = 0x1234_5678_9abc_def0_1357_2468_ace0_bdf1u128;
    for value in input {
        x = x
            .wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb)
            .rotate_left(37);
        *value = x & PACKED_MASK;
    }
}

fn checksum(values: &[u128]) -> u128 {
    let mut out = 0u128;
    for (idx, &value) in values.iter().enumerate().step_by(37) {
        out ^= value.rotate_left((idx % 127) as u32);
    }
    out
}

fn thread_counts(all_threads: usize) -> Vec<usize> {
    if let Ok(raw) = std::env::var("KECCAK_WITNESS_THREADS") {
        let mut counts: Vec<usize> = raw
            .split([',', ' '])
            .filter(|part| !part.is_empty())
            .map(|part| part.parse().expect("invalid KECCAK_WITNESS_THREADS entry"))
            .collect();
        counts.sort_unstable();
        counts.dedup();
        return counts;
    }

    let mut counts = Vec::new();
    let mut n = 1usize;
    while n < all_threads {
        counts.push(n);
        n *= 2;
    }
    counts.push(all_threads);
    counts.sort_unstable();
    counts.dedup();
    counts
}

fn witness_bytes(blocks: usize) -> usize {
    let witness_words = blocks * KECCAK_BITS * (2 + 24);
    witness_words * core::mem::size_of::<u128>()
}

fn scratch_bytes(workers: usize) -> usize {
    workers * KECCAK_BITS * core::mem::size_of::<u128>()
}

fn mib() -> f64 {
    1024.0 * 1024.0
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
