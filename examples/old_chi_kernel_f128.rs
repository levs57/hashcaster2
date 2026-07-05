use std::hint::black_box;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use hashcaster2::boolpoly::{self, BoolPoly, WideBoolPoly};
use hashcaster2::field::F128;
use hashcaster2::{protocol_state, rmfe};
use rayon::prelude::*;

const PACKED_BITS: usize = protocol_state::PACKED_KECCAKS;
const WORD_BYTES: usize = rmfe::RMFE_BITS / 8;
const MAX_BUCKET_BITS: usize = 8;
const MIN_BUCKET_BITS: usize = 6;
const BUCKET_COUNT: usize = 1 << MAX_BUCKET_BITS;
const BUCKET_LIMBS: usize = 128_usize.div_ceil(MIN_BUCKET_BITS);

#[derive(Clone, Copy, Debug, Default)]
struct Timings {
    eq: Duration,
    fused_accumulate: Duration,
    merge: Duration,
    product_recover: Duration,
    linear_recover: Duration,
    response: Duration,
}

struct ResultSet {
    all: Vec<Timings>,
    guard: u128,
}

struct Scratch {
    eq_low: Vec<F128>,
    eq_high: Vec<F128>,
    product_buckets: Vec<[WideBoolPoly; BUCKET_COUNT]>,
    linear_buckets: Vec<[BoolPoly; BUCKET_COUNT]>,
    worker_product_buckets: Vec<[WideBoolPoly; BUCKET_COUNT]>,
    worker_linear_buckets: Vec<[BoolPoly; BUCKET_COUNT]>,
    coeffs: Vec<F128>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows_log = arg_usize(&args, 1, "OLD_CHI_ROWS_LOG", 21);
    let runs = arg_usize(&args, 2, "OLD_CHI_RUNS", 5).max(1);
    let workers = arg_usize(
        &args,
        3,
        "OLD_CHI_WORKERS",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    )
    .max(1);
    let bucket_bits = bucket_bits_arg(&args);

    let rows = 1usize << rows_log;
    let embed_tables = embed_tables();
    let one_poly = embed_word(protocol_state::PACKED_MASK, embed_tables);
    let mut rng = 0x1234_5678_9abc_def0_1357_2468_ace0_bdf1u128;
    let state: [Vec<u128>; 5] = core::array::from_fn(|_| {
        let mut lane = vec![0u128; rows];
        for value in &mut lane {
            *value = next_random(&mut rng) & protocol_state::PACKED_MASK;
        }
        lane
    });
    let point: Vec<F128> = (0..rows_log)
        .map(|_| F128::from_raw(next_random(&mut rng)))
        .collect();
    let t = F128::from_raw(next_random(&mut rng));
    let gamma = F128::from_raw(next_random(&mut rng));
    let mut gamma_powers = [F128::ONE; 5];
    for idx in 1..5 {
        gamma_powers[idx] = gamma_powers[idx - 1] * gamma;
    }

    let mut scratch = Scratch {
        eq_low: vec![F128::ZERO; 1usize << (rows_log / 2)],
        eq_high: vec![F128::ZERO; 1usize << rows_log.div_ceil(2)],
        product_buckets: vec![[WideBoolPoly::ZERO; BUCKET_COUNT]; BUCKET_LIMBS],
        linear_buckets: vec![[BoolPoly::ZERO; BUCKET_COUNT]; BUCKET_LIMBS],
        worker_product_buckets: vec![[WideBoolPoly::ZERO; BUCKET_COUNT]; BUCKET_LIMBS * workers],
        worker_linear_buckets: vec![[BoolPoly::ZERO; BUCKET_COUNT]; BUCKET_LIMBS * workers],
        coeffs: vec![F128::ZERO; rmfe::PRODUCT_BITS],
    };

    println!("Old chi fused kernel over F2^128");
    println!("rows:        2^{rows_log} = {rows}");
    println!("packed bits: {PACKED_BITS}");
    println!("poly coeffs: {}", rmfe::PRODUCT_DEGREE);
    println!("workers:     {workers}");
    println!("rayon:       {} thread(s)", rayon::current_num_threads());
    println!("runs:        {runs}");
    println!("buckets:     {:?}", bucket_bits);
    println!();

    for bucket_bits in bucket_bits {
        let result = run_case(
            runs,
            bucket_bits,
            workers,
            &state,
            &point,
            &gamma_powers,
            t,
            one_poly,
            &mut scratch,
        );
        print_result(bucket_bits, rows, &result);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    runs: usize,
    bucket_bits: usize,
    workers: usize,
    state: &[Vec<u128>; 5],
    point: &[F128],
    gamma_powers: &[F128; 5],
    t: F128,
    one_poly: BoolPoly,
    scratch: &mut Scratch,
) -> ResultSet {
    assert!((MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bucket_bits));
    let mut all = Vec::with_capacity(runs);
    let mut guard = 0u128;
    for run_idx in 0..runs {
        let (timings, run_guard) = run_kernel(
            bucket_bits,
            workers,
            state,
            point,
            gamma_powers,
            t,
            one_poly,
            scratch,
        );
        all.push(timings);
        guard ^= run_guard.rotate_left((run_idx % 127) as u32);
    }
    ResultSet { all, guard }
}

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    bucket_bits: usize,
    workers: usize,
    state: &[Vec<u128>; 5],
    point: &[F128],
    gamma_powers: &[F128; 5],
    t: F128,
    one_poly: BoolPoly,
    scratch: &mut Scratch,
) -> (Timings, u128) {
    let started = Instant::now();
    let low_len = fill_eq_factors(point, &mut scratch.eq_low, &mut scratch.eq_high);
    let eq_time = started.elapsed();
    let low_bits = low_len.trailing_zeros() as usize;
    let low_mask = low_len - 1;

    let rows = state[0].len();
    let bucket_limbs = bucket_limbs(bucket_bits);
    let bucket_count = 1usize << bucket_bits;
    let worker_count = workers.max(1).min(rows.max(1));
    let use_parallel = worker_count > 1 && rows >= 1 << 15;

    let started = Instant::now();
    if use_parallel {
        for slab in scratch
            .worker_product_buckets
            .chunks_mut(BUCKET_LIMBS)
            .take(worker_count)
        {
            for buckets in &mut slab[..bucket_limbs] {
                buckets[..bucket_count].fill(WideBoolPoly::ZERO);
            }
        }
        for slab in scratch
            .worker_linear_buckets
            .chunks_mut(BUCKET_LIMBS)
            .take(worker_count)
        {
            for buckets in &mut slab[..bucket_limbs] {
                buckets[..bucket_count].fill(BoolPoly::ZERO);
            }
        }
        let chunk = rows.div_ceil(worker_count);
        scratch
            .worker_product_buckets
            .par_chunks_mut(BUCKET_LIMBS)
            .zip(scratch.worker_linear_buckets.par_chunks_mut(BUCKET_LIMBS))
            .take(worker_count)
            .enumerate()
            .for_each(|(worker_idx, (product_slab, linear_slab))| {
                let start = worker_idx * chunk;
                let end = (start + chunk).min(rows);
                for idx in start..end {
                    accumulate_row(
                        idx,
                        bucket_bits,
                        low_bits,
                        low_mask,
                        state,
                        gamma_powers,
                        &scratch.eq_low,
                        &scratch.eq_high,
                        &mut product_slab[..bucket_limbs],
                        &mut linear_slab[..bucket_limbs],
                    );
                }
            });
    } else {
        for buckets in &mut scratch.product_buckets[..bucket_limbs] {
            buckets[..bucket_count].fill(WideBoolPoly::ZERO);
        }
        for buckets in &mut scratch.linear_buckets[..bucket_limbs] {
            buckets[..bucket_count].fill(BoolPoly::ZERO);
        }
        for idx in 0..rows {
            accumulate_row(
                idx,
                bucket_bits,
                low_bits,
                low_mask,
                state,
                gamma_powers,
                &scratch.eq_low,
                &scratch.eq_high,
                &mut scratch.product_buckets[..bucket_limbs],
                &mut scratch.linear_buckets[..bucket_limbs],
            );
        }
    }
    let fused_time = started.elapsed();

    let started = Instant::now();
    if use_parallel {
        for buckets in &mut scratch.product_buckets[..bucket_limbs] {
            buckets[..bucket_count].fill(WideBoolPoly::ZERO);
        }
        for buckets in &mut scratch.linear_buckets[..bucket_limbs] {
            buckets[..bucket_count].fill(BoolPoly::ZERO);
        }
        for worker_idx in 0..worker_count {
            let product_slab =
                &scratch.worker_product_buckets[worker_idx * BUCKET_LIMBS..][..bucket_limbs];
            let linear_slab =
                &scratch.worker_linear_buckets[worker_idx * BUCKET_LIMBS..][..bucket_limbs];
            for limb_idx in 0..bucket_limbs {
                for value in 0..bucket_count {
                    scratch.product_buckets[limb_idx][value] ^= product_slab[limb_idx][value];
                    scratch.linear_buckets[limb_idx][value] ^= linear_slab[limb_idx][value];
                }
            }
        }
    }
    let merge_time = started.elapsed();

    scratch.coeffs.fill(F128::ZERO);

    let started = Instant::now();
    recover_wide_buckets(
        &scratch.product_buckets[..bucket_limbs],
        bucket_bits,
        &mut scratch.coeffs,
    );
    let product_recover_time = started.elapsed();

    let started = Instant::now();
    recover_linear_buckets(
        &scratch.linear_buckets[..bucket_limbs],
        bucket_bits,
        one_poly,
        &mut scratch.coeffs,
    );
    let linear_recover_time = started.elapsed();

    let started = Instant::now();
    let response = eval_coeffs_at(&scratch.coeffs, t);
    let response_time = started.elapsed();

    let guard = response.raw() ^ scratch.coeffs[17].raw() ^ scratch.coeffs[rmfe::PRODUCT_DEGREE].raw();
    (
        Timings {
            eq: eq_time,
            fused_accumulate: fused_time,
            merge: merge_time,
            product_recover: product_recover_time,
            linear_recover: linear_recover_time,
            response: response_time,
        },
        guard,
    )
}

#[allow(clippy::too_many_arguments)]
fn accumulate_row(
    idx: usize,
    bucket_bits: usize,
    low_bits: usize,
    low_mask: usize,
    state: &[Vec<u128>; 5],
    gamma_powers: &[F128; 5],
    eq_low: &[F128],
    eq_high: &[F128],
    product_buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    linear_buckets: &mut [[BoolPoly; BUCKET_COUNT]],
) {
    let eq = eq_high[idx >> low_bits] * eq_low[idx & low_mask];
    let polys: [BoolPoly; 5] = core::array::from_fn(|x| embed_word(state[x][idx], embed_tables()));
    for x in 0..5 {
        let scalar = if x == 0 { eq } else { eq * gamma_powers[x] };
        accumulate_wide(
            product_buckets,
            boolpoly::clmul_192(polys[x], polys[(x + 2) % 5]),
            scalar.raw(),
            bucket_bits,
        );
        accumulate_poly(linear_buckets, polys[(x + 1) % 5], scalar.raw(), bucket_bits);
    }
}

fn fill_eq_factors(point: &[F128], low: &mut [F128], high: &mut [F128]) -> usize {
    let low_vars = point.len() / 2;
    let low_len = 1usize << low_vars;
    let high_len = 1usize << (point.len() - low_vars);
    fill_eq_poly(&point[..low_vars], &mut low[..low_len]);
    fill_eq_poly(&point[low_vars..], &mut high[..high_len]);
    low_len
}

fn fill_eq_poly(point: &[F128], out: &mut [F128]) {
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

#[inline]
fn bucket_limbs(bucket_bits: usize) -> usize {
    128_usize.div_ceil(bucket_bits)
}

fn accumulate_wide(
    buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    product: WideBoolPoly,
    scalar: u128,
    bucket_bits: usize,
) {
    match bucket_bits {
        8 => {
            for (limb_idx, &value) in scalar.to_le_bytes().iter().enumerate() {
                buckets[limb_idx][value as usize] ^= product;
            }
        }
        7 => {
            for limb_idx in 0..bucket_limbs(7) {
                buckets[limb_idx][((scalar >> (7 * limb_idx)) & 0x7f) as usize] ^= product;
            }
        }
        6 => {
            for limb_idx in 0..bucket_limbs(6) {
                buckets[limb_idx][((scalar >> (6 * limb_idx)) & 0x3f) as usize] ^= product;
            }
        }
        _ => panic!("unsupported bucket size"),
    }
}

fn accumulate_poly(
    buckets: &mut [[BoolPoly; BUCKET_COUNT]],
    poly: BoolPoly,
    scalar: u128,
    bucket_bits: usize,
) {
    match bucket_bits {
        8 => {
            for (limb_idx, &value) in scalar.to_le_bytes().iter().enumerate() {
                buckets[limb_idx][value as usize] ^= poly;
            }
        }
        7 => {
            for limb_idx in 0..bucket_limbs(7) {
                buckets[limb_idx][((scalar >> (7 * limb_idx)) & 0x7f) as usize] ^= poly;
            }
        }
        6 => {
            for limb_idx in 0..bucket_limbs(6) {
                buckets[limb_idx][((scalar >> (6 * limb_idx)) & 0x3f) as usize] ^= poly;
            }
        }
        _ => panic!("unsupported bucket size"),
    }
}

fn recover_wide_buckets(
    buckets: &[[WideBoolPoly; BUCKET_COUNT]],
    bucket_bits: usize,
    out: &mut [F128],
) {
    let bucket_count = 1usize << bucket_bits;
    for (limb_idx, bucket_set) in buckets.iter().enumerate() {
        for value in 1..bucket_count {
            let product = bucket_set[value];
            if product.is_zero() {
                continue;
            }
            let mut limb = value;
            while limb != 0 {
                let bit = limb.trailing_zeros() as usize;
                let scalar_bit = limb_idx * bucket_bits + bit;
                if scalar_bit < 128 {
                    add_wide_bit(out, product, F128::from_raw(1u128 << scalar_bit));
                }
                limb &= limb - 1;
            }
        }
    }
}

fn recover_linear_buckets(
    buckets: &[[BoolPoly; BUCKET_COUNT]],
    bucket_bits: usize,
    one_poly: BoolPoly,
    out: &mut [F128],
) {
    let bucket_count = 1usize << bucket_bits;
    for (limb_idx, bucket_set) in buckets.iter().enumerate() {
        for value in 1..bucket_count {
            let poly = bucket_set[value];
            if poly.is_zero() {
                continue;
            }
            let product = boolpoly::clmul_192(poly, one_poly);
            let mut limb = value;
            while limb != 0 {
                let bit = limb.trailing_zeros() as usize;
                let scalar_bit = limb_idx * bucket_bits + bit;
                if scalar_bit < 128 {
                    add_wide_bit(out, product, F128::from_raw(1u128 << scalar_bit));
                }
                limb &= limb - 1;
            }
        }
    }
}

fn add_wide_bit(out: &mut [F128], product: WideBoolPoly, scalar_bit: F128) {
    let limbs = product.limbs();
    for (word_idx, mut word) in limbs.into_iter().enumerate() {
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            out[word_idx * 64 + bit] += scalar_bit;
            word &= word - 1;
        }
    }
}

fn eval_coeffs_at(coeffs: &[F128], t: F128) -> F128 {
    let mut out = F128::ZERO;
    for &coeff in coeffs.iter().rev() {
        out *= t;
        out += coeff;
    }
    out
}

fn embed_word(word: u128, tables: &[[BoolPoly; 256]; WORD_BYTES]) -> BoolPoly {
    let bytes = word.to_le_bytes();
    let mut out = BoolPoly::ZERO;
    for idx in 0..WORD_BYTES {
        out ^= tables[idx][bytes[idx] as usize];
    }
    out
}

fn embed_tables() -> &'static [[BoolPoly; 256]; WORD_BYTES] {
    static TABLES: OnceLock<[[BoolPoly; 256]; WORD_BYTES]> = OnceLock::new();
    TABLES.get_or_init(|| {
        let matrix = rmfe::embedding_matrix();
        let mut basis = [BoolPoly::ZERO; rmfe::RMFE_BITS];
        for input_bit in 0..rmfe::RMFE_BITS {
            let mut limbs = [0u64; 4];
            for coeff in 0..rmfe::PRODUCT_DEGREE {
                if matrix.get(coeff, input_bit) {
                    limbs[coeff / 64] ^= 1u64 << (coeff % 64);
                }
            }
            basis[input_bit] = BoolPoly::from_limbs(limbs);
        }

        let mut tables = [[BoolPoly::ZERO; 256]; WORD_BYTES];
        for byte_idx in 0..WORD_BYTES {
            for value in 1usize..256 {
                let bit = value.trailing_zeros() as usize;
                tables[byte_idx][value] =
                    tables[byte_idx][value & (value - 1)] ^ basis[8 * byte_idx + bit];
            }
        }
        tables
    })
}

fn print_result(bucket_bits: usize, rows: usize, result: &ResultSet) {
    let mut totals: Vec<Duration> = result.all.iter().map(|&timing| total_time(timing)).collect();
    totals.sort_unstable();
    let median_total = median_duration(&totals);
    let best_total = totals[0];
    let chi_instances = rows as f64 * PACKED_BITS as f64 / 320.0;
    let projected_f = chi_instances / 24.0;

    println!("bucket bits:             {bucket_bits}");
    print_phase("synthesize eq", &result.all, |timing| timing.eq);
    print_phase("fused chi acc", &result.all, |timing| timing.fused_accumulate);
    print_phase("merge buckets", &result.all, |timing| timing.merge);
    print_phase("product recover", &result.all, |timing| timing.product_recover);
    print_phase("linear recover", &result.all, |timing| timing.linear_recover);
    print_phase("response eval", &result.all, |timing| timing.response);
    println!("best chi skip:           {}", fmt_duration(best_total));
    println!("median chi skip:         {}", fmt_duration(median_total));
    println!(
        "median payload:          {:.2} Mbit/s",
        rows as f64 * PACKED_BITS as f64 / median_total.as_secs_f64() / 1e6,
    );
    println!(
        "projected Keccak-f[1600]:{:.2} K/s",
        projected_f / median_total.as_secs_f64() / 1e3,
    );
    println!("guard:                   {:032x}", black_box(result.guard));
    println!();
}

fn print_phase(name: &str, timings: &[Timings], get: impl Fn(Timings) -> Duration) {
    let mut values: Vec<Duration> = timings.iter().map(|&timing| get(timing)).collect();
    values.sort_unstable();
    println!(
        "{name:<24} best {:>10} | median {:>10}",
        fmt_duration(values[0]),
        fmt_duration(median_duration(&values)),
    );
}

fn total_time(timing: Timings) -> Duration {
    timing.eq
        + timing.fused_accumulate
        + timing.merge
        + timing.product_recover
        + timing.linear_recover
        + timing.response
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

fn next_random(state: &mut u128) -> u128 {
    *state = state
        .wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb)
        .wrapping_add(0x9e37_79b9_7f4a_7c15_6a09_e667_f3bc_c909)
        .rotate_left(37);
    *state
}

fn arg_usize(args: &[String], index: usize, env: &str, default: usize) -> usize {
    args.get(index)
        .cloned()
        .or_else(|| std::env::var(env).ok())
        .map(|value| value.parse().expect("expected usize"))
        .unwrap_or(default)
}

fn bucket_bits_arg(args: &[String]) -> Vec<usize> {
    let mode = args
        .get(4)
        .cloned()
        .or_else(|| std::env::var("OLD_CHI_BUCKET_BITS").ok())
        .unwrap_or_else(|| "8,7,6".to_string());
    mode.split(',')
        .map(|part| {
            let bits: usize = part.trim().parse().expect("bucket bits must be usize");
            assert!((MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bits));
            bits
        })
        .collect()
}
