//! Single-threaded microbenchmarks for the GF(2^128) field primitives.
//!
//! Reports best-of-many (minimum) ns/op, which is far more robust than
//! medians/averages when the machine is under load.  Run with:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example field_bench
//! ```

use std::hint::black_box;
use std::time::Instant;

use hashcaster2::field::{F128, F128Acc};

/// splitmix64 -> u128
struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f128(&mut self) -> F128 {
        // Force a nonzero value so it is a valid field element for inverse.
        let raw = (self.u64() as u128) | ((self.u64() as u128) << 64) | 1;
        F128::from_raw(raw)
    }
}

/// Run `f` `rounds` times over `inner` op-iterations each, return best ns/op.
fn bench<F: FnMut() -> u64>(name: &str, ops: u64, rounds: u32, mut f: F) {
    // Warmup.
    for _ in 0..3 {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        let sink = f();
        let dt = t.elapsed().as_secs_f64();
        black_box(sink);
        let per = dt / ops as f64 * 1e9;
        if per < best {
            best = per;
        }
    }
    println!("  {name:<34} {best:8.3} ns/op");
}

fn main() {
    let mut rng = Rng(0xC0FF_EE12);

    // ---- latency-bound dependent multiply chain ----
    let c = rng.f128();
    let iters: u64 = 20_000_000;
    bench("mul  (latency, dependent chain)", iters, 15, || {
        let mut x = black_box(c);
        let cc = black_box(c);
        for _ in 0..iters {
            x = x * cc;
        }
        x.raw() as u64
    });

    // ---- latency-bound dependent square chain ----
    bench("square (latency, dependent chain)", iters, 15, || {
        let mut x = black_box(c);
        for _ in 0..iters {
            x = x.square();
        }
        x.raw() as u64
    });

    // ---- throughput-bound independent multiplies ----
    // 8 independent accumulators broken into independent dependency chains.
    let seeds: [F128; 8] = std::array::from_fn(|_| rng.f128());
    let muls: [F128; 8] = std::array::from_fn(|_| rng.f128());
    let titers: u64 = 4_000_000;
    bench("mul  (throughput, 8 indep chains)", titers * 8, 15, || {
        let mut x = black_box(seeds);
        let m = black_box(muls);
        for _ in 0..titers {
            for k in 0..8 {
                x[k] = x[k] * m[k];
            }
        }
        x.iter().fold(0u64, |a, v| a ^ v.raw() as u64)
    });

    // ---- batched multiply (mul_batch) vs scalar loop, N independent pairs ----
    let n = 4096usize;
    let av: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let bv: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let mut outv = vec![F128::ZERO; n];
    let reps: u64 = 20_000;
    bench("mul  (scalar loop, N=4096)", reps * n as u64, 15, || {
        let a = black_box(&av);
        let b = black_box(&bv);
        let out = black_box(&mut outv);
        for _ in 0..reps {
            for i in 0..n {
                out[i] = a[i] * b[i];
            }
        }
        out[0].raw() as u64
    });
    bench("mul_batch (N=4096)", reps * n as u64, 15, || {
        let a = black_box(&av);
        let b = black_box(&bv);
        let out = black_box(&mut outv);
        for _ in 0..reps {
            F128::mul_batch(a, b, out);
        }
        out[0].raw() as u64
    });

    // ---- dot product: naive fold vs deferred-reduction ----
    bench("dot  (naive fold of muls, N=4096)", reps * n as u64, 15, || {
        let a = black_box(&av);
        let b = black_box(&bv);
        let mut acc = F128::ZERO;
        for _ in 0..reps {
            let mut s = F128::ZERO;
            for i in 0..n {
                s += a[i] * b[i];
            }
            acc += s;
        }
        acc.raw() as u64
    });
    bench("dot_product (deferred, N=4096)", reps * n as u64, 15, || {
        let a = black_box(&av);
        let b = black_box(&bv);
        let mut acc = F128::ZERO;
        for _ in 0..reps {
            acc += F128::dot_product(a, b);
        }
        acc.raw() as u64
    });
    bench("F128Acc accumulate (N=4096)", reps * n as u64, 15, || {
        let a = black_box(&av);
        let b = black_box(&bv);
        let mut sink = F128::ZERO;
        for _ in 0..reps {
            let mut acc = F128Acc::new();
            for i in 0..n {
                acc.accumulate(a[i], b[i]);
            }
            sink += acc.finalize();
        }
        sink.raw() as u64
    });

    // ---- inverse: old (127 mul + 127 sq) vs new (addition chain) ----
    let inv_inputs: Vec<F128> = (0..1024).map(|_| rng.f128()).collect();
    let inv_reps: u64 = 2_000;
    bench("inverse (naive 127-mul)", inv_reps * inv_inputs.len() as u64, 10, || {
        let xs = black_box(&inv_inputs);
        let mut sink = F128::ZERO;
        for _ in 0..inv_reps {
            for &x in xs {
                sink += old_inverse(x);
            }
        }
        sink.raw() as u64
    });
    bench("inverse (addition chain)", inv_reps * inv_inputs.len() as u64, 10, || {
        let xs = black_box(&inv_inputs);
        let mut sink = F128::ZERO;
        for _ in 0..inv_reps {
            for &x in xs {
                sink += x.inverse();
            }
        }
        sink.raw() as u64
    });
}

/// Previous inverse algorithm (127 squarings + 127 multiplies) for A/B.
fn old_inverse(a: F128) -> F128 {
    let mut x = a;
    let mut out = F128::ONE;
    for _ in 1..128 {
        x = x * x;
        out = out * x;
    }
    out
}
