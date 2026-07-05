use std::sync::OnceLock;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::{
    boolpoly::{self, BoolPoly, WideBoolPoly},
    challenger::{Challenger, ProofWriter},
    chi_round::verifier::HybridClaim,
    field::{F128, F128Acc},
    protocol_state::{self, KeccakWitness},
    rmfe,
    util::{eq_poly_v, fill_eq_poly_v},
};

/// Five independent field multiplies by a single loop-invariant scalar `r`,
/// issued together so the backend overlaps their PMULL+reduction dependency
/// chains instead of serialising them one product at a time.
#[inline(always)]
fn mul5_by_scalar(r: F128, v: [F128; 5]) -> [F128; 5] {
    [r * v[0], r * v[1], r * v[2], r * v[3], r * v[4]]
}

/// Sumcheck fold of five independent elements: `out[x] = a[x] + r*(a[x]+b[x])`.
/// The five `r*(...)` multiplies are the latency-bound part; batching them lets
/// their PMULL+reduction chains overlap.
#[inline(always)]
fn fold5(a: [F128; 5], b: [F128; 5], r: F128) -> [F128; 5] {
    let diff = [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
    ];
    let prod = mul5_by_scalar(r, diff);
    [
        a[0] + prod[0],
        a[1] + prod[1],
        a[2] + prod[2],
        a[3] + prod[3],
        a[4] + prod[4],
    ]
}

const MAX_BUCKET_BITS: usize = 8;
const MIN_BUCKET_BITS: usize = 6;
const BUCKET_COUNT: usize = 1 << MAX_BUCKET_BITS;
const BUCKET_LIMBS: usize = 128_usize.div_ceil(MIN_BUCKET_BITS);
const WORD_BYTES: usize = rmfe::RMFE_BITS / 8;
const HOT_STRIPS: usize = 5 * 64;

#[derive(Clone, Copy)]
pub struct ProverCfg {
    pub log_packed_instances: usize,
    pub round: usize,
}

pub struct ProverScratch {
    workers: usize,
    bucket_bits: usize,
    u: Vec<F128>,
    buckets: Vec<[WideBoolPoly; BUCKET_COUNT]>,
    worker_buckets: Vec<[WideBoolPoly; BUCKET_COUNT]>,
    hot_state: Vec<F128>,
    state: [Vec<F128>; 5],
    eq_out: Vec<F128>,
    eq_yz: Vec<F128>,
    eq_y: Vec<F128>,
    eq_z: Vec<F128>,
    eq_strips: Vec<F128>,
    embed_evals: [[F128; 256]; WORD_BYTES],
}

pub struct ProverOutput {
    pub claim: HybridClaim,
    pub pre_chi_state: [F128; protocol_state::KECCAK_BITS],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProverProfile {
    pub build_u: Duration,
    pub build_u_clear: Duration,
    pub build_u_accumulate: Duration,
    pub build_u_merge: Duration,
    pub build_u_recover: Duration,
    pub setup: Duration,
    pub out_messages: Duration,
    pub out_folds: Duration,
    pub out_first_message: Duration,
    pub out_first_fold: Duration,
    pub out_later_messages: Duration,
    pub out_later_folds: Duration,
    pub expand: Duration,
    pub tail_messages: Duration,
    pub tail_folds: Duration,
    pub final_claim: Duration,
}

impl ProverScratch {
    pub fn new(log_packed_instances: usize) -> Self {
        Self::new_with_workers(log_packed_instances, 1)
    }

    pub fn new_with_workers(log_packed_instances: usize, workers: usize) -> Self {
        Self::new_with_workers_and_bucket_bits(log_packed_instances, workers, 8)
    }

    pub fn new_with_workers_and_bucket_bits(
        log_packed_instances: usize,
        workers: usize,
        bucket_bits: usize,
    ) -> Self {
        assert!((MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bucket_bits));
        let out_len = 1usize << log_packed_instances;
        let yz_len = 1usize << (3 + 6);
        let workers = workers.max(1).min(out_len.max(1));
        Self {
            workers,
            bucket_bits,
            u: vec![F128::ZERO; rmfe::PRODUCT_BITS],
            buckets: vec![[WideBoolPoly::ZERO; BUCKET_COUNT]; BUCKET_LIMBS],
            worker_buckets: vec![[WideBoolPoly::ZERO; BUCKET_COUNT]; BUCKET_LIMBS * workers],
            hot_state: vec![F128::ZERO; out_len * protocol_state::KECCAK_BITS],
            state: core::array::from_fn(|_| vec![F128::ZERO; yz_len]),
            eq_out: vec![F128::ZERO; out_len],
            eq_yz: vec![F128::ZERO; yz_len],
            eq_y: vec![F128::ZERO; 1 << 3],
            eq_z: vec![F128::ZERO; 1 << 6],
            eq_strips: vec![F128::ZERO; HOT_STRIPS],
            embed_evals: [[F128::ZERO; 256]; WORD_BYTES],
        }
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    pub fn bucket_bits(&self) -> usize {
        self.bucket_bits
    }

    fn assert_shape(&self, log_packed_instances: usize) {
        let out_len = 1usize << log_packed_instances;
        let yz_len = 1usize << (3 + 6);
        assert_eq!(self.u.len(), rmfe::PRODUCT_BITS);
        assert_eq!(self.buckets.len(), BUCKET_LIMBS);
        assert_eq!(self.worker_buckets.len(), BUCKET_LIMBS * self.workers);
        assert_eq!(self.hot_state.len(), out_len * protocol_state::KECCAK_BITS);
        assert_eq!(self.eq_out.len(), out_len);
        assert_eq!(self.eq_yz.len(), yz_len);
        assert_eq!(self.eq_y.len(), 1 << 3);
        assert_eq!(self.eq_z.len(), 1 << 6);
        assert_eq!(self.eq_strips.len(), HOT_STRIPS);
        assert_eq!(self.embed_evals.len(), WORD_BYTES);
        for table in &self.state {
            assert_eq!(table.len(), yz_len);
        }
    }
}

impl ProverCfg {
    pub fn prove<Ch: Challenger>(
        &self,
        ctx: &mut ProofWriter<Ch>,
        witness: &KeccakWitness,
        claim: HybridClaim,
        scratch: &mut ProverScratch,
    ) -> ProverOutput {
        self.prove_inner(ctx, witness, claim, scratch, None)
    }

    pub fn prove_profiled<Ch: Challenger>(
        &self,
        ctx: &mut ProofWriter<Ch>,
        witness: &KeccakWitness,
        claim: HybridClaim,
        scratch: &mut ProverScratch,
        profile: &mut ProverProfile,
    ) -> ProverOutput {
        *profile = ProverProfile::default();
        self.prove_inner(ctx, witness, claim, scratch, Some(profile))
    }

    fn prove_inner<Ch: Challenger>(
        &self,
        ctx: &mut ProofWriter<Ch>,
        witness: &KeccakWitness,
        claim: HybridClaim,
        scratch: &mut ProverScratch,
        mut profile: Option<&mut ProverProfile>,
    ) -> ProverOutput {
        debug_assert_eq!(claim.r_out.len(), self.log_packed_instances);
        debug_assert_eq!(claim.r_x.len(), 3);
        debug_assert_eq!(claim.r_y.len(), 3);
        debug_assert_eq!(claim.r_z.len(), 6);
        debug_assert!(witness.blocks() <= (1usize << self.log_packed_instances));
        scratch.assert_shape(self.log_packed_instances);

        let started = Instant::now();
        fill_eq_poly_v(&claim.r_out, &mut scratch.eq_out);
        fill_yz_eq_tables(
            &claim,
            &mut scratch.eq_y,
            &mut scratch.eq_z,
            &mut scratch.eq_yz,
            &mut scratch.eq_strips,
        );
        let eq_x = eq_poly_v(&claim.r_x);
        if let Some(profile) = profile.as_deref_mut() {
            profile.setup += started.elapsed();
        }

        let started = Instant::now();
        if let Some(profile) = profile.as_deref_mut() {
            build_u(
                witness,
                self.round,
                &claim,
                &eq_x,
                &scratch.eq_out,
                &scratch.eq_strips,
                &mut scratch.buckets,
                &mut scratch.worker_buckets,
                scratch.workers,
                scratch.bucket_bits,
                &mut scratch.u,
                Some(profile),
            );
            profile.build_u += started.elapsed();
        } else {
            build_u(
                witness,
                self.round,
                &claim,
                &eq_x,
                &scratch.eq_out,
                &scratch.eq_strips,
                &mut scratch.buckets,
                &mut scratch.worker_buckets,
                scratch.workers,
                scratch.bucket_bits,
                &mut scratch.u,
                None,
            );
        }
        ctx.write_f128_slice(&scratch.u);

        let started = Instant::now();
        let t = ctx.sample_f128();
        fill_embed_eval_tables(t, &mut scratch.embed_evals);
        let mut active_blocks = 1usize << self.log_packed_instances;
        if active_blocks == 1 {
            build_hot_state(
                witness,
                self.round,
                self.log_packed_instances,
                &scratch.embed_evals,
                scratch.workers,
                &mut scratch.hot_state,
            );
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.setup += started.elapsed();
        }
        let mut bound_out = Vec::with_capacity(claim.r_out.len());
        let mut bound_y = Vec::with_capacity(claim.r_y.len());
        let mut bound_z = Vec::with_capacity(claim.r_z.len());
        let mut active_real_blocks = witness.blocks().min(active_blocks);

        let mut pending_out_msg = None;
        for round_idx in 0..self.log_packed_instances {
            let started = Instant::now();
            let msg = if let Some(msg) = pending_out_msg.take() {
                msg
            } else if round_idx == 0 {
                out_round_gruen_from_witness(
                    witness,
                    self.round,
                    &scratch.embed_evals,
                    &scratch.eq_out,
                    &scratch.eq_strips,
                    active_blocks,
                    active_real_blocks,
                    &eq_x,
                    scratch.workers,
                )
            } else {
                out_round_gruen(
                    &scratch.hot_state,
                    &scratch.eq_out,
                    &scratch.eq_strips,
                    active_blocks,
                    active_real_blocks,
                    &eq_x,
                    scratch.workers,
                )
            };
            ctx.write_f128_slice(&msg);
            if let Some(profile) = profile.as_deref_mut() {
                let elapsed = started.elapsed();
                profile.out_messages += elapsed;
                if round_idx == 0 {
                    profile.out_first_message += elapsed;
                } else {
                    profile.out_later_messages += elapsed;
                }
            }

            let challenge = ctx.sample_f128();
            let started = Instant::now();
            if round_idx == 0 {
                if round_idx + 1 < self.log_packed_instances {
                    pending_out_msg = Some(build_folded_hot_state_and_next_gruen_from_witness(
                        witness,
                        self.round,
                        &scratch.embed_evals,
                        &mut scratch.eq_out,
                        &scratch.eq_strips,
                        active_blocks,
                        active_real_blocks,
                        challenge,
                        &eq_x,
                        scratch.workers,
                        &mut scratch.hot_state,
                    ));
                    active_blocks /= 2;
                    active_real_blocks = active_real_blocks.div_ceil(2);
                } else {
                    build_folded_hot_state_from_witness(
                        witness,
                        self.round,
                        &scratch.embed_evals,
                        active_blocks,
                        active_real_blocks,
                        challenge,
                        scratch.workers,
                        &mut scratch.hot_state,
                    );
                    fold_eq_table_for_gruen(&mut scratch.eq_out, active_blocks);
                    active_blocks /= 2;
                    active_real_blocks = active_real_blocks.div_ceil(2);
                }
            } else if round_idx + 1 < self.log_packed_instances {
                pending_out_msg = Some(fold_hot_state_and_next_gruen(
                    &mut scratch.hot_state,
                    &mut scratch.eq_out,
                    &scratch.eq_strips,
                    active_blocks,
                    active_real_blocks,
                    challenge,
                    &eq_x,
                    scratch.workers,
                ));
                active_blocks /= 2;
                active_real_blocks = active_real_blocks.div_ceil(2);
            } else {
                fold_hot_state(
                    &mut scratch.hot_state,
                    active_real_blocks,
                    challenge,
                    scratch.workers,
                );
                fold_eq_table_for_gruen(&mut scratch.eq_out, active_blocks);
                active_blocks /= 2;
                active_real_blocks = active_real_blocks.div_ceil(2);
            }
            bound_out.push(challenge);
            if let Some(profile) = profile.as_deref_mut() {
                let elapsed = started.elapsed();
                profile.out_folds += elapsed;
                if round_idx == 0 {
                    profile.out_first_fold += elapsed;
                } else {
                    profile.out_later_folds += elapsed;
                }
            }
        }

        let started = Instant::now();
        let out_eq = scratch.eq_out[0];
        for value in &mut scratch.eq_yz {
            *value *= out_eq;
        }
        expand_hot_state(&scratch.hot_state, &mut scratch.state);
        let mut pre_chi_state = [F128::ZERO; protocol_state::KECCAK_BITS];
        snapshot_expanded_hot_state(&scratch.state, &mut pre_chi_state);
        if let Some(profile) = profile.as_deref_mut() {
            profile.expand += started.elapsed();
        }

        let mut active_len = scratch.eq_yz.len();
        for round in 0..(3 + 6) {
            let started = Instant::now();
            let msg = tail_round_gruen(
                &scratch.state,
                &scratch.eq_yz,
                active_len,
                &eq_x,
            );
            ctx.write_f128_slice(&msg);
            if let Some(profile) = profile.as_deref_mut() {
                profile.tail_messages += started.elapsed();
            }

            let challenge = ctx.sample_f128();
            let started = Instant::now();
            fold_state(&mut scratch.state, active_len, challenge);
            fold_eq_table_for_gruen(&mut scratch.eq_yz, active_len);
            active_len /= 2;
            if round < 3 {
                bound_y.push(challenge);
            } else {
                bound_z.push(challenge);
            }
            if let Some(profile) = profile.as_deref_mut() {
                profile.tail_folds += started.elapsed();
            }
        }

        let started = Instant::now();
        let values: [F128; 5] = core::array::from_fn(|x| scratch.state[x][0]);
        ctx.write_f128_slice(&values);

        let r_x = ctx.sample_f128_vec(3);
        let eq_x = eq_poly_v(&r_x);
        let mut ev = F128::ZERO;
        for x in 0..5 {
            ev += eq_x[x] * values[x];
        }

        let output = HybridClaim {
            t,
            r_x,
            r_y: bound_y,
            r_z: bound_z,
            r_out: bound_out,
            ev,
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.final_claim += started.elapsed();
        }
        ProverOutput {
            claim: output,
            pre_chi_state,
        }
    }
}

fn build_u(
    witness: &KeccakWitness,
    round: usize,
    claim: &HybridClaim,
    eq_x: &[F128],
    eq_out: &[F128],
    eq_strips: &[F128],
    buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    worker_buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    workers: usize,
    bucket_bits: usize,
    out: &mut [F128],
    mut profile: Option<&mut ProverProfile>,
) {
    debug_assert!((MIN_BUCKET_BITS..=MAX_BUCKET_BITS).contains(&bucket_bits));
    let bucket_limbs = bucket_limbs(bucket_bits);
    let bucket_count = 1usize << bucket_bits;
    let out_len = 1usize << claim.r_out.len();
    let real_out_len = witness.blocks().min(out_len);
    let mut virtual_out_eq = F128::ZERO;
    for &value in &eq_out[real_out_len..out_len] {
        virtual_out_eq += value;
    }
    let started = Instant::now();
    out.fill(F128::ZERO);
    for bucket_set in &mut buckets[..bucket_limbs] {
        bucket_set[..bucket_count].fill(WideBoolPoly::ZERO);
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.build_u_clear += started.elapsed();
    }

    let use_parallel = workers > 1 && HOT_STRIPS >= workers * 2;
    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let started = Instant::now();
        for worker_bucket_set in worker_buckets.chunks_mut(BUCKET_LIMBS).take(worker_count) {
            for bucket_set in &mut worker_bucket_set[..bucket_limbs] {
                bucket_set[..bucket_count].fill(WideBoolPoly::ZERO);
            }
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.build_u_clear += started.elapsed();
        }

        let started = Instant::now();
        let strip_chunk = HOT_STRIPS.div_ceil(worker_count);
        worker_buckets
            .par_chunks_mut(BUCKET_LIMBS)
            .take(worker_count)
            .enumerate()
            .for_each(|(worker_idx, worker_buckets)| {
                let strip_start = worker_idx * strip_chunk;
                let strip_end = (strip_start + strip_chunk).min(HOT_STRIPS);
                accumulate_u_strip_range(
                    witness,
                    round,
                    strip_start,
                    strip_end,
                    real_out_len,
                    eq_out,
                    eq_strips,
                    eq_x,
                    worker_buckets,
                    bucket_bits,
                );
                if virtual_out_eq != F128::ZERO {
                    accumulate_u_virtual_strip_range(
                        round,
                        strip_start,
                        strip_end,
                        virtual_out_eq,
                        eq_x,
                        eq_strips,
                        worker_buckets,
                        bucket_bits,
                    );
                }
            });
        if let Some(profile) = profile.as_deref_mut() {
            profile.build_u_accumulate += started.elapsed();
        }

        let started = Instant::now();
        for worker_idx in 0..worker_count {
            let worker_buckets = &worker_buckets[worker_idx * BUCKET_LIMBS..][..BUCKET_LIMBS];
            xor_wide_bucket_sets(buckets, worker_buckets, bucket_limbs, bucket_count);
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.build_u_merge += started.elapsed();
        }
    } else {
        let started = Instant::now();
        accumulate_u_strip_range(
            witness,
            round,
            0,
            HOT_STRIPS,
            real_out_len,
            eq_out,
            eq_strips,
            eq_x,
            buckets,
            bucket_bits,
        );
        if virtual_out_eq != F128::ZERO {
            accumulate_u_virtual_strip_range(
                round,
                0,
                HOT_STRIPS,
                virtual_out_eq,
                eq_x,
                eq_strips,
                buckets,
                bucket_bits,
            );
        }
        if let Some(profile) = profile.as_deref_mut() {
            profile.build_u_accumulate += started.elapsed();
        }
    }

    let started = Instant::now();
    recover_wide_buckets(buckets, bucket_bits, out);
    if let Some(profile) = profile.as_deref_mut() {
        profile.build_u_recover += started.elapsed();
    }
}

fn accumulate_u_strip_range(
    witness: &KeccakWitness,
    round: usize,
    start_strip: usize,
    end_strip: usize,
    real_out_len: usize,
    eq_out: &[F128],
    eq_strips: &[F128],
    eq_x: &[F128],
    buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    bucket_bits: usize,
) {
    debug_assert!(real_out_len > 0);

    for strip_idx in start_strip..end_strip {
        let y = strip_idx / 64;
        let z = strip_idx % 64;
        let yz_eq = eq_strips[strip_idx];
        let strip_scalars = scale_eq_x(yz_eq, eq_x);
        let strip = witness.pre_chi_strip(round, y, z);
        for out in 0..real_out_len {
            let out_eq = eq_out[out];
            let base = out * 5;
            let words = [
                strip[base],
                strip[base + 1],
                strip[base + 2],
                strip[base + 3],
                strip[base + 4],
            ];
            accumulate_chi_row(words, out_eq, strip_scalars, buckets, bucket_bits);
        }
    }
}

fn accumulate_u_virtual_strip_range(
    round: usize,
    start_strip: usize,
    end_strip: usize,
    virtual_out_eq: F128,
    eq_x: &[F128],
    eq_strips: &[F128],
    buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    bucket_bits: usize,
) {
    for strip_idx in start_strip..end_strip {
        let y = strip_idx / 64;
        let z = strip_idx % 64;
        let out_eq = virtual_out_eq;
        let strip_scalars = scale_eq_x(eq_strips[strip_idx], eq_x);
        if out_eq != F128::ZERO && strip_scalars.iter().any(|&value| value != F128::ZERO) {
            let words: [u128; 5] =
                core::array::from_fn(|x| protocol_state::zero_pre_chi_word(round, x, y, z));
            accumulate_chi_row(words, out_eq, strip_scalars, buckets, bucket_bits);
        }
    }
}

#[inline(always)]
fn scale_eq_x(scale: F128, eq_x: &[F128]) -> [F128; 5] {
    [
        scale * eq_x[0],
        scale * eq_x[1],
        scale * eq_x[2],
        scale * eq_x[3],
        scale * eq_x[4],
    ]
}

#[inline(always)]
fn accumulate_chi_row(
    words: [u128; 5],
    out_eq: F128,
    strip_scalars: [F128; 5],
    buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    bucket_bits: usize,
) {
    let p0 = embed_word_poly(words[0]);
    let p1 = embed_word_poly(words[1]);
    let p2 = embed_word_poly(words[2]);
    let p3 = embed_word_poly(words[3]);
    let p4 = embed_word_poly(words[4]);

    accumulate_chi_term(buckets, p1, p2, p0 ^ p2, out_eq * strip_scalars[0], bucket_bits);
    accumulate_chi_term(buckets, p2, p3, p1 ^ p3, out_eq * strip_scalars[1], bucket_bits);
    accumulate_chi_term(buckets, p3, p4, p2 ^ p4, out_eq * strip_scalars[2], bucket_bits);
    accumulate_chi_term(buckets, p4, p0, p3 ^ p0, out_eq * strip_scalars[3], bucket_bits);
    accumulate_chi_term(buckets, p0, p1, p4 ^ p1, out_eq * strip_scalars[4], bucket_bits);
}

#[inline(always)]
fn accumulate_chi_term(
    buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    left: BoolPoly,
    right: BoolPoly,
    linear: BoolPoly,
    scalar: F128,
    bucket_bits: usize,
) {
    accumulate_wide(
        buckets,
        boolpoly::clmul_192(left, right) ^ boolpoly::square_192(linear),
        scalar,
        bucket_bits,
    );
}

#[inline(always)]
fn build_hot_state(
    witness: &KeccakWitness,
    round: usize,
    log_packed_instances: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    workers: usize,
    state: &mut [F128],
) {
    let out_len = 1usize << log_packed_instances;
    assert_eq!(state.len(), out_len * protocol_state::KECCAK_BITS);
    let strip_len = out_len * 5;
    let use_parallel = workers > 1 && HOT_STRIPS >= workers * 2;
    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let chunk_strips = HOT_STRIPS.div_ceil(worker_count);
        let chunk_len = chunk_strips * strip_len;
        state
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_strip = chunk_idx * chunk_strips;
                fill_hot_state_strips(witness, round, out_len, start_strip, embed_evals, chunk);
            });
        return;
    }

    fill_hot_state_strips(witness, round, out_len, 0, embed_evals, state);
}

#[inline(always)]
fn fill_hot_state_strips(
    witness: &KeccakWitness,
    round: usize,
    out_len: usize,
    start_strip: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    state: &mut [F128],
) {
    let strip_len = out_len * 5;
    debug_assert_eq!(state.len() % strip_len, 0);
    let strips = state.len() / strip_len;
    for local_strip in 0..strips {
        let strip_idx = start_strip + local_strip;
        let y = strip_idx / 64;
        let z = strip_idx % 64;
        let out_strip = &mut state[local_strip * strip_len..][..strip_len];
        let real_strip = witness.pre_chi_strip(round, y, z);
        let real_blocks = witness.blocks().min(out_len);
        for out in 0..real_blocks {
            for x in 0..5 {
                let word = real_strip[out * 5 + x];
                out_strip[out * 5 + x] = if word == 0 {
                    F128::ZERO
                } else {
                    eval_word(word, embed_evals)
                };
            }
        }
        for out in real_blocks..out_len {
            for x in 0..5 {
                let word = protocol_state::zero_pre_chi_word(round, x, y, z);
                out_strip[out * 5 + x] = if word == 0 {
                    F128::ZERO
                } else {
                    eval_word(word, embed_evals)
                };
            }
        }
    }
}

fn expand_hot_state(hot_state: &[F128], state: &mut [Vec<F128>; 5]) {
    debug_assert_eq!(hot_state.len() % (HOT_STRIPS * 5), 0);
    let out_len = hot_state.len() / (HOT_STRIPS * 5);
    for table in state.iter_mut() {
        table.fill(F128::ZERO);
    }
    for y in 0..5 {
        for z in 0..64 {
            let idx = y + (z << 3);
            let base = hot_idx(out_len, 0, y, z, 0);
            for x in 0..5 {
                state[x][idx] = hot_state[base + x];
            }
        }
    }
}

fn snapshot_expanded_hot_state(state: &[Vec<F128>; 5], out: &mut [F128; protocol_state::KECCAK_BITS]) {
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                out[protocol_state::state_idx(x, y, z)] = state[x][y + (z << 3)];
            }
        }
    }
}

fn fill_yz_eq_tables(
    claim: &HybridClaim,
    eq_y: &mut [F128],
    eq_z: &mut [F128],
    eq_yz: &mut [F128],
    eq_strips: &mut [F128],
) {
    debug_assert_eq!(eq_y.len(), 1 << claim.r_y.len());
    debug_assert_eq!(eq_z.len(), 1 << claim.r_z.len());
    debug_assert_eq!(eq_yz.len(), eq_y.len() * eq_z.len());
    debug_assert_eq!(eq_strips.len(), HOT_STRIPS);

    fill_eq_poly_v(&claim.r_y, eq_y);
    fill_eq_poly_v(&claim.r_z, eq_z);

    for z in 0..64 {
        let z_eq = eq_z[z];
        for y in 0..8 {
            eq_yz[y + (z << 3)] = eq_y[y] * z_eq;
        }
        for y in 0..5 {
            eq_strips[y * 64 + z] = eq_y[y] * z_eq;
        }
    }
}

fn eq_pair_sum_for_gruen(eq: &[F128], start_pair: usize, end_pair: usize) -> F128 {
    let mut sum = F128::ZERO;
    for pair_idx in start_pair..end_pair {
        let idx = 2 * pair_idx;
        sum += eq[idx] + eq[idx + 1];
    }
    sum
}

#[inline]
fn bucket_limbs(bucket_bits: usize) -> usize {
    128_usize.div_ceil(bucket_bits)
}

fn out_round_gruen(
    state: &[F128],
    eq_out: &[F128],
    eq_strips: &[F128],
    active_blocks: usize,
    real_blocks: usize,
    eq_x: &[F128],
    workers: usize,
) -> [F128; 2] {
    let pair_count = active_blocks / 2;
    let real_pair_count = real_blocks / 2;
    let has_mixed_pair = real_blocks % 2 == 1;
    let virtual_pair_start = real_blocks.div_ceil(2);
    let virtual_eq = eq_pair_sum_for_gruen(eq_out, virtual_pair_start, pair_count);
    let use_parallel = workers > 1 && active_blocks >= 16 && HOT_STRIPS >= workers * 2;
    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let chunk = HOT_STRIPS.div_ceil(worker_count);
        return (0..worker_count)
            .into_par_iter()
            .map(|worker_idx| {
                let start_strip = worker_idx * chunk;
                let end_strip = (start_strip + chunk).min(HOT_STRIPS);
                if start_strip >= end_strip {
                    return [F128::ZERO; 2];
                }
                out_round_gruen_strip_range(
                    state,
                    eq_out,
                    eq_strips,
                    start_strip,
                    end_strip,
                    real_blocks,
                    real_pair_count,
                    has_mixed_pair,
                    virtual_eq,
                    eq_x,
                )
            })
            .reduce(
                || [F128::ZERO; 2],
                |mut acc, part| {
                    for idx in 0..2 {
                        acc[idx] += part[idx];
                    }
                    acc
                },
            );
    }
    out_round_gruen_strip_range(
        state,
        eq_out,
        eq_strips,
        0,
        HOT_STRIPS,
        real_blocks,
        real_pair_count,
        has_mixed_pair,
        virtual_eq,
        eq_x,
    )
}

fn out_round_gruen_from_witness(
    witness: &KeccakWitness,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    eq_out: &[F128],
    eq_strips: &[F128],
    active_blocks: usize,
    real_blocks: usize,
    eq_x: &[F128],
    workers: usize,
) -> [F128; 2] {
    let pair_count = active_blocks / 2;
    let real_pair_count = real_blocks / 2;
    let has_mixed_pair = real_blocks % 2 == 1;
    let virtual_pair_start = real_blocks.div_ceil(2);
    let virtual_eq = eq_pair_sum_for_gruen(eq_out, virtual_pair_start, pair_count);
    let use_parallel = workers > 1 && active_blocks >= 16 && HOT_STRIPS >= workers * 2;
    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let chunk = HOT_STRIPS.div_ceil(worker_count);
        return (0..worker_count)
            .into_par_iter()
            .map(|worker_idx| {
                let start_strip = worker_idx * chunk;
                let end_strip = (start_strip + chunk).min(HOT_STRIPS);
                if start_strip >= end_strip {
                    return [F128::ZERO; 2];
                }
                out_round_gruen_from_witness_strip_range(
                    witness,
                    round,
                    embed_evals,
                    eq_out,
                    eq_strips,
                    start_strip,
                    end_strip,
                    real_blocks,
                    real_pair_count,
                    has_mixed_pair,
                    virtual_eq,
                    eq_x,
                )
            })
            .reduce(
                || [F128::ZERO; 2],
                |mut acc, part| {
                    for idx in 0..2 {
                        acc[idx] += part[idx];
                    }
                    acc
                },
            );
    }
    out_round_gruen_from_witness_strip_range(
        witness,
        round,
        embed_evals,
        eq_out,
        eq_strips,
        0,
        HOT_STRIPS,
        real_blocks,
        real_pair_count,
        has_mixed_pair,
        virtual_eq,
        eq_x,
    )
}

fn out_round_gruen_from_witness_strip_range(
    witness: &KeccakWitness,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    eq_out: &[F128],
    eq_strips: &[F128],
    start_strip: usize,
    end_strip: usize,
    real_blocks: usize,
    real_pair_count: usize,
    has_mixed_pair: bool,
    virtual_eq: F128,
    eq_x: &[F128],
) -> [F128; 2] {
    let mut acc = [F128::ZERO; 2];
    debug_assert!(real_blocks <= witness.blocks().min(eq_out.len()));
    for strip_idx in start_strip..end_strip {
        let y = strip_idx / 64;
        let z = strip_idx % 64;
        let yz_eq = eq_strips[strip_idx];
        if yz_eq == F128::ZERO {
            continue;
        }
        let strip = witness.pre_chi_strip(round, y, z);
        for pair_idx in 0..real_pair_count {
            let out_idx = 2 * pair_idx;
            let eq = eq_out[out_idx] + eq_out[out_idx + 1];
            let pairs: [(F128, F128); 5] = core::array::from_fn(|x| {
                eval_hot_word_pair_one_inf_from_strip(
                    strip,
                    real_blocks,
                    out_idx,
                    x,
                    y,
                    z,
                    round,
                    embed_evals,
                )
            });
            let values_one: [F128; 5] = core::array::from_fn(|x| pairs[x].0);
            let values_inf: [F128; 5] = core::array::from_fn(|x| pairs[x].1);
            accumulate_chi_gruen_one_inf(
                &mut acc,
                yz_eq * eq,
                values_one,
                values_inf,
                eq_x,
            );
        }
        if has_mixed_pair {
            let out_idx = 2 * real_pair_count;
            let eq = eq_out[out_idx] + eq_out[out_idx + 1];
            let pairs: [(F128, F128); 5] = core::array::from_fn(|x| {
                eval_hot_word_pair_one_inf_from_strip(
                    strip,
                    real_blocks,
                    out_idx,
                    x,
                    y,
                    z,
                    round,
                    embed_evals,
                )
            });
            let values_one: [F128; 5] = core::array::from_fn(|x| pairs[x].0);
            let values_inf: [F128; 5] = core::array::from_fn(|x| pairs[x].1);
            accumulate_chi_gruen_one_inf(
                &mut acc,
                yz_eq * eq,
                values_one,
                values_inf,
                eq_x,
            );
        }
        if virtual_eq != F128::ZERO {
            let pairs: [(F128, F128); 5] = core::array::from_fn(|x| {
                eval_hot_word_pair_one_inf_from_strip(
                    strip,
                    real_blocks,
                    real_blocks,
                    x,
                    y,
                    z,
                    round,
                    embed_evals,
                )
            });
            let values_one: [F128; 5] = core::array::from_fn(|x| pairs[x].0);
            let values_inf: [F128; 5] = core::array::from_fn(|x| pairs[x].1);
            accumulate_chi_gruen_one_inf(
                &mut acc,
                yz_eq * virtual_eq,
                values_one,
                values_inf,
                eq_x,
            );
        }
    }
    acc
}

fn out_round_gruen_strip_range(
    state: &[F128],
    eq_out: &[F128],
    eq_strips: &[F128],
    start_strip: usize,
    end_strip: usize,
    real_blocks: usize,
    real_pair_count: usize,
    has_mixed_pair: bool,
    virtual_eq: F128,
    eq_x: &[F128],
) -> [F128; 2] {
    let mut acc = [F128::ZERO; 2];
    let out_len = eq_out.len();
    for strip_idx in start_strip..end_strip {
        let yz_eq = eq_strips[strip_idx];
        if yz_eq == F128::ZERO {
            continue;
        }
        let strip_base = strip_idx * out_len * 5;
        for pair_idx in 0..real_pair_count {
            let out_idx = 2 * pair_idx;
            let eq = eq_out[out_idx] + eq_out[out_idx + 1];
            let lo_base = strip_base + out_idx * 5;
            let hi_base = strip_base + (out_idx + 1) * 5;
            let values_lo: [F128; 5] = core::array::from_fn(|x| state[lo_base + x]);
            let values_hi: [F128; 5] = core::array::from_fn(|x| state[hi_base + x]);
            accumulate_chi_gruen(
                &mut acc,
                yz_eq * eq,
                values_lo,
                values_hi,
                eq_x,
            );
        }
        if has_mixed_pair {
            let out_idx = 2 * real_pair_count;
            let eq = eq_out[out_idx] + eq_out[out_idx + 1];
            let lo_base = strip_base + out_idx * 5;
            let hi_base = strip_base + real_blocks * 5;
            let values_lo: [F128; 5] = core::array::from_fn(|x| state[lo_base + x]);
            let values_hi: [F128; 5] = core::array::from_fn(|x| state[hi_base + x]);
            accumulate_chi_gruen(
                &mut acc,
                yz_eq * eq,
                values_lo,
                values_hi,
                eq_x,
            );
        }
        if virtual_eq != F128::ZERO {
            let default_base = strip_base + real_blocks * 5;
            let values_lo: [F128; 5] = core::array::from_fn(|x| state[default_base + x]);
            accumulate_chi_gruen(
                &mut acc,
                yz_eq * virtual_eq,
                values_lo,
                values_lo,
                eq_x,
            );
        }
    }
    acc
}

fn build_folded_hot_state_and_next_gruen_from_witness(
    witness: &KeccakWitness,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    eq_out: &mut [F128],
    eq_strips: &[F128],
    active_blocks: usize,
    real_blocks: usize,
    r: F128,
    eq_x: &[F128],
    workers: usize,
    state: &mut [F128],
) -> [F128; 2] {
    debug_assert!(active_blocks >= 4);
    debug_assert_eq!(state.len() % (HOT_STRIPS * 5), 0);
    let out_len = state.len() / (HOT_STRIPS * 5);
    debug_assert_eq!(out_len, active_blocks);
    fold_eq_table_for_gruen(eq_out, active_blocks);

    let next_active = active_blocks / 2;
    let next_real = real_blocks.div_ceil(2);
    let next_pair_count = next_active / 2;
    let next_real_pair_count = next_real / 2;
    let has_mixed_pair = next_real % 2 == 1;
    let virtual_pair_start = next_real.div_ceil(2);
    let virtual_eq = eq_pair_sum_for_gruen(eq_out, virtual_pair_start, next_pair_count);
    let use_parallel = workers > 1 && HOT_STRIPS >= workers * 2;

    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let chunk_strips = HOT_STRIPS.div_ceil(worker_count);
        let strip_len = out_len * 5;
        let chunk_len = chunk_strips * strip_len;
        return state
            .par_chunks_mut(chunk_len)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let start_strip = chunk_idx * chunk_strips;
                build_folded_hot_state_and_next_gruen_from_witness_strips(
                    witness,
                    round,
                    embed_evals,
                    chunk,
                    out_len,
                    real_blocks,
                    next_real,
                    next_real_pair_count,
                    has_mixed_pair,
                    virtual_eq,
                    start_strip,
                    eq_out,
                    eq_strips,
                    r,
                    eq_x,
                )
            })
            .reduce(
                || [F128::ZERO; 2],
                |mut acc, part| {
                    acc[0] += part[0];
                    acc[1] += part[1];
                    acc
                },
            );
    }

    build_folded_hot_state_and_next_gruen_from_witness_strips(
        witness,
        round,
        embed_evals,
        state,
        out_len,
        real_blocks,
        next_real,
        next_real_pair_count,
        has_mixed_pair,
        virtual_eq,
        0,
        eq_out,
        eq_strips,
        r,
        eq_x,
    )
}

#[inline(always)]
fn folded_witness_pair_values(
    strip: &[u128],
    real_blocks: usize,
    old_idx: usize,
    x_base: (usize, usize, usize),
    embed_evals: &[[F128; 256]; WORD_BYTES],
    inf_scale: F128,
) -> [F128; 5] {
    let (round, y, z) = x_base;
    core::array::from_fn(|x| {
        let (one, inf) = eval_hot_word_pair_one_inf_from_strip(
            strip,
            real_blocks,
            old_idx,
            x,
            y,
            z,
            round,
            embed_evals,
        );
        one + inf_scale * inf
    })
}

fn build_folded_hot_state_and_next_gruen_from_witness_strips(
    witness: &KeccakWitness,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    state: &mut [F128],
    out_len: usize,
    real_blocks: usize,
    next_real: usize,
    next_real_pair_count: usize,
    has_mixed_pair: bool,
    virtual_eq: F128,
    start_strip: usize,
    eq_out: &[F128],
    eq_strips: &[F128],
    r: F128,
    eq_x: &[F128],
) -> [F128; 2] {
    let strip_len = out_len * 5;
    debug_assert_eq!(state.len() % strip_len, 0);
    let mut acc = [F128::ZERO; 2];

    for local_strip in 0..state.len() / strip_len {
        let strip_idx = start_strip + local_strip;
        let y = strip_idx / 64;
        let z = strip_idx % 64;
        let yz_eq = eq_strips[strip_idx];
        let real_strip = witness.pre_chi_strip(round, y, z);
        let out_strip = &mut state[local_strip * strip_len..][..strip_len];
        let coords = (round, y, z);
        let inf_scale = F128::ONE + r;

        if yz_eq != F128::ZERO {
            for pair_idx in 0..next_real_pair_count {
                let out_idx = 2 * pair_idx;
                let values_lo =
                    folded_witness_pair_values(real_strip, real_blocks, 2 * out_idx, coords, embed_evals, inf_scale);
                let values_hi = folded_witness_pair_values(
                    real_strip,
                    real_blocks,
                    2 * out_idx + 2,
                    coords,
                    embed_evals,
                    inf_scale,
                );
                for x in 0..5 {
                    out_strip[out_idx * 5 + x] = values_lo[x];
                    out_strip[(out_idx + 1) * 5 + x] = values_hi[x];
                }
                let eq = eq_out[out_idx] + eq_out[out_idx + 1];
                accumulate_chi_gruen(&mut acc, yz_eq * eq, values_lo, values_hi, eq_x);
            }

            if has_mixed_pair {
                let out_idx = 2 * next_real_pair_count;
                let values_lo =
                    folded_witness_pair_values(real_strip, real_blocks, 2 * out_idx, coords, embed_evals, inf_scale);
                let values_hi =
                    folded_witness_pair_values(real_strip, real_blocks, real_blocks, coords, embed_evals, inf_scale);
                for x in 0..5 {
                    out_strip[out_idx * 5 + x] = values_lo[x];
                    out_strip[(out_idx + 1) * 5 + x] = values_hi[x];
                }
                let eq = eq_out[out_idx] + eq_out[out_idx + 1];
                accumulate_chi_gruen(&mut acc, yz_eq * eq, values_lo, values_hi, eq_x);
            }

            if virtual_eq != F128::ZERO {
                let values =
                    folded_witness_pair_values(real_strip, real_blocks, real_blocks, coords, embed_evals, inf_scale);
                accumulate_chi_gruen(&mut acc, yz_eq * virtual_eq, values, values, eq_x);
            }
        } else {
            for out in 0..next_real {
                let values =
                    folded_witness_pair_values(real_strip, real_blocks, 2 * out, coords, embed_evals, inf_scale);
                for x in 0..5 {
                    out_strip[out * 5 + x] = values[x];
                }
            }
        }

        if next_real < out_len {
            let values =
                folded_witness_pair_values(real_strip, real_blocks, real_blocks, coords, embed_evals, inf_scale);
            for x in 0..5 {
                out_strip[next_real * 5 + x] = values[x];
            }
        }
    }

    acc
}

fn build_folded_hot_state_from_witness(
    witness: &KeccakWitness,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    active_blocks: usize,
    real_blocks: usize,
    r: F128,
    workers: usize,
    state: &mut [F128],
) {
    debug_assert_eq!(state.len() % (HOT_STRIPS * 5), 0);
    let out_len = state.len() / (HOT_STRIPS * 5);
    debug_assert_eq!(active_blocks, out_len);
    let strip_len = out_len * 5;
    let use_parallel = workers > 1 && HOT_STRIPS >= workers * 2;
    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let chunk_strips = HOT_STRIPS.div_ceil(worker_count);
        let chunk_len = chunk_strips * strip_len;
        state
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_strip = chunk_idx * chunk_strips;
                fill_folded_hot_state_strips_from_witness(
                    witness,
                    round,
                    embed_evals,
                    out_len,
                    real_blocks,
                    start_strip,
                    r,
                    chunk,
                );
            });
        return;
    }
    fill_folded_hot_state_strips_from_witness(
        witness,
        round,
        embed_evals,
        out_len,
        real_blocks,
        0,
        r,
        state,
    );
}

fn fill_folded_hot_state_strips_from_witness(
    witness: &KeccakWitness,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
    out_len: usize,
    real_blocks: usize,
    start_strip: usize,
    r: F128,
    state: &mut [F128],
) {
    let strip_len = out_len * 5;
    debug_assert_eq!(state.len() % strip_len, 0);
    debug_assert!(real_blocks <= witness.blocks().min(out_len));
    let next_real_blocks = real_blocks.div_ceil(2);
    for local_strip in 0..state.len() / strip_len {
        let strip_idx = start_strip + local_strip;
        let y = strip_idx / 64;
        let z = strip_idx % 64;
        let strip = witness.pre_chi_strip(round, y, z);
        let out_strip = &mut state[local_strip * strip_len..][..strip_len];
        for out in 0..next_real_blocks {
            let a: [F128; 5] = core::array::from_fn(|x| {
                eval_hot_word_from_strip(strip, real_blocks, 2 * out, x, y, z, round, embed_evals)
            });
            let b: [F128; 5] = core::array::from_fn(|x| {
                eval_hot_word_from_strip(strip, real_blocks, 2 * out + 1, x, y, z, round, embed_evals)
            });
            let folded = fold5(a, b, r);
            for x in 0..5 {
                out_strip[out * 5 + x] = folded[x];
            }
        }
        if next_real_blocks < out_len {
            for x in 0..5 {
                out_strip[next_real_blocks * 5 + x] =
                    eval_default_hot_word(round, x, y, z, embed_evals);
            }
        }
    }
}

#[inline(always)]
fn hot_word_from_strip(
    strip: &[u128],
    real_blocks: usize,
    out: usize,
    x: usize,
    y: usize,
    z: usize,
    round: usize,
) -> u128 {
    if out < real_blocks {
        strip[out * 5 + x]
    } else {
        protocol_state::zero_pre_chi_word(round, x, y, z)
    }
}

#[inline(always)]
fn eval_word_or_zero(word: u128, tables: &[[F128; 256]; WORD_BYTES]) -> F128 {
    if word == 0 {
        F128::ZERO
    } else {
        eval_word(word, tables)
    }
}

#[inline(always)]
fn eval_hot_word_pair_one_inf_from_strip(
    strip: &[u128],
    real_blocks: usize,
    out: usize,
    x: usize,
    y: usize,
    z: usize,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
) -> (F128, F128) {
    let lo = hot_word_from_strip(strip, real_blocks, out, x, y, z, round);
    let hi = hot_word_from_strip(strip, real_blocks, out + 1, x, y, z, round);
    (eval_word_or_zero(hi, embed_evals), eval_word_or_zero(lo ^ hi, embed_evals))
}

#[inline(always)]
fn eval_hot_word_from_strip(
    strip: &[u128],
    real_blocks: usize,
    out: usize,
    x: usize,
    y: usize,
    z: usize,
    round: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
) -> F128 {
    eval_word_or_zero(
        hot_word_from_strip(strip, real_blocks, out, x, y, z, round),
        embed_evals,
    )
}

#[inline(always)]
fn eval_default_hot_word(
    round: usize,
    x: usize,
    y: usize,
    z: usize,
    embed_evals: &[[F128; 256]; WORD_BYTES],
) -> F128 {
    let word = protocol_state::zero_pre_chi_word(round, x, y, z);
    eval_word_or_zero(word, embed_evals)
}

#[inline(always)]
fn tail_round_gruen(
    state: &[Vec<F128>; 5],
    eq: &[F128],
    active_len: usize,
    eq_x: &[F128],
) -> [F128; 2] {
    let mut acc = [F128::ZERO; 2];
    for idx in (0..active_len).step_by(2) {
        let scale = eq[idx] + eq[idx + 1];
        let values_lo: [F128; 5] = core::array::from_fn(|x| state[x][idx]);
        let values_hi: [F128; 5] = core::array::from_fn(|x| state[x][idx + 1]);
        accumulate_chi_gruen(
            &mut acc,
            scale,
            values_lo,
            values_hi,
            eq_x,
        );
    }
    acc
}

#[inline(always)]
fn accumulate_chi_gruen(
    acc: &mut [F128; 2],
    scale: F128,
    values_lo: [F128; 5],
    values_hi: [F128; 5],
    eq_x: &[F128],
) {
    // `s1 = sum_x eq_x[x]*g1_x` and `sinf = sum_x eq_x[x]*ginf_x` are dot
    // products; accumulate them unreduced and reduce once. Each `g*` is itself
    // a sum of two products, so it too is folded into a deferred accumulator
    // and reduced a single time. This cuts the per-term reduction count from
    // ~6 to ~2 and leaves the independent PMULLs free to pipeline.
    let mut g1 = [F128::ZERO; 5];
    let mut ginf = [F128::ZERO; 5];
    for x in 0..5 {
        let left = (x + 1) % 5;
        let right = (x + 2) % 5;
        let c1 = values_hi[x] + values_hi[right];
        let delta_left = values_lo[left] + values_hi[left];
        let delta_right = values_lo[right] + values_hi[right];
        let delta_c = values_lo[x] + values_hi[x] + delta_right;

        let mut g1_acc = F128Acc::new();
        g1_acc.accumulate(values_hi[left], values_hi[right]);
        g1_acc.accumulate(c1, c1);
        g1[x] = g1_acc.finalize();

        let mut ginf_acc = F128Acc::new();
        ginf_acc.accumulate(delta_left, delta_right);
        ginf_acc.accumulate(delta_c, delta_c);
        ginf[x] = ginf_acc.finalize();
    }
    acc[0] += scale * F128::dot_product(&eq_x[..5], &g1);
    acc[1] += scale * F128::dot_product(&eq_x[..5], &ginf);
}

#[inline(always)]
fn accumulate_chi_gruen_one_inf(
    acc: &mut [F128; 2],
    scale: F128,
    values_one: [F128; 5],
    values_inf: [F128; 5],
    eq_x: &[F128],
) {
    let mut g1 = [F128::ZERO; 5];
    let mut ginf = [F128::ZERO; 5];
    for x in 0..5 {
        let left = (x + 1) % 5;
        let right = (x + 2) % 5;
        let c1 = values_one[x] + values_one[right];
        let c_inf = values_inf[x] + values_inf[right];

        let mut g1_acc = F128Acc::new();
        g1_acc.accumulate(values_one[left], values_one[right]);
        g1_acc.accumulate(c1, c1);
        g1[x] = g1_acc.finalize();

        let mut ginf_acc = F128Acc::new();
        ginf_acc.accumulate(values_inf[left], values_inf[right]);
        ginf_acc.accumulate(c_inf, c_inf);
        ginf[x] = ginf_acc.finalize();
    }
    acc[0] += scale * F128::dot_product(&eq_x[..5], &g1);
    acc[1] += scale * F128::dot_product(&eq_x[..5], &ginf);
}

fn fold_state(state: &mut [Vec<F128>; 5], active_len: usize, r: F128) {
    for table in state {
        fold_table(table, active_len, r);
    }
}

#[inline(always)]
fn fold_eq_table_for_gruen(table: &mut [F128], active_len: usize) {
    for idx in 0..active_len / 2 {
        table[idx] = table[2 * idx] + table[2 * idx + 1];
    }
}

fn fold_hot_state_and_next_gruen(
    state: &mut [F128],
    eq_out: &mut [F128],
    eq_strips: &[F128],
    active_blocks: usize,
    real_blocks: usize,
    r: F128,
    eq_x: &[F128],
    workers: usize,
) -> [F128; 2] {
    debug_assert!(active_blocks >= 4);
    debug_assert_eq!(state.len() % (HOT_STRIPS * 5), 0);
    let out_len = state.len() / (HOT_STRIPS * 5);
    debug_assert_eq!(out_len, active_blocks);
    fold_eq_table_for_gruen(eq_out, active_blocks);

    let next_active = active_blocks / 2;
    let next_real = real_blocks.div_ceil(2);
    let next_pair_count = next_active / 2;
    let next_real_pair_count = next_real / 2;
    let has_mixed_pair = next_real % 2 == 1;
    let virtual_pair_start = next_real.div_ceil(2);
    let virtual_eq = eq_pair_sum_for_gruen(eq_out, virtual_pair_start, next_pair_count);

    let use_parallel = workers > 1 && HOT_STRIPS >= workers * 2;
    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let chunk_strips = HOT_STRIPS.div_ceil(worker_count);
        let strip_len = out_len * 5;
        let chunk_len = chunk_strips * strip_len;
        return state
            .par_chunks_mut(chunk_len)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let start_strip = chunk_idx * chunk_strips;
                fold_hot_state_and_next_gruen_strips(
                    chunk,
                    out_len,
                    real_blocks,
                    next_real,
                    next_real_pair_count,
                    has_mixed_pair,
                    virtual_eq,
                    start_strip,
                    eq_out,
                    eq_strips,
                    r,
                    eq_x,
                )
            })
            .reduce(
                || [F128::ZERO; 2],
                |mut acc, part| {
                    acc[0] += part[0];
                    acc[1] += part[1];
                    acc
                },
            );
    }

    fold_hot_state_and_next_gruen_strips(
        state,
        out_len,
        real_blocks,
        next_real,
        next_real_pair_count,
        has_mixed_pair,
        virtual_eq,
        0,
        eq_out,
        eq_strips,
        r,
        eq_x,
    )
}

#[inline(always)]
fn fold_hot_pair_values(strip: &[F128], old_idx: usize, real_blocks: usize, r: F128) -> [F128; 5] {
    let lo = old_idx * 5;
    let hi = if old_idx + 1 < real_blocks {
        (old_idx + 1) * 5
    } else {
        real_blocks * 5
    };
    let a = [
        strip[lo],
        strip[lo + 1],
        strip[lo + 2],
        strip[lo + 3],
        strip[lo + 4],
    ];
    let b = [
        strip[hi],
        strip[hi + 1],
        strip[hi + 2],
        strip[hi + 3],
        strip[hi + 4],
    ];
    fold5(a, b, r)
}

fn fold_hot_state_and_next_gruen_strips(
    state: &mut [F128],
    out_len: usize,
    real_blocks: usize,
    next_real: usize,
    next_real_pair_count: usize,
    has_mixed_pair: bool,
    virtual_eq: F128,
    start_strip: usize,
    eq_out: &[F128],
    eq_strips: &[F128],
    r: F128,
    eq_x: &[F128],
) -> [F128; 2] {
    let strip_len = out_len * 5;
    debug_assert_eq!(state.len() % strip_len, 0);
    let mut acc = [F128::ZERO; 2];
    for local_strip in 0..state.len() / strip_len {
        let strip_idx = start_strip + local_strip;
        let yz_eq = eq_strips[strip_idx];
        let strip = &mut state[local_strip * strip_len..][..strip_len];

        if yz_eq != F128::ZERO {
            for pair_idx in 0..next_real_pair_count {
                let out_idx = 2 * pair_idx;
                let values_lo = fold_hot_pair_values(strip, 2 * out_idx, real_blocks, r);
                let values_hi = fold_hot_pair_values(strip, 2 * out_idx + 2, real_blocks, r);
                for x in 0..5 {
                    strip[out_idx * 5 + x] = values_lo[x];
                    strip[(out_idx + 1) * 5 + x] = values_hi[x];
                }
                let eq = eq_out[out_idx] + eq_out[out_idx + 1];
                accumulate_chi_gruen(&mut acc, yz_eq * eq, values_lo, values_hi, eq_x);
            }

            if has_mixed_pair {
                let out_idx = 2 * next_real_pair_count;
                let values_lo = fold_hot_pair_values(strip, 2 * out_idx, real_blocks, r);
                let values_hi = fold_hot_pair_values(strip, real_blocks, real_blocks, r);
                for x in 0..5 {
                    strip[out_idx * 5 + x] = values_lo[x];
                    strip[(out_idx + 1) * 5 + x] = values_hi[x];
                }
                let eq = eq_out[out_idx] + eq_out[out_idx + 1];
                accumulate_chi_gruen(&mut acc, yz_eq * eq, values_lo, values_hi, eq_x);
            }

            if virtual_eq != F128::ZERO {
                let values = fold_hot_pair_values(strip, real_blocks, real_blocks, r);
                accumulate_chi_gruen(&mut acc, yz_eq * virtual_eq, values, values, eq_x);
            }
        } else {
            for out in 0..next_real {
                let values = fold_hot_pair_values(strip, 2 * out, real_blocks, r);
                for x in 0..5 {
                    strip[out * 5 + x] = values[x];
                }
            }
        }

        if next_real < out_len {
            let values = fold_hot_pair_values(strip, real_blocks, real_blocks, r);
            for x in 0..5 {
                strip[next_real * 5 + x] = values[x];
            }
        }
    }
    acc
}

#[inline(always)]
fn fold_hot_state(state: &mut [F128], real_blocks: usize, r: F128, workers: usize) {
    debug_assert_eq!(state.len() % (HOT_STRIPS * 5), 0);
    let out_len = state.len() / (HOT_STRIPS * 5);
    let strip_len = out_len * 5;
    let use_parallel = workers > 1 && HOT_STRIPS >= workers * 2;
    if use_parallel {
        let worker_count = workers.min(HOT_STRIPS);
        let chunk_strips = HOT_STRIPS.div_ceil(worker_count);
        let chunk_len = chunk_strips * strip_len;
        state.par_chunks_mut(chunk_len).for_each(|chunk| {
            fold_hot_state_strips(chunk, out_len, real_blocks, r);
        });
        return;
    }
    fold_hot_state_strips(state, out_len, real_blocks, r);
}

#[inline(always)]
fn fold_hot_state_strips(
    state: &mut [F128],
    out_len: usize,
    real_blocks: usize,
    r: F128,
) {
    let strip_len = out_len * 5;
    debug_assert_eq!(state.len() % strip_len, 0);
    let next_real_blocks = real_blocks.div_ceil(2);
    for strip in state.chunks_mut(strip_len) {
        for out in 0..next_real_blocks {
            let dst = out * 5;
            let lo = (2 * out) * 5;
            let hi = if 2 * out + 1 < real_blocks {
                (2 * out + 1) * 5
            } else {
                real_blocks * 5
            };
            let a = [
                strip[lo],
                strip[lo + 1],
                strip[lo + 2],
                strip[lo + 3],
                strip[lo + 4],
            ];
            let b = [
                strip[hi],
                strip[hi + 1],
                strip[hi + 2],
                strip[hi + 3],
                strip[hi + 4],
            ];
            let folded = fold5(a, b, r);
            for x in 0..5 {
                strip[dst + x] = folded[x];
            }
        }
        if next_real_blocks < out_len {
            for x in 0..5 {
                strip[next_real_blocks * 5 + x] = strip[real_blocks * 5 + x];
            }
        }
    }
}

#[inline(always)]
fn fold_table(table: &mut [F128], active_len: usize, r: F128) {
    for idx in 0..active_len / 2 {
        let lo = table[2 * idx];
        let hi = table[2 * idx + 1];
        table[idx] = lo + r * (lo + hi);
    }
}

#[inline]
fn hot_idx(out_len: usize, out: usize, y: usize, z: usize, x: usize) -> usize {
    debug_assert!(out < out_len);
    debug_assert!(x < 5 && y < 5 && z < 64);
    ((y * 64 + z) * out_len + out) * 5 + x
}

fn embed_word_poly(word: u128) -> BoolPoly {
    let tables = embed_tables();
    let mut out = BoolPoly::ZERO;
    let bytes = word.to_le_bytes();
    for byte_idx in 0..WORD_BYTES {
        out ^= tables[byte_idx][bytes[byte_idx] as usize];
    }
    out
}

fn embed_word_poly_matrix(word: u128) -> BoolPoly {
    let input = [word as u64, (word >> 64) as u64];
    let matrix = rmfe::embedding_matrix();
    let mut limbs = [0u64; 4];
    for coeff in 0..rmfe::PRODUCT_DEGREE {
        let row = matrix.row(coeff);
        let parity = ((row[0] & input[0]).count_ones() ^ (row[1] & input[1]).count_ones()) & 1;
        if parity != 0 {
            limbs[coeff / 64] ^= 1u64 << (coeff % 64);
        }
    }
    BoolPoly::from_limbs(limbs)
}

fn embed_tables() -> &'static [[BoolPoly; 256]; WORD_BYTES] {
    static TABLES: OnceLock<[[BoolPoly; 256]; WORD_BYTES]> = OnceLock::new();
    TABLES.get_or_init(|| {
        core::array::from_fn(|byte_idx| {
            core::array::from_fn(|value| embed_word_poly_matrix((value as u128) << (8 * byte_idx)))
        })
    })
}

fn fill_embed_eval_tables(t: F128, out: &mut [[F128; 256]; WORD_BYTES]) {
    let mut basis_evals = [F128::ZERO; rmfe::RMFE_BITS];
    let matrix = rmfe::embedding_matrix();
    for input_bit in 0..rmfe::RMFE_BITS {
        let mut acc = F128::ZERO;
        for coeff in (0..rmfe::PRODUCT_DEGREE).rev() {
            acc *= t;
            if matrix.row(coeff)[input_bit / 64] & (1u64 << (input_bit % 64)) != 0 {
                acc += F128::ONE;
            }
        }
        basis_evals[input_bit] = acc;
    }

    for byte_idx in 0..WORD_BYTES {
        out[byte_idx][0] = F128::ZERO;
        for bit in 0..8 {
            out[byte_idx][1 << bit] = basis_evals[8 * byte_idx + bit];
        }
        for value in 0usize..256 {
            if value == 0 || value.is_power_of_two() {
                continue;
            }
            let low_bit = value & value.wrapping_neg();
            out[byte_idx][value] = out[byte_idx][value ^ low_bit] + out[byte_idx][low_bit];
        }
    }
}

fn eval_word(word: u128, tables: &[[F128; 256]; WORD_BYTES]) -> F128 {
    let mut out = F128::ZERO;
    let bytes = word.to_le_bytes();
    for byte_idx in 0..WORD_BYTES {
        out += tables[byte_idx][bytes[byte_idx] as usize];
    }
    out
}

fn accumulate_wide(
    buckets: &mut [[WideBoolPoly; BUCKET_COUNT]],
    product: WideBoolPoly,
    scalar: F128,
    bucket_bits: usize,
) {
    let raw = scalar.raw();
    match bucket_bits {
        8 => {
            for (limb_idx, &value) in raw.to_le_bytes().iter().enumerate() {
                buckets[limb_idx][value as usize] ^= product;
            }
        }
        7 => {
            for limb_idx in 0..bucket_limbs(7) {
                let value = ((raw >> (7 * limb_idx)) & 0x7f) as usize;
                buckets[limb_idx][value] ^= product;
            }
        }
        6 => {
            for limb_idx in 0..bucket_limbs(6) {
                let value = ((raw >> (6 * limb_idx)) & 0x3f) as usize;
                buckets[limb_idx][value] ^= product;
            }
        }
        _ => unreachable!(),
    }
}

fn xor_wide_bucket_sets(
    out: &mut [[WideBoolPoly; BUCKET_COUNT]],
    rhs: &[[WideBoolPoly; BUCKET_COUNT]],
    bucket_limbs: usize,
    bucket_count: usize,
) {
    for limb_idx in 0..bucket_limbs {
        for value in 0..bucket_count {
            out[limb_idx][value] ^= rhs[limb_idx][value];
        }
    }
}

fn recover_wide_buckets(
    buckets: &[[WideBoolPoly; BUCKET_COUNT]],
    bucket_bits: usize,
    out: &mut [F128],
) {
    let bucket_count = 1usize << bucket_bits;
    for (limb_idx, bucket_set) in buckets[..bucket_limbs(bucket_bits)].iter().enumerate() {
        for value in 1..bucket_count {
            let product = bucket_set[value];
            if product.is_zero() {
                continue;
            }
            // Fold every set bit of `value` into a single scalar first, so the
            // (dense, ~384-bit) product is scanned once per bucket instead of
            // once per set bit — a ~popcount(value)x reduction in XOR traffic.
            let mut scalar = 0u128;
            let mut bits = value;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let scalar_bit = bucket_bits * limb_idx + bit;
                if scalar_bit < 128 {
                    scalar |= 1u128 << scalar_bit;
                }
                bits &= bits - 1;
            }
            if scalar != 0 {
                add_wide_raw_bit(out, product, F128::from_raw(scalar));
            }
        }
    }
}

fn add_wide_raw_bit(out: &mut [F128], product: WideBoolPoly, raw_bit: F128) {
    let limbs = product.limbs();
    for (limb_idx, &limb) in limbs.iter().enumerate() {
        let mut bits = limb;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            out[64 * limb_idx + bit] += raw_bit;
            bits &= bits - 1;
        }
    }
}
