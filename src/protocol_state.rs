//! Protocol-owned witness and reusable scratch storage.
//!
//! The Keccak witness is stored in the protocol layout directly: each word is
//! a 96-bit packed column, where bit `i` belongs to Keccak instance `i`.
//! Pre-chi states are laid out as `(round, y, z, block, x)` so a fixed
//! chi row is contiguous across packed blocks.

use std::{sync::OnceLock, thread};

pub const PACKED_KECCAKS: usize = 96;
pub const KECCAK_LANES: usize = 25;
pub const LANE_BITS: usize = 64;
pub const KECCAK_BITS: usize = KECCAK_LANES * LANE_BITS;
pub const KECCAK_ROUNDS: usize = 24;
pub const PACKED_MASK: u128 = (1u128 << PACKED_KECCAKS) - 1;

const RHO_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

const ROUND_CONSTANTS: [u64; KECCAK_ROUNDS] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

pub struct ProtocolState {
    pub witness: KeccakWitness,
    pub scratches: ReusableScratches,
}

impl ProtocolState {
    pub fn new() -> Self {
        Self::new_blocks(1, 1)
    }

    pub fn new_for_keccaks(n_keccaks: usize, workers: usize) -> Self {
        Self::new_blocks(n_keccaks.div_ceil(PACKED_KECCAKS).max(1), workers)
    }

    pub fn new_blocks(blocks: usize, workers: usize) -> Self {
        assert!(blocks > 0);
        let workers = workers.max(1).min(blocks);
        Self {
            witness: KeccakWitness::new(blocks),
            scratches: ReusableScratches::new(workers),
        }
    }

    pub fn generate_keccak(&mut self) {
        self.witness.generate(&mut self.scratches);
    }
}

pub struct ReusableScratches {
    workers: Vec<WorkerScratch>,
}

impl ReusableScratches {
    pub fn new(workers: usize) -> Self {
        Self {
            workers: (0..workers.max(1)).map(|_| WorkerScratch::new()).collect(),
        }
    }
}

struct WorkerScratch {
    current: Box<[u128]>,
}

impl WorkerScratch {
    fn new() -> Self {
        Self {
            current: zero_boxed_state(),
        }
    }
}

pub struct KeccakWitness {
    blocks: usize,
    input: Box<[u128]>,
    pre_chi: Box<[u128]>,
    output: Box<[u128]>,
}

impl KeccakWitness {
    pub fn new(blocks: usize) -> Self {
        assert!(blocks > 0);
        Self {
            blocks,
            input: zero_boxed_states(blocks),
            pre_chi: vec![0u128; blocks * KECCAK_ROUNDS * KECCAK_BITS].into_boxed_slice(),
            output: zero_boxed_states(blocks),
        }
    }

    pub fn generate(&mut self, scratches: &mut ReusableScratches) {
        assert!(!scratches.workers.is_empty());
        assert!(scratches.workers.len() <= self.blocks);

        let blocks_per_worker = self.blocks.div_ceil(scratches.workers.len());
        let pre_chi_ptr = PreChiPtr(self.pre_chi.as_mut_ptr());
        let total_blocks = self.blocks;
        thread::scope(|scope| {
            for (worker_idx, ((input, output), worker)) in self
                .input
                .chunks(blocks_per_worker * KECCAK_BITS)
                .zip(self.output.chunks_mut(blocks_per_worker * KECCAK_BITS))
                .zip(scratches.workers.iter_mut())
                .enumerate()
            {
                let start_block = worker_idx * blocks_per_worker;
                scope.spawn(move || {
                    generate_blocks(
                        input,
                        pre_chi_ptr,
                        total_blocks,
                        start_block,
                        output,
                        &mut worker.current,
                    );
                });
            }
        });
    }

    #[inline]
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    #[inline]
    pub fn n_keccaks_capacity(&self) -> usize {
        self.blocks * PACKED_KECCAKS
    }

    #[inline]
    pub fn input(&self) -> &[u128] {
        &self.input
    }

    #[inline]
    pub fn input_mut(&mut self) -> &mut [u128] {
        &mut self.input
    }

    #[inline]
    pub fn input_block(&self, block: usize) -> &[u128] {
        assert!(block < self.blocks);
        &self.input[block * KECCAK_BITS..][..KECCAK_BITS]
    }

    #[inline]
    pub fn input_block_mut(&mut self, block: usize) -> &mut [u128] {
        assert!(block < self.blocks);
        &mut self.input[block * KECCAK_BITS..][..KECCAK_BITS]
    }

    #[inline]
    pub fn pre_chi_word(&self, block: usize, round: usize, x: usize, y: usize, z: usize) -> u128 {
        assert!(block < self.blocks);
        assert!(round < KECCAK_ROUNDS);
        assert!(x < 5 && y < 5 && z < 64);
        self.pre_chi[pre_chi_idx(self.blocks, block, round, x, y, z)]
    }

    pub fn set_pre_chi_word(
        &mut self,
        block: usize,
        round: usize,
        x: usize,
        y: usize,
        z: usize,
        value: u128,
    ) {
        assert!(block < self.blocks);
        assert!(round < KECCAK_ROUNDS);
        assert!(x < 5 && y < 5 && z < 64);
        let idx = pre_chi_idx(self.blocks, block, round, x, y, z);
        self.pre_chi[idx] = value;
    }

    #[inline]
    pub fn pre_chi_strip(&self, round: usize, y: usize, z: usize) -> &[u128] {
        assert!(round < KECCAK_ROUNDS);
        assert!(y < 5 && z < 64);
        let offset = pre_chi_idx(self.blocks, 0, round, 0, y, z);
        &self.pre_chi[offset..][..self.blocks * 5]
    }

    #[inline]
    pub fn output(&self) -> &[u128] {
        &self.output
    }

    #[inline]
    pub fn output_block(&self, block: usize) -> &[u128] {
        assert!(block < self.blocks);
        &self.output[block * KECCAK_BITS..][..KECCAK_BITS]
    }
}

fn generate_blocks(
    input: &[u128],
    pre_chi_ptr: PreChiPtr,
    total_blocks: usize,
    start_block: usize,
    output: &mut [u128],
    current: &mut [u128],
) {
    let blocks = input.len() / KECCAK_BITS;
    assert_eq!(input.len(), blocks * KECCAK_BITS);
    assert_eq!(output.len(), blocks * KECCAK_BITS);
    // rho_pi fully overwrites this scratch every round, so it never needs
    // re-zeroing; allocate once per worker call instead of once per round.
    let mut pre_chi = vec![0u128; KECCAK_BITS].into_boxed_slice();
    for block in 0..blocks {
        let global_block = start_block + block;
        let input = &input[block * KECCAK_BITS..][..KECCAK_BITS];
        for idx in 0..KECCAK_BITS {
            current[idx] = input[idx] & PACKED_MASK;
        }

        for round in 0..KECCAK_ROUNDS {
            theta_packed(current);
            rho_pi_packed(current, &mut pre_chi);
            write_pre_chi_block(pre_chi_ptr, total_blocks, global_block, round, &pre_chi);
            chi_iota_packed(&pre_chi, current, round);
        }
        output[block * KECCAK_BITS..][..KECCAK_BITS].copy_from_slice(current);
    }
}

#[derive(Clone, Copy)]
struct PreChiPtr(*mut u128);

unsafe impl Send for PreChiPtr {}

fn write_pre_chi_block(
    pre_chi: PreChiPtr,
    blocks: usize,
    block: usize,
    round: usize,
    state: &[u128],
) {
    // Destination layout is (round, y, z, block, x): the five x words for a
    // fixed (round, y, z, block) are contiguous, and each y advances by
    // 64 * blocks * 5 words. Walk the source in (y, z, x) order so the loads
    // stay sequential-ish and the stores hit five contiguous slots at a time.
    let round_base = round * blocks * KECCAK_BITS;
    let block_x = block * 5;
    for y in 0..5 {
        let src_y = y * 5 * 64; // state_idx(0, y, 0)
        let dst_y = round_base + (y * 64) * blocks * 5 + block_x;
        for z in 0..64 {
            let src = src_y + z; // state_idx(0, y, z) with stride 64 over x
            let dst = dst_y + z * blocks * 5;
            unsafe {
                let d = pre_chi.0.add(dst);
                *d = *state.get_unchecked(src);
                *d.add(1) = *state.get_unchecked(src + 64);
                *d.add(2) = *state.get_unchecked(src + 128);
                *d.add(3) = *state.get_unchecked(src + 192);
                *d.add(4) = *state.get_unchecked(src + 256);
            }
        }
    }
}

fn zero_boxed_state() -> Box<[u128]> {
    vec![0u128; KECCAK_BITS].into_boxed_slice()
}

fn zero_boxed_states(blocks: usize) -> Box<[u128]> {
    vec![0u128; blocks * KECCAK_BITS].into_boxed_slice()
}

#[inline]
fn theta_packed(state: &mut [u128]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    unsafe {
        theta_packed_neon(state);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    theta_packed_scalar(state);
}

#[inline]
fn rho_pi_packed(input: &[u128], output: &mut [u128]) {
    // rho+pi is a pure permutation with a per-lane cyclic rotation over z.
    // Each destination lane is contiguous in z, so express the rotation as two
    // contiguous slice copies instead of an element-wise gather.
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = RHO_OFFSETS[a][b] as usize;
            let in_base = (a + 5 * b) * 64; // state_idx(a, b, 0)
            let out_base = (x + 5 * y) * 64; // state_idx(x, y, 0)
            let src = &input[in_base..in_base + 64];
            let dst = &mut output[out_base..out_base + 64];
            if r == 0 {
                dst.copy_from_slice(src);
            } else {
                // output[z] = input[(z + 64 - r) % 64]
                dst[..r].copy_from_slice(&src[64 - r..]);
                dst[r..].copy_from_slice(&src[..64 - r]);
            }
        }
    }
}

#[inline]
fn chi_iota_packed(pre_chi: &[u128], output: &mut [u128], round: usize) {
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    unsafe {
        chi_iota_packed_neon(pre_chi, output, round);
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    chi_iota_packed_scalar(pre_chi, output, round);
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[target_feature(enable = "sha3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn theta_packed_neon(state: &mut [u128]) {
    use core::arch::aarch64::*;
    let base = state.as_mut_ptr();
    // c[x][z] = xor over y of state[x, y, z]. Two 3-way EOR3 collapse the
    // five-term parity into two instructions per column.
    let mut c = [[vdupq_n_u64(0); LANE_BITS]; 5];
    for x in 0..5 {
        let col = &mut c[x];
        for z in 0..LANE_BITS {
            let p = base.add(state_idx(x, 0, z)) as *const u64;
            let s0 = vld1q_u64(p);
            let s1 = vld1q_u64(p.add(2 * 5 * 64)); // + one y row (5*64 u128 = 2*.. u64)
            let s2 = vld1q_u64(p.add(2 * 2 * 5 * 64));
            let s3 = vld1q_u64(p.add(2 * 3 * 5 * 64));
            let s4 = vld1q_u64(p.add(2 * 4 * 5 * 64));
            col[z] = veor3q_u64(veor3q_u64(s0, s1, s2), s3, s4);
        }
    }

    // state[x, y, z] ^= c[(x+4)%5][z] ^ c[(x+1)%5][(z+63)%64], fused as EOR3.
    for y in 0..5 {
        for x in 0..5 {
            let xm = (x + 4) % 5;
            let xp = (x + 1) % 5;
            let cm = &c[xm];
            let cp = &c[xp];
            for z in 0..LANE_BITS {
                let idx = state_idx(x, y, z);
                let p = base.add(idx) as *mut u64;
                let sv = vld1q_u64(p);
                let ca = cm[z];
                let cb = cp[(z + 63) % 64];
                vst1q_u64(p, veor3q_u64(sv, ca, cb));
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[target_feature(enable = "sha3")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn chi_iota_packed_neon(pre_chi: &[u128], output: &mut [u128], round: usize) {
    use core::arch::aarch64::*;
    let src = pre_chi.as_ptr();
    let dst = output.as_mut_ptr();
    // chi: out = a ^ ((!b) & c). BCAX computes a ^ (b & !c) directly, so pass
    // (a, c, b). Inputs are 96-bit masked, so !b's high bits are cleared by
    // & c and the result stays masked without an explicit AND.
    for y in 0..5 {
        for x in 0..5 {
            let x1 = (x + 1) % 5;
            let x2 = (x + 2) % 5;
            let pa = src.add(state_idx(x, y, 0)) as *const u64;
            let pb = src.add(state_idx(x1, y, 0)) as *const u64;
            let pc = src.add(state_idx(x2, y, 0)) as *const u64;
            let po = dst.add(state_idx(x, y, 0)) as *mut u64;
            for z in 0..LANE_BITS {
                let a = vld1q_u64(pa.add(2 * z));
                let b = vld1q_u64(pb.add(2 * z));
                let cc = vld1q_u64(pc.add(2 * z));
                vst1q_u64(po.add(2 * z), vbcaxq_u64(a, cc, b));
            }
        }
    }
    let rc = ROUND_CONSTANTS[round];
    for z in 0..LANE_BITS {
        if ((rc >> z) & 1) != 0 {
            output[state_idx(0, 0, z)] ^= PACKED_MASK;
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
fn theta_packed_scalar(state: &mut [u128]) {
    let mut c = [[0u128; LANE_BITS]; 5];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            c[x][z] = state[state_idx(x, 0, z)]
                ^ state[state_idx(x, 1, z)]
                ^ state[state_idx(x, 2, z)]
                ^ state[state_idx(x, 3, z)]
                ^ state[state_idx(x, 4, z)];
        }
    }

    let mut d = [[0u128; LANE_BITS]; 5];
    for x in 0..5 {
        for z in 0..LANE_BITS {
            d[x][z] = c[(x + 4) % 5][z] ^ c[(x + 1) % 5][(z + 63) % 64];
        }
    }

    for y in 0..5 {
        for x in 0..5 {
            for z in 0..LANE_BITS {
                state[state_idx(x, y, z)] ^= d[x][z];
            }
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
fn chi_iota_packed_scalar(pre_chi: &[u128], output: &mut [u128], round: usize) {
    for y in 0..5 {
        for x in 0..5 {
            for z in 0..LANE_BITS {
                let a = pre_chi[state_idx(x, y, z)];
                let b = pre_chi[state_idx((x + 1) % 5, y, z)];
                let c = pre_chi[state_idx((x + 2) % 5, y, z)];
                output[state_idx(x, y, z)] = (a ^ ((!b) & c)) & PACKED_MASK;
            }
        }
    }
    let rc = ROUND_CONSTANTS[round];
    for z in 0..LANE_BITS {
        if ((rc >> z) & 1) != 0 {
            output[state_idx(0, 0, z)] ^= PACKED_MASK;
        }
    }
}

#[inline]
pub fn state_idx(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < 5 && y < 5 && z < 64);
    (x + 5 * y) * 64 + z
}

#[inline]
pub fn pre_chi_idx(
    blocks: usize,
    block: usize,
    round: usize,
    x: usize,
    y: usize,
    z: usize,
) -> usize {
    debug_assert!(block < blocks);
    debug_assert!(round < KECCAK_ROUNDS);
    debug_assert!(x < 5 && y < 5 && z < 64);
    round * blocks * KECCAK_BITS + ((y * 64 + z) * blocks + block) * 5 + x
}

pub fn zero_pre_chi_word(round: usize, x: usize, y: usize, z: usize) -> u128 {
    assert!(round < KECCAK_ROUNDS);
    let trace = ZERO_PRE_CHI.get_or_init(|| {
        let mut witness = KeccakWitness::new(1);
        let mut scratches = ReusableScratches::new(1);
        witness.generate(&mut scratches);
        witness.pre_chi
    });
    trace[pre_chi_idx(1, 0, round, x, y, z)]
}

pub fn round_constant(round: usize) -> u64 {
    assert!(round < KECCAK_ROUNDS);
    ROUND_CONSTANTS[round]
}

static ZERO_PRE_CHI: OnceLock<Box<[u128]>> = OnceLock::new();

pub fn zero_output_word(x: usize, y: usize, z: usize) -> u128 {
    let trace = ZERO_OUTPUT.get_or_init(|| {
        let mut witness = KeccakWitness::new(1);
        let mut scratches = ReusableScratches::new(1);
        witness.generate(&mut scratches);
        witness.output
    });
    trace[state_idx(x, y, z)]
}

static ZERO_OUTPUT: OnceLock<Box<[u128]>> = OnceLock::new();

#[cfg(test)]
pub(crate) fn keccak_f_lanes_with_pre_chi(
    state: &mut [u64; KECCAK_LANES],
    pre_chi_trace: &mut [[u64; KECCAK_LANES]; KECCAK_ROUNDS],
) {
    for round in 0..KECCAK_ROUNDS {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[lane_idx(x, 0)]
                ^ state[lane_idx(x, 1)]
                ^ state[lane_idx(x, 2)]
                ^ state[lane_idx(x, 3)]
                ^ state[lane_idx(x, 4)];
        }

        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for y in 0..5 {
            for x in 0..5 {
                state[lane_idx(x, y)] ^= d[x];
            }
        }

        let mut b = [0u64; KECCAK_LANES];
        for x in 0..5 {
            for y in 0..5 {
                b[lane_idx(y, (2 * x + 3 * y) % 5)] =
                    state[lane_idx(x, y)].rotate_left(RHO_OFFSETS[x][y]);
            }
        }
        pre_chi_trace[round] = b;

        for y in 0..5 {
            let a0 = b[lane_idx(0, y)];
            let a1 = b[lane_idx(1, y)];
            let a2 = b[lane_idx(2, y)];
            let a3 = b[lane_idx(3, y)];
            let a4 = b[lane_idx(4, y)];
            state[lane_idx(0, y)] = a0 ^ ((!a1) & a2);
            state[lane_idx(1, y)] = a1 ^ ((!a2) & a3);
            state[lane_idx(2, y)] = a2 ^ ((!a3) & a4);
            state[lane_idx(3, y)] = a3 ^ ((!a4) & a0);
            state[lane_idx(4, y)] = a4 ^ ((!a0) & a1);
        }

        state[lane_idx(0, 0)] ^= ROUND_CONSTANTS[round];
    }
}

#[inline]
#[cfg(test)]
fn lane_idx(x: usize, y: usize) -> usize {
    x + 5 * y
}

#[cfg(test)]
mod packed_layout_tests {
    use super::*;

    fn fill_random(input: &mut [u128], seed: u128) {
        let mut rng = seed | 1;
        for slot in input.iter_mut() {
            rng = rng
                .wrapping_mul(0xda94_2042_e4dd_58b5_94d0_49bb_1331_11eb)
                .rotate_left(41);
            let lo = rng as u64 as u128;
            rng = rng
                .wrapping_mul(0x2545_f491_4f6c_dd1d_9e37_79b9_7f4a_7c15)
                .rotate_left(29);
            let hi = rng as u64 as u128;
            *slot = (lo | (hi << 64)) & PACKED_MASK;
        }
    }

    // The single-block generator is validated bit-for-bit against a scalar
    // Keccak reference elsewhere; here we check that the multi-block, multi-
    // worker path (including the transposed pre_chi write and worker split)
    // reproduces the single-block result for every block, over random inputs.
    #[test]
    fn multi_block_matches_single_block() {
        for &(blocks, workers) in &[(1usize, 1usize), (3, 1), (4, 2), (7, 3), (16, 16), (5, 4)] {
            let workers = workers.min(blocks);
            let mut multi = KeccakWitness::new(blocks);
            let seed = (blocks as u128) << 64 | (workers as u128) | 0x9e37_79b9u128 << 96;
            fill_random(multi.input_mut(), seed);

            // Reference: run each block on its own single-block witness.
            let mut singles = Vec::with_capacity(blocks);
            for block in 0..blocks {
                let mut single = KeccakWitness::new(1);
                single
                    .input_block_mut(0)
                    .copy_from_slice(multi.input_block(block));
                let mut scr = ReusableScratches::new(1);
                single.generate(&mut scr);
                singles.push(single);
            }

            let mut scr = ReusableScratches::new(workers);
            multi.generate(&mut scr);

            for block in 0..blocks {
                let single = &singles[block];
                assert_eq!(
                    multi.output_block(block),
                    single.output_block(0),
                    "output mismatch blocks={blocks} workers={workers} block={block}",
                );
                for round in 0..KECCAK_ROUNDS {
                    for x in 0..5 {
                        for y in 0..5 {
                            for z in 0..64 {
                                assert_eq!(
                                    multi.pre_chi_word(block, round, x, y, z),
                                    single.pre_chi_word(0, round, x, y, z),
                                    "pre_chi mismatch blocks={blocks} workers={workers} \
                                     block={block} round={round} ({x},{y},{z})",
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
