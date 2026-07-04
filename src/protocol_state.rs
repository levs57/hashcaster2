//! Protocol-owned witness and reusable scratch storage.
//!
//! The Keccak witness is stored in the protocol layout directly: each entry is
//! a 96-bit packed column, where bit `i` belongs to Keccak instance `i`.

pub const PACKED_KECCAKS: usize = 96;
pub const KECCAK_LANES: usize = 25;
pub const LANE_BITS: usize = 64;
pub const KECCAK_BITS: usize = KECCAK_LANES * LANE_BITS;
pub const KECCAK_ROUNDS: usize = 24;
pub const KECCAK_STATES: usize = KECCAK_ROUNDS + 1;
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
    pub fn new_keccak(initial: &[u128; KECCAK_BITS]) -> Self {
        Self {
            witness: KeccakWitness::generate(initial),
            scratches: ReusableScratches::new(),
        }
    }
}

#[derive(Default)]
pub struct ReusableScratches;

impl ReusableScratches {
    pub fn new() -> Self {
        Self
    }
}

pub struct KeccakWitness {
    states: Box<[u128]>,
}

impl KeccakWitness {
    pub fn generate(initial: &[u128; KECCAK_BITS]) -> Self {
        let mut states = vec![0u128; KECCAK_STATES * KECCAK_BITS].into_boxed_slice();
        states[..KECCAK_BITS].copy_from_slice(initial);
        for value in &mut states[..KECCAK_BITS] {
            *value &= PACKED_MASK;
        }

        let mut tmp = [0u128; KECCAK_BITS];
        for round in 0..KECCAK_ROUNDS {
            let (prev, next) = split_round_pair(&mut states, round);
            next.copy_from_slice(prev);
            keccak_round_packed(next, round, &mut tmp);
        }

        Self { states }
    }

    #[inline]
    pub fn round_state(&self, round: usize) -> &[u128] {
        assert!(round <= KECCAK_ROUNDS);
        &self.states[round * KECCAK_BITS..][..KECCAK_BITS]
    }

    #[inline]
    pub fn round_state_mut(&mut self, round: usize) -> &mut [u128] {
        assert!(round <= KECCAK_ROUNDS);
        &mut self.states[round * KECCAK_BITS..][..KECCAK_BITS]
    }

    #[inline]
    pub fn bit(&self, round: usize, x: usize, y: usize, z: usize) -> u128 {
        self.round_state(round)[state_idx(x, y, z)]
    }

    #[inline]
    pub fn final_state(&self) -> &[u128] {
        self.round_state(KECCAK_ROUNDS)
    }
}

fn split_round_pair(states: &mut [u128], round: usize) -> (&[u128], &mut [u128]) {
    let prev_start = round * KECCAK_BITS;
    let next_start = (round + 1) * KECCAK_BITS;
    let (left, right) = states.split_at_mut(next_start);
    (&left[prev_start..][..KECCAK_BITS], &mut right[..KECCAK_BITS])
}

fn keccak_round_packed(state: &mut [u128], round: usize, tmp: &mut [u128; KECCAK_BITS]) {
    theta_packed(state);
    rho_pi_packed(state, tmp);
    chi_packed(state, tmp);
    iota_packed(state, round);
}

fn theta_packed(state: &mut [u128]) {
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

fn rho_pi_packed(state: &mut [u128], tmp: &mut [u128; KECCAK_BITS]) {
    tmp.copy_from_slice(state);
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = RHO_OFFSETS[a][b] as usize;
            for z in 0..LANE_BITS {
                state[state_idx(x, y, z)] = tmp[state_idx(a, b, (z + 64 - r) % 64)];
            }
        }
    }
}

fn chi_packed(state: &mut [u128], tmp: &mut [u128; KECCAK_BITS]) {
    tmp.copy_from_slice(state);
    for y in 0..5 {
        for x in 0..5 {
            for z in 0..LANE_BITS {
                let a = tmp[state_idx(x, y, z)];
                let b = tmp[state_idx((x + 1) % 5, y, z)];
                let c = tmp[state_idx((x + 2) % 5, y, z)];
                state[state_idx(x, y, z)] = (a ^ ((!b) & c)) & PACKED_MASK;
            }
        }
    }
}

fn iota_packed(state: &mut [u128], round: usize) {
    let rc = ROUND_CONSTANTS[round];
    for z in 0..LANE_BITS {
        if ((rc >> z) & 1) != 0 {
            state[state_idx(0, 0, z)] ^= PACKED_MASK;
        }
    }
}

#[inline]
pub fn state_idx(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < 5 && y < 5 && z < 64);
    (x + 5 * y) * 64 + z
}

#[cfg(test)]
pub(crate) fn keccak_f_lanes(state: &mut [u64; KECCAK_LANES]) {
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
