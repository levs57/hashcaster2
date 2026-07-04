//! Protocol-owned witness and reusable scratch storage.
//!
//! The Keccak witness is stored in the protocol layout directly: each entry is
//! a 96-bit packed column, where bit `i` belongs to Keccak instance `i`.

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
        Self {
            witness: KeccakWitness::new(),
            scratches: ReusableScratches::new(),
        }
    }

    pub fn generate_keccak(&mut self) {
        self.witness.generate(&mut self.scratches);
    }
}

pub struct ReusableScratches {
    current: Box<[u128]>,
}

impl ReusableScratches {
    pub fn new() -> Self {
        Self {
            current: zero_boxed_state(),
        }
    }
}

pub struct KeccakWitness {
    input: Box<[u128]>,
    pre_chi: Box<[u128]>,
    output: Box<[u128]>,
}

impl KeccakWitness {
    pub fn new() -> Self {
        Self {
            input: zero_boxed_state(),
            pre_chi: vec![0u128; KECCAK_ROUNDS * KECCAK_BITS].into_boxed_slice(),
            output: zero_boxed_state(),
        }
    }

    pub fn generate(&mut self, scratches: &mut ReusableScratches) {
        for value in &mut *self.input {
            *value &= PACKED_MASK;
        }

        scratches.current.copy_from_slice(&self.input);
        for round in 0..KECCAK_ROUNDS {
            theta_packed(&mut scratches.current);
            rho_pi_packed(&scratches.current, self.pre_chi_mut(round));
            chi_iota_packed(self.pre_chi(round), &mut scratches.current, round);
        }
        self.output.copy_from_slice(&scratches.current);
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
    pub fn pre_chi(&self, round: usize) -> &[u128] {
        assert!(round < KECCAK_ROUNDS);
        &self.pre_chi[round * KECCAK_BITS..][..KECCAK_BITS]
    }

    #[inline]
    pub fn pre_chi_mut(&mut self, round: usize) -> &mut [u128] {
        assert!(round < KECCAK_ROUNDS);
        &mut self.pre_chi[round * KECCAK_BITS..][..KECCAK_BITS]
    }

    #[inline]
    pub fn output(&self) -> &[u128] {
        &self.output
    }
}

fn zero_boxed_state() -> Box<[u128]> {
    vec![0u128; KECCAK_BITS].into_boxed_slice()
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

fn rho_pi_packed(input: &[u128], output: &mut [u128]) {
    for y in 0..5 {
        for x in 0..5 {
            let a = (x + 3 * y) % 5;
            let b = x;
            let r = RHO_OFFSETS[a][b] as usize;
            for z in 0..LANE_BITS {
                output[state_idx(x, y, z)] = input[state_idx(a, b, (z + 64 - r) % 64)];
            }
        }
    }
}

fn chi_iota_packed(pre_chi: &[u128], output: &mut [u128], round: usize) {
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
