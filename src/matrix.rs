//! Boolean matrix kernels for packed RMFE-kernel coordinates.
//!
//! The protocol frequently applies Boolean matrices to packed 96-coordinate
//! RMFE payloads.  We store each value by its 96-coordinate chart, and pack
//! four chart values into three `u128`s so the projection loop moves dense
//! 384-bit payloads.
//!
//! The first concrete shapes we need are 96 input rows into either 128 or 256
//! output rows.  The implementation below is generic in the output count, but
//! exposes those two aliases explicitly.

pub const KERNEL_COORD_BITS: usize = 96;
pub const PACKED_LANES: usize = 4;
pub const PACKED_WORDS: usize = 3;

const COORD_MASK_96: u128 = (1u128 << KERNEL_COORD_BITS) - 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Packed4x96 {
    words: [u128; PACKED_WORDS],
}

impl Packed4x96 {
    #[inline(always)]
    pub fn pack(values: [u128; PACKED_LANES]) -> Self {
        debug_assert!(values.iter().all(|&v| v >> KERNEL_COORD_BITS == 0));
        Self {
            words: [
                values[0] | (values[1] << 96),
                (values[1] >> 32) | (values[2] << 64),
                (values[2] >> 64) | (values[3] << 32),
            ],
        }
    }

    #[inline(always)]
    pub fn unpack(self) -> [u128; PACKED_LANES] {
        [
            self.words[0] & COORD_MASK_96,
            ((self.words[0] >> 96) | (self.words[1] << 32)) & COORD_MASK_96,
            ((self.words[1] >> 64) | (self.words[2] << 64)) & COORD_MASK_96,
            self.words[2] >> 32,
        ]
    }

    #[inline(always)]
    pub fn xor(self, rhs: Self) -> Self {
        Self {
            words: [
                self.words[0] ^ rhs.words[0],
                self.words[1] ^ rhs.words[1],
                self.words[2] ^ rhs.words[2],
            ],
        }
    }

    #[inline(always)]
    pub fn xor_assign(&mut self, rhs: Self) {
        self.words[0] ^= rhs.words[0];
        self.words[1] ^= rhs.words[1];
        self.words[2] ^= rhs.words[2];
    }
}

pub type FourRussians128 = FourRussiansMatrix<128>;
pub type FourRussians256 = FourRussiansMatrix<256>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FourRussiansMatrix<const OUT: usize> {
    input_len: usize,
    windows: Vec<u8>,
}

impl FourRussians128 {
    pub fn from_rows_96x128(rows: &[u128; 128]) -> Self {
        Self::from_row_masks(96, rows)
    }
}

impl FourRussians256 {
    pub fn from_rows_96x256(rows: &[u128; 256]) -> Self {
        Self::from_row_masks(96, rows)
    }
}

impl<const OUT: usize> FourRussiansMatrix<OUT> {
    pub fn from_row_masks(input_len: usize, rows: &[u128; OUT]) -> Self {
        assert!(input_len > 0);
        assert_eq!(input_len % 8, 0);
        assert!(input_len <= u128::BITS as usize);

        let groups = input_len / 8;
        let mut windows = vec![0u8; groups * OUT];
        for group in 0..groups {
            let shift = group * 8;
            for row_idx in 0..OUT {
                windows[group * OUT + row_idx] = ((rows[row_idx] >> shift) & 0xff) as u8;
            }
        }
        Self { input_len, windows }
    }

    #[inline]
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    #[inline]
    pub fn output_len(&self) -> usize {
        OUT
    }

    pub fn apply(&self, input: &[Packed4x96], out: &mut [Packed4x96]) {
        assert_eq!(input.len(), self.input_len);
        assert_eq!(out.len(), OUT);
        out.fill(Packed4x96::default());

        let groups = self.input_len / 8;
        let mut table = [Packed4x96::default(); 256];
        for group in 0..groups {
            let base = group * 8;
            fill_table(
                &mut table,
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
                input[base + 4],
                input[base + 5],
                input[base + 6],
                input[base + 7],
            );
            let windows = &self.windows[group * OUT..][..OUT];
            for row_idx in 0..OUT {
                out[row_idx].xor_assign(table[windows[row_idx] as usize]);
            }
        }
    }
}

#[inline(always)]
fn fill_table(
    table: &mut [Packed4x96; 256],
    v0: Packed4x96,
    v1: Packed4x96,
    v2: Packed4x96,
    v3: Packed4x96,
    v4: Packed4x96,
    v5: Packed4x96,
    v6: Packed4x96,
    v7: Packed4x96,
) {
    table[0] = Packed4x96::default();
    table[1] = v0;
    table[2] = v1;
    table[3] = v0.xor(v1);
    table[4] = v2;
    table[5] = v0.xor(v2);
    table[6] = v1.xor(v2);
    table[7] = table[3].xor(v2);
    table[8] = v3;
    table[9] = v0.xor(v3);
    table[10] = v1.xor(v3);
    table[11] = table[3].xor(v3);
    table[12] = v2.xor(v3);
    table[13] = table[5].xor(v3);
    table[14] = table[6].xor(v3);
    table[15] = table[7].xor(v3);

    let mut high_bit = 16usize;
    while high_bit < 256 {
        let value = match high_bit {
            16 => v4,
            32 => v5,
            64 => v6,
            128 => v7,
            _ => unreachable!(),
        };
        for mask in 0..high_bit {
            table[high_bit + mask] = table[mask].xor(value);
        }
        high_bit <<= 1;
    }
}
