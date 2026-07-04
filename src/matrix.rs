//! Boolean matrix kernels over raw packed payload streams.
//!
//! A packed block contains four 96-bit payloads, stored densely in six `u64`
//! words.  Matrix application consumes `input_len` such blocks and writes
//! `OUT` such blocks.  The API intentionally deals in raw `u64` slices so
//! callers can stream data without materializing wrapper objects.

pub const COORD_BITS: usize = 96;
pub const PACKED_LANES: usize = 4;
pub const PACKED_U64S: usize = 6;

pub type FourRussians128 = FourRussiansMatrix<128>;
pub type FourRussians192 = FourRussiansMatrix<192>;
pub type FourRussians256 = FourRussiansMatrix<256>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanMatrix<const OUT: usize> {
    input_len: usize,
    row_words: Vec<u64>,
}

impl<const OUT: usize> BooleanMatrix<OUT> {
    pub fn zero(input_len: usize) -> Self {
        assert!(input_len > 0);
        Self {
            input_len,
            row_words: vec![0; OUT * input_len.div_ceil(64)],
        }
    }

    pub fn from_rows_u128(input_len: usize, rows: &[u128; OUT]) -> Self {
        assert!(input_len <= u128::BITS as usize);
        let mut matrix = Self::zero(input_len);
        for (row_idx, &row) in rows.iter().enumerate() {
            matrix.row_mut(row_idx)[0] = row as u64;
            if matrix.words_per_row() > 1 {
                matrix.row_mut(row_idx)[1] = (row >> 64) as u64;
            }
        }
        matrix
    }

    #[inline]
    pub fn input_len(&self) -> usize {
        self.input_len
    }

    #[inline]
    pub fn output_len(&self) -> usize {
        OUT
    }

    #[inline]
    pub fn words_per_row(&self) -> usize {
        self.input_len.div_ceil(64)
    }

    #[inline]
    pub fn row(&self, row: usize) -> &[u64] {
        let width = self.words_per_row();
        &self.row_words[row * width..][..width]
    }

    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [u64] {
        let width = self.words_per_row();
        &mut self.row_words[row * width..][..width]
    }

    pub fn set(&mut self, row: usize, col: usize) {
        assert!(row < OUT);
        assert!(col < self.input_len);
        self.row_mut(row)[col / 64] |= 1u64 << (col % 64);
    }

    pub fn get(&self, row: usize, col: usize) -> bool {
        assert!(row < OUT);
        assert!(col < self.input_len);
        ((self.row(row)[col / 64] >> (col % 64)) & 1) != 0
    }

    fn window(&self, row: usize, group: usize) -> u8 {
        let bit = group * 8;
        let row = self.row(row);
        let word = bit / 64;
        let shift = bit % 64;
        let mut value = row[word] >> shift;
        if shift > 56 && word + 1 < row.len() {
            value |= row[word + 1] << (64 - shift);
        }
        (value & 0xff) as u8
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FourRussiansMatrix<const OUT: usize> {
    input_len: usize,
    windows: Vec<u8>,
}

impl FourRussians128 {
    pub fn from_rows_96x128(rows: &[u128; 128]) -> Self {
        Self::from_boolean_matrix(&BooleanMatrix::from_rows_u128(96, rows))
    }
}

impl FourRussians192 {
    pub fn from_rows_96x192(rows: &[u128; 192]) -> Self {
        Self::from_boolean_matrix(&BooleanMatrix::from_rows_u128(96, rows))
    }
}

impl FourRussians256 {
    pub fn from_rows_96x256(rows: &[u128; 256]) -> Self {
        Self::from_boolean_matrix(&BooleanMatrix::from_rows_u128(96, rows))
    }
}

impl<const OUT: usize> FourRussiansMatrix<OUT> {
    pub fn from_boolean_matrix(matrix: &BooleanMatrix<OUT>) -> Self {
        assert_eq!(matrix.input_len() % 8, 0);

        let input_len = matrix.input_len();
        let groups = input_len / 8;
        let mut windows = vec![0u8; groups * OUT];
        for group in 0..groups {
            for row_idx in 0..OUT {
                windows[group * OUT + row_idx] = matrix.window(row_idx, group);
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

    pub fn apply(&self, input: &[u64], out: &mut [u64]) {
        assert_eq!(input.len(), self.input_len * PACKED_U64S);
        assert_eq!(out.len(), OUT * PACKED_U64S);
        out.fill(0);

        let groups = self.input_len / 8;
        let mut table = [[0u64; PACKED_U64S]; 256];
        for group in 0..groups {
            let base = group * 8;
            fill_table(&mut table, input, base);
            let windows = &self.windows[group * OUT..][..OUT];
            for row_idx in 0..OUT {
                xor_block(&mut out[row_idx * PACKED_U64S..][..PACKED_U64S], &table[windows[row_idx] as usize]);
            }
        }
    }
}

fn fill_table(table: &mut [[u64; PACKED_U64S]; 256], input: &[u64], base: usize) {
    table[0] = [0; PACKED_U64S];
    for bit in 0..8 {
        table[1usize << bit].copy_from_slice(&input[(base + bit) * PACKED_U64S..][..PACKED_U64S]);
    }
    let mut high_bit = 2usize;
    while high_bit < 256 {
        for mask in 1..high_bit {
            table[high_bit + mask] = xor_blocks(table[high_bit], table[mask]);
        }
        high_bit <<= 1;
    }
}

#[inline(always)]
fn xor_blocks(lhs: [u64; PACKED_U64S], rhs: [u64; PACKED_U64S]) -> [u64; PACKED_U64S] {
    [
        lhs[0] ^ rhs[0],
        lhs[1] ^ rhs[1],
        lhs[2] ^ rhs[2],
        lhs[3] ^ rhs[3],
        lhs[4] ^ rhs[4],
        lhs[5] ^ rhs[5],
    ]
}

#[inline(always)]
fn xor_block(out: &mut [u64], rhs: &[u64; PACKED_U64S]) {
    out[0] ^= rhs[0];
    out[1] ^= rhs[1];
    out[2] ^= rhs[2];
    out[3] ^= rhs[3];
    out[4] ^= rhs[4];
    out[5] ^= rhs[5];
}
