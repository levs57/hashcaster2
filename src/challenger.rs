//! Verifier-randomness abstraction.
//!
//! A [`Challenger`] is the source of Fiat-Shamir challenges. Proof messages
//! should normally flow through [`ProofWriter`] and [`ProofReader`], which bind
//! serialized elements automatically as they are written or read.

use crate::field::F128;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProofTranscript {
    elements: Vec<F128>,
}

impl ProofTranscript {
    pub fn new(elements: Vec<F128>) -> Self {
        Self { elements }
    }

    pub fn elements(&self) -> &[F128] {
        &self.elements
    }

    pub fn into_elements(self) -> Vec<F128> {
        self.elements
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofTranscriptError {
    UnexpectedEof { requested: usize, remaining: usize },
    TrailingElements { unread: usize },
}

pub struct ProofWriter<Ch> {
    challenger: Ch,
    proof: ProofTranscript,
}

impl<Ch: Challenger> ProofWriter<Ch> {
    pub fn new(challenger: Ch) -> Self {
        Self {
            challenger,
            proof: ProofTranscript::default(),
        }
    }

    pub fn with_capacity(challenger: Ch, capacity: usize) -> Self {
        Self {
            challenger,
            proof: ProofTranscript {
                elements: Vec::with_capacity(capacity),
            },
        }
    }

    pub fn write_f128(&mut self, value: F128) {
        self.proof.elements.push(value);
        self.challenger.observe_f128(value);
    }

    pub fn write_f128_slice(&mut self, values: &[F128]) {
        self.proof.elements.extend_from_slice(values);
        self.challenger.observe_f128_slice(values);
    }

    /// Write proof-carried auxiliary data without binding it to Fiat-Shamir.
    ///
    /// This is only for hints that are already bound by an observed commitment
    /// or by later verification against observed data. Ordinary prover messages
    /// must use [`Self::write_f128`] or [`Self::write_f128_slice`].
    pub fn write_unsafe_hint(&mut self, value: F128) {
        self.proof.elements.push(value);
    }

    /// Write proof-carried auxiliary data without binding it to Fiat-Shamir.
    ///
    /// See [`Self::write_unsafe_hint`].
    pub fn write_unsafe_hint_slice(&mut self, values: &[F128]) {
        self.proof.elements.extend_from_slice(values);
    }

    pub fn sample_f128(&mut self) -> F128 {
        self.challenger.sample_f128()
    }

    pub fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.challenger.sample_f128_vec(n)
    }

    pub fn proof(&self) -> &ProofTranscript {
        &self.proof
    }

    pub fn into_proof(self) -> ProofTranscript {
        self.proof
    }

    pub fn into_parts(self) -> (Ch, ProofTranscript) {
        (self.challenger, self.proof)
    }
}

pub struct ProofReader<Ch> {
    challenger: Ch,
    proof: ProofTranscript,
    cursor: usize,
}

impl<Ch: Challenger> ProofReader<Ch> {
    pub fn new(challenger: Ch, proof: ProofTranscript) -> Self {
        Self {
            challenger,
            proof,
            cursor: 0,
        }
    }

    pub fn read_f128(&mut self) -> Result<F128, ProofTranscriptError> {
        let Some(&value) = self.proof.elements.get(self.cursor) else {
            return Err(ProofTranscriptError::UnexpectedEof {
                requested: 1,
                remaining: 0,
            });
        };
        self.cursor += 1;
        self.challenger.observe_f128(value);
        Ok(value)
    }

    /// Read proof-carried auxiliary data without binding it to Fiat-Shamir.
    ///
    /// This is only for hints that are already bound by an observed commitment
    /// or by later verification against observed data. Ordinary prover messages
    /// must use [`Self::read_f128`] or [`Self::read_f128_vec`].
    pub fn read_unsafe_hint(&mut self) -> Result<F128, ProofTranscriptError> {
        let Some(&value) = self.proof.elements.get(self.cursor) else {
            return Err(ProofTranscriptError::UnexpectedEof {
                requested: 1,
                remaining: 0,
            });
        };
        self.cursor += 1;
        Ok(value)
    }

    pub fn read_f128_vec(&mut self, n: usize) -> Result<Vec<F128>, ProofTranscriptError> {
        let remaining = self.proof.elements.len() - self.cursor;
        if remaining < n {
            return Err(ProofTranscriptError::UnexpectedEof {
                requested: n,
                remaining,
            });
        }
        let values = self.proof.elements[self.cursor..self.cursor + n].to_vec();
        self.cursor += n;
        self.challenger.observe_f128_slice(&values);
        Ok(values)
    }

    /// Read proof-carried auxiliary data without binding it to Fiat-Shamir.
    ///
    /// See [`Self::read_unsafe_hint`].
    pub fn read_unsafe_hint_vec(&mut self, n: usize) -> Result<Vec<F128>, ProofTranscriptError> {
        let remaining = self.proof.elements.len() - self.cursor;
        if remaining < n {
            return Err(ProofTranscriptError::UnexpectedEof {
                requested: n,
                remaining,
            });
        }
        let values = self.proof.elements[self.cursor..self.cursor + n].to_vec();
        self.cursor += n;
        Ok(values)
    }

    pub fn sample_f128(&mut self) -> F128 {
        self.challenger.sample_f128()
    }

    pub fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.challenger.sample_f128_vec(n)
    }

    pub fn finish(self) -> Result<Ch, ProofTranscriptError> {
        let unread = self.proof.elements.len() - self.cursor;
        if unread == 0 {
            Ok(self.challenger)
        } else {
            Err(ProofTranscriptError::TrailingElements { unread })
        }
    }
}

pub trait Challenger {
    /// Absorb a domain-separation label. Each protocol entry should call this
    /// once on entry so a transcript from one protocol cannot be replayed as
    /// another.
    fn observe_label(&mut self, _label: &[u8]) {}

    /// Absorb a single `F128` prover message.
    fn observe_f128(&mut self, value: F128);

    /// Absorb a slice of `F128` prover messages.
    fn observe_f128_slice(&mut self, values: &[F128]) {
        for v in values {
            self.observe_f128(*v);
        }
    }

    /// Absorb arbitrary bytes, e.g. a commitment root or statement digest.
    fn observe_bytes(&mut self, _bytes: &[u8]) {}

    /// Produce one `F128` challenge.
    fn sample_f128(&mut self) -> F128;

    /// Produce `n` `F128` challenges, in order.
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.sample_f128()).collect()
    }

    /// Prover-side PoW grinding. Default implementation is a no-op.
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        0
    }

    /// Verifier-side mirror of [`Self::grind_pow`]. Default accepts
    /// unconditionally.
    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        true
    }
}

pub trait ChallengeSource {
    fn draw_f128(&mut self) -> F128;

    fn draw_f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.draw_f128()).collect()
    }
}

impl<T: Challenger + ?Sized> ChallengeSource for T {
    fn draw_f128(&mut self) -> F128 {
        Challenger::sample_f128(self)
    }

    fn draw_f128_vec(&mut self, n: usize) -> Vec<F128> {
        Challenger::sample_f128_vec(self, n)
    }
}

impl<Ch: Challenger> ChallengeSource for ProofWriter<Ch> {
    fn draw_f128(&mut self) -> F128 {
        ProofWriter::sample_f128(self)
    }

    fn draw_f128_vec(&mut self, n: usize) -> Vec<F128> {
        ProofWriter::sample_f128_vec(self, n)
    }
}

impl<Ch: Challenger> ChallengeSource for ProofReader<Ch> {
    fn draw_f128(&mut self) -> F128 {
        ProofReader::sample_f128(self)
    }

    fn draw_f128_vec(&mut self, n: usize) -> Vec<F128> {
        ProofReader::sample_f128_vec(self, n)
    }
}

impl<T: Challenger + ?Sized> Challenger for &mut T {
    fn observe_label(&mut self, label: &[u8]) {
        (**self).observe_label(label);
    }

    fn observe_f128(&mut self, value: F128) {
        (**self).observe_f128(value);
    }

    fn observe_f128_slice(&mut self, values: &[F128]) {
        (**self).observe_f128_slice(values);
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        (**self).observe_bytes(bytes);
    }

    fn sample_f128(&mut self) -> F128 {
        (**self).sample_f128()
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        (**self).sample_f128_vec(n)
    }

    fn grind_pow(&mut self, bits: u32) -> u64 {
        (**self).grind_pow(bits)
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        (**self).verify_pow(nonce, bits)
    }
}

const OP_DOMAIN: u8 = 0x01;
const OP_LABEL: u8 = 0x02;
const OP_OBSERVE: u8 = 0x03;
const OP_SQUEEZE: u8 = 0x04;
const OP_BYTES: u8 = 0x05;

const KIND_SCALAR: u8 = 0x01;
const KIND_SLICE: u8 = 0x02;

/// Global Fiat-Shamir hash counters, enabled with `--features hash-count`.
#[cfg(feature = "hash-count")]
pub mod fs_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static SQUEEZES: AtomicU64 = AtomicU64::new(0);
    pub static POW_SHA256: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        SQUEEZES.store(0, Relaxed);
        POW_SHA256.store(0, Relaxed);
    }

    pub fn snapshot() -> (u64, u64) {
        (SQUEEZES.load(Relaxed), POW_SHA256.load(Relaxed))
    }
}

#[derive(Clone)]
pub struct FsChallenger {
    hasher: blake3::Hasher,
}

impl FsChallenger {
    /// New challenger seeded with a length-prefixed domain-separation tag.
    pub fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[OP_DOMAIN]);
        hasher.update(&(domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        Self { hasher }
    }

    #[inline]
    fn absorb_f128(&mut self, v: F128) {
        self.hasher.update(&v.raw().to_le_bytes());
    }

    #[cfg(feature = "hash-count")]
    pub fn absorbed_bytes(&self) -> u64 {
        self.hasher.count()
    }
}

impl Challenger for FsChallenger {
    fn observe_label(&mut self, label: &[u8]) {
        self.hasher.update(&[OP_LABEL]);
        self.hasher.update(&(label.len() as u64).to_le_bytes());
        self.hasher.update(label);
    }

    fn observe_f128(&mut self, value: F128) {
        self.hasher.update(&[OP_OBSERVE, KIND_SCALAR]);
        self.absorb_f128(value);
    }

    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.hasher.update(&[OP_OBSERVE, KIND_SLICE]);
        self.hasher.update(&(values.len() as u64).to_le_bytes());
        for v in values {
            self.absorb_f128(*v);
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.hasher.update(&[OP_BYTES]);
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
    }

    fn sample_f128(&mut self) -> F128 {
        #[cfg(feature = "hash-count")]
        fs_count::SQUEEZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.hasher.update(&[OP_SQUEEZE, KIND_SCALAR]);
        let snapshot = self.hasher.clone();
        let mut reader = snapshot.finalize_xof();
        let mut buf = [0u8; 16];
        reader.fill(&mut buf);
        self.hasher.update(&buf);
        // Interpret 128 Fiat-Shamir bits directly as an `F2^128` element.
        F128::from_raw(u128::from_le_bytes(buf))
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        #[cfg(feature = "hash-count")]
        fs_count::SQUEEZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.hasher.update(&[OP_SQUEEZE, KIND_SLICE]);
        self.hasher.update(&(n as u64).to_le_bytes());
        let snapshot = self.hasher.clone();
        let mut reader = snapshot.finalize_xof();
        let mut buf = vec![0u8; n * 16];
        reader.fill(&mut buf);
        self.hasher.update(&buf);
        buf.chunks_exact(16)
            .map(|chunk| F128::from_raw(u128::from_le_bytes(chunk.try_into().unwrap())))
            .collect()
    }

    fn grind_pow(&mut self, bits: u32) -> u64 {
        let state_digest = fs_pow_state_digest(&self.hasher);
        let nonce = if bits == 0 {
            0
        } else {
            let mut nonce = 0u64;
            loop {
                if sha256_has_leading_zero_bits(&state_digest, nonce, bits) {
                    break nonce;
                }
                nonce = nonce.wrapping_add(1);
            }
        };
        self.observe_bytes(&nonce.to_le_bytes());
        nonce
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        let state_digest = fs_pow_state_digest(&self.hasher);
        let ok = bits == 0 || sha256_has_leading_zero_bits(&state_digest, nonce, bits);
        self.observe_bytes(&nonce.to_le_bytes());
        ok
    }
}

#[inline]
fn fs_pow_state_digest(hasher: &blake3::Hasher) -> [u8; 32] {
    #[cfg(feature = "hash-count")]
    fs_count::SQUEEZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let snapshot = hasher.clone();
    let mut reader = snapshot.finalize_xof();
    let mut out = [0u8; 32];
    reader.fill(&mut out);
    out
}

#[inline]
fn sha256_has_leading_zero_bits(state_digest: &[u8; 32], nonce: u64, bits: u32) -> bool {
    use sha2::{Digest, Sha256};
    #[cfg(feature = "hash-count")]
    fs_count::POW_SHA256.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut hasher = Sha256::new();
    hasher.update(state_digest);
    hasher.update(nonce.to_le_bytes());
    let h: [u8; 32] = hasher.finalize().into();
    let full_bytes = (bits / 8) as usize;
    let extra = bits % 8;
    for &b in h.iter().take(full_bytes) {
        if b != 0 {
            return false;
        }
    }
    if extra > 0 && (h[full_bytes] >> (8 - extra)) != 0 {
        return false;
    }
    true
}
