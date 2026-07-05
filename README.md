## Hashcaster 2

New keccak GKR protocol based on new cool RMFE-in-polynomial-ring and univariate skip. Full GKR proving routine (24 rounds of keccak permutation) is integrated, commitments are not integrated yet.

Uses $\mathbb{F}_{2^{128}}$ in POLYVAL basis.

On my 32-core x86 it proves ~220k / s Keccak permutations, Neon benchmarks and optimizations pending. There are also probably some tricks related to AVX-512 SIMD, this requires AVX-256 or Neon+aes only.

Protocol description: https://hackmd.io/@levs57/HJkT2mEXMx (described for the bitwise AND gate, for Keccak specifically I of course use Keccak's quadratic gate, but the main idea can be read there; another note is that I use 96-dimensional subspace in $\mathbb{F}_2[x]_{\leq 192}$, not 64-dimensional subspace in $\mathbb{F}_2[x]_{\leq 128}$).

Example run:

`RUSTFLAGS="-C target-cpu=native" cargo run --release --example main_protocol -- 200000 10 $(nproc) 6`

Here, you can replace `200000` with your desired amount of keccak permutations (there is no harm if your instances are not full power of `2`), `10` is the amount of samples, `$(nproc)` can be replaced with amount of threads if you want to restrict parallelism, and you can choose either `6` or `7` as the last parameter ~~because sixseven~~ depending on size of your L1 cache and phase of the moon. For me, `6` seems to work better.