# hashcaster2

Clean staging crate for the RMFE-based Hashcaster direction.

This crate intentionally starts small.  It contains only primitives that are
expected to survive into the protocol implementation:

- `field`: Hashcaster-compatible `F128`.
- `challenger`: Fiat-Shamir challenger plus proof reader/writer discipline.
- `rmfe`: reserved for the actual `F2^128` RMFE/subspace constants.
- `matrix`: 4-Russian Boolean matrix kernels over packed 96-coordinate values.

## Boolean Matrix Shapes

The matrix module currently exposes:

- `FourRussians128::from_rows_96x128`
- `FourRussians256::from_rows_96x256`

Both apply Boolean row masks to arrays of `Packed4x96`.  A `Packed4x96` stores
four 96-bit kernel-coordinate payloads in three `u128`s.  This is the layout
used by the projection-heavy path: Boolean matrix coefficients select whole
kernel payloads, while the payload itself remains densely packed.

If we later need wider composed matrices, the generic
`FourRussiansMatrix<OUT>` already supports any `OUT`; the input length is kept
runtime-configurable but must be divisible by 4.

## RMFE Constants

The previous staging crate accidentally imported constants from a different
field/subspace experiment.  Those are intentionally not present here.  The
`rmfe` module is a placeholder for the actual `F2^128` subspace constants and
validator.

Run:

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release
```
