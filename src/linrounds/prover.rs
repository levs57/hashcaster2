use crate::{
    challenger::{Challenger, ProofWriter},
    chi_round::verifier::HybridClaim,
    field::F128,
    protocol_state::{self, KECCAK_BITS},
};

use super::{apply_transposed, matrix_eval, physical_eq, Linround};

const LOGICAL_STATE_BITS: usize = 1 << 12;

#[derive(Clone, Copy)]
pub struct ProverCfg {
    pub round: Linround,
}

impl ProverCfg {
    pub fn prove<Ch: Challenger>(
        &self,
        ctx: &mut ProofWriter<Ch>,
        output_claim: HybridClaim,
        input_state: &[F128],
    ) -> HybridClaim {
        assert_eq!(input_state.len(), KECCAK_BITS);

        let output_eq = physical_eq(&output_claim.r_x, &output_claim.r_y, &output_claim.r_z);
        let mut weights = [F128::ZERO; KECCAK_BITS];
        apply_transposed(self.round, &output_eq, &mut weights);

        let mut p = vec![F128::ZERO; LOGICAL_STATE_BITS];
        let mut q = vec![F128::ZERO; LOGICAL_STATE_BITS];
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    let physical = protocol_state::state_idx(x, y, z);
                    let logical = logical_idx(x, y, z);
                    p[logical] = input_state[physical];
                    q[logical] = weights[physical];
                }
            }
        }

        let mut active = LOGICAL_STATE_BITS;
        let mut point = Vec::with_capacity(12);
        for _ in 0..12 {
            let half = active / 2;
            let mut g1 = F128::ZERO;
            let mut g_inf = F128::ZERO;
            // Chunked over pairs so the compiler keeps the two independent
            // multiply chains (g1 and g_inf) in flight and elides bounds checks.
            for (pp, qq) in p[..active]
                .chunks_exact(2)
                .zip(q[..active].chunks_exact(2))
            {
                let p0 = pp[0];
                let p1 = pp[1];
                let q0 = qq[0];
                let q1 = qq[1];
                g1 += p1 * q1;
                g_inf += (p0 + p1) * (q0 + q1);
            }
            ctx.write_f128_slice(&[g1, g_inf]);

            let rho = ctx.sample_f128();
            // p0*(1+rho) + p1*rho = p0 + (p0+p1)*rho, halving the field
            // multiplies in the fold (the dominant cost of this sumcheck).
            for idx in 0..half {
                let p0 = p[2 * idx];
                let p1 = p[2 * idx + 1];
                let q0 = q[2 * idx];
                let q1 = q[2 * idx + 1];
                p[idx] = p0 + (p0 + p1) * rho;
                q[idx] = q0 + (q0 + q1) * rho;
            }
            active = half;
            point.push(rho);
        }

        ctx.write_f128(p[0]);
        debug_assert_eq!(
            q[0],
            matrix_eval(
                self.round,
                &point[..3],
                &point[3..6],
                &point[6..],
                &output_claim.r_x,
                &output_claim.r_y,
                &output_claim.r_z,
            ),
        );

        HybridClaim {
            t: output_claim.t,
            r_x: point[..3].to_vec(),
            r_y: point[3..6].to_vec(),
            r_z: point[6..].to_vec(),
            r_out: output_claim.r_out,
            ev: p[0],
        }
    }
}

#[inline]
fn logical_idx(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < 8 && y < 8 && z < 64);
    x + 8 * y + 64 * z
}
