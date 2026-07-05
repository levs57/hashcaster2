use crate::{
    challenger::{Challenger, ProofReader},
    chi_round::verifier::{HybridClaim, VerifierError},
    field::F128,
};
use super::{matrix_eval, Linround};

#[derive(Clone, Copy)]
pub struct VerifierCfg {
    pub round: Linround,
}

#[derive(Clone, Copy, Debug)]
struct ProductMessage {
    g1: F128,
    g_inf: F128,
}

impl ProductMessage {
    fn read<Ch: Challenger>(ctx: &mut ProofReader<Ch>) -> Result<Self, VerifierError> {
        let values = ctx.read_f128_vec(2)?;
        Ok(Self { g1: values[0], g_inf: values[1] })
    }

    #[inline]
    fn bind(self, running_claim: F128, rho: F128) -> F128 {
        let g0 = running_claim + self.g1;
        let one_plus_rho = F128::ONE + rho;
        g0 * one_plus_rho + self.g1 * rho + self.g_inf * rho * one_plus_rho
    }
}

impl VerifierCfg {
    pub fn verify<Ch: Challenger>(
        &self,
        ctx: &mut ProofReader<Ch>,
        claim: HybridClaim,
    ) -> Result<HybridClaim, VerifierError> {
        debug_assert_eq!(claim.r_x.len(), 3);
        debug_assert_eq!(claim.r_y.len(), 3);
        debug_assert_eq!(claim.r_z.len(), 6);

        let mut running_claim = claim.ev;
        let mut point = Vec::with_capacity(12);
        for _ in 0..12 {
            let msg = ProductMessage::read(ctx)?;
            let rho = ctx.sample_f128();
            running_claim = msg.bind(running_claim, rho);
            point.push(rho);
        }

        let p_eval = ctx.read_f128()?;
        let weight = matrix_eval(
            self.round,
            &point[..3],
            &point[3..6],
            &point[6..],
            &claim.r_x,
            &claim.r_y,
            &claim.r_z,
        );
        if weight == F128::ZERO {
            return Err(VerifierError::NegligibleEvent);
        }
        if running_claim != weight * p_eval {
            return Err(VerifierError::InvalidProof);
        }

        Ok(HybridClaim {
            t: claim.t,
            r_x: point[..3].to_vec(),
            r_y: point[3..6].to_vec(),
            r_z: point[6..].to_vec(),
            r_out: claim.r_out,
            ev: p_eval,
        })
    }
}
