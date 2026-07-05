use crate::{
    challenger::{Challenger, ProofReader},
    chi_round,
    chi_round::verifier::{HybridClaim, VerifierError},
    iota,
    linrounds::{verifier::VerifierCfg as LinroundVerifierCfg, Linround},
    protocol_state,
};

#[derive(Clone, Copy)]
pub struct GkrVerifierCfg {
    pub log_packed_instances: usize,
}

impl GkrVerifierCfg {
    pub fn verify<Ch: Challenger>(
        &self,
        ctx: &mut ProofReader<Ch>,
        mut claim: HybridClaim,
    ) -> Result<HybridClaim, VerifierError> {
        for round in (0..protocol_state::KECCAK_ROUNDS).rev() {
            claim = iota::VerifierCfg {
                log_packed_instances: self.log_packed_instances,
                round,
            }
            .verify(claim);

            claim = chi_round::verifier::VerifierCfg {
                log_packed_instances: self.log_packed_instances,
            }
            .verify(ctx, claim)?;

            claim = LinroundVerifierCfg {
                round: Linround::RhoPi,
            }
            .verify(ctx, claim)?;

            claim = LinroundVerifierCfg {
                round: Linround::Theta,
            }
            .verify(ctx, claim)?;
        }

        Ok(claim)
    }
}
