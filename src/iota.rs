use crate::{
    chi_round::verifier::HybridClaim,
    field::F128,
    protocol_state,
    util::{eq_poly_v, rmfe_one_eval},
};

#[derive(Clone, Copy)]
pub struct VerifierCfg {
    pub log_packed_instances: usize,
    pub round: usize,
}

impl VerifierCfg {
    pub fn verify(&self, mut claim: HybridClaim) -> HybridClaim {
        debug_assert_eq!(claim.r_out.len(), self.log_packed_instances);
        debug_assert_eq!(claim.r_x.len(), 3);
        debug_assert_eq!(claim.r_y.len(), 3);
        debug_assert_eq!(claim.r_z.len(), 6);

        claim.ev += iota_eval(self.round, &claim);
        claim
    }
}

fn iota_eval(round: usize, claim: &HybridClaim) -> F128 {
    let rc = protocol_state::round_constant(round);
    if rc == 0 {
        return F128::ZERO;
    }

    let eq_x = eq_poly_v(&claim.r_x);
    let eq_y = eq_poly_v(&claim.r_y);
    let eq_z = eq_poly_v(&claim.r_z);

    let mut z_eval = F128::ZERO;
    for z in 0..64 {
        if ((rc >> z) & 1) != 0 {
            z_eval += eq_z[z];
        }
    }

    eq_x[0] * eq_y[0] * z_eval * rmfe_one_eval(claim.t)
}
