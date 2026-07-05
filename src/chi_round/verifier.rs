use crate::{
    challenger::{Challenger, ProofReader, ProofTranscriptError},
    field::F128,
    rmfe,
    util::{apply_boolean_matrix, eq_poly_v, evaluate_univar},
};

#[derive(Clone, Copy)]
pub struct VerifierCfg {
    pub log_packed_instances: usize, // verifier is doing all the proofs for 96 * 2^n keccak instances
}

#[derive(Clone, Copy, Debug)]
pub enum VerifierError {
    InvalidProof, // catch-all for any malformed proof
    NegligibleEvent, // this never happens in reality
}

impl From<ProofTranscriptError> for VerifierError {
    fn from(_: ProofTranscriptError) -> Self {
        VerifierError::InvalidProof
    }
}

#[derive(Clone, Copy, Debug)]
struct GruenMessage {
    g1: F128,
    g_inf: F128,
}

impl GruenMessage {
    fn read<Ch: Challenger>(ctx: &mut ProofReader<Ch>) -> Result<Self, VerifierError> {
        let values = ctx.read_f128_vec(2)?;
        Ok(Self { g1: values[0], g_inf: values[1] })
    }

    #[inline]
    fn bind(self, running_claim: F128, eq_challenge: F128, rho: F128) -> Result<F128, VerifierError> {
        let denom = F128::ONE + eq_challenge;
        if denom == F128::ZERO {
            return Err(VerifierError::NegligibleEvent);
        }
        let g0 = (running_claim + eq_challenge * self.g1) * denom.inverse();
        let one_plus_rho = F128::ONE + rho;
        Ok(g0 * one_plus_rho + self.g1 * rho + self.g_inf * rho * one_plus_rho)
    }
}

#[derive(Clone)]
pub struct HybridClaim {
    pub t: F128, // univariate evaluation point
    pub r_x: Vec<F128>,
    pub r_y: Vec<F128>,
    pub r_z: Vec<F128>,
    pub r_out: Vec<F128>,
    pub ev: F128,
}

// Chi round protocol description
// Recall that physically state is (5 x 5 x 64) x N and logically it is
// (8 x 8 x 64) x n where n = log_ceil(N).
// denote the corresponding variables x, y, z and out
// 
// we, initially, get the following claim:
// sum_{x, y, z, out} EQ(r_x, r_y, r_z, r_out; x, y, z, out)(
//  pi(STATE(x+1, y, z, out) * STATE(x-1, y, z, out) + STATE(x, y, z, out) * ENCODED_ONE)
// ) (t)
// here indexing in x, y is modulo 5, note that state is empty for x, y = 5, 6, 7
//
// We first communicate a polynomial U which is
// sum_{x, y, z, out} EQ(r_x, r_y, r_z, r_out; x, y, z, out)(
//  (STATE(x+1, y, z, out) * STATE(x-1, y, z, out) + STATE(x, y, z, out) * ENCODED_ONE)
// )
//
// Verifier checks that if you apply (pi U)(t) you get the original claim.
// A new value t' is picked and U is evaluated there. We now have claim of the form:
// 
// sum_{x, y, z, out} EQ(r_x, r_y, r_z, r_out; x, y, z, out)(
//  (STATE(x+1, y, z, out)(t') * STATE(x-1, y, z, out)(t') + STATE(x, y, z, out)(t') * ENCODED_ONE(t'))
// )
//
// This is a normal sumcheck that can be ran across the variables out, y and z.
// We first run along out in ascending order, then along (y, z) in ascending order.
//
// Now, we get claim on 3-dimensional hypercube (with 5 nonzero entries).
// We communicate them directly and check the equality
// We sample 3 variables, substitute for x, fold back.
// This is our new claim.

impl VerifierCfg {
    pub fn verify<Ch: Challenger>(
        &self,
        ctx: &mut ProofReader<Ch>,
        claim: HybridClaim,
    ) -> Result<HybridClaim, VerifierError> {
        debug_assert_eq!(claim.r_out.len(), self.log_packed_instances);
        debug_assert_eq!(claim.r_x.len(), 3);
        debug_assert_eq!(claim.r_y.len(), 3);
        debug_assert_eq!(claim.r_z.len(), 6);

        let u = ctx.read_f128_vec(rmfe::PRODUCT_BITS)?;
        let pi_u = apply_boolean_matrix(rmfe::projection_matrix(), &u);
        let projected = evaluate_univar(&pi_u, claim.t);
        if projected != claim.ev {
            return Err(VerifierError::InvalidProof);
        }

        let t = ctx.sample_f128();
        let ev = evaluate_univar(&u, t);

        let mut running_claim = ev;
        let sumcheck_rounds = claim.r_out.len() + claim.r_y.len() + claim.r_z.len();

        let mut bound_out = Vec::with_capacity(claim.r_out.len());
        let mut bound_y = Vec::with_capacity(claim.r_y.len());
        let mut bound_z = Vec::with_capacity(claim.r_z.len());
        for round in 0..sumcheck_rounds {
            let msg = GruenMessage::read(ctx)?;
            let challenge = ctx.sample_f128();
            let eq_challenge = if round < claim.r_out.len() {
                claim.r_out[round]
            } else if round < claim.r_out.len() + claim.r_y.len() {
                claim.r_y[round - claim.r_out.len()]
            } else {
                claim.r_z[round - claim.r_out.len() - claim.r_y.len()]
            };
            running_claim = msg.bind(running_claim, eq_challenge, challenge)?;
            if round < claim.r_out.len() {
                bound_out.push(challenge);
            } else if round < claim.r_out.len() + claim.r_y.len() {
                bound_y.push(challenge);
            } else {
                bound_z.push(challenge);
            }
        }

        let inputs = ctx.read_f128_vec(5)?;
        let eq_x = eq_poly_v(&claim.r_x);
        let mut expected = F128::ZERO;
        for x in 0..5 {
            let left = inputs[(x + 1) % 5];
            let right = inputs[(x + 2) % 5];
            let c = inputs[x] + right;
            expected += eq_x[x] * (left * right + c * c);
        }
        if expected != running_claim {
            return Err(VerifierError::InvalidProof);
        }

        let r_x = ctx.sample_f128_vec(3);
        let eq_x = eq_poly_v(&r_x);
        let mut ev = F128::ZERO;
        for x in 0..5 {
            ev += eq_x[x] * inputs[x];
        }

        Ok(HybridClaim {
            t,
            r_x,
            r_y: bound_y,
            r_z: bound_z,
            r_out: bound_out,
            ev,
        })
    }
}
