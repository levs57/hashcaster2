//! RMFE/subspace constants will live here.
//!
//! Deliberately empty for the moment: the previous staging constants came from
//! a different field/subspace experiment.  That subspace is not the target for
//! this crate.
//!
//! The next step is to install the actual `F2^128` subspace used by the clean
//! Hashcaster2 protocol, together with a validator for the multiplicative
//! friendly property.

use crate::field::F128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RmfeValidationError {
    MissingSubspaceConstants,
}

pub fn validate_rmfe() -> Result<(), RmfeValidationError> {
    Err(RmfeValidationError::MissingSubspaceConstants)
}

pub fn basis_elements() -> &'static [F128] {
    &[]
}
