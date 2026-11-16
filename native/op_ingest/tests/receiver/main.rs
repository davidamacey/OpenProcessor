//! The upload receiver through its router: the Python cases, streaming and back pressure, and what
//! a refused or abandoned request leaves behind.
#![expect(
    clippy::unwrap_used,
    reason = "the request tests and their rig are test code outside #[test] functions"
)]

mod cancel;
mod cases;
mod common;
mod contract;
mod fields;
mod streaming;
