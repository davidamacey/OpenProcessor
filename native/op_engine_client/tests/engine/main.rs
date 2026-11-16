//! The engine client against the Python client's own output and the contract file.

#![expect(
    clippy::unwrap_used,
    reason = "the helpers of the suites are test code, outside #[test] functions"
)]

mod plans;
mod support;
