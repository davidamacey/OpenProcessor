//! The front end to end over real sockets: a stub Python API, a running front, a client.
//!
//! One test crate, so the helpers in `support` are compiled once and every one is used.

// `cfg(test)` is always set for a test target; the attribute is what lets clippy's
// `allow-unwrap-in-tests` cover the helpers as well as the `#[test]` functions.
#[cfg(test)]
mod edges;
#[cfg(test)]
mod forwarding;
#[cfg(test)]
mod serving;
#[cfg(test)]
mod streaming;
#[cfg(test)]
mod support;
