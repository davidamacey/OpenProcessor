//! The durable object store, derived files, erase and reconcile through their public interface,
//! on `InMemory` and on a temporary directory. One crate so the helpers are shared.
#![cfg(test)]

mod durable_tests;
mod s3_tests;
mod support;
