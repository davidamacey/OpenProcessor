//! The general part of the native frame engine: nothing here knows the plan, the wire, a branch
//! or a product kernel (plan 17 section 2.0).

#[cfg(test)]
mod tests {
    #[test]
    fn crate_is_named_for_its_directory() {
        assert_eq!(env!("CARGO_PKG_NAME"), "gpu_frame");
    }
}
