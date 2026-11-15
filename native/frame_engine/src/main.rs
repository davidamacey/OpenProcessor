//! `op-frame-engine`: the frame engine process (plan 17 section 5 N0).

fn main() {}

#[cfg(test)]
mod tests {
    #[test]
    fn binary_carries_the_name_the_image_and_compose_use() {
        assert_eq!(env!("CARGO_BIN_NAME"), "op-frame-engine");
    }
}
