//! The product part of the native frame engine: kernels, branches, the KServe front of protocol
//! version 1 (plan 17 section 2.0). Every CUDA or nvJPEG call goes through `gpu_frame`.

fn main() {}

#[cfg(test)]
mod tests {
    #[test]
    fn binary_carries_the_name_the_image_and_compose_use() {
        assert_eq!(env!("CARGO_BIN_NAME"), "op-frame-engine");
    }
}
