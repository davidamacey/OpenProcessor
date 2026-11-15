//! Generates the model-server client from Triton's protos (`proto/`, tag r26.10) with protox, a
//! protobuf compiler written in Rust, so the build needs no `protoc`
//! (https://docs.rs/protox/0.10.0/protox/fn.compile.html,
//! https://docs.rs/tonic-prost-build/0.14.9/tonic_prost_build/struct.Builder.html).
//!
//! Nothing is linked here: CUDA, NVRTC and nvJPEG are loaded at run time, so the crate builds and
//! its tests run on a machine without a GPU.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto");
    let files = protox::compile(["proto/grpc_service.proto"], ["proto"])?;
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        // Output tensors arrive as reference-counted slices of the response buffer, not copies.
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .compile_fds(files)?;
    Ok(())
}
