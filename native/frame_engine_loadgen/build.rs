//! Generates the KServe v2 client (and, for the tests' in-process engine, the server) from the
//! model server's protos in `../gpu_frame/proto`, the one copy both crates and this client read
//! (Triton tag r26.10), with protox so the build needs no `protoc`
//! (https://docs.rs/protox/0.10.0/protox/fn.compile.html,
//! https://docs.rs/tonic-prost-build/0.14.9/tonic_prost_build/struct.Builder.html).
//!
//! The client is written against the protocol, not against the engine's own crate, so a defect in
//! the engine's wire code cannot hide in the instrument that measures it.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../gpu_frame/proto");
    let files = protox::compile(["grpc_service.proto"], ["../gpu_frame/proto"])?;
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // A photo is up to tens of MB and a response carries crop JPEGs: slices, not copies.
        .bytes(".inference.ModelInferRequest.raw_input_contents")
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .compile_fds(files)?;
    Ok(())
}
