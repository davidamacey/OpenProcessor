//! Generates the KServe v2 server from the model server's protos (`../gpu_frame/proto`, Triton
//! tag r26.10, one copy for both crates) with protox, a protobuf compiler written in Rust, so the
//! build needs no `protoc` (https://docs.rs/protox/0.10.0/protox/fn.compile.html,
//! https://docs.rs/tonic-prost-build/0.14.9/tonic_prost_build/struct.Builder.html).
//!
//! The service keeps the five calls the engine answers (protocol version 1). Every other call of
//! the model server's service is left out of the generated router, so a client that sends one
//! gets gRPC `UNIMPLEMENTED` from the router, as it does from the Python engine's front. The
//! client of the same five calls is what `op-frame-engine health` asks the running engine with.

const ANSWERED: [&str; 5] = [
    "ServerLive",
    "ServerReady",
    "ModelReady",
    "ModelMetadata",
    "ModelInfer",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../gpu_frame/proto");
    let mut files = protox::compile(["grpc_service.proto"], ["../gpu_frame/proto"])?;
    for file in &mut files.file {
        for service in &mut file.service {
            service
                .method
                .retain(|m| m.name.as_deref().is_some_and(|n| ANSWERED.contains(&n)));
        }
    }
    tonic_prost_build::configure()
        .build_client(true)
        .build_transport(false)
        .build_server(true)
        // Inputs arrive as reference-counted slices of the request buffer, and outputs leave as
        // slices of the engine's buffers, not copies.
        .bytes(".inference.ModelInferRequest.raw_input_contents")
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .compile_fds(files)?;
    Ok(())
}
