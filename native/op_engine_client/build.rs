//! Generates the KServe v2 client (and, for the tests' fake servers, the server) from the model
//! server's protos in `../gpu_frame/proto`, the one copy every crate that speaks the protocol
//! reads (Triton tag r26.10), with protox so the build needs no `protoc`
//! (<https://docs.rs/protox/0.10.0/protox/fn.compile.html>,
//! <https://docs.rs/tonic-prost-build/0.14.9/tonic_prost_build/struct.Builder.html>).
//!
//! Only the calls this client makes are generated: `ModelInfer` and `ModelReady` for the
//! engine, `ModelReady` and `RepositoryModelLoad` for the model server.
//!
//! The client is written against the protocol, not against the engine's own crate, so a defect in
//! the engine's wire code cannot hide in the client that talks to it.

const PROTO_DIR: &str = "../gpu_frame/proto";
const USED: [&str; 3] = ["ModelInfer", "ModelReady", "RepositoryModelLoad"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={PROTO_DIR}");
    let mut set = protox::compile(["grpc_service.proto"], [PROTO_DIR])?;
    let method_lists = set
        .file
        .iter_mut()
        .flat_map(|file| file.service.iter_mut())
        .map(|service| &mut service.method);
    for methods in method_lists {
        methods.retain(|m| m.name.as_deref().is_some_and(|name| USED.contains(&name)));
    }
    let builder = tonic_prost_build::configure()
        .build_client(true)
        .build_server(true);
    // A photo is up to tens of MB and a response carries crop JPEGs: slices, not copies.
    builder
        .bytes(".inference.ModelInferRequest.raw_input_contents")
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .compile_fds(set)?;
    Ok(())
}
