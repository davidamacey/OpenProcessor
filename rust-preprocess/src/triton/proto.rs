/// Re-exports of generated Triton protobuf types.

// The generated code will be in the OUT_DIR from tonic-build
pub mod inference {
    tonic::include_proto!("inference");
}

pub use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
pub use inference::{
    InferTensorContents, ModelInferRequest, ModelInferResponse,
};
