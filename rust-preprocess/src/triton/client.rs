/// Triton gRPC client with connection pooling and retry logic.
///
/// Matches Python's channel options from src/clients/triton_pool.py:52-74:
/// - keepalive 30s
/// - 100MB message limits
/// - max_concurrent_streams 1000

use super::proto::*;
use crate::error::AppError;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Endpoint};

const MAX_RETRIES: u32 = 3;
const RETRY_BASE_DELAY_MS: u64 = 100;

#[derive(Clone)]
pub struct TritonClient {
    channels: Arc<Vec<Channel>>,
    next_channel: Arc<AtomicUsize>,
}

impl TritonClient {
    pub async fn new(triton_url: &str, pool_size: usize) -> Result<Self, AppError> {
        let mut channels = Vec::with_capacity(pool_size);

        for _ in 0..pool_size {
            let endpoint = Endpoint::from_shared(format!("http://{triton_url}"))
                .map_err(|e| AppError::Internal(format!("Invalid Triton URL: {e}")))?
                .keep_alive_timeout(Duration::from_secs(30))
                .http2_keep_alive_interval(Duration::from_secs(30))
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .initial_stream_window_size(Some(100 * 1024 * 1024)) // 100MB
                .initial_connection_window_size(Some(100 * 1024 * 1024))
                .timeout(Duration::from_secs(60))
                .concurrency_limit(1000);

            let channel = endpoint
                .connect()
                .await
                .map_err(|e| AppError::TritonInference(format!("Failed to connect to Triton: {e}")))?;

            channels.push(channel);
        }

        Ok(Self {
            channels: Arc::new(channels),
            next_channel: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Get next channel using round-robin.
    fn get_channel(&self) -> Channel {
        let idx = self.next_channel.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        self.channels[idx].clone()
    }

    /// Infer YOLO End2End model.
    ///
    /// Input: [N, 3, 640, 640] FP32
    /// Outputs: num_dets [N, 1] INT32, det_boxes [N, 300, 4] FP32,
    ///          det_scores [N, 300] FP32, det_classes [N, 300] FP32
    pub async fn infer_yolo(
        &self,
        batch_tensor: &[f32],
        batch_size: usize,
    ) -> Result<YoloResult, AppError> {
        let shape = vec![batch_size as i64, 3, 640, 640];

        let mut request = ModelInferRequest {
            model_name: "yolov11_small_trt_end2end".to_string(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            inputs: vec![inference::model_infer_request::InferInputTensor {
                name: "images".to_string(),
                datatype: "FP32".to_string(),
                shape: shape.clone(),
                parameters: Default::default(),
                contents: Some(inference::InferTensorContents {
                    fp32_contents: batch_tensor.to_vec(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            outputs: vec![
                inference::model_infer_request::InferRequestedOutputTensor {
                    name: "num_dets".to_string(),
                    ..Default::default()
                },
                inference::model_infer_request::InferRequestedOutputTensor {
                    name: "det_boxes".to_string(),
                    ..Default::default()
                },
                inference::model_infer_request::InferRequestedOutputTensor {
                    name: "det_scores".to_string(),
                    ..Default::default()
                },
                inference::model_infer_request::InferRequestedOutputTensor {
                    name: "det_classes".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let response = self.infer_with_retry(request).await?;

        // Parse outputs
        let num_dets_tensor = response.outputs.iter()
            .find(|o| o.name == "num_dets")
            .ok_or_else(|| AppError::TritonInference("num_dets output missing".to_string()))?;
        let boxes_tensor = response.outputs.iter()
            .find(|o| o.name == "det_boxes")
            .ok_or_else(|| AppError::TritonInference("det_boxes output missing".to_string()))?;
        let scores_tensor = response.outputs.iter()
            .find(|o| o.name == "det_scores")
            .ok_or_else(|| AppError::TritonInference("det_scores output missing".to_string()))?;
        let classes_tensor = response.outputs.iter()
            .find(|o| o.name == "det_classes")
            .ok_or_else(|| AppError::TritonInference("det_classes output missing".to_string()))?;

        // Extract data from raw_output_contents (fallback to contents)
        let num_dets_data = if let Some(ref raw) = response.raw_output_contents.get(0) {
            bytemuck::cast_slice::<u8, i32>(raw).to_vec()
        } else {
            num_dets_tensor.contents.as_ref()
                .ok_or_else(|| AppError::TritonInference("num_dets contents missing".to_string()))?
                .int_contents.clone()
        };

        let boxes_data = if let Some(ref raw) = response.raw_output_contents.get(1) {
            bytemuck::cast_slice::<u8, f32>(raw).to_vec()
        } else {
            boxes_tensor.contents.as_ref()
                .ok_or_else(|| AppError::TritonInference("det_boxes contents missing".to_string()))?
                .fp32_contents.clone()
        };

        let scores_data = if let Some(ref raw) = response.raw_output_contents.get(2) {
            bytemuck::cast_slice::<u8, f32>(raw).to_vec()
        } else {
            scores_tensor.contents.as_ref()
                .ok_or_else(|| AppError::TritonInference("det_scores contents missing".to_string()))?
                .fp32_contents.clone()
        };

        let classes_data = if let Some(ref raw) = response.raw_output_contents.get(3) {
            bytemuck::cast_slice::<u8, i32>(raw).to_vec()
        } else {
            classes_tensor.contents.as_ref()
                .ok_or_else(|| AppError::TritonInference("det_classes contents missing".to_string()))?
                .int_contents.clone()
        };

        Ok(YoloResult {
            num_dets: num_dets_data,
            boxes: boxes_data,
            scores: scores_data,
            classes: classes_data,
        })
    }

    /// Infer MobileCLIP image encoder.
    ///
    /// Input: [N, 3, 256, 256] FP32
    /// Output: image_embeddings [N, 512] FP32
    pub async fn infer_clip(
        &self,
        batch_tensor: &[f32],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        let shape = vec![batch_size as i64, 3, 256, 256];

        let request = ModelInferRequest {
            model_name: "mobileclip2_s2_image_encoder".to_string(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            inputs: vec![inference::model_infer_request::InferInputTensor {
                name: "images".to_string(),
                datatype: "FP32".to_string(),
                shape: shape.clone(),
                parameters: Default::default(),
                contents: Some(inference::InferTensorContents {
                    fp32_contents: batch_tensor.to_vec(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            outputs: vec![
                inference::model_infer_request::InferRequestedOutputTensor {
                    name: "image_embeddings".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let response = self.infer_with_retry(request).await?;

        // Parse output
        let embeddings_tensor = response.outputs.iter()
            .find(|o| o.name == "image_embeddings")
            .ok_or_else(|| AppError::TritonInference("image_embeddings output missing".to_string()))?;

        let embeddings_data = if let Some(ref raw) = response.raw_output_contents.first() {
            bytemuck::cast_slice::<u8, f32>(raw).to_vec()
        } else {
            embeddings_tensor.contents.as_ref()
                .ok_or_else(|| AppError::TritonInference("image_embeddings contents missing".to_string()))?
                .fp32_contents.clone()
        };

        // Split into per-image embeddings (512 dims each)
        Ok(embeddings_data.chunks(512).map(|c| c.to_vec()).collect())
    }

    /// Execute inference with exponential backoff retry.
    async fn infer_with_retry(
        &self,
        request: ModelInferRequest,
    ) -> Result<ModelInferResponse, AppError> {
        let mut delay_ms = RETRY_BASE_DELAY_MS;

        for attempt in 0..MAX_RETRIES {
            let channel = self.get_channel();
            let mut client = GrpcInferenceServiceClient::new(channel);

            match client.model_infer(request.clone()).await {
                Ok(response) => return Ok(response.into_inner()),
                Err(e) => {
                    let code = e.code();
                    if code == tonic::Code::Unavailable || code == tonic::Code::ResourceExhausted {
                        if attempt < MAX_RETRIES - 1 {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                            delay_ms = (delay_ms * 2).min(5000);
                            continue;
                        }
                    }
                    return Err(AppError::TritonInference(format!("Inference failed: {e}")));
                }
            }
        }

        Err(AppError::TritonInference(
            "Max retries exceeded".to_string(),
        ))
    }
}

pub struct YoloResult {
    pub num_dets: Vec<i32>,   // [N, 1] flattened
    pub boxes: Vec<f32>,      // [N, 300, 4] flattened
    pub scores: Vec<f32>,     // [N, 300] flattened
    pub classes: Vec<i32>,    // [N, 300] flattened - INT32 per model config
}
