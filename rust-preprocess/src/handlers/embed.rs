/// Embedding endpoints: /embed/image and /embed/batch

use crate::error::AppError;
use crate::preprocess::clip::center_crop;
use crate::preprocess::decode::decode_image;
use crate::triton::TritonClient;
use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

pub struct AppState {
    pub triton: TritonClient,
}

/// POST /embed/image - Single image embedding
pub async fn embed_single(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<ImageEmbeddingResponse>, AppError> {
    let start = Instant::now();

    // Extract image bytes
    let mut image_bytes = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Multipart error: {e}")))?
    {
        if field.name() == Some("image") {
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::BadRequest(format!("Failed to read image: {e}")))?;
            image_bytes = Some(data.to_vec());
            break;
        }
    }

    let image_bytes = image_bytes
        .ok_or_else(|| AppError::BadRequest("No image field in request".to_string()))?;

    // Decode and preprocess
    let img = decode_image(&image_bytes).map_err(|e| AppError::ImageDecode(e))?;
    let clip_tensor = center_crop(&img, 256);

    // Run CLIP inference
    let embeddings = state.triton.infer_clip(&clip_tensor, 1).await?;
    let embedding = embeddings.into_iter().next()
        .ok_or_else(|| AppError::Internal("No embedding returned".to_string()))?;

    Ok(Json(ImageEmbeddingResponse {
        embedding,
        dimensions: 512,
        inference_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

/// POST /embed/batch - Batch image embeddings
pub async fn embed_batch(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<BatchEmbeddingResponse>, AppError> {
    let start = Instant::now();

    let mut images = Vec::new();

    // Extract all images
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Multipart error: {e}")))?
    {
        if field.name() == Some("images") {
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::BadRequest(format!("Failed to read image: {e}")))?;
            images.push(data.to_vec());
        }
    }

    if images.is_empty() {
        return Err(AppError::BadRequest("No images in request".to_string()));
    }

    if images.len() > 64 {
        return Err(AppError::BadRequest("Too many images (max 64)".to_string()));
    }

    // Decode and preprocess all images
    let mut clip_tensors = Vec::new();
    for img_bytes in &images {
        let img = decode_image(img_bytes).map_err(|e| AppError::ImageDecode(e))?;
        let clip_tensor = center_crop(&img, 256);
        clip_tensors.push(clip_tensor);
    }

    // Stack into batch
    let batch_size = clip_tensors.len();
    let mut batch_tensor = Vec::with_capacity(batch_size * 3 * 256 * 256);
    for tensor in &clip_tensors {
        batch_tensor.extend_from_slice(tensor);
    }

    // Run batch inference
    let embeddings = state.triton.infer_clip(&batch_tensor, batch_size).await?;

    Ok(Json(BatchEmbeddingResponse {
        embeddings,
        dimensions: 512,
        total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

#[derive(Serialize)]
pub struct ImageEmbeddingResponse {
    pub embedding: Vec<f32>,
    pub dimensions: usize,
    pub inference_time_ms: f64,
}

#[derive(Serialize)]
pub struct BatchEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dimensions: usize,
    pub total_time_ms: f64,
}
