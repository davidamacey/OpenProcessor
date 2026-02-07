/// Detection endpoints: /detect and /detect/batch

use crate::error::AppError;
use crate::postprocess::detection::transform_detections;
use crate::preprocess::decode::decode_image;
use crate::preprocess::yolo::letterbox;
use crate::triton::TritonClient;
use axum::extract::{Multipart, State};
use axum::Json;
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

pub struct AppState {
    pub triton: TritonClient,
}

/// POST /detect - Single image detection
pub async fn detect_single(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<InferenceResult>, AppError> {
    let start = Instant::now();

    // Extract image bytes from multipart form
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

    // Decode image
    let img = decode_image(&image_bytes)
        .map_err(|e| AppError::ImageDecode(e))?;

    // Letterbox preprocessing
    let letterbox_result = letterbox(&img, 640);

    // Run YOLO inference
    let yolo_result = state.triton.infer_yolo(&letterbox_result.tensor, 1).await?;

    // Transform detections to original coordinates
    let num_dets = yolo_result.num_dets[0];
    let detections = transform_detections(
        num_dets,
        &yolo_result.boxes,
        &yolo_result.scores,
        &yolo_result.classes,
        letterbox_result.scale,
        letterbox_result.pad_w,
        letterbox_result.pad_h,
        letterbox_result.orig_w,
        letterbox_result.orig_h,
    );

    Ok(Json(InferenceResult {
        detections,
        image: ImageInfo {
            width: letterbox_result.orig_w,
            height: letterbox_result.orig_h,
        },
        inference_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

/// POST /detect/batch - Batch detection
pub async fn detect_batch(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<BatchInferenceResult>, AppError> {
    let start = Instant::now();

    let mut images = Vec::new();

    // Extract all image fields
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
    let mut letterbox_results = Vec::new();
    for img_bytes in &images {
        let img = decode_image(img_bytes).map_err(|e| AppError::ImageDecode(e))?;
        let lb = letterbox(&img, 640);
        letterbox_results.push(lb);
    }

    // Stack tensors into batch
    let batch_size = letterbox_results.len();
    let mut batch_tensor = Vec::with_capacity(batch_size * 3 * 640 * 640);
    for lb in &letterbox_results {
        batch_tensor.extend_from_slice(&lb.tensor);
    }

    // Run batch inference
    let yolo_result = state.triton.infer_yolo(&batch_tensor, batch_size).await?;

    // Transform each image's detections
    let mut results = Vec::new();
    for (i, lb) in letterbox_results.iter().enumerate() {
        let num_dets = yolo_result.num_dets[i];
        let box_offset = i * 300 * 4;
        let score_offset = i * 300;
        let class_offset = i * 300;

        let detections = transform_detections(
            num_dets,
            &yolo_result.boxes[box_offset..box_offset + 300 * 4],
            &yolo_result.scores[score_offset..score_offset + 300],
            &yolo_result.classes[class_offset..class_offset + 300],
            lb.scale,
            lb.pad_w,
            lb.pad_h,
            lb.orig_w,
            lb.orig_h,
        );

        results.push(InferenceResult {
            detections,
            image: ImageInfo {
                width: lb.orig_w,
                height: lb.orig_h,
            },
            inference_time_ms: 0.0, // Per-image timing not tracked in batch
        });
    }

    Ok(Json(BatchInferenceResult {
        results,
        total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    }))
}

#[derive(Serialize)]
pub struct InferenceResult {
    pub detections: Vec<crate::postprocess::detection::Detection>,
    pub image: ImageInfo,
    pub inference_time_ms: f64,
}

#[derive(Serialize)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize)]
pub struct BatchInferenceResult {
    pub results: Vec<InferenceResult>,
    pub total_time_ms: f64,
}
