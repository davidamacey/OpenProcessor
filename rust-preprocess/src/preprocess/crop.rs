/// Crop bounding boxes from images for per-detection embeddings.

use super::clip::center_crop;
use super::decode::DecodedImage;

/// Crop a bounding box from an image and preprocess for CLIP.
///
/// Box coordinates are normalized [0, 1] relative to image dimensions.
pub fn crop_and_preprocess_clip(
    img: &DecodedImage,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> Vec<f32> {
    // Convert normalized coords to pixels
    let px1 = (x1 * img.width as f32).max(0.0) as usize;
    let py1 = (y1 * img.height as f32).max(0.0) as usize;
    let px2 = (x2 * img.width as f32).min(img.width as f32) as usize;
    let py2 = (y2 * img.height as f32).min(img.height as f32) as usize;

    // Ensure valid crop dimensions
    let crop_w = px2.saturating_sub(px1).max(1);
    let crop_h = py2.saturating_sub(py1).max(1);

    // Extract crop pixels (RGB, row-major)
    let mut crop_data = Vec::with_capacity(crop_w * crop_h * 3);
    for y in py1..py2 {
        let row_start = (y * img.width as usize + px1) * 3;
        let row_end = (y * img.width as usize + px2) * 3;
        crop_data.extend_from_slice(&img.data[row_start..row_end]);
    }

    // Create temporary DecodedImage for the crop
    let crop_img = DecodedImage {
        width: crop_w as u32,
        height: crop_h as u32,
        data: crop_data,
    };

    // Apply CLIP preprocessing (resize + center crop to 256x256)
    center_crop(&crop_img, 256)
}
