/// Detection coordinate transformation (inverse letterbox).
///
/// Matches src/utils/affine.py:155-214 exactly.

use super::coco_classes::class_name as get_class_name;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: i32,
    pub class_name: &'static str,
}

/// Transform YOLO detections from letterbox coordinates to original image coordinates.
///
/// EfficientNMS with normalize_boxes=True outputs coords normalized to the 640x640
/// letterboxed input [0,1]. This function applies inverse letterbox transformation
/// to convert them to [0,1] normalized coordinates relative to original image dimensions.
///
/// Matches Python's src/utils/affine.py:155-214 exactly.
pub fn transform_detections(
    num_dets: i32,
    boxes: &[f32],       // [300, 4] flattened
    scores: &[f32],      // [300] flattened
    classes: &[i32],     // [300] flattened - INT32 per model config
    scale: f64,
    pad_w: f64,
    pad_h: f64,
    orig_w: u32,
    orig_h: u32,
) -> Vec<Detection> {
    let mut detections = Vec::new();

    if num_dets == 0 {
        return detections;
    }

    // Auto-detect if boxes are normalized [0,1] or pixel coordinates
    // If max value > 1.0, boxes are already in pixel coords (640x640 space)
    let max_coord = boxes.iter()
        .take(num_dets as usize * 4)
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let is_pixel_coords = max_coord > 1.0;
    let input_size = 640.0;

    for i in 0..(num_dets as usize) {
        let box_offset = i * 4;
        let mut x1 = boxes[box_offset] as f64;
        let mut y1 = boxes[box_offset + 1] as f64;
        let mut x2 = boxes[box_offset + 2] as f64;
        let mut y2 = boxes[box_offset + 3] as f64;
        let score = scores[i];
        let class_id = classes[i];

        // 1. Scale letterbox-normalized [0,1] to letterbox pixels (skip if already pixels)
        if !is_pixel_coords {
            x1 *= input_size;
            y1 *= input_size;
            x2 *= input_size;
            y2 *= input_size;
        }

        // 2. Remove padding and scale to get original pixels
        x1 = (x1 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
        x2 = (x2 - pad_w) / scale;
        y2 = (y2 - pad_h) / scale;

        // 3. Normalize to original image dimensions
        x1 /= orig_w as f64;
        y1 /= orig_h as f64;
        x2 /= orig_w as f64;
        y2 /= orig_h as f64;

        // 4. Clip to [0, 1]
        let x1_clipped = x1.clamp(0.0, 1.0) as f32;
        let y1_clipped = y1.clamp(0.0, 1.0) as f32;
        let x2_clipped = x2.clamp(0.0, 1.0) as f32;
        let y2_clipped = y2.clamp(0.0, 1.0) as f32;

        detections.push(Detection {
            x1: x1_clipped,
            y1: y1_clipped,
            x2: x2_clipped,
            y2: y2_clipped,
            confidence: score,
            class_id,
            class_name: get_class_name(class_id),
        });
    }

    detections
}
