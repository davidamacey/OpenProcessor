/// Bulk indexing helpers for OpenSearch.
///
/// Formats documents for _bulk API (NDJSON format).

use serde_json::json;

/// Vehicle class IDs (car, motorcycle, bus, truck, boat)
const VEHICLE_CLASSES: &[i32] = &[2, 3, 5, 7, 8];
/// Person class ID
const PERSON_CLASS: i32 = 0;

/// Build bulk index operations for the global index.
///
/// Each document contains: image_id, image_path, global_embedding, imohash, width, height
pub fn build_global_bulk(
    image_ids: &[String],
    image_paths: &[String],
    embeddings: &[Vec<f32>],
    imohashes: &[String],
    dimensions: &[(u32, u32)],
) -> String {
    let mut bulk = String::new();
    let now = chrono::Utc::now().to_rfc3339();

    for i in 0..image_ids.len() {
        // Action line
        bulk.push_str(&format!(
            "{{\"index\":{{\"_index\":\"visual_search_global\",\"_id\":\"{}\"}}}}\n",
            image_ids[i]
        ));

        // Document line
        let doc = json!({
            "image_id": image_ids[i],
            "image_path": image_paths[i],
            "global_embedding": embeddings[i],
            "imohash": imohashes[i],
            "width": dimensions[i].0,
            "height": dimensions[i].1,
            "indexed_at": now
        });
        bulk.push_str(&serde_json::to_string(&doc).unwrap());
        bulk.push('\n');
    }

    bulk
}

/// Build bulk index operations for detection-based indexes (vehicles, people).
///
/// Each detection gets its own embedding document if it's a vehicle or person.
pub fn build_detection_bulk(
    image_ids: &[String],
    image_paths: &[String],
    yolo_results: &[DetectionResult],
    box_embeddings: &[Vec<Vec<f32>>], // [image_idx][detection_idx] -> embedding
) -> String {
    let mut bulk = String::new();
    let now = chrono::Utc::now().to_rfc3339();

    for (img_idx, result) in yolo_results.iter().enumerate() {
        for (det_idx, det) in result.detections.iter().enumerate() {
            let class_id = det.class_id;
            let index = if VEHICLE_CLASSES.contains(&class_id) {
                "visual_search_vehicles"
            } else if class_id == PERSON_CLASS {
                "visual_search_people"
            } else {
                continue; // Skip other classes
            };

            // Get embedding for this detection
            let embedding = if let Some(img_embeddings) = box_embeddings.get(img_idx) {
                if let Some(emb) = img_embeddings.get(det_idx) {
                    emb
                } else {
                    continue;
                }
            } else {
                continue;
            };

            let detection_id = format!("{}_{}", image_ids[img_idx], det_idx);

            // Action line
            bulk.push_str(&format!(
                "{{\"index\":{{\"_index\":\"{index}\",\"_id\":\"{detection_id}\"}}}}\n"
            ));

            // Document line
            let doc = json!({
                "detection_id": detection_id,
                "image_id": image_ids[img_idx],
                "image_path": image_paths[img_idx],
                "embedding": embedding,
                "class_id": class_id,
                "class_name": get_class_name(class_id),
                "box": [det.x1, det.y1, det.x2, det.y2],
                "confidence": det.confidence,
                "indexed_at": now
            });
            bulk.push_str(&serde_json::to_string(&doc).unwrap());
            bulk.push('\n');
        }
    }

    bulk
}

pub struct DetectionResult {
    pub detections: Vec<Detection>,
}

pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: i32,
}

/// Get COCO class name from class ID.
fn get_class_name(class_id: i32) -> &'static str {
    match class_id {
        0 => "person",
        2 => "car",
        3 => "motorcycle",
        5 => "bus",
        7 => "truck",
        8 => "boat",
        _ => "unknown",
    }
}
