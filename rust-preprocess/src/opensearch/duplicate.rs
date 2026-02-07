/// Near-duplicate detection using k-NN search and clustering.
///
/// Matches Python's logic from src/services/duplicate_detection.py:173-284.

use super::client::{KnnResult, OpenSearchClient};
use crate::error::AppError;
use std::collections::{HashMap, HashSet};

const KNN_K: usize = 10;
const KNN_THRESHOLD: f32 = 0.95; // cosine similarity threshold

/// Detect near-duplicates for a batch of images using k-NN search.
///
/// Returns map of image_id -> near_duplicate_group_id.
pub async fn detect_near_duplicates(
    client: &OpenSearchClient,
    index: &str,
    image_ids: &[String],
    embeddings: &[Vec<f32>],
) -> Result<HashMap<String, String>, AppError> {
    let mut assignments: HashMap<String, String> = HashMap::new();
    let mut next_group_id = 0;

    for (i, (image_id, embedding)) in image_ids.iter().zip(embeddings.iter()).enumerate() {
        // k-NN search for similar images
        let similar = client
            .knn_search(index, embedding, KNN_K, KNN_THRESHOLD)
            .await?;

        if similar.is_empty() {
            // No similar images found - this is a unique image
            continue;
        }

        // Find the highest-scoring match
        let best_match = similar
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();

        // Assign to the same group as the best match (or create a new group)
        let group_id = if let Some(existing_group) = assignments.get(&best_match.image_id) {
            existing_group.clone()
        } else {
            let new_id = format!("ndg_{}", next_group_id);
            next_group_id += 1;
            new_id
        };

        assignments.insert(image_id.clone(), group_id);
    }

    Ok(assignments)
}

/// Update documents with near_duplicate_group assignments using _bulk update.
pub async fn assign_near_duplicate_groups(
    client: &OpenSearchClient,
    index: &str,
    assignments: &HashMap<String, String>,
) -> Result<(), AppError> {
    if assignments.is_empty() {
        return Ok(());
    }

    let mut bulk = String::new();

    for (image_id, group_id) in assignments {
        // Update action
        bulk.push_str(&format!(
            "{{\"update\":{{\"_index\":\"{index}\",\"_id\":\"{image_id}\"}}}}\n"
        ));
        // Doc update
        bulk.push_str(&format!(
            "{{\"doc\":{{\"near_duplicate_group\":\"{group_id}\"}}}}\n"
        ));
    }

    client.bulk(&bulk).await?;
    Ok(())
}
