/// OpenSearch REST client for bulk indexing and search operations.

use crate::error::AppError;
use reqwest::Client;
use serde_json::{json, Value};

#[derive(Clone)]
pub struct OpenSearchClient {
    base_url: String,
    client: Client,
}

impl OpenSearchClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: Client::new(),
        }
    }

    /// Bulk index documents using OpenSearch _bulk API.
    ///
    /// Format: newline-delimited JSON (NDJSON)
    /// Each operation is 2 lines: action metadata + document source
    pub async fn bulk(&self, operations: &str) -> Result<Value, AppError> {
        let url = format!("{base_url}/_bulk", base_url = self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/x-ndjson")
            .body(operations.to_string())
            .send()
            .await
            .map_err(|e| AppError::OpenSearch(format!("Bulk request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(AppError::OpenSearch(format!(
                "Bulk indexing failed ({status}): {text}"
            )));
        }

        response
            .json()
            .await
            .map_err(|e| AppError::OpenSearch(format!("Failed to parse bulk response: {e}")))
    }

    /// Multi-search API for checking duplicates by imohash.
    ///
    /// Returns list of image_ids that already exist in the index.
    pub async fn msearch_duplicates(
        &self,
        index: &str,
        imohashes: &[String],
    ) -> Result<Vec<String>, AppError> {
        if imohashes.is_empty() {
            return Ok(Vec::new());
        }

        // Build msearch request (NDJSON format)
        let mut ndjson = String::new();
        for hash in imohashes {
            // Header line
            ndjson.push_str(&format!("{{\"index\":\"{index}\"}}\n"));
            // Query line
            let query = json!({
                "query": {
                    "term": {
                        "imohash": hash
                    }
                },
                "_source": ["image_id"],
                "size": 1
            });
            ndjson.push_str(&serde_json::to_string(&query).unwrap());
            ndjson.push('\n');
        }

        let url = format!("{base_url}/_msearch", base_url = self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/x-ndjson")
            .body(ndjson)
            .send()
            .await
            .map_err(|e| AppError::OpenSearch(format!("msearch failed: {e}")))?;

        let result: Value = response
            .json()
            .await
            .map_err(|e| AppError::OpenSearch(format!("Failed to parse msearch response: {e}")))?;

        // Extract image_ids from hits
        let mut existing_ids = Vec::new();
        if let Some(responses) = result["responses"].as_array() {
            for resp in responses {
                if let Some(hits) = resp["hits"]["hits"].as_array() {
                    if let Some(first_hit) = hits.first() {
                        if let Some(image_id) = first_hit["_source"]["image_id"].as_str() {
                            existing_ids.push(image_id.to_string());
                        }
                    }
                }
            }
        }

        Ok(existing_ids)
    }

    /// k-NN search for near-duplicate detection.
    ///
    /// Searches for similar embeddings using cosine similarity.
    pub async fn knn_search(
        &self,
        index: &str,
        embedding: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<KnnResult>, AppError> {
        let url = format!("{base_url}/{index}/_search", base_url = self.base_url);

        let query = json!({
            "size": k,
            "query": {
                "knn": {
                    "global_embedding": {
                        "vector": embedding,
                        "k": k
                    }
                }
            },
            "_source": ["image_id", "image_path"],
            "min_score": threshold
        });

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&query)
            .send()
            .await
            .map_err(|e| AppError::OpenSearch(format!("kNN search failed: {e}")))?;

        let result: Value = response
            .json()
            .await
            .map_err(|e| AppError::OpenSearch(format!("Failed to parse kNN response: {e}")))?;

        let mut results = Vec::new();
        if let Some(hits) = result["hits"]["hits"].as_array() {
            for hit in hits {
                if let (Some(image_id), Some(score)) = (
                    hit["_source"]["image_id"].as_str(),
                    hit["_score"].as_f64(),
                ) {
                    results.push(KnnResult {
                        image_id: image_id.to_string(),
                        score: score as f32,
                    });
                }
            }
        }

        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct KnnResult {
    pub image_id: String,
    pub score: f32,
}
