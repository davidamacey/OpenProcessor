//! The wire types of the route's answer, written by hand against `contracts/openapi/curation.json`
//! (`IngestImageResponse`, `BatchIngestSummaryResponse`, `CurationBatchIngestResponse`; a test
//! compares the property names, defaults and enums with the committed file), and the merge of the
//! photos refused before the pipeline with the pipeline's result (`_batch_response` in
//! `src/routers/curation/ingest.py`).

use serde::{Deserialize, Serialize};

use super::form::{ERROR_KIND_EMPTY, ERROR_KIND_UNSUPPORTED_TYPE};

/// How one photo ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageStatus {
    /// Ingested.
    Success,
    /// Already in the project.
    Duplicate,
    /// Refused or failed.
    Failed,
}

/// The result for one photo (`IngestImageResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestImageResponse {
    /// How the photo ended.
    pub status: ImageStatus,
    /// The photo's record id; empty when none was made.
    #[serde(default)]
    pub image_id: String,
    /// The server-side path of the stored photo; for a photo refused before the pipeline, the
    /// client's identifier.
    pub image_path: String,
    /// The duplicate probe hash; empty when none was made.
    #[serde(default)]
    pub imohash: String,
    /// Crops stored.
    #[serde(default)]
    pub n_crops: u64,
    /// Regions queued.
    #[serde(default)]
    pub n_regions: u64,
    /// Crops stored with a vector.
    #[serde(default)]
    pub n_embedded: u64,
    /// Crops stored without a vector.
    #[serde(default)]
    pub n_not_embedded: u64,
    /// Of those, the ones whose encoder failed.
    #[serde(default)]
    pub n_embed_failed: u64,
    /// Detections the project's filter dropped.
    #[serde(default)]
    pub n_filtered: u64,
    /// The failure, when the photo failed.
    #[serde(default)]
    pub error: Option<String>,
    /// The stable machine code of the failure.
    #[serde(default)]
    pub error_kind: Option<String>,
    /// The client's identifier for the photo.
    #[serde(default)]
    pub source_identifier: Option<String>,
    /// The secondary detector's failure, when it failed and was skipped.
    #[serde(default)]
    pub secondary_detector_error: Option<String>,
}

impl IngestImageResponse {
    /// A photo refused before the pipeline: `image_path` and `source_identifier` are the
    /// identifier the client sent.
    #[must_use]
    pub fn early_failure(identifier: &str, error: String, error_kind: &str) -> Self {
        Self {
            status: ImageStatus::Failed,
            image_id: String::new(),
            image_path: identifier.to_string(),
            imohash: String::new(),
            n_crops: 0,
            n_regions: 0,
            n_embedded: 0,
            n_not_embedded: 0,
            n_embed_failed: 0,
            n_filtered: 0,
            error: Some(error),
            error_kind: Some(error_kind.to_string()),
            source_identifier: Some(identifier.to_string()),
            secondary_detector_error: None,
        }
    }

    /// An empty photo (`empty upload`).
    #[must_use]
    pub fn empty_upload(identifier: &str) -> Self {
        Self::early_failure(identifier, "empty upload".into(), ERROR_KIND_EMPTY)
    }

    /// A photo whose extension is not accepted.
    #[must_use]
    pub fn unsupported(identifier: &str, message: String) -> Self {
        Self::early_failure(identifier, message, ERROR_KIND_UNSUPPORTED_TYPE)
    }
}

/// The totals of one batch (`BatchIngestSummaryResponse`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSummary {
    /// Photos ingested.
    #[serde(default)]
    pub successful: u64,
    /// Photos already in the project.
    #[serde(default)]
    pub duplicates: u64,
    /// Photos refused or failed.
    #[serde(default)]
    pub failed: u64,
    /// Crops stored.
    #[serde(default)]
    pub crops_indexed: u64,
    /// Photos where the secondary detector failed and was skipped.
    #[serde(default)]
    pub secondary_detector_failures: u64,
    /// Crops stored with a vector.
    #[serde(default)]
    pub n_embedded: u64,
    /// Crops stored without a vector.
    #[serde(default)]
    pub n_not_embedded: u64,
    /// Of those, the ones whose encoder failed.
    #[serde(default)]
    pub n_embed_failed: u64,
    /// Detections the project's filter dropped.
    #[serde(default)]
    pub n_filtered: u64,
}

/// How the whole request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BatchStatus {
    /// Nothing failed.
    Success,
    /// Something failed and something succeeded.
    Partial,
    /// Everything failed.
    Error,
}

/// The route's 200 answer (`CurationBatchIngestResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchResponse {
    /// How the request ended.
    pub status: BatchStatus,
    /// The totals.
    pub summary: BatchSummary,
    /// One result per photo: the refused ones first, then the pipeline's in order.
    pub results: Vec<IngestImageResponse>,
}

/// What the sink returns for the photos it took: their results and the batch totals.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SinkBatch {
    /// One result per photo handed over, in hand-over order.
    pub results: Vec<IngestImageResponse>,
    /// The totals of those photos.
    pub summary: BatchSummary,
}

/// The answer for the refused photos `early` and the sink's `batch` (`None` when no photo was
/// handed over).
#[must_use]
pub fn merge(early: Vec<IngestImageResponse>, batch: Option<SinkBatch>) -> BatchResponse {
    let mut summary = BatchSummary {
        failed: early.len() as u64,
        ..BatchSummary::default()
    };
    let mut results = early;
    if let Some(batch) = batch {
        let b = batch.summary;
        summary.successful += b.successful;
        summary.duplicates += b.duplicates;
        summary.failed += b.failed;
        summary.crops_indexed += b.crops_indexed;
        summary.secondary_detector_failures += b.secondary_detector_failures;
        summary.n_embedded += b.n_embedded;
        summary.n_not_embedded += b.n_not_embedded;
        summary.n_embed_failed += b.n_embed_failed;
        summary.n_filtered += b.n_filtered;
        results.extend(batch.results);
    }
    let status = if summary.failed == 0 {
        BatchStatus::Success
    } else if summary.successful == 0 {
        BatchStatus::Error
    } else {
        BatchStatus::Partial
    };
    BatchResponse {
        status,
        summary,
        results,
    }
}
