//! What the models-alone mode calls for one photo: the same model calls the engine makes for the
//! plan, as pure data. Round 1 is the detector and the frame encoder at once; round 2 is the crop
//! encoder in chunks of the plan's `batch`, all in flight together (as the engine submits them,
//! `frame_engine/branches/embed.py`), one chunk per `batch` crops of the photo.
//!
//! The tensor names are the models' configs' (`models/*/config.pbtxt`): input `images` (UINT8,
//! `[batch, 3, side, side]`), output `output0` for the detector and `image_embeddings` for the
//! encoders.

use serde::Deserialize;

use crate::plan::PlanFile;

/// One model call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSpec {
    /// The model's name on the model server.
    pub model: String,
    /// The output tensor's name.
    pub output: &'static str,
    /// `[batch, 3, side, side]`.
    pub shape: [i64; 4],
}

impl CallSpec {
    /// The input's size in bytes (UINT8).
    pub fn bytes(&self) -> usize {
        self.shape
            .iter()
            .map(|d| usize::try_from(*d).unwrap_or(0))
            .product()
    }
}

/// The calls of one photo, by round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhotoCalls {
    /// The detector and frame-encoder calls, all in flight together.
    pub round1: Vec<CallSpec>,
    /// The crop-encoder calls, one per chunk of the photo's crops, all in flight together after round 1.
    pub round2: Vec<CallSpec>,
}

fn call(model: &str, output: &'static str, batch: usize, side: u32) -> CallSpec {
    CallSpec {
        model: model.to_string(),
        output,
        shape: [
            i64::try_from(batch).unwrap_or(0),
            3,
            i64::from(side),
            i64::from(side),
        ],
    }
}

/// The chunk sizes of `crops` crops in chunks of `batch`: `[batch, batch, ..., rest]`.
pub fn chunks(crops: usize, batch: usize) -> Vec<usize> {
    if batch == 0 {
        return Vec::new();
    }
    (0..crops.div_ceil(batch))
        .map(|k| batch.min(crops - k * batch))
        .collect()
}

/// The calls for a photo that has `crops` crops to embed (0 for a plan without crops).
///
/// # Errors
///
/// Returns `Err` when the plan names a branch without its model, or calls no model in round 1.
pub fn photo_calls(plan: &PlanFile, crops: usize) -> Result<PhotoCalls, String> {
    let mut round1 = Vec::new();
    if plan.branches.iter().any(|b| b == "detect") {
        let d = plan
            .detector
            .as_ref()
            .ok_or("the plan detects but names no detector")?;
        round1.push(call(&d.model, "output0", 1, d.input_size));
    }
    if plan.branches.iter().any(|b| b == "embed_frame") {
        let e = plan
            .embed
            .as_ref()
            .ok_or("the plan embeds the frame but names no encoder")?;
        round1.push(call(&e.model, "image_embeddings", 1, e.size));
    }
    let mut round2 = Vec::new();
    if plan.runs_crops() {
        let c = plan
            .crop_embed
            .as_ref()
            .ok_or("the plan embeds crops but names no crop encoder")?;
        let batch = c.batch as usize;
        round2 = chunks(crops, batch)
            .into_iter()
            .map(|n| call(&c.model, "image_embeddings", n, c.size))
            .collect();
    }
    if round1.is_empty() {
        return Err("the plan calls no model in round 1".into());
    }
    Ok(PhotoCalls { round1, round2 })
}

/// The crop counts a parity run recorded for one list, in list order.
#[derive(Debug, Deserialize)]
pub struct CropCounts {
    /// SHA-256 of the list file the counts were recorded for.
    pub list_sha256: String,
    /// Crops of each photo, in list order.
    pub crops: Vec<usize>,
}

impl CropCounts {
    /// Parse and check against the list that is about to be replayed: a count file recorded for
    /// another list would be a silent error, so it is refused.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the text is not a crop-count file, was recorded for another list, or holds a different number of photos than `photos`.
    pub fn parse(text: &str, list_sha256: &str, photos: usize) -> Result<CropCounts, String> {
        let counts: CropCounts =
            serde_json::from_str(text).map_err(|e| format!("crop counts: {e}"))?;
        if counts.list_sha256 != list_sha256 {
            return Err(format!(
                "crop counts were recorded for list {} but the list is {}",
                counts.list_sha256, list_sha256
            ));
        }
        if counts.crops.len() != photos {
            return Err(format!(
                "crop counts hold {} photos, the list {photos}",
                counts.crops.len()
            ));
        }
        Ok(counts)
    }
}

/// Region bytes for `in_flight` photos holding every call's input at once, with each slice
/// rounded up to the slab's alignment and a quarter more for first-fit fragmentation.
pub fn region_bytes(per_photo_max: &[usize], align: usize, in_flight: usize) -> usize {
    let one: usize = per_photo_max
        .iter()
        .map(|b| b.div_ceil(align) * align)
        .sum();
    let total = one.saturating_mul(in_flight);
    (total + total / 4).div_ceil(align).max(1) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(text: &str) -> PlanFile {
        PlanFile::parse(text).unwrap()
    }

    const FRAME: &str = r#"{"version": 1, "branches": ["detect", "embed_frame"], "detector": {"model": "det", "input_size": 640}, "embed": {"model": "pe", "size": 336}}"#;
    const INGEST: &str = r#"{"version": 1, "branches": ["detect", "embed_frame", "embed_crops"], "detector": {"model": "det", "input_size": 640}, "embed": {"model": "pe", "size": 336}, "crop_embed": {"model": "pe_crop", "size": 224, "batch": 32}}"#;

    #[test]
    fn chunks_fill_the_batch_and_leave_the_rest_last() {
        assert_eq!(chunks(0, 32), Vec::<usize>::new());
        assert_eq!(chunks(5, 32), [5]);
        assert_eq!(chunks(32, 32), [32]);
        assert_eq!(chunks(33, 32), [32, 1]);
        assert_eq!(chunks(70, 32), [32, 32, 6]);
    }

    #[test]
    fn the_frame_plan_calls_the_detector_and_the_frame_encoder_once() {
        let c = photo_calls(&plan(FRAME), 7).unwrap();
        assert_eq!(c.round1.len(), 2);
        assert_eq!(
            c.round1[0],
            CallSpec {
                model: "det".into(),
                output: "output0",
                shape: [1, 3, 640, 640]
            }
        );
        assert_eq!(c.round1[1].shape, [1, 3, 336, 336]);
        assert!(
            c.round2.is_empty(),
            "no crops in plan F, whatever the count"
        );
    }

    #[test]
    fn the_ingest_plan_adds_one_crop_call_per_chunk() {
        let c = photo_calls(&plan(INGEST), 70).unwrap();
        let shapes: Vec<_> = c.round2.iter().map(|s| s.shape[0]).collect();
        assert_eq!(shapes, [32, 32, 6]);
        assert!(
            c.round2
                .iter()
                .all(|s| s.model == "pe_crop" && s.shape[2] == 224)
        );
        assert_eq!(c.round2[2].bytes(), 6 * 3 * 224 * 224);
        assert!(photo_calls(&plan(INGEST), 0).unwrap().round2.is_empty());
    }

    #[test]
    fn a_plan_without_the_models_it_names_is_refused() {
        let bad = plan(r#"{"version": 1, "branches": ["detect"]}"#);
        assert!(photo_calls(&bad, 0).is_err());
        let none = plan(r#"{"version": 1, "branches": ["ocr"]}"#);
        assert!(photo_calls(&none, 0).is_err());
    }

    #[test]
    fn counts_recorded_for_another_list_are_refused() {
        let ok =
            CropCounts::parse(r#"{"list_sha256": "aa", "crops": [1, 2, 3]}"#, "aa", 3).unwrap();
        assert_eq!(ok.crops, [1, 2, 3]);
        assert!(
            CropCounts::parse(r#"{"list_sha256": "bb", "crops": [1, 2, 3]}"#, "aa", 3).is_err()
        );
        assert!(CropCounts::parse(r#"{"list_sha256": "aa", "crops": [1, 2]}"#, "aa", 3).is_err());
        assert!(CropCounts::parse("[]", "aa", 0).is_err());
    }

    #[test]
    fn the_region_holds_every_photos_inputs_at_once_with_room_to_spare() {
        // 3 slices of 300, 600 and 1 bytes, align 256: 512 + 768 + 256 = 1536 per photo
        let bytes = region_bytes(&[300, 600, 1], 256, 4);
        assert_eq!(bytes % 256, 0);
        assert!(bytes >= 4 * 1536 * 5 / 4, "{bytes}");
        assert!(bytes < 4 * 1536 * 2);
    }
}
