//! The plan every photo of a row carries. The text is written by the API's own builder
//! (`scripts/bench/gpu_prep_parity.py plan`, which calls `src.domain.frame_engine_wire`), so the
//! rule that builds a plan exists once, in Python; this client sends the text byte for byte and
//! reads from it only what the models-alone mode needs to repeat the same model calls.

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

use crate::list::sha256_hex;

/// The plan's detector model.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Detector {
    /// The detector's model name on the model server.
    pub model: String,
    /// Side of the square detector input, in pixels.
    pub input_size: u32,
}

/// The plan's frame encoder.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Embed {
    /// The encoder's model name on the model server.
    pub model: String,
    /// Side of the square encoder input, in pixels.
    pub size: u32,
}

/// The plan's crop encoder.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CropEmbed {
    /// The encoder's model name on the model server.
    pub model: String,
    /// Side of the square encoder input, in pixels.
    pub size: u32,
    /// Most crops one model call carries.
    pub batch: u32,
}

#[derive(Debug, Deserialize)]
struct Fields {
    version: u32,
    branches: Vec<String>,
    detector: Option<Detector>,
    embed: Option<Embed>,
    crop_embed: Option<CropEmbed>,
}

/// A plan text and the parts of it the models-alone mode replays.
#[derive(Debug, Clone)]
pub struct PlanFile {
    /// The plan JSON exactly as sent with every photo, without the trailing newline.
    pub text: String,
    /// Lowercase hexadecimal SHA-256 of `text`.
    pub sha256: String,
    /// The plan's branch names, for example `detect` and `embed_frame`.
    pub branches: Vec<String>,
    /// Present when the plan has a detector.
    pub detector: Option<Detector>,
    /// Present when the plan embeds the whole frame.
    pub embed: Option<Embed>,
    /// Present when the plan embeds crops.
    pub crop_embed: Option<CropEmbed>,
}

impl PlanFile {
    /// Read the parts the models-alone mode needs from the plan JSON `text`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `text` is not the plan JSON or its `version` is not 1.
    pub fn parse(text: &str) -> Result<PlanFile, String> {
        let f: Fields = serde_json::from_str(text).map_err(|e| format!("plan JSON: {e}"))?;
        if f.version != 1 {
            return Err(format!(
                "plan version {} is not protocol version 1",
                f.version
            ));
        }
        Ok(PlanFile {
            text: text.to_string(),
            sha256: sha256_hex(text.as_bytes()),
            branches: f.branches,
            detector: f.detector,
            embed: f.embed,
            crop_embed: f.crop_embed,
        })
    }

    /// Read the plan at `path`, ignoring trailing newlines.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the file cannot be read or does not parse as a plan.
    pub fn load(path: &Path) -> io::Result<PlanFile> {
        let text = fs::read_to_string(path)?;
        PlanFile::parse(text.trim_end_matches('\n')).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })
    }

    /// Whether the plan embeds crops, so the models-alone mode needs each photo's crop count.
    pub fn runs_crops(&self) -> bool {
        self.branches.iter().any(|b| b == "embed_crops")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INGEST: &str = r#"{"version": 1, "branches": ["detect", "embed_frame", "embed_crops"], "crops": "inline", "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114}, "embed": {"model": "pe", "size": 336}, "crop_embed": {"model": "pe_crop", "size": 224, "batch": 32}}"#;

    #[test]
    fn the_replayed_parts_are_read_and_the_text_is_kept_verbatim() {
        let p = PlanFile::parse(INGEST).unwrap();
        assert_eq!(p.text, INGEST);
        assert_eq!(p.sha256, sha256_hex(INGEST.as_bytes()));
        assert_eq!(
            p.detector,
            Some(Detector {
                model: "det".into(),
                input_size: 640
            })
        );
        assert_eq!(
            p.embed,
            Some(Embed {
                model: "pe".into(),
                size: 336
            })
        );
        assert_eq!(
            p.crop_embed,
            Some(CropEmbed {
                model: "pe_crop".into(),
                size: 224,
                batch: 32
            })
        );
        assert!(p.runs_crops());
    }

    #[test]
    fn a_frame_plan_has_no_crops() {
        let p = PlanFile::parse(
            r#"{"version": 1, "branches": ["detect", "embed_frame"], "detector": {"model": "d", "input_size": 640}, "embed": {"model": "e", "size": 336}}"#,
        )
        .unwrap();
        assert!(!p.runs_crops());
        assert_eq!(p.crop_embed, None);
    }

    #[test]
    fn a_plan_of_another_version_or_not_json_is_refused() {
        assert!(PlanFile::parse(r#"{"version": 2, "branches": []}"#).is_err());
        assert!(PlanFile::parse("not json").is_err());
        assert!(PlanFile::parse(r#"{"branches": []}"#).is_err());
    }
}
