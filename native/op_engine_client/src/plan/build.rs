//! The two plans the Python server builds (`ingest_gpu_decode.ingest_engine_plan`,
//! `reprocess_gpu_embed.embed_on_gpu`) and the encoder names they read from the settings.

use std::collections::BTreeSet;

use op_config::{Env, EnvError};

use super::{Branch, CropEmbedSpec, CropsMode, DetectorSpec, EmbedSpec, EnginePlan, FrameRecipe};

/// Squares per encoder call inside the engine (`PE_BATCH`, the crop encoder's maximum batch).
const PE_BATCH: u32 = 32;
/// The frame encoder's input square (`PE_SIZE`).
const PE_FRAME_SIZE: u32 = 336;
const CROP_SIZES: [u32; 2] = [224, 336];
const DEFAULT_CROP_SIZE: u32 = 224;

/// The encoders every plan names: the model server's names and the crop encoder's size, read
/// from `PE_IMAGE_MODEL`, `PE_CROP_MODEL` and `PE_CROP_SIZE` as `src/config/settings.py` does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderModels {
    /// The whole-frame encoder's model name.
    pub image_model: String,
    /// The crop encoder's model name.
    pub crop_model: String,
    /// The crop encoder's input square: 224 or 336, each its own vector space.
    pub crop_size: u32,
}

impl EncoderModels {
    /// The deployment's encoders.
    ///
    /// # Errors
    ///
    /// [`EnvError::Invalid`] naming the key when `PE_CROP_SIZE` is not 224 or 336 (a project's
    /// vectors belong to one size, so a typo refuses start-up rather than guessing) or a model
    /// name is empty.
    pub fn from_env(env: &Env) -> Result<Self, EnvError> {
        let model = |key: &str, default: &str| -> Result<String, EnvError> {
            let value: String = env.get(key, default.to_owned())?;
            if value.trim().is_empty() {
                return Err(EnvError::Invalid {
                    key: key.into(),
                    value,
                    reason: "a model name is not empty".into(),
                });
            }
            Ok(value)
        };
        let crop_size: u32 = env.get("PE_CROP_SIZE", DEFAULT_CROP_SIZE)?;
        if !CROP_SIZES.contains(&crop_size) {
            return Err(EnvError::Invalid {
                key: "PE_CROP_SIZE".into(),
                value: crop_size.to_string(),
                reason: format!("one of {CROP_SIZES:?}"),
            });
        }
        Ok(Self {
            image_model: model("PE_IMAGE_MODEL", "pe_image_encoder")?,
            crop_model: model("PE_CROP_MODEL", "pe_crop_encoder")?,
            crop_size,
        })
    }

    fn embed(&self) -> EmbedSpec {
        EmbedSpec {
            model: self.image_model.clone(),
            size: PE_FRAME_SIZE,
            frame_recipe: FrameRecipe::WholeFrame,
        }
    }

    fn crop_embed(&self) -> CropEmbedSpec {
        CropEmbedSpec::new(self.crop_model.clone(), self.crop_size, PE_BATCH)
    }
}

/// Which boxes of an ingested photo get a crop vector at ingest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CropVectors {
    /// None: the policy defers them (`lazy`, `none`).
    Deferred,
    /// Every box.
    Every,
    /// The boxes of these detector class ids; the ids are the engine's addressing, the policy
    /// names classes by name and the caller maps names to ids.
    Classes(BTreeSet<u32>),
}

/// What decides the plan every photo of an ingest carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestPlan {
    /// The project's detector.
    pub detector: DetectorSpec,
    /// Which boxes get a crop vector.
    pub crop_vectors: CropVectors,
}

impl EnginePlan {
    /// The one plan every photo of an ingest carries: detect, the whole-frame vector, the crop
    /// JPEGs, and the crop vectors unless the policy defers them.
    #[must_use]
    pub fn ingest(spec: &IngestPlan, encoders: &EncoderModels) -> Self {
        let mut branches = BTreeSet::from([Branch::Detect, Branch::EmbedFrame]);
        let mut plan = Self {
            detector: Some(spec.detector.clone()),
            embed: Some(encoders.embed()),
            crops: CropsMode::Inline,
            ..Self::default()
        };
        match &spec.crop_vectors {
            CropVectors::Deferred => {}
            CropVectors::Every | CropVectors::Classes(_) => {
                branches.insert(Branch::EmbedCrops);
                plan.crop_embed = Some(encoders.crop_embed());
                if let CropVectors::Classes(ids) = &spec.crop_vectors {
                    plan.embed_classes = Some(ids.clone());
                }
            }
        }
        plan.branches = branches;
        plan
    }

    /// The plan of a photo whose boxes are already stored: the frame vector and one crop vector
    /// per stored box, no detector (the `boxes` input stands in for it).
    #[must_use]
    pub fn stored_boxes_embed(encoders: &EncoderModels) -> Self {
        Self {
            branches: BTreeSet::from([Branch::EmbedFrame, Branch::EmbedCrops]),
            embed: Some(encoders.embed()),
            crop_embed: Some(encoders.crop_embed()),
            ..Self::default()
        }
    }
}
