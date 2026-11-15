//! The request plan of protocol version 1 and its parser: a port of `parse_plan` in
//! `src/domain/frame_engine_wire.py`, answering the same fault with the same message for every
//! text (`contracts/json/frame_engine_plan_cases.json` holds the two to it, case by case). The
//! order of the checks is the Python order, because the first fault wins:
//! the document, the top-level fields, the version, the branches, then each settings object in
//! the order the Python constructor evaluates them, then the dependencies between them.
//!
//! The vocabularies (branch names, recipes, formats, the rendition side range, the OCR limits)
//! come from the generated contract; the numeric ranges of the other fields are written here as
//! in the Python module and held to it by the boundary cases of the generated plan cases.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Serialize;

use super::contract::contract;
use super::json::{Json, parse};

/// The fault names of the contract (`plan.faults`).
pub mod fault {
    pub const BAD_JSON: &str = "bad_json";
    pub const BAD_VERSION: &str = "bad_version";
    pub const UNKNOWN_FIELD: &str = "unknown_field";
    pub const MISSING_FIELD: &str = "missing_field";
    pub const BAD_FIELD: &str = "bad_field";
    pub const EMPTY_PLAN: &str = "empty_plan";
    pub const UNKNOWN_BRANCH: &str = "unknown_branch";
    pub const DUPLICATE_BRANCH: &str = "duplicate_branch";
    pub const MISSING_DEPENDENCY: &str = "missing_dependency";
    pub const CONDITION_ON_UNCONDITIONAL: &str = "condition_on_unconditional";
    pub const BOXES_WITH_DETECT: &str = "boxes_with_detect";
    pub const UNUSED_BOXES: &str = "unused_boxes";
}

/// A plan the engine refuses as a whole (gRPC `INVALID_ARGUMENT`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError {
    pub fault: &'static str,
    pub detail: String,
}

impl PlanError {
    fn new(fault: &'static str, detail: impl Into<String>) -> Self {
        Self {
            fault,
            detail: detail.into(),
        }
    }
}

/// `fault: detail`, or the fault alone: the status details the Python front sends.
impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.detail.is_empty() {
            f.write_str(self.fault)
        } else {
            write!(f, "{}: {}", self.fault, self.detail)
        }
    }
}

impl std::error::Error for PlanError {}

type Parsed<T> = Result<T, PlanError>;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DetectorSpec {
    pub model: String,
    pub input_size: u32,
    pub letterbox_fill: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbedSpec {
    pub model: String,
    pub size: u32,
    pub frame_recipe: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CropEmbedSpec {
    pub model: String,
    pub size: u32,
    pub batch: u32,
    pub crop_recipe: String,
    pub fill: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FaceSpec {
    pub detector: String,
    pub input_size: u32,
    pub recogniser: String,
    pub batch: u32,
    pub score_min: f64,
    pub nms: f64,
    pub max_faces: u32,
    pub coarse: u32,
    pub tile_side: u32,
    pub overlap: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OcrSpec {
    pub detector: String,
    pub recogniser: String,
    pub batch: u32,
    pub max_lines: u32,
    pub det_side: u32,
    pub det_limit: String,
    pub det_max_side: u32,
    pub det_antialias: bool,
    pub thresh: f64,
    pub box_thresh: f64,
    pub unclip_ratio: f64,
    pub taps: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RenditionSpec {
    pub name: String,
    pub side: u32,
    pub quality: u32,
    pub format: String,
}

/// `crops`: `inline` returns each detection's JPEG, `none` does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Crops {
    Inline,
    None,
}

/// A parsed plan; serialised, it is the `parsed` object of the plan cases (Python's
/// `EnginePlan.to_json`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnginePlan {
    pub version: u32,
    /// In the vocabulary's order, whatever order the text had.
    pub branches: Vec<String>,
    pub crops: Crops,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detector: Option<DetectorSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed: Option<EmbedSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop_embed: Option<CropEmbedSpec>,
    /// Detector class ids that get a crop vector; `None` means every detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_classes: Option<BTreeSet<u32>>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub if_detected: BTreeMap<String, Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub faces: Option<FaceSpec>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr: Option<OcrSpec>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub derivatives: Vec<RenditionSpec>,
}

impl EnginePlan {
    pub fn runs(&self, branch: &str) -> bool {
        self.branches.iter().any(|b| b == branch)
    }

    /// The names `branch_status` carries: the branches, then `derivatives` when there are
    /// renditions.
    pub fn reported(&self) -> Vec<String> {
        let mut names = self.branches.clone();
        if !self.derivatives.is_empty() {
            names.push(contract().derivatives.field.clone());
        }
        names
    }

    /// The branch tensors an `ok` answer carries besides the four always present, in the
    /// vocabulary's order (`expected_outputs`).
    pub fn expected_outputs(&self) -> Vec<&'static str> {
        let c = contract();
        let mut names: Vec<&'static str> = Vec::new();
        for branch in &self.branches {
            if let Some(outputs) = c.branch_outputs.get(branch) {
                names.extend(outputs.iter().map(String::as_str));
            }
        }
        if self.crops == Crops::Inline {
            names.extend(c.crops_inline_outputs.iter().map(String::as_str));
        }
        if !self.derivatives.is_empty() {
            names.push(c.derivatives.output.as_str());
        }
        names
    }
}

type Object<'a> = &'a [(String, Json)];

fn get<'a>(object: Object<'a>, key: &str) -> Option<&'a Json> {
    object.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn as_object<'a>(raw: &'a Json, name: &str, keys: &[&str]) -> Parsed<Object<'a>> {
    let Json::Object(pairs) = raw else {
        return Err(PlanError::new(fault::BAD_FIELD, name));
    };
    let mut extra: Vec<&str> = pairs
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| !keys.contains(k))
        .collect();
    extra.sort_unstable();
    match extra.first() {
        Some(first) => Err(PlanError::new(
            fault::UNKNOWN_FIELD,
            format!("{name}.{first}"),
        )),
        None => Ok(pairs),
    }
}

fn need<'a>(object: Object<'a>, key: &str, name: &str) -> Parsed<&'a Json> {
    get(object, key).ok_or_else(|| PlanError::new(fault::MISSING_FIELD, name))
}

fn bad(name: &str) -> PlanError {
    PlanError::new(fault::BAD_FIELD, name)
}

fn int(raw: &Json, name: &str, lo: i64, hi: i64) -> Parsed<u32> {
    raw.int_in(lo, hi)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| bad(name))
}

fn float(raw: &Json, name: &str, lo: f64, hi: f64) -> Parsed<f64> {
    raw.number_in(lo, hi).ok_or_else(|| bad(name))
}

fn string(raw: &Json, name: &str, choices: Option<&[String]>) -> Parsed<String> {
    match raw {
        Json::Str(s) if !s.is_empty() && choices.is_none_or(|c| c.contains(s)) => Ok(s.clone()),
        _ => Err(bad(name)),
    }
}

/// `object.get(key, default)`: the value, or `default` when the key is absent (a present `null`
/// is a value, and fails its check).
fn or<'a>(object: Object<'a>, key: &str, default: &'a Json) -> &'a Json {
    get(object, key).unwrap_or(default)
}

fn detector(raw: &Json) -> Parsed<DetectorSpec> {
    let d = as_object(raw, "detector", &["model", "input_size", "letterbox_fill"])?;
    Ok(DetectorSpec {
        model: string(need(d, "model", "detector.model")?, "detector.model", None)?,
        input_size: int(
            need(d, "input_size", "detector.input_size")?,
            "detector.input_size",
            32,
            4096,
        )?,
        letterbox_fill: int(
            need(d, "letterbox_fill", "detector.letterbox_fill")?,
            "detector.letterbox_fill",
            0,
            255,
        )?,
    })
}

fn embed(raw: &Json) -> Parsed<EmbedSpec> {
    let c = contract();
    let e = as_object(raw, "embed", &["model", "size", "frame_recipe"])?;
    let default = Json::Str(c.plan.frame_recipes[0].clone());
    Ok(EmbedSpec {
        model: string(need(e, "model", "embed.model")?, "embed.model", None)?,
        size: int(need(e, "size", "embed.size")?, "embed.size", 16, 4096)?,
        frame_recipe: string(
            or(e, "frame_recipe", &default),
            "embed.frame_recipe",
            Some(&c.plan.frame_recipes),
        )?,
    })
}

fn crop_embed(raw: &Json) -> Parsed<CropEmbedSpec> {
    let c = contract();
    let o = as_object(
        raw,
        "crop_embed",
        &["model", "size", "batch", "crop_recipe", "fill"],
    )?;
    let default_recipe = Json::Str(c.plan.crop_recipes[0].clone());
    let default_fill = Json::Int(c.plan.crop_fill_default.to_string());
    Ok(CropEmbedSpec {
        model: string(
            need(o, "model", "crop_embed.model")?,
            "crop_embed.model",
            None,
        )?,
        size: int(
            need(o, "size", "crop_embed.size")?,
            "crop_embed.size",
            16,
            4096,
        )?,
        batch: int(
            need(o, "batch", "crop_embed.batch")?,
            "crop_embed.batch",
            1,
            1024,
        )?,
        crop_recipe: string(
            or(o, "crop_recipe", &default_recipe),
            "crop_embed.crop_recipe",
            Some(&c.plan.crop_recipes),
        )?,
        fill: int(or(o, "fill", &default_fill), "crop_embed.fill", 0, 255)?,
    })
}

const FACE_FIELDS: [&str; 10] = [
    "detector",
    "input_size",
    "recogniser",
    "batch",
    "score_min",
    "nms",
    "max_faces",
    "coarse",
    "tile_side",
    "overlap",
];

fn faces(raw: &Json) -> Parsed<FaceSpec> {
    let f = as_object(raw, "faces", &FACE_FIELDS)?;
    let zero = Json::Int("0".into());
    let net = int(
        need(f, "input_size", "faces.input_size")?,
        "faces.input_size",
        32,
        4096,
    )?;
    if net % 32 != 0 {
        return Err(bad("faces.input_size"));
    }
    let coarse = int(or(f, "coarse", &zero), "faces.coarse", 0, i64::from(net))?;
    if coarse != 0 && coarse < 32 {
        return Err(bad("faces.coarse"));
    }
    Ok(FaceSpec {
        detector: string(
            need(f, "detector", "faces.detector")?,
            "faces.detector",
            None,
        )?,
        input_size: net,
        recogniser: string(
            need(f, "recogniser", "faces.recogniser")?,
            "faces.recogniser",
            None,
        )?,
        batch: int(need(f, "batch", "faces.batch")?, "faces.batch", 1, 1024)?,
        score_min: float(
            need(f, "score_min", "faces.score_min")?,
            "faces.score_min",
            0.0,
            1.0,
        )?,
        nms: float(need(f, "nms", "faces.nms")?, "faces.nms", 0.0, 1.0)?,
        max_faces: int(
            need(f, "max_faces", "faces.max_faces")?,
            "faces.max_faces",
            1,
            100_000,
        )?,
        coarse,
        tile_side: int(or(f, "tile_side", &zero), "faces.tile_side", 0, 65_535)?,
        overlap: int(
            or(f, "overlap", &zero),
            "faces.overlap",
            0,
            i64::from(net) - 1,
        )?,
    })
}

const OCR_FIELDS: [&str; 12] = [
    "detector",
    "recogniser",
    "batch",
    "max_lines",
    "det_side",
    "det_limit",
    "det_max_side",
    "det_antialias",
    "thresh",
    "box_thresh",
    "unclip_ratio",
    "taps",
];

fn ocr(raw: &Json) -> Parsed<OcrSpec> {
    let c = contract();
    let o = as_object(raw, "ocr", &OCR_FIELDS)?;
    let det_antialias = match get(o, "det_antialias") {
        None => false,
        Some(Json::Bool(b)) => *b,
        Some(_) => return Err(bad("ocr.det_antialias")),
    };
    let default_limit = Json::Str("max".into());
    let default_max_side = Json::Int(c.plan.ocr_max_det_side.to_string());
    let one = Json::Int("1".into());
    let thresh = Json::Float(0.2);
    let box_thresh = Json::Float(0.45);
    let unclip = Json::Float(1.4);
    Ok(OcrSpec {
        detector: string(need(o, "detector", "ocr.detector")?, "ocr.detector", None)?,
        recogniser: string(
            need(o, "recogniser", "ocr.recogniser")?,
            "ocr.recogniser",
            None,
        )?,
        batch: int(
            need(o, "batch", "ocr.batch")?,
            "ocr.batch",
            1,
            c.plan.ocr_max_batch,
        )?,
        max_lines: int(
            need(o, "max_lines", "ocr.max_lines")?,
            "ocr.max_lines",
            1,
            10_000,
        )?,
        det_side: int(
            need(o, "det_side", "ocr.det_side")?,
            "ocr.det_side",
            32,
            4096,
        )?,
        det_limit: string(
            or(o, "det_limit", &default_limit),
            "ocr.det_limit",
            Some(&c.plan.ocr_det_limits),
        )?,
        det_max_side: int(
            or(o, "det_max_side", &default_max_side),
            "ocr.det_max_side",
            32,
            c.plan.ocr_max_det_side,
        )?,
        det_antialias,
        thresh: float(or(o, "thresh", &thresh), "ocr.thresh", 0.0, 1.0)?,
        box_thresh: float(or(o, "box_thresh", &box_thresh), "ocr.box_thresh", 0.0, 1.0)?,
        unclip_ratio: float(
            or(o, "unclip_ratio", &unclip),
            "ocr.unclip_ratio",
            0.0,
            10.0,
        )?,
        taps: int(or(o, "taps", &one), "ocr.taps", 1, 8)?,
    })
}

fn class_ids(raw: &Json, name: &str) -> Parsed<Vec<u32>> {
    let Json::List(items) = raw else {
        return Err(bad(name));
    };
    items.iter().map(|v| int(v, name, 0, 1_000_000)).collect()
}

fn renditions(items: &[Json]) -> Parsed<Vec<RenditionSpec>> {
    let c = contract();
    let field = &c.derivatives.field;
    let [side_lo, side_hi] = c.derivatives.side_range;
    let default_quality = Json::Int(c.derivatives.default_quality.to_string());
    let default_format = Json::Str(c.derivatives.formats[0].clone());
    let mut specs: Vec<RenditionSpec> = Vec::new();
    for (i, entry) in items.iter().enumerate() {
        let at = format!("{field}[{i}]");
        let r = as_object(entry, &at, &["name", "side", "quality", "format"])?;
        let name = string(
            need(r, "name", &format!("{at}.name"))?,
            &format!("{at}.name"),
            None,
        )?;
        if specs.iter().any(|s| s.name == name) {
            return Err(bad(&format!("{at}.name")));
        }
        specs.push(RenditionSpec {
            name,
            side: int(
                need(r, "side", &format!("{at}.side"))?,
                &format!("{at}.side"),
                side_lo,
                side_hi,
            )?,
            quality: int(
                or(r, "quality", &default_quality),
                &format!("{at}.quality"),
                1,
                100,
            )?,
            format: string(
                or(r, "format", &default_format),
                &format!("{at}.format"),
                Some(&c.derivatives.formats),
            )?,
        });
    }
    Ok(specs)
}

fn branches(raw: &Json) -> Parsed<Vec<String>> {
    let vocabulary = &contract().branches;
    let Json::List(items) = raw else {
        return Err(bad("branches"));
    };
    if items.is_empty() {
        return Err(PlanError::new(fault::EMPTY_PLAN, ""));
    }
    let mut names: Vec<&str> = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Json::Str(s) if vocabulary.contains(s) => names.push(s),
            other => return Err(PlanError::new(fault::UNKNOWN_BRANCH, other.text())),
        }
    }
    for name in &names {
        if names.iter().filter(|n| *n == name).count() > 1 {
            return Err(PlanError::new(fault::DUPLICATE_BRANCH, *name));
        }
    }
    Ok(vocabulary
        .iter()
        .filter(|b| names.contains(&b.as_str()))
        .cloned()
        .collect())
}

/// A branch's settings object comes with the branch and only with it.
fn check_settings(plan: &EnginePlan, branch: &str, present: bool, name: &str) -> Parsed<()> {
    match (plan.runs(branch), present) {
        (true, false) => Err(PlanError::new(fault::MISSING_FIELD, name)),
        (false, true) => Err(bad(name)),
        _ => Ok(()),
    }
}

fn check_dependencies(plan: &EnginePlan, has_boxes: bool) -> Parsed<()> {
    let runs = |b: &str| plan.runs(b);
    let inline = plan.crops == Crops::Inline;
    if has_boxes && runs("detect") {
        return Err(PlanError::new(fault::BOXES_WITH_DETECT, ""));
    }
    if has_boxes && !(runs("embed_crops") || inline) {
        return Err(PlanError::new(fault::UNUSED_BOXES, ""));
    }
    let rects = runs("detect") || has_boxes;
    if (runs("embed_crops") || inline) && !rects {
        return Err(PlanError::new(fault::MISSING_DEPENDENCY, "detect"));
    }
    if runs("face_embed") && !runs("faces") {
        return Err(PlanError::new(fault::MISSING_DEPENDENCY, "faces"));
    }
    check_settings(plan, "faces", plan.faces.is_some(), "faces")?;
    check_settings(plan, "ocr", plan.ocr.is_some(), "ocr")?;
    if runs("detect") && plan.detector.is_none() {
        return Err(PlanError::new(fault::MISSING_FIELD, "detector"));
    }
    check_settings(plan, "embed_frame", plan.embed.is_some(), "embed")?;
    check_settings(plan, "embed_crops", plan.crop_embed.is_some(), "crop_embed")?;
    if plan.embed_classes.is_some() && (!runs("embed_crops") || has_boxes) {
        return Err(bad("embed_classes"));
    }
    for name in plan.if_detected.keys() {
        if !contract().plan.conditional_branches.contains(name) || !runs(name) {
            return Err(PlanError::new(fault::CONDITION_ON_UNCONDITIONAL, name));
        }
        if !runs("detect") {
            return Err(PlanError::new(fault::MISSING_DEPENDENCY, "detect"));
        }
    }
    Ok(())
}

const TOP_FIELDS: [&str; 11] = [
    "version",
    "branches",
    "detector",
    "embed",
    "crop_embed",
    "embed_classes",
    "if_detected",
    "faces",
    "ocr",
    "crops",
    "derivatives",
];

/// The plan in `text` exactly as written, or the fault. `has_boxes` says whether the request
/// carries the `boxes` input: stored boxes stand in for the detector and so exclude `detect`.
pub fn parse_plan(text: &[u8], has_boxes: bool) -> Parsed<EnginePlan> {
    let c = contract();
    let document = parse(text).map_err(|e| PlanError::new(fault::BAD_JSON, e.to_string()))?;
    let body = as_object(&document, "plan", &TOP_FIELDS)?;
    let version = get(body, "version");
    if version
        .and_then(|v| v.int_in(c.model.protocol_version, c.model.protocol_version))
        .is_none()
    {
        let shown = version.map_or_else(|| Json::Null.repr(), Json::repr);
        return Err(PlanError::new(fault::BAD_VERSION, shown));
    }
    let branch_list = branches(need(body, "branches", "branches")?)?;
    let default_crops = Json::Str("none".into());
    let crops = string(
        or(body, "crops", &default_crops),
        "crops",
        Some(&c.plan.crops_modes),
    )?;
    let raw_classes = get(body, "embed_classes").filter(|v| **v != Json::Null);
    let empty_object = Json::Object(Vec::new());
    let Json::Object(conditions) = or(body, "if_detected", &empty_object) else {
        return Err(bad("if_detected"));
    };
    let empty_list = Json::List(Vec::new());
    let Json::List(rendition_list) = or(body, &c.derivatives.field, &empty_list) else {
        return Err(bad(&c.derivatives.field));
    };
    let plan = EnginePlan {
        version: u32::try_from(c.model.protocol_version).unwrap_or(1),
        branches: branch_list,
        crops: if crops == "inline" {
            Crops::Inline
        } else {
            Crops::None
        },
        detector: get(body, "detector").map(detector).transpose()?,
        embed: get(body, "embed").map(embed).transpose()?,
        crop_embed: get(body, "crop_embed").map(crop_embed).transpose()?,
        embed_classes: raw_classes
            .map(|raw| class_ids(raw, "embed_classes").map(|ids| ids.into_iter().collect()))
            .transpose()?,
        if_detected: conditions
            .iter()
            .map(|(k, v)| class_ids(v, &format!("if_detected.{k}")).map(|ids| (k.clone(), ids)))
            .collect::<Parsed<_>>()?,
        faces: get(body, "faces").map(faces).transpose()?,
        ocr: get(body, "ocr").map(ocr).transpose()?,
        derivatives: renditions(rendition_list)?,
    };
    check_dependencies(&plan, has_boxes)?;
    Ok(plan)
}
