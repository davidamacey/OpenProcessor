//! A pipeline that needs no CUDA and no model server: it answers every branch of the plan with
//! deterministic tensors of the right shapes and types. It is the subject of the contract test,
//! which drives the release binary (`--pipeline stub`) with the API's own client.
//!
//! The photo steers it. An image whose bytes start with `stub:` is a list of `key=value` words
//! separated by commas; any other image is a plain photo and gets the plain answer.
//!
//! | word | effect |
//! |---|---|
//! | `fail=overload`, `unavailable`, `context_lost`, `internal` | the photo fails that way |
//! | `hold_ms=N` | the photo takes N ms, in the stage named by `stage=` (default `detect`) |
//! | `stage=NAME` | the stage the held photo is in (a name of the contract's stage list) |
//! | `status=NAME` | the photo is not decoded: that status, route `none`, every branch `image_failed` |
//!
//! Readiness follows a file: while the file named by `--stub-not-ready-file` exists the pipeline
//! is not ready. The health tick reads it, as it probes the model server in the real pipeline.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::Notify;

use crate::admission::Overload;
use crate::branches::detect::{self, Detections};
use crate::geometry::{crop_rect, letterbox_geometry, undo_letterbox};
use crate::pipeline::{Photo, PhotoAnswer, Pipeline, PipelineError, Tensor};
use crate::wire::EnginePlan;
use crate::wire::branch_status::{BranchState, IMAGE_FAILED};
use crate::wire::contract::contract;
use crate::wire::plan::DetectorSpec;

const PREFIX: &[u8] = b"stub:";
/// The number of detections the stub finds, and of faces, text lines and vectors it makes.
const DETECTIONS: usize = 2;
/// The stub's photo is this many pixels wide and high (the `size` answer).
const FRAME: (u32, u32) = (96, 64);
/// Width of the frame and crop vectors: the width of the API's `pe_embedding` index field
/// (`encoder_embedding_dim` in `src/config/curation.py`). The real pipeline takes it from the
/// encoder's answer; the stub has no encoder, so it must say the width the API stores.
const ENCODER_DIM: usize = 1024;
/// Width of the face vectors: the API's `embedding` field (`embedding_dim`, same file).
const FACE_DIM: usize = 512;

#[derive(Debug, Default, PartialEq, Eq)]
struct Script {
    fail: Option<String>,
    hold: Option<Duration>,
    stage: Option<&'static str>,
    status: Option<String>,
}

fn script(image: &[u8]) -> Result<Script, PipelineError> {
    let mut out = Script::default();
    let Some(words) = image.strip_prefix(PREFIX) else {
        return Ok(out);
    };
    let bad = |what: String| PipelineError::Engine(format!("stub photo: {what}"));
    let text = std::str::from_utf8(words).map_err(|e| bad(e.to_string()))?;
    for word in text.split(',').filter(|w| !w.is_empty()) {
        let (key, value) = word
            .split_once('=')
            .ok_or_else(|| bad(format!("{word:?} is no key=value")))?;
        match key {
            "fail" => out.fail = Some(value.to_owned()),
            "hold_ms" => {
                let ms: u64 = value
                    .parse()
                    .map_err(|_| bad(format!("hold_ms={value:?}")))?;
                out.hold = Some(Duration::from_millis(ms));
            }
            "stage" => {
                let stage = contract().stages.iter().find(|s| s.as_str() == value);
                out.stage = Some(
                    stage
                        .map(String::as_str)
                        .ok_or_else(|| bad(format!("stage={value:?}")))?,
                );
            }
            "status" => out.status = Some(value.to_owned()),
            other => return Err(bad(format!("unknown word {other:?}"))),
        }
    }
    Ok(out)
}

fn fp32(shape: &[i64], value: f32) -> Tensor {
    let n: i64 = shape.iter().product();
    Tensor::Fp32 {
        shape: shape.to_vec(),
        data: vec![value; usize::try_from(n).unwrap_or(0)],
    }
}

/// `rows` unit-length vectors of width `dim`, each a different direction, so that a search over
/// them has something to rank and the API's L2 normalisation divides by a non-zero norm.
fn vectors(rows: usize, dim: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|r| (0..dim).map(move |c| if (c + r) % 2 == 0 { 1.0 } else { -1.0 }))
        .collect()
}

fn embeddings(rows: usize, dim: usize) -> Tensor {
    Tensor::Fp32 {
        shape: vec![rows as i64, dim as i64],
        data: vectors(rows, dim),
    }
}

/// The two detections of a frame: boxes in letterboxed input pixels inside the canvas, as the
/// detector's `settle` keeps them, with the rectangles the blur sums read.
fn detections(spec: &DetectorSpec) -> Result<Detections, PipelineError> {
    let (w, h) = (f64::from(FRAME.0), f64::from(FRAME.1));
    let lb = letterbox_geometry(
        i64::from(FRAME.0),
        i64::from(FRAME.1),
        i64::from(spec.input_size),
    )
    .map_err(|e| PipelineError::Engine(format!("stub letterbox: {e}")))?;
    let mut d = Detections {
        boxes: Vec::new(),
        scores: vec![0.9; DETECTIONS],
        classes: vec![1, 2],
        rects: Vec::new(),
        letterbox: lb,
    };
    for [x1, y1, x2, y2] in [[0.1, 0.1, 0.5, 0.6], [0.5, 0.3, 0.9, 0.9]] {
        let canvas = [
            x1 * w * lb.scale + lb.pad_w,
            y1 * h * lb.scale + lb.pad_h,
            x2 * w * lb.scale + lb.pad_w,
            y2 * h * lb.scale + lb.pad_h,
        ];
        d.rects.push(crop_rect(
            undo_letterbox(canvas, lb.scale, (lb.pad_w, lb.pad_h)),
            i64::from(FRAME.0),
            i64::from(FRAME.1),
        ));
        d.boxes.push(canvas.map(|v| v as f32));
    }
    Ok(d)
}

/// Laplacian sums (sum, sum of squares) of a rectangle of `pixels`: a mean response of 8 and a
/// variance of 36, a plausible sharp photo, so `(n * squares - sum^2) / n^2` is positive.
fn sums_of(pixels: i64) -> [i64; 2] {
    [8 * pixels, 100 * pixels]
}

fn detect_tensor(name: &str, d: &Detections) -> Option<Tensor> {
    if let Some((_, t)) = detect::outputs(d).into_iter().find(|(n, _)| *n == name) {
        return Some(t);
    }
    if name != "blur_sums" {
        return None;
    }
    let (rects, clamped) = detect::blur_rects(d, FRAME.0, FRAME.1);
    let sums: Vec<[i64; 2]> = rects
        .iter()
        .map(|[x1, y1, x2, y2]| sums_of((x2 - x1) * (y2 - y1)))
        .collect();
    detect::blur_sums(FRAME.0, FRAME.1, &clamped, &sums).ok()
}

/// The two faces, in frame pixels inside the frame, five landmarks inside each box, and the
/// `[pixels, sum, squares]` rows of their rectangles.
fn face_tensor(name: &str) -> Option<Tensor> {
    let boxes = [[8.0_f32, 8.0, 40.0, 40.0], [48.0, 16.0, 88.0, 56.0]];
    let d = DETECTIONS as i64;
    Some(match name {
        "face_boxes" => Tensor::Fp32 {
            shape: vec![d, 4],
            data: boxes.concat(),
        },
        "face_landmarks" => Tensor::Fp32 {
            shape: vec![d, 10],
            data: boxes
                .iter()
                .flat_map(|[x1, y1, x2, y2]| {
                    let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
                    [
                        cx - 4.0,
                        cy - 4.0,
                        cx + 4.0,
                        cy - 4.0,
                        cx,
                        cy,
                        cx - 3.0,
                        cy + 6.0,
                        cx + 3.0,
                        cy + 6.0,
                    ]
                })
                .collect(),
        },
        "face_blur_sums" => Tensor::Int64 {
            shape: vec![d, 3],
            data: boxes
                .iter()
                .flat_map(|[x1, y1, x2, y2]| {
                    let pixels = ((x2 - x1) * (y2 - y1)) as i64;
                    let [s, q] = sums_of(pixels);
                    [pixels, s, q]
                })
                .collect(),
        },
        _ => return None,
    })
}

/// The stub's value for the tensor `name`; `rows` is the number of crops (the stored boxes, or
/// the stub's detections); `det` the detections of a plan that has the `detect` branch.
fn tensor(name: &str, rows: usize, plan: &EnginePlan, det: Option<&Detections>) -> Tensor {
    let n = i64::try_from(rows).unwrap_or(0);
    let d = i64::try_from(DETECTIONS).unwrap_or(0);
    let bytes =
        |items: &[&[u8]]| Tensor::Bytes(items.iter().map(|b| Bytes::copy_from_slice(b)).collect());
    if let Some(t) = det
        .and_then(|det| detect_tensor(name, det))
        .or_else(|| face_tensor(name))
    {
        return t;
    }
    match name {
        "whole_embedding" => Tensor::Fp32 {
            shape: vec![ENCODER_DIM as i64],
            data: vectors(1, ENCODER_DIM),
        },
        "crop_embeddings" => embeddings(rows, ENCODER_DIM),
        "crop_embedded" => Tensor::Bool {
            shape: vec![n],
            data: vec![true; rows],
        },
        "crop_jpegs" => Tensor::Bytes(
            (0..rows)
                .map(|r| Bytes::from([b"\xff\xd8".as_slice(), r.to_string().as_bytes()].concat()))
                .collect(),
        ),
        "face_scores" => fp32(&[d], 0.9),
        "face_overflow" | "ocr_overflow" => Tensor::Int32 {
            shape: vec![1],
            data: vec![0],
        },
        "face_embeddings" => embeddings(DETECTIONS, FACE_DIM),
        "face_embedded" => Tensor::Bool {
            shape: vec![d],
            data: vec![true; DETECTIONS],
        },
        "ocr_quads" => fp32(&[d, 8], 0.0),
        "ocr_texts" => bytes(&[b"STOP".as_slice(), "caf\u{e9}".as_bytes()]),
        "ocr_scores" => fp32(&[d], 0.9),
        "ocr_char_min" => fp32(&[d], 0.8),
        "ocr_det_scores" => fp32(&[d], 0.7),
        "derivative_jpegs" => Tensor::Bytes(
            plan.derivatives
                .iter()
                .map(|r| Bytes::from([b"\xff\xd8".as_slice(), r.name.as_bytes()].concat()))
                .collect(),
        ),
        other => {
            unreachable!("{other} is a branch output of the contract, and the stub knows them all")
        }
    }
}

fn plain_answer(photo: &Photo) -> Result<PhotoAnswer, PipelineError> {
    let plan = &photo.plan;
    let det = plan.detector.as_ref().map(detections).transpose()?;
    let rows = photo.boxes.as_ref().map_or(DETECTIONS, Vec::len);
    let mut outputs = vec![(
        contract().size_output.clone(),
        Tensor::Int32 {
            shape: vec![2],
            data: vec![
                i32::try_from(FRAME.0).unwrap_or(0),
                i32::try_from(FRAME.1).unwrap_or(0),
            ],
        },
    )];
    outputs.extend(
        plan.expected_outputs()
            .into_iter()
            .map(|name| (name.to_owned(), tensor(name, rows, plan, det.as_ref()))),
    );
    Ok(PhotoAnswer {
        status: "ok".into(),
        decode_route: "gpu".into(),
        branches: plan
            .reported()
            .into_iter()
            .map(|b| (b, BranchState::Ok))
            .collect(),
        outputs,
    })
}

fn refused_answer(plan: &EnginePlan, status: String) -> PhotoAnswer {
    PhotoAnswer {
        status,
        decode_route: "none".into(),
        branches: plan
            .reported()
            .into_iter()
            .map(|b| (b, BranchState::Error(IMAGE_FAILED)))
            .collect::<BTreeMap<_, _>>(),
        outputs: Vec::new(),
    }
}

/// See the module documentation.
pub struct StubPipeline {
    ready: AtomicBool,
    not_ready_file: Option<PathBuf>,
    wake: Notify,
}

impl StubPipeline {
    /// A stub whose health check reports not ready while `not_ready_file` is set and exists.
    pub fn new(not_ready_file: Option<PathBuf>) -> Self {
        Self {
            ready: AtomicBool::new(true),
            not_ready_file,
            wake: Notify::new(),
        }
    }
}

impl Pipeline for StubPipeline {
    fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    fn lost(&self) -> bool {
        false
    }

    fn wake(&self) -> &Notify {
        &self.wake
    }

    async fn check(&self) {
        let blocked = self.not_ready_file.as_ref().is_some_and(|p| p.exists());
        self.ready.store(!blocked, Ordering::Release);
    }

    async fn process(&self, photo: Photo) -> Result<PhotoAnswer, PipelineError> {
        let script = script(&photo.image)?;
        if let Some(hold) = script.hold {
            photo.request.enter(script.stage.unwrap_or("detect"));
            tokio::time::sleep(hold).await;
        }
        match script.fail.as_deref() {
            None => {}
            Some("overload") => {
                return Err(PipelineError::Overload(Overload {
                    cause: "budget".into(),
                    detail: "stub".into(),
                }));
            }
            Some("unavailable") => {
                self.wake.notify_one();
                return Err(PipelineError::ModelServerUnavailable("stub: down".into()));
            }
            Some("context_lost") => return Err(PipelineError::ContextLost("stub: lost".into())),
            Some("internal") => return Err(PipelineError::Engine("stub failure".into())),
            Some(other) => {
                return Err(PipelineError::Engine(format!("stub photo: fail={other:?}")));
            }
        }
        Ok(match script.status {
            Some(status) => refused_answer(&photo.plan, status),
            None => plain_answer(&photo)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use super::*;
    use crate::admission::RequestClock;
    use crate::pipeline::RequestState;
    use crate::wire::parse_plan;

    const FRAME: &str = r#"{"version": 1, "branches": ["detect", "embed_frame"], "detector": {"model": "d", "input_size": 640, "letterbox_fill": 114}, "embed": {"model": "pe", "size": 336}}"#;
    const INGEST: &str = r#"{"version": 1, "branches": ["detect", "embed_frame", "embed_crops"], "crops": "inline", "detector": {"model": "d", "input_size": 640, "letterbox_fill": 114}, "embed": {"model": "pe", "size": 336}, "crop_embed": {"model": "c", "size": 224, "batch": 32}}"#;

    fn photo(image: &[u8], plan: &str, boxes: Option<Vec<[f64; 4]>>) -> Photo {
        let clock = RequestClock::start(Instant::now(), 30.0);
        Photo {
            image: Bytes::copy_from_slice(image),
            plan: parse_plan(plan.as_bytes(), boxes.is_some()).expect("plan"),
            boxes,
            request: Arc::new(RequestState::new(clock)),
        }
    }

    #[tokio::test]
    async fn a_plain_photo_gets_every_tensor_of_its_plan_with_the_contracts_types() {
        let stub = StubPipeline::new(None);
        for plan in [FRAME, INGEST] {
            let p = photo(b"\xff\xd8x", plan, None);
            let expected = p.plan.expected_outputs();
            let answer = stub.process(p).await.expect("answer");
            assert_eq!(
                (answer.status.as_str(), answer.decode_route.as_str()),
                ("ok", "gpu")
            );
            let names: Vec<&str> = answer.outputs.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names[0], "size");
            assert_eq!(&names[1..], expected.as_slice());
            for (name, tensor) in &answer.outputs {
                let declared = contract().tensor(name).expect("a protocol tensor");
                assert_eq!(tensor.datatype(), declared.datatype, "{name}");
            }
        }
    }

    /// What the API does with an answer (`src/services/curation/ingest_gpu_decode.py`): it
    /// divides by the blur pixel counts, L2-normalises the vectors and stores them in indexes of
    /// fixed width, and indexes the crop JPEGs by detection row.
    #[tokio::test]
    async fn a_plain_answer_is_one_the_api_can_store() {
        let stub = StubPipeline::new(None);
        for plan in [FRAME, INGEST] {
            let answer = stub
                .process(photo(b"\xff\xd8x", plan, None))
                .await
                .expect("answer");
            let get = |name: &str| {
                answer
                    .outputs
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, t)| t)
            };
            let Some(Tensor::Int32 { data: size, .. }) = get("size") else {
                panic!("size")
            };
            let (w, h) = (i64::from(size[0]), i64::from(size[1]));
            let Some(Tensor::Fp32 { data: whole, .. }) = get("whole_embedding") else {
                panic!("whole_embedding")
            };
            assert_eq!(whole.len(), 1024, "the width of the pe_embedding index");
            assert!(
                whole.iter().any(|v| *v != 0.0),
                "a zero vector has no direction"
            );
            let Some(Tensor::Int64 { shape, data: blur }) = get("blur_sums") else {
                panic!("blur_sums")
            };
            let detections = usize::try_from(shape[0]).expect("rows") - 1;
            for (i, row) in blur.chunks(3).enumerate() {
                let [n, total, squares] = [row[0], row[1], row[2]];
                assert!(n > 0, "row {i}: the API divides by the pixel count");
                assert!(n <= w * h, "row {i}: more pixels than the frame has");
                assert!(
                    n * squares - total * total > 0,
                    "row {i}: a flat photo, no variance"
                );
            }
            let Some(Tensor::Fp64 { data: lb, .. }) = get("letterbox") else {
                panic!("letterbox")
            };
            let canvas = 640.0;
            let Some(Tensor::Fp32 { data: boxes, .. }) = get("det_boxes") else {
                panic!("det_boxes")
            };
            assert_eq!(boxes.len(), 4 * detections);
            for b in boxes.chunks(4) {
                assert!(b[0] < b[2] && b[1] < b[3], "{b:?} is empty");
                assert!(
                    b.iter().all(|v| (0.0..=canvas).contains(&f64::from(*v))),
                    "{b:?} is outside the canvas"
                );
                assert!(
                    f64::from(b[0]) >= lb[1] && f64::from(b[1]) >= lb[2] - 1e-3,
                    "{b:?} is in the padding"
                );
            }
            let Some(Tensor::Fp32 { shape, data }) = get("crop_embeddings") else {
                continue;
            };
            assert_eq!(shape, &vec![detections as i64, 1024]);
            assert!(data.iter().any(|v| *v != 0.0));
            let Some(Tensor::Bytes(jpegs)) = get("crop_jpegs") else {
                panic!("crop_jpegs")
            };
            assert_eq!(
                jpegs.len(),
                detections,
                "the API indexes the JPEGs by detection row"
            );
        }
    }

    #[tokio::test]
    async fn the_rows_follow_the_stored_boxes_or_the_two_detections() {
        let stub = StubPipeline::new(None);
        let plan = r#"{"version": 1, "branches": ["embed_crops"], "crop_embed": {"model": "c", "size": 224, "batch": 32}, "crops": "inline"}"#;
        let rows = vec![[0.1, 0.1, 0.5, 0.5]; 3];
        let answer = stub
            .process(photo(b"x", plan, Some(rows)))
            .await
            .expect("answer");
        let get = |name: &str| {
            answer
                .outputs
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.clone())
                .expect(name)
        };
        assert_eq!(
            get("crop_embeddings"),
            Tensor::Fp32 {
                shape: vec![3, 1024],
                data: vectors(3, 1024)
            }
        );
        let Tensor::Bytes(jpegs) = get("crop_jpegs") else {
            panic!("bytes")
        };
        assert_eq!(jpegs.len(), 3, "one crop JPEG per stored box");
        let answer = stub
            .process(photo(b"x", INGEST, None))
            .await
            .expect("answer");
        let Some((_, Tensor::Fp32 { shape, .. })) =
            answer.outputs.iter().find(|(n, _)| n == "crop_embeddings")
        else {
            panic!("crop_embeddings");
        };
        assert_eq!(shape, &vec![2, 1024]);
    }

    #[test]
    fn the_scripts_in_the_image_are_read() {
        assert_eq!(script(b"\xff\xd8").expect("plain"), Script::default());
        let s =
            script(b"stub:fail=overload,hold_ms=40,stage=embed_pe,status=corrupt").expect("script");
        assert_eq!(s.fail.as_deref(), Some("overload"));
        assert_eq!(s.hold, Some(Duration::from_millis(40)));
        assert_eq!(s.stage, Some("embed_pe"));
        assert_eq!(s.status.as_deref(), Some("corrupt"));
        for bad in [
            &b"stub:nonsense"[..],
            b"stub:hold_ms=x",
            b"stub:stage=nope",
            b"stub:what=1",
        ] {
            assert!(
                matches!(script(bad), Err(PipelineError::Engine(_))),
                "{bad:?}"
            );
        }
    }

    #[tokio::test]
    async fn each_failure_word_is_its_error_class() {
        let stub = StubPipeline::new(None);
        let fail = |word: &'static str| {
            let stub = &stub;
            async move {
                stub.process(photo(format!("stub:fail={word}").as_bytes(), FRAME, None))
                    .await
                    .expect_err(word)
            }
        };
        assert!(matches!(fail("overload").await, PipelineError::Overload(_)));
        assert!(matches!(
            fail("unavailable").await,
            PipelineError::ModelServerUnavailable(_)
        ));
        assert!(matches!(
            fail("context_lost").await,
            PipelineError::ContextLost(_)
        ));
        assert!(matches!(fail("internal").await, PipelineError::Engine(_)));
    }

    #[tokio::test]
    async fn a_refused_photo_reports_image_failed_for_every_branch_and_no_tensor() {
        let answer = StubPipeline::new(None)
            .process(photo(b"stub:status=corrupt", INGEST, None))
            .await
            .expect("answer");
        assert_eq!(
            (answer.status.as_str(), answer.decode_route.as_str()),
            ("corrupt", "none")
        );
        assert!(answer.outputs.is_empty());
        assert_eq!(answer.branches.len(), 3);
        assert!(
            answer
                .branches
                .values()
                .all(|b| *b == BranchState::Error(IMAGE_FAILED))
        );
    }

    #[tokio::test]
    async fn a_held_photo_is_in_the_named_stage_and_a_failed_call_wakes_the_health_tick() {
        let stub = Arc::new(StubPipeline::new(None));
        let p = photo(b"stub:hold_ms=300,stage=embed_pe", FRAME, None);
        let state = Arc::clone(&p.request);
        let running = tokio::spawn({
            let stub = Arc::clone(&stub);
            async move { stub.process(p).await }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(state.stage(), "embed_pe");
        running.await.expect("joined").expect("answered");
        let waiting = tokio::time::timeout(Duration::from_millis(50), stub.wake().notified()).await;
        assert!(waiting.is_err(), "no call failed yet");
        let _ = stub
            .process(photo(b"stub:fail=unavailable", FRAME, None))
            .await;
        tokio::time::timeout(Duration::from_secs(1), stub.wake().notified())
            .await
            .expect("woken");
    }

    #[tokio::test]
    async fn readiness_follows_the_file_when_the_health_tick_reads_it() {
        let file = std::env::temp_dir().join(format!("fe-stub-notready-{}", std::process::id()));
        let _ = std::fs::remove_file(&file);
        let stub = StubPipeline::new(Some(file.clone()));
        assert!(stub.ready());
        std::fs::write(&file, b"").expect("create");
        assert!(
            stub.ready(),
            "readiness is one atomic: it changes when the tick has probed"
        );
        stub.check().await;
        assert!(!stub.ready());
        std::fs::remove_file(&file).expect("remove");
        stub.check().await;
        assert!(stub.ready());
        assert!(!stub.lost());
    }
}
