//! `op-frame-engine self-check`: the GPU pipeline started as the server starts it, against the
//! model server at `OP_FRAME_ENGINE_TRITON_URL`, then one photo through the frame plan (F:
//! `detect`, `embed_frame`, crops `none`) and the ingest plan (C: plus `embed_crops` and crops
//! `inline`), each printed as one line per stage with its seconds and a result line. A short hold
//! on the card then says where the engine breaks: at the start (each step is logged with its
//! time), at readiness (the model server or the region), or at a stage of a plan.
//!
//! The built-in photo is a 640 x 480 test pattern encoded here; it has no objects, so the crop
//! calls run only with `--photo` (a photo of objects, for example one of the COCO set).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;

use crate::admission::RequestClock;
use crate::branches::crops;
use crate::engine::GpuPipeline;
use crate::metrics::{ByStage, Metrics};
use crate::photo::clock;
use crate::pipeline::{Photo, Pipeline, RequestState, Tensor};
use crate::settings::Settings;
use crate::wire::parse_plan;

/// The models the plans name (the API's ingest plan, `ingest_gpu_decode.ingest_engine_plan`).
#[derive(Debug, Clone)]
pub struct Models {
    pub detector: String,
    pub frame_encoder: String,
    pub crop_encoder: String,
}

/// Plan F and plan C as the API writes them for these models.
pub fn plans(m: &Models) -> [(&'static str, String); 2] {
    let detector = format!(
        r#""detector": {{"model": "{}", "input_size": 640, "letterbox_fill": 114}}"#,
        m.detector
    );
    let embed = format!(
        r#""embed": {{"model": "{}", "size": 336}}"#,
        m.frame_encoder
    );
    [
        (
            "F",
            format!(
                r#"{{"version": 1, "branches": ["detect", "embed_frame"], "crops": "none", {detector}, {embed}}}"#
            ),
        ),
        (
            "C",
            format!(
                r#"{{"version": 1, "branches": ["detect", "embed_frame", "embed_crops"], "crops": "inline", {detector}, {embed}, "crop_embed": {{"model": "{}", "size": 224, "batch": 32, "crop_recipe": "letterbox", "fill": 128}}}}"#,
                m.crop_encoder
            ),
        ),
    ]
}

/// The built-in photo: a 640 x 480 pattern of gradients and blocks, JPEG at quality 90.
pub fn built_in_photo() -> Result<Bytes, String> {
    let (w, h) = (640usize, 480usize);
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let block = ((x / 80) + (y / 80)) % 2 == 0;
            let v = [
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                if block { 200 } else { 40 },
            ];
            px.extend(v);
        }
    }
    let mut encoder = crops::encoder(90)?;
    crops::encode(&mut encoder, &px, w, h).map(Bytes::from)
}

fn stage_seconds(metrics: &Metrics) -> Vec<f64> {
    clock::ALL
        .iter()
        .map(|s| {
            metrics
                .stage_seconds
                .get_or_create(&ByStage {
                    stage: (*s).to_owned(),
                })
                .get()
        })
        .collect()
}

fn rows(answer: &[(String, Tensor)], name: &str) -> i64 {
    match answer.iter().find(|(n, _)| n == name).map(|(_, t)| t) {
        Some(Tensor::Fp32 { shape, .. }) => shape.first().copied().unwrap_or(0),
        Some(Tensor::Bytes(items)) => items.len() as i64,
        _ => -1,
    }
}

/// Runs the check; `true` when both plans answered `ok` with every branch `ok`.
pub async fn run(
    settings: &Settings,
    models: &Models,
    photo: Option<PathBuf>,
    ready_wait: Duration,
) -> bool {
    let started = Instant::now();
    let metrics = Metrics::new();
    let pipeline = match GpuPipeline::start(settings, Arc::clone(&metrics)) {
        Ok(p) => p,
        Err(e) => {
            println!("self-check step=start FAILED {e}");
            return false;
        }
    };
    println!(
        "self-check step=start seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    let waiting = Instant::now();
    loop {
        pipeline.check().await;
        if pipeline.ready() {
            break;
        }
        if pipeline.lost() || waiting.elapsed() > ready_wait {
            println!(
                "self-check step=ready FAILED the model server at {} is not ready or the region \
                 is not registered after {:.1} s",
                settings.triton_url,
                waiting.elapsed().as_secs_f64()
            );
            return false;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    println!(
        "self-check step=ready seconds={:.3}",
        waiting.elapsed().as_secs_f64()
    );
    let image = match photo {
        Some(path) => std::fs::read(&path)
            .map(Bytes::from)
            .map_err(|e| format!("{}: {e}", path.display())),
        None => built_in_photo(),
    };
    let image = match image {
        Ok(i) => i,
        Err(e) => {
            println!("self-check step=photo FAILED {e}");
            return false;
        }
    };
    let mut passed = true;
    for (name, text) in plans(models) {
        let plan = match parse_plan(text.as_bytes(), false) {
            Ok(p) => p,
            Err(e) => {
                println!("self-check plan={name} FAILED the plan: {e}");
                passed = false;
                continue;
            }
        };
        let before = stage_seconds(&metrics);
        let clock = RequestClock::start(Instant::now(), settings.request_deadline_seconds);
        let request = Arc::new(RequestState::new(clock));
        let photo = Photo {
            image: image.clone(),
            plan,
            boxes: None,
            request: Arc::clone(&request),
        };
        let t = Instant::now();
        let answer = pipeline.process(photo).await;
        let total = t.elapsed().as_secs_f64();
        for (stage, (b, a)) in clock::ALL
            .iter()
            .zip(before.iter().zip(stage_seconds(&metrics)))
        {
            if a > *b {
                println!("self-check plan={name} stage={stage} seconds={:.6}", a - b);
            }
        }
        match answer {
            Ok(a) => {
                let branches: Vec<String> = a
                    .branches
                    .iter()
                    .map(|(b, s)| format!("{b}:{s:?}"))
                    .collect();
                let ok = a.status == "ok"
                    && a.branches
                        .values()
                        .all(|s| *s == crate::wire::branch_status::BranchState::Ok);
                println!(
                    "self-check plan={name} {} status={} route={} detections={} crop_vectors={} \
                     crop_jpegs={} branches={} seconds={total:.3}",
                    if ok { "ok" } else { "FAILED" },
                    a.status,
                    a.decode_route,
                    rows(&a.outputs, "det_boxes"),
                    rows(&a.outputs, "crop_embeddings"),
                    rows(&a.outputs, "crop_jpegs"),
                    branches.join(","),
                );
                passed &= ok;
            }
            Err(e) => {
                println!(
                    "self-check plan={name} FAILED in stage {}: {e:?} seconds={total:.3}",
                    request.stage()
                );
                passed = false;
            }
        }
    }
    pipeline.close().await;
    println!("self-check {}", if passed { "passed" } else { "FAILED" });
    passed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plans_parse_as_the_api_writes_them_and_the_built_in_photo_routes_to_the_gpu() {
        let m = Models {
            detector: "yolov11_small_trt_end2end".into(),
            frame_encoder: "pe_image_encoder".into(),
            crop_encoder: "pe_crop_encoder".into(),
        };
        let [(f, frame), (c, ingest)] = plans(&m);
        assert_eq!((f, c), ("F", "C"));
        let frame = parse_plan(frame.as_bytes(), false).expect("plan F");
        assert_eq!(frame.branches, ["detect", "embed_frame"]);
        let ingest = parse_plan(ingest.as_bytes(), false).expect("plan C");
        let crop = ingest.crop_embed.expect("crop_embed");
        assert_eq!(
            (crop.model.as_str(), crop.size, crop.batch, crop.fill),
            ("pe_crop_encoder", 224, 32, 128)
        );
        let photo = built_in_photo().expect("photo");
        let route = crate::jpeg_header::route(&photo, 0.0, 100.0);
        assert_eq!(route.decoder, crate::jpeg_header::Decoder::Gpu);
        assert_eq!((route.header.width, route.header.height), (640, 480));
    }
}
