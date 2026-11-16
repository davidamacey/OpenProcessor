//! The host route against the Python engine on the format and damaged-file corpus
//! (`scripts/codegen/frame_engine_host_corpus.py` wrote the files and what the Python engine
//! answered for each, up to the card): the route, the status, the decoded size and pixels.

use std::path::Path;

use frame_engine::host::{admit, decode_checked};
use frame_engine::jpeg_header::{Decoder, route};
use serde_json::Value;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/host");

/// Formats the Python engine opens through Pillow and this engine does not decode (each answers
/// `decoder` here), with the reason. A format that starts to decode must leave this list.
const NOT_DECODED: [(&str, &str); 8] = [
    (
        "tiff_p",
        "palette TIFF: the tiff crate 0.11.3 refuses RGBPalette (libtiff would decode it)",
    ),
    (
        "ppm_rgb",
        "PPM: not in the accepted set (no ingest walker lists it)",
    ),
    (
        "tga_rgb",
        "TGA: no magic number; the content guess cannot find it",
    ),
    ("ico_rgba", "ICO: not in the accepted set"),
    ("pcx_rgb", "PCX: no maintained Rust decoder"),
    ("qoi_rgb", "QOI: not in the accepted set"),
    (
        "jpeg2000_rgb",
        "JPEG 2000: needs OpenJPEG 2.5.4 bound as Pillow bundles it",
    ),
    ("sgi_rgb", "SGI: no maintained Rust decoder"),
];

/// Lossy formats other than JPEG: within one level of Pillow's pixels (plan 17 section 1.2). A
/// JPEG-compressed TIFF is decoded by zune-jpeg inside the tiff crate, not libjpeg-turbo.
const WITHIN_ONE_LEVEL: [&str; 2] = ["webp_lossy", "tiff_jpeg"];

/// A decoded frame: width, height, RGB pixels.
type Frame = (u32, u32, Vec<u8>);

/// The engine's answer up to the card: route, status and the decoded frame.
fn answer(data: &[u8], lo: f64, hi: f64) -> (String, Option<String>, Option<Frame>) {
    let r = route(data, lo, hi);
    let reason = match r.decoder {
        Decoder::Gpu => return ("gpu".into(), None, None),
        Decoder::Refused(s) => return ("none".into(), Some(s.into()), None),
        Decoder::Cpu(reason) => reason,
    };
    let refused = |s: &str| ("none".to_owned(), Some(s.to_owned()), None);
    let admitted = match admit(data, hi) {
        Ok(a) => a,
        Err(s) => return refused(s),
    };
    match decode_checked(data, admitted.size) {
        Ok(f) => (
            format!("cpu_{reason}"),
            Some("ok".into()),
            Some((f.width, f.height, f.rgb)),
        ),
        Err(s) => refused(s),
    }
}

fn max_level_difference(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

#[test]
fn every_corpus_file_gets_the_python_engines_answer() {
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(Path::new(DIR).join("cases.json")).unwrap())
            .unwrap();
    let mut failures = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let data = std::fs::read(Path::new(DIR).join(case["file"].as_str().unwrap())).unwrap();
        let lo = case["min_megapixels"].as_f64().unwrap();
        let hi = case["max_megapixels"].as_f64().unwrap();
        let (route, status, frame) = answer(&data, lo, hi);
        if let Some((_, why)) = NOT_DECODED.iter().find(|(n, _)| *n == name) {
            if status.as_deref() != Some("decoder") || case["status"] != "ok" {
                failures.push(format!(
                    "{name}: listed as not decoded ({why}) but got {status:?}"
                ));
            }
            continue;
        }
        if route != case["route"] || status.as_deref() != case["status"].as_str() {
            failures.push(format!(
                "{name}: route {route} status {status:?}, Python {} {}",
                case["route"], case["status"]
            ));
            continue;
        }
        let (Some((w, h, got)), Some(want)) = (frame, case["pixels"].as_str()) else {
            continue;
        };
        let want = std::fs::read(Path::new(DIR).join(want)).unwrap();
        if [u64::from(w), u64::from(h)]
            != [
                case["size"][0].as_u64().unwrap(),
                case["size"][1].as_u64().unwrap(),
            ]
        {
            failures.push(format!("{name}: size {w}x{h}, Python {}", case["size"]));
            continue;
        }
        let bound = u8::from(WITHIN_ONE_LEVEL.contains(&name));
        let diff = max_level_difference(&got, &want);
        if diff > bound {
            failures.push(format!(
                "{name}: pixels differ by up to {diff} levels (bound {bound})"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
