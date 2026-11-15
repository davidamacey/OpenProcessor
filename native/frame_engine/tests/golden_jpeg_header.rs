//! The header reads and the route against the goldens the Python functions wrote for
//! byte strings built marker by marker and for seeded damage to them
//! (`scripts/codegen/frame_engine_goldens.py`).

use base64::Engine as _;
use frame_engine::jpeg_header::{
    Decoder, JpegHeader, Kind, jpeg_end, jpeg_header, oriented_size, route,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const CASES: &str = include_str!("golden/jpeg_header.json");

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn check_header(got: &JpegHeader, want: &Value, name: &str) {
    let kind = match got.kind {
        Kind::Jpeg => "jpeg",
        Kind::Corrupt => "corrupt",
        Kind::Other => "other",
    };
    assert_eq!(kind, want["kind"], "{name}: kind");
    assert_eq!(
        u64::from(got.width),
        want["width"].as_u64().expect("width"),
        "{name}: width"
    );
    assert_eq!(
        u64::from(got.height),
        want["height"].as_u64().expect("height"),
        "{name}: height"
    );
    assert_eq!(
        u64::from(got.components),
        want["components"].as_u64().expect("c"),
        "{name}"
    );
    assert_eq!(
        u64::from(got.precision),
        want["precision"].as_u64().expect("p"),
        "{name}"
    );
    assert_eq!(
        got.progressive,
        want["progressive"].as_bool().expect("flag"),
        "{name}"
    );
    assert_eq!(
        u64::from(got.orientation),
        want["orientation"].as_u64().expect("o"),
        "{name}"
    );
    assert_eq!(
        got.lossless,
        want["lossless"].as_bool().expect("flag"),
        "{name}"
    );
    assert_eq!(
        got.end as u64,
        want["end"].as_u64().expect("end"),
        "{name}: end"
    );
    match &got.icc {
        Some(icc) => {
            assert_eq!(want["icc_sha256"], sha(icc).as_str(), "{name}: icc");
            assert_eq!(
                want["icc_len"].as_u64(),
                Some(icc.len() as u64),
                "{name}: icc length"
            );
        }
        None => assert!(
            want["icc_sha256"].is_null(),
            "{name}: the profile is missing"
        ),
    }
}

#[test]
fn every_case_reads_and_routes_as_python_does() {
    let document: Value = serde_json::from_str(CASES).expect("the goldens are JSON");
    let cases = document["cases"].as_array().expect("cases");
    assert!(cases.len() >= 450, "{} cases", cases.len());
    let mut routes = std::collections::BTreeSet::new();
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let data = base64::engine::general_purpose::STANDARD
            .decode(case["data"].as_str().expect("data"))
            .expect("base64");
        let header = jpeg_header(&data);
        check_header(&header, &case["header"], name);
        assert_eq!(
            jpeg_end(&data).map(|e| e as u64),
            case["jpeg_end"].as_u64(),
            "{name}: jpeg_end"
        );
        let (lo, hi) = (
            case["min_megapixels"].as_f64().expect("lo"),
            case["max_megapixels"].as_f64().expect("hi"),
        );
        let routed = route(&data, lo, hi);
        let (kind, reason) = match &routed.decoder {
            Decoder::Gpu => ("gpu", ""),
            Decoder::Cpu(r) => ("cpu", *r),
            Decoder::Refused(r) => ("refused", *r),
        };
        assert_eq!(kind, case["route"]["kind"], "{name}: route");
        assert_eq!(reason, case["route"]["reason"], "{name}: reason");
        check_header(&routed.header, &case["route"]["header"], name);
        let size = oriented_size(&header);
        let want = case["oriented_size"].as_array().expect("size");
        assert_eq!(
            [u64::from(size.0), u64::from(size.1)],
            [want[0].as_u64().expect("w"), want[1].as_u64().expect("h")],
            "{name}: oriented size"
        );
        routes.insert(format!("{kind}:{reason}"));
    }
    for seen in [
        "gpu:",
        "cpu:truncated",
        "cpu:not_jpeg",
        "cpu:precision",
        "cpu:components",
        "cpu:progressive",
        "cpu:small",
        "refused:corrupt",
        "refused:huge",
    ] {
        assert!(routes.contains(seen), "no case routes {seen}");
    }
}
