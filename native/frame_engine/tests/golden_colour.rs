//! The colour rule's port against what `src/utils/icc_srgb.py` answered through Pillow's Little
//! CMS (`tests/golden/colour.json`, `scripts/codegen/frame_engine_colour_goldens.py`): the same
//! tag verdict, the numbers it rests on, the same kind and, for `convert`, a table equal to the
//! byte. The COCO corpus (85 profiles, kept out of the tree) runs through the same check with
//! `cargo test --test golden_colour -- --ignored` and `OP_TEST_ICC_CORPUS` set to the directory
//! of profiles and the record the script wrote (`--corpus DIR --out FILE`).

use std::path::Path;

use base64::Engine;
use frame_engine::colour::classify::{Kind, classify, colorant_distance, tag_verdict};
use lcms2::Profile;
use serde_json::Value;
use sha2::{Digest, Sha256};

const COLOUR: &str = include_str!("golden/colour.json");

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn kind(v: &Value) -> Option<&str> {
    v.as_str()
}

fn close(got: Option<f64>, want: &Value, what: &str, name: &str) {
    match (got, want.as_f64()) {
        (None, None) => {}
        (Some(g), Some(w)) => assert!(
            (g - w).abs() <= 1e-9 * w.abs().max(1.0),
            "{name} {what}: {g} against {w}"
        ),
        _ => panic!("{name} {what}: {got:?} against {want}"),
    }
}

/// One profile's record against the port.
fn check(name: &str, icc: &[u8], want: &Value) {
    assert_eq!(
        tag_verdict(icc).map(Kind::as_str),
        kind(&want["tag_verdict"]),
        "{name} tag verdict"
    );
    let distance = Profile::new_icc(icc)
        .ok()
        .and_then(|p| colorant_distance(&p));
    close(
        distance,
        &want["colorant_distance"],
        "colorant distance",
        name,
    );
    let (got, table) = classify(icc);
    assert_eq!(Some(got.as_str()), kind(&want["kind"]), "{name} kind");
    assert_eq!(
        table.as_deref().map(hex),
        want["table_sha256"].as_str().map(str::to_owned),
        "{name} table"
    );
}

#[test]
fn every_golden_profile_is_classified_and_converted_as_python_does() {
    let golden: Value = serde_json::from_str(COLOUR).expect("colour goldens");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let profiles = golden["profiles"].as_array().expect("profiles");
    assert!(profiles.len() >= 16);
    for p in profiles {
        let name = p["name"].as_str().expect("a name");
        let icc = match (p["path"].as_str(), p["icc"].as_str()) {
            (Some(path), _) => std::fs::read(root.join(path)).expect("a fixture"),
            (None, Some(b64)) => base64::engine::general_purpose::STANDARD
                .decode(b64)
                .expect("base64"),
            _ => panic!("{name} has neither a path nor bytes"),
        };
        if let Some(digest) = p["sha256"].as_str() {
            assert_eq!(hex(&icc), digest, "{name} bytes");
        }
        check(name, &icc, p);
    }
}

#[test]
#[ignore = "needs the COCO profile corpus, kept out of the tree: set OP_TEST_ICC_CORPUS=DIR:RECORD"]
fn the_coco_corpus_is_classified_and_converted_as_python_does() {
    let spec = std::env::var("OP_TEST_ICC_CORPUS").expect("OP_TEST_ICC_CORPUS=DIR:RECORD");
    let (dir, record) = spec.split_once(':').expect("DIR:RECORD");
    let golden: Value =
        serde_json::from_str(&std::fs::read_to_string(record).expect("record")).expect("json");
    let profiles = golden["profiles"].as_array().expect("profiles");
    for p in profiles {
        let name = p["name"].as_str().expect("a name");
        let icc = std::fs::read(Path::new(dir).join(format!("{name}.icc"))).expect("a profile");
        assert_eq!(
            hex(&icc),
            p["sha256"].as_str().expect("a digest"),
            "{name} bytes"
        );
        check(name, &icc, p);
    }
    println!("{} profiles checked", profiles.len());
}
