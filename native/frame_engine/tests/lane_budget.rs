//! `lanes_that_fit` against `native/frame_engine/tests/golden/lane_budget.json`: the cases
//! `scripts/codegen/frame_engine_lane_budget.py` answered, and the constants it was run with.

use frame_engine::lane_budget::{
    HEADROOM_PERCENT, LaneFit, NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE, SIZED_FOR_MEGAPIXELS,
    default_lanes, lanes_that_fit, sized_megapixels,
};
use serde_json::Value;

const CASES: &str = include_str!("golden/lane_budget.json");

#[test]
fn every_case_of_the_python_function_is_answered_the_same() {
    let doc: Value = serde_json::from_str(CASES).expect("JSON");
    assert_eq!(
        doc["bytes_per_megapixel"],
        NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE
    );
    assert_eq!(doc["headroom_percent"], HEADROOM_PERCENT);
    assert_eq!(
        doc["sized_for_megapixels"].as_f64(),
        Some(SIZED_FOR_MEGAPIXELS)
    );
    let cases = doc["cases"].as_array().expect("cases");
    assert!(cases.len() > 90);
    for c in cases {
        let got = lanes_that_fit(
            c["free_bytes"].as_u64().expect("free"),
            c["pool_bytes"].as_u64().expect("pool"),
            c["megapixels"].as_f64().expect("mp"),
        );
        assert_eq!(got.lanes, c["lanes"], "{c}");
        assert_eq!(got.fits, c["fits"], "{c}");
    }
}

#[test]
fn the_photo_size_lanes_are_sized_for_is_the_pythons() {
    let doc: Value = serde_json::from_str(CASES).expect("JSON");
    for c in doc["sized"].as_array().expect("sized") {
        let got = sized_megapixels(c["max_megapixels"].as_f64().expect("max"));
        assert_eq!(Some(got), c["sized"].as_f64(), "{c}");
    }
}

#[test]
fn never_more_than_fit_and_never_fewer_than_one() {
    let mib = 1u64 << 20;
    let none = lanes_that_fit(100 * mib, 4096 * mib, 26.0);
    assert_eq!(
        none,
        LaneFit {
            lanes: 1,
            fits: false
        }
    );
    let some = lanes_that_fit(10_000 * mib, 4096 * mib, 26.0);
    let per_lane = 26 * NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE;
    assert!(u64::from(some.lanes) * per_lane <= (10_000 - 4096) * mib);
    assert!(u64::from(some.lanes + 1) * per_lane > (10_000 - 4096) * mib * 90 / 100);
    assert_eq!(lanes_that_fit(u64::MAX, 0, 1.0).lanes, 256);
}

#[test]
fn the_default_is_the_cores_or_what_fits() {
    let tight = LaneFit {
        lanes: 6,
        fits: true,
    };
    assert_eq!(default_lanes(24, tight), 6);
    assert_eq!(default_lanes(4, tight), 4);
    assert_eq!(sized_megapixels(100.0), 26.0);
    assert_eq!(sized_megapixels(12.0), 12.0);
}
