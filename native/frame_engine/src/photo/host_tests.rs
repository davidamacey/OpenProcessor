//! The host route through the photo flow on the fake (the lane's host decode is CPU work, so the
//! fake runs it): the route each corpus file takes, the statuses and when they are answered, and
//! every photo status and decode route of the wire contract produced by a scenario.

use super::*;
use crate::photo::source;
use crate::wire::contract::contract;

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/host");

fn corpus(name: &str) -> Bytes {
    Bytes::from(std::fs::read(format!("{CORPUS}/{name}")).expect("corpus file"))
}

fn size_of(a: &PhotoAnswer) -> Option<Vec<i32>> {
    match output(a, "size") {
        Some(Tensor::Int32 { data, .. }) => Some(data.clone()),
        _ => None,
    }
}

#[tokio::test]
async fn host_route_photos_are_decoded_on_the_lane_and_answer_their_route_and_upright_size() {
    let b = bed();
    for (file, route, size) in [
        ("png_rgb.png", "cpu_not_jpeg", [23, 15]),
        ("png_exif_orientation_6.png", "cpu_not_jpeg", [15, 23]),
        ("webp_lossless.webp", "cpu_not_jpeg", [23, 15]),
        ("tiff_exif_orientation_6.tif", "cpu_not_jpeg", [15, 23]),
        ("jpeg_progressive.jpg", "cpu_progressive", [23, 15]),
        ("jpeg_cmyk.jpg", "cpu_components", [23, 15]),
        ("gif_p.gif", "cpu_not_jpeg", [23, 15]),
    ] {
        let answer = b
            .flow
            .process(photo_of(corpus(file), INGEST))
            .await
            .expect("answer");
        assert_eq!(
            (answer.status.as_str(), answer.decode_route.as_str()),
            ("ok", route),
            "{file}"
        );
        assert_eq!(size_of(&answer), Some(size.to_vec()), "{file}");
        assert!(
            answer.branches.values().all(|s| *s == BranchState::Ok),
            "{file}: {:?}",
            answer.branches
        );
    }
    assert_released(&b).await;
}

#[tokio::test]
async fn host_refusals_come_before_the_budget_and_decode_failures_after_it() {
    let b = bed();
    for (file, status) in [
        ("damaged_text.bin", "decoder"),
        ("damaged_empty.bin", "decoder"),
        ("huge_png_header.png", "huge"),
        ("huge_png_bomb.png", "huge"),
        ("damaged_soi_only.jpg", "corrupt"),
    ] {
        let answer = b
            .flow
            .process(photo_of(corpus(file), INGEST))
            .await
            .expect("answer");
        assert_eq!(
            (answer.status.as_str(), answer.decode_route.as_str()),
            (status, "none"),
            "{file}"
        );
    }
    assert_eq!(b.fake.events(), [], "refused before any decode");
    for file in [
        "damaged_truncated_no_eoi_gpu.jpg",
        "damaged_progressive_truncated.jpg",
        "damaged_png_truncated.png",
    ] {
        let answer = b
            .flow
            .process(photo_of(corpus(file), INGEST))
            .await
            .expect("answer");
        assert_eq!(
            (answer.status.as_str(), answer.decode_route.as_str()),
            ("decoder", "none"),
            "{file}"
        );
    }
    assert_eq!(
        b.fake
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Decode { .. }))
            .count(),
        3,
        "each damaged file was decoded on a lane and refused there"
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn a_host_frame_reserves_one_upright_frame_and_its_need_decides_over_budget() {
    // 23 x 15 x 3 = 1035 bytes, rotated or not: the host frame arrives upright, so no copy is held
    // beside it (twice, 2070 bytes, would pass this budget)
    let b = bed_with(64 * MIB, 4 * MIB, 2048, 4);
    let answer = b
        .flow
        .process(photo_of(corpus("png_exif_orientation_6.png"), FRAME))
        .await
        .expect("answer");
    assert_eq!(answer.status, "ok");
    let b = bed_with(64 * MIB, 4 * MIB, 1034, 4);
    let answer = b
        .flow
        .process(photo_of(corpus("png_rgb.png"), FRAME))
        .await
        .expect("answer");
    assert_eq!(answer.status, "over_budget");
    assert_eq!(b.fake.events(), []);
}

/// The scenario that makes a photo answer `status`; a status the contract gains without a
/// scenario here fails the test.
async fn scenario(status: &str) -> PhotoAnswer {
    let (b, image) = match status {
        "ok" => (bed(), corpus("png_rgb.png")),
        "corrupt" => (bed(), corpus("damaged_soi_garbage.jpg")),
        "huge" => (bed(), jpeg(12000, 9000, 1)),
        "over_budget" => (
            bed_with(64 * MIB, 4 * MIB, MIB as u64, 4),
            jpeg(1000, 750, 1),
        ),
        "gpu_memory" => {
            let b = bed();
            b.fake
                .set(|s| s.decode_fault = Some(GpuFault::OutOfMemory("cap".into())));
            (b, jpeg(1000, 750, 1))
        }
        "decoder" => (bed(), corpus("damaged_truncated_half_small.jpg")),
        other => panic!("no scenario answers {other}: add one with the status's producer"),
    };
    b.flow
        .process(photo_of(image, INGEST))
        .await
        .expect("answer")
}

#[tokio::test]
async fn every_photo_status_of_the_contract_has_its_producer() {
    let statuses = &contract().photo_statuses;
    assert!(!statuses.is_empty());
    for status in statuses {
        let answer = scenario(status).await;
        assert_eq!(&answer.status, status);
        let refused = status != "ok";
        assert_eq!(answer.decode_route == "none", refused, "{status}");
        assert_eq!(answer.outputs.is_empty(), refused, "{status}");
        if refused {
            assert!(
                answer
                    .branches
                    .values()
                    .all(|s| *s == BranchState::Error(IMAGE_FAILED)),
                "{status}"
            );
        }
    }
}

/// A frame header of `precision` bits (no image data): routed `cpu_precision`.
fn twelve_bit_jpeg() -> Bytes {
    let mut b = jpeg(64, 48, 1).to_vec();
    let sof = b.windows(2).position(|w| w == [0xFF, 0xC0]).expect("SOF");
    b[sof + 4] = 12;
    Bytes::from(b)
}

#[test]
fn every_decode_route_of_the_contract_is_one_the_source_decides() {
    let decide = |image: &Bytes, lo: f64| {
        source::decide(image, route(image, lo, 100.0), 100.0, &[], |_| vec![640])
            .map(|s| s.decode_route)
    };
    for name in &contract().decode_routes {
        let got = match name.as_str() {
            "gpu" => decide(&jpeg(1000, 750, 1), 0.0),
            "cpu_not_jpeg" => decide(&corpus("png_rgb.png"), 0.0),
            "cpu_precision" => decide(&twelve_bit_jpeg(), 0.0),
            "cpu_components" => decide(&corpus("jpeg_cmyk.jpg"), 0.0),
            "cpu_progressive" => decide(&corpus("jpeg_progressive.jpg"), 0.0),
            "cpu_truncated" => decide(&corpus("damaged_truncated_no_eoi_gpu.jpg"), 0.0),
            "cpu_small" => decide(&corpus("jpeg_baseline_420_small.jpg"), 1.0),
            "none" => decide(&corpus("damaged_soi_only.jpg"), 0.0)
                .map_err(|_| ())
                .err()
                .map(|()| "none".to_owned())
                .ok_or("refused"),
            other => panic!("no input routes {other}: add one"),
        };
        assert_eq!(got.as_deref(), Ok(name.as_str()), "{name}");
    }
}
