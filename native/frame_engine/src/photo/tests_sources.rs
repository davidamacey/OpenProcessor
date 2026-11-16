//! Where a photo's rectangles and colours come from: stored boxes and the colour step, on the host fake.

use super::*;

const STORED: &str = r#"{"version": 1, "branches": ["embed_crops"], "crops": "inline", "crop_embed": {"model": "pe_crop", "size": 224, "batch": 32}}"#;

fn stored(plan: &str, boxes: Vec<[f64; 4]>) -> Photo {
    Photo {
        image: ImageInput::Bytes(jpeg(100, 100, 1)),
        plan: parse_plan(plan.as_bytes(), true).expect("plan"),
        boxes: Some(boxes),
        request: Arc::new(RequestState::new(RequestClock::start(Instant::now(), 30.0))),
    }
}

fn jpeg_names(a: &PhotoAnswer) -> Vec<String> {
    let Some(Tensor::Bytes(items)) = output(a, "crop_jpegs") else {
        panic!("crop_jpegs is BYTES");
    };
    items
        .iter()
        .map(|i| String::from_utf8_lossy(i).into_owned())
        .collect()
}

#[tokio::test]
async fn stored_boxes_stand_in_for_the_detector_in_the_requests_order() {
    let b = bed();
    // one inside, one past the right edge and above the top: cut to a one-pixel column
    let boxes = vec![[0.1, 0.1, 0.5, 0.5], [1.2, -0.3, 1.5, 0.2]];
    let answer = b.flow.process(stored(STORED, boxes)).await.expect("answer");
    assert_eq!(
        (answer.status.as_str(), answer.decode_route.as_str()),
        ("ok", "gpu")
    );
    assert_eq!(answer.branches["embed_crops"], BranchState::Ok);
    let names: Vec<&str> = answer.outputs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["size", "crop_embeddings", "crop_embedded", "crop_jpegs"]
    );
    assert_eq!(
        shape(output(&answer, "crop_embeddings").expect("vectors")),
        [2, 8]
    );
    assert_eq!(
        jpeg_names(&answer),
        ["jpeg [10, 10, 50, 50]", "jpeg [99, 0, 100, 20]"]
    );
    assert_eq!(b.fake.calls(), [("pe_crop".to_owned(), [2, 3, 224, 224])]);
    assert!(
        b.fake
            .events()
            .iter()
            .all(|e| !matches!(e, Event::Launch { blur, .. } if *blur > 0)),
        "no blur sums without the detector"
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn zero_stored_boxes_answer_empty_crop_outputs_without_a_call() {
    let b = bed();
    let answer = b
        .flow
        .process(stored(STORED, vec![]))
        .await
        .expect("answer");
    assert_eq!(answer.status, "ok");
    assert_eq!(
        shape(output(&answer, "crop_embeddings").expect("vectors")),
        [0, 0]
    );
    assert_eq!(shape(output(&answer, "crop_embedded").expect("mask")), [0]);
    assert_eq!(jpeg_names(&answer), Vec::<String>::new());
    assert_eq!(b.fake.calls(), []);
    assert_released(&b).await;
}

#[tokio::test]
async fn stored_boxes_with_crops_alone_still_encode_every_crop() {
    let b = bed();
    let plan = r#"{"version": 1, "branches": ["embed_frame"], "crops": "inline", "embed": {"model": "pe", "size": 336}}"#;
    let answer = b
        .flow
        .process(stored(
            plan,
            vec![[0.0, 0.0, 1.0, 1.0], [0.25, 0.5, 0.75, 1.0]],
        ))
        .await
        .expect("answer");
    assert_eq!(
        jpeg_names(&answer),
        ["jpeg [0, 0, 100, 100]", "jpeg [25, 50, 75, 100]"]
    );
    assert_eq!(b.fake.calls(), [("pe".to_owned(), [1, 3, 336, 336])]);
    assert_released(&b).await;
}

#[tokio::test]
async fn a_stored_box_that_is_not_a_number_fails_the_photo_as_pythons_round_does() {
    let b = bed();
    let failed = b
        .flow
        .process(stored(STORED, vec![[0.1, f64::NAN, 0.5, 0.5]]))
        .await
        .expect_err("not a number");
    assert!(matches!(&failed, PipelineError::Engine(m) if m.contains("finite")));
    assert_released(&b).await;
}

pub(super) const P3: &[u8] = include_bytes!("../../../../tests/fixtures/icc/DisplayP3-v4.icc");
const SRGB: &[u8] = include_bytes!("../../../../tests/fixtures/icc/colord-sRGB.icc");
pub(super) const PROGRESSIVE: &[u8] =
    include_bytes!("../../tests/golden/host/jpeg_progressive.jpg");

/// `plain` with an ICC profile in one APP2 chunk after the start marker.
pub(super) fn with_icc(plain: &[u8], icc: &[u8]) -> Bytes {
    let mut payload = b"ICC_PROFILE\0\x01\x01".to_vec();
    payload.extend_from_slice(icc);
    let mut b = plain[..2].to_vec();
    b.extend_from_slice(&[0xFF, 0xE2]);
    b.extend_from_slice(
        &u16::try_from(payload.len() + 2)
            .expect("short")
            .to_be_bytes(),
    );
    b.extend_from_slice(&payload);
    b.extend_from_slice(&plain[2..]);
    Bytes::from(b)
}

/// `jpeg` with an ICC profile.
fn tagged(w: u16, h: u16, icc: &[u8]) -> Bytes {
    with_icc(&jpeg(w, h, 1), icc)
}

fn gathers(b: &Bed) -> usize {
    b.fake
        .events()
        .iter()
        .filter(|e| matches!(e, Event::Gather))
        .count()
}

fn colour_count(b: &Bed, outcome: &str) -> u64 {
    b.flow
        .metrics
        .colour
        .get_or_create(&ByOutcome {
            outcome: outcome.into(),
        })
        .get()
}

#[tokio::test]
async fn a_photo_with_a_converting_profile_is_gathered_through_one_table_per_profile() {
    let b = bed();
    for _ in 0..2 {
        let answer = b
            .flow
            .process(photo_of(tagged(1000, 750, P3), FRAME))
            .await
            .expect("answer");
        assert_eq!(
            (answer.status.as_str(), answer.decode_route.as_str()),
            ("ok", "gpu")
        );
    }
    let events = b.fake.events();
    let tables = events
        .iter()
        .filter(|e| matches!(e, Event::ColourTable(_)))
        .count();
    let gathers = events.iter().filter(|e| matches!(e, Event::Gather)).count();
    assert_eq!(
        (tables, gathers),
        (1, 2),
        "one table for the profile, a gather per photo"
    );
    assert!(events.contains(&Event::ColourTable(crate::colour::TABLE_BYTES)));
    assert_eq!(colour_count(&b, "converted"), 2);
    // untagged: no gather
    b.flow.process(photo(FRAME)).await.expect("answer");
    assert_eq!(colour_count(&b, "srgb"), 1);
    assert_eq!(
        b.fake
            .events()
            .iter()
            .filter(|e| matches!(e, Event::Gather))
            .count(),
        2
    );
    assert_released(&b).await;
}

#[tokio::test]
async fn a_host_route_photo_with_a_converting_profile_is_gathered_after_its_upload() {
    let b = bed();
    let answer = b
        .flow
        .process(photo_of(with_icc(PROGRESSIVE, P3), FRAME))
        .await
        .expect("answer");
    assert_eq!(
        (answer.status.as_str(), answer.decode_route.as_str()),
        ("ok", "cpu_progressive")
    );
    assert_eq!(
        gathers(&b),
        1,
        "the upright host frame goes through the table"
    );
    assert_eq!(colour_count(&b, "converted"), 1);
    for image in [with_icc(PROGRESSIVE, SRGB), Bytes::from_static(PROGRESSIVE)] {
        let answer = b
            .flow
            .process(photo_of(image, FRAME))
            .await
            .expect("answer");
        assert_eq!(answer.decode_route, "cpu_progressive");
    }
    assert_eq!(gathers(&b), 1, "an sRGB profile or none: left as decoded");
    assert_eq!(colour_count(&b, "srgb"), 2);
    assert_released(&b).await;
}
