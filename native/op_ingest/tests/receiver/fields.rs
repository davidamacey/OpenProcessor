//! Which photos are handed on, refused or held, and in what order the answer lists them.

use axum::body::Body;
use axum::http::StatusCode;
use op_ingest::upload::router;

use crate::common::{answer, form, landing_files, multipart_request, pattern, rig};

#[tokio::test]
async fn a_photo_refused_for_its_extension_is_listed_first_and_the_rest_go_through() {
    let rig = rig(&[]);
    let (a, b, c) = (pattern(1, 900), pattern(2, 900), pattern(3, 900));
    let (content_type, body) = form(&[
        ("image_paths", None, br#"["a.jpg","b.gif","c.png"]"#),
        ("source", None, b"camera"),
        ("images", Some("a.jpg"), &a),
        ("images", Some("b.gif"), &b),
        ("images", Some("c.png"), &c),
    ]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    assert_eq!(got.body["status"], "partial");
    assert_eq!(got.body["summary"]["failed"], 1);
    assert_eq!(got.body["summary"]["successful"], 2);
    let results = got.body["results"].as_array().unwrap();
    let order: Vec<&str> = results
        .iter()
        .map(|r| r["source_identifier"].as_str().unwrap())
        .collect();
    assert_eq!(order, ["b.gif", "a.jpg", "c.png"]);
    assert_eq!(results[0]["error_kind"], "unsupported_type");
    let seen = rig.sink.seen.lock().unwrap().clone();
    assert_eq!(seen.iter().map(|s| s.index).collect::<Vec<_>>(), [0, 2]);
    assert_eq!(seen[1].extension, ".png");
    let end = rig.sink.ends.lock().unwrap()[0].clone();
    assert_eq!(end.source, "camera");
    assert_eq!(end.run_id, None);
    // the refused photo's file is gone, the two accepted ones stay with the sink
    assert_eq!(landing_files(&rig.landing_root).len(), 2);
}

#[tokio::test]
async fn image_paths_in_the_middle_keeps_photo_order_and_defaults_fill_the_rest() {
    let rig = rig(&[]);
    let (a, b) = (pattern(4, 700), pattern(5, 700));
    let (content_type, body) = form(&[
        ("images", Some("a.jpg"), &a),
        ("image_paths", None, br#"["x/first.jpeg","x/second.jpg"]"#),
        ("images", Some("b.jpg"), &b),
    ]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    let seen = rig.sink.seen.lock().unwrap().clone();
    assert_eq!(seen[0].identifier, "x/first.jpeg");
    assert_eq!(seen[0].extension, ".jpeg");
    assert_eq!(seen[1].identifier, "x/second.jpg");
    assert_eq!(seen[0].bytes, a);
    assert_eq!(seen[1].bytes, b);
    assert_eq!(rig.sink.ends.lock().unwrap()[0].source, "upload");
}

#[tokio::test]
async fn without_image_paths_the_file_names_are_the_identifiers() {
    let rig = rig(&[]);
    let (content_type, body) = form(&[
        ("images", Some("one.jpg"), &pattern(6, 300)),
        ("images", Some(""), &pattern(7, 300)),
    ]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    let ids: Vec<String> = rig
        .sink
        .seen
        .lock()
        .unwrap()
        .iter()
        .map(|s| s.identifier.clone())
        .collect();
    assert_eq!(ids, ["one.jpg", "upload_1"]);
}

#[tokio::test]
async fn a_request_without_a_resolved_project_is_refused_before_its_body_is_read() {
    let rig = rig(&[]);
    let bare = router(std::sync::Arc::clone(&rig.state));
    let (content_type, body) = form(&[("images", Some("a.jpg"), b"abc")]);
    let got = answer(
        &bare,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        rig.sink
            .accept_calls
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert!(landing_files(&rig.landing_root).is_empty());
}
