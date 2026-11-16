//! The hand-written wire types held to the committed OpenAPI file: contract first, so the file
//! decides and a drift in either side fails here.

use std::collections::BTreeSet;

use op_ingest::upload::{
    BatchResponse, BatchStatus, BatchSummary, ImageStatus, IngestImageResponse, merge,
};
use serde_json::Value;

fn schemas() -> Value {
    let doc: Value =
        serde_json::from_str(include_str!("../../../../contracts/openapi/curation.json")).unwrap();
    doc["components"]["schemas"].clone()
}

fn keys(value: &Value) -> BTreeSet<String> {
    value.as_object().unwrap().keys().cloned().collect()
}

/// The names and defaults of `schema`'s properties, and the ones it requires.
fn check(schema: &Value, served: &Value) {
    let props = &schema["properties"];
    assert_eq!(
        keys(props),
        keys(served),
        "the properties served differ from the contract"
    );
    for (name, prop) in props.as_object().unwrap() {
        if let Some(default) = prop.get("default") {
            assert_eq!(&served[name], default, "default of {name}");
        }
    }
    for required in schema["required"].as_array().into_iter().flatten() {
        assert!(
            served.get(required.as_str().unwrap()).is_some(),
            "{required} is required"
        );
    }
}

fn enum_values(schema: &Value, property: &str) -> BTreeSet<String> {
    schema["properties"][property]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn the_image_result_is_the_contract_s() {
    let schemas = schemas();
    let schema = &schemas["IngestImageResponse"];
    let blank = IngestImageResponse::empty_upload("a.jpg");
    let mut served = serde_json::to_value(&blank).unwrap();
    // the defaults are those of a result that carries nothing
    served["error"] = Value::Null;
    served["error_kind"] = Value::Null;
    served["source_identifier"] = Value::Null;
    check(schema, &served);
    let statuses: BTreeSet<String> = [
        ImageStatus::Success,
        ImageStatus::Duplicate,
        ImageStatus::Failed,
    ]
    .iter()
    .map(|s| {
        serde_json::to_value(s)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    })
    .collect();
    assert_eq!(statuses, enum_values(schema, "status"));
    // a client's payload with only the required fields reads, as the contract's defaults say
    let minimal: IngestImageResponse =
        serde_json::from_value(serde_json::json!({"status": "success", "image_path": "/p"}))
            .unwrap();
    assert_eq!(minimal.n_crops, 0);
    assert_eq!(minimal.image_id, "");
}

#[test]
fn the_summary_and_the_batch_are_the_contract_s() {
    let schemas = schemas();
    check(
        &schemas["BatchIngestSummaryResponse"],
        &serde_json::to_value(BatchSummary::default()).unwrap(),
    );
    let schema = &schemas["CurationBatchIngestResponse"];
    let served = serde_json::to_value(merge(Vec::new(), None)).unwrap();
    check(schema, &served);
    let statuses: BTreeSet<String> = [
        BatchStatus::Success,
        BatchStatus::Partial,
        BatchStatus::Error,
    ]
    .iter()
    .map(|s| {
        serde_json::to_value(s)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    })
    .collect();
    assert_eq!(statuses, enum_values(schema, "status"));
    let _: BatchResponse = serde_json::from_value(served).unwrap();
}

#[test]
fn the_route_s_413_is_documented_and_its_request_has_the_four_fields() {
    let doc: Value =
        serde_json::from_str(include_str!("../../../../contracts/openapi/curation.json")).unwrap();
    let op = &doc["paths"]["/curation/projects/{project}/ingest/upload"]["post"];
    assert!(op["responses"].get("413").is_some());
    let body = &doc["components"]["schemas"]["Body_curation_ingest_upload_curation_projects__project__ingest_upload_post"];
    assert_eq!(
        keys(&body["properties"]),
        ["image_paths", "images", "run_id", "source"]
            .map(String::from)
            .into_iter()
            .collect()
    );
}
