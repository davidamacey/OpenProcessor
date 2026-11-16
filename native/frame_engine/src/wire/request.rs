//! One `ModelInferRequest` of protocol version 1 as the engine reads it: the image, the validated
//! plan and the stored boxes (`parse_request` in `frame_engine/front.py`). Every refusal is
//! `INVALID_ARGUMENT`; the texts are the contract's request rules and the plan fault text, and
//! the first fault in the Python order wins.
//!
//! The native engine adds one optional input beside `image`, `source` (`read_by_key` in the
//! contract): a key and the claim for the file it names, read by the engine itself. Exactly one of
//! the two travels in a request; what it makes of a request without `source` is the Python
//! engine's, but for the message of a request that has neither.

use std::collections::HashMap;
use std::fmt;

use bytes::Bytes;
use op_local_file::FileRef;

use super::bytes_tensor::unpack;
use super::contract::contract;
use super::plan::{EnginePlan, PlanError, parse_plan};
use super::source::parse_source;
use crate::proto::ModelInferRequest;
use crate::proto::model_infer_request::InferInputTensor;

/// A request the engine refuses before any work: the status details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestError(pub String);

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RequestError {}

impl From<PlanError> for RequestError {
    fn from(e: PlanError) -> Self {
        Self(e.to_string())
    }
}

/// Where a photo's encoded bytes come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageInput {
    /// The `image` input: a slice of the request message.
    Bytes(Bytes),
    /// The `source` input: a file the engine reads itself under one of its read-only roots.
    File(FileRef),
}

/// The image (a slice of the request message, or the file to read), the plan and the stored
/// boxes: `Some` when the request carries the `boxes` input, even with zero rows (a reprocess
/// sends `(0, 4)`).
#[derive(Debug)]
pub struct ParsedRequest {
    /// The encoded image, or the file that holds it.
    pub image: ImageInput,
    /// The validated plan.
    pub plan: EnginePlan,
    /// Stored boxes as float64 `x1, y1, x2, y2` rows, as the request sent them.
    pub boxes: Option<Vec<[f64; 4]>>,
}

fn rule(id: &str, name: &str) -> RequestError {
    RequestError(contract().request_message(id, name))
}

/// The one element of a BYTES `[1]` input.
fn one_element(tensor: &InferInputTensor, raw: &Bytes) -> Result<Bytes, RequestError> {
    if tensor.datatype != "BYTES" || tensor.shape != [1] {
        return Err(rule("bytes_input_shape", &tensor.name));
    }
    let mut items = unpack(raw).map_err(|e| rule(e.rule(), ""))?;
    match (items.pop(), items.is_empty()) {
        (Some(item), true) => Ok(item),
        _ => Err(rule("bytes_input_one_element", &tensor.name)),
    }
}

fn stored_boxes(tensor: &InferInputTensor, raw: &Bytes) -> Result<Vec<[f64; 4]>, RequestError> {
    let shape = &tensor.shape;
    let boxes = &contract().inputs[2];
    if tensor.datatype != boxes.datatype || shape.len() != 2 || shape[1] != 4 {
        return Err(rule("boxes_datatype_shape", ""));
    }
    let rows = usize::try_from(shape[0])
        .ok()
        .filter(|n| n.checked_mul(32) == Some(raw.len()));
    let Some(rows) = rows else {
        return Err(rule("boxes_byte_size", ""));
    };
    let value = |at: usize| {
        let mut word = [0u8; 8];
        word.copy_from_slice(&raw[at * 8..at * 8 + 8]);
        f64::from_le_bytes(word)
    };
    Ok((0..rows)
        .map(|r| {
            [
                value(4 * r),
                value(4 * r + 1),
                value(4 * r + 2),
                value(4 * r + 3),
            ]
        })
        .collect())
}

/// The image or source, the plan and the stored boxes of `request`.
///
/// # Errors
///
/// An input is unknown or missing, the raw contents do not match the inputs, a tensor has the wrong datatype or shape, both or neither of `image` and `source` is sent, the `source` document is invalid, or the plan is invalid.
///
/// # Panics
///
/// Never in practice: both lookups follow a check that the inputs they read are present.
pub fn parse_request(request: &ModelInferRequest) -> Result<ParsedRequest, RequestError> {
    let c = contract();
    if request.raw_input_contents.len() != request.inputs.len() {
        return Err(rule("raw_contents_count", ""));
    }
    // a name given twice keeps its last tensor, as Python's dict does
    let inputs: HashMap<&str, usize> = request
        .inputs
        .iter()
        .enumerate()
        .map(|(k, t)| (t.name.as_str(), k))
        .collect();
    let source_name = c.read_by_key.input.name.as_str();
    let mut unknown: Vec<&str> = inputs
        .keys()
        .copied()
        .filter(|name| *name != source_name && !c.inputs.iter().any(|t| t.name == *name))
        .collect();
    unknown.sort_unstable();
    if let Some(first) = unknown.first() {
        return Err(rule("unknown_input", first));
    }
    let (image_name, plan_name) = (c.inputs[0].name.as_str(), c.inputs[1].name.as_str());
    let (has_image, has_source) = (
        inputs.contains_key(image_name),
        inputs.contains_key(source_name),
    );
    if !has_image && !has_source {
        return Err(rule("image_or_source_missing", ""));
    }
    if !inputs.contains_key(plan_name) {
        return Err(rule("missing_input", plan_name));
    }
    if has_image && has_source {
        return Err(rule("image_and_source", ""));
    }
    let at = |name: &str| {
        inputs
            .get(name)
            .map(|&k| (&request.inputs[k], &request.raw_input_contents[k]))
    };
    let (image_tensor, image_raw) =
        at(if has_image { image_name } else { source_name }).expect("checked present");
    let (plan_tensor, plan_raw) = at(plan_name).expect("checked present");
    let element = one_element(image_tensor, image_raw)?;
    let plan_text = one_element(plan_tensor, plan_raw)?;
    let boxes = at(&c.inputs[2].name)
        .map(|(t, raw)| stored_boxes(t, raw))
        .transpose()?;
    let plan = parse_plan(&plan_text, boxes.is_some())?;
    let image = if has_image {
        ImageInput::Bytes(element)
    } else {
        ImageInput::File(parse_source(&element)?)
    };
    Ok(ParsedRequest { image, plan, boxes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::bytes_tensor::pack;

    const PLAN: &str = r#"{"version": 1, "branches": ["detect"], "detector": {"model": "det", "input_size": 640, "letterbox_fill": 114}}"#;

    fn tensor(
        request: &mut ModelInferRequest,
        name: &str,
        datatype: &str,
        shape: &[i64],
        raw: Bytes,
    ) {
        request.inputs.push(InferInputTensor {
            name: name.into(),
            datatype: datatype.into(),
            shape: shape.to_vec(),
            ..Default::default()
        });
        request.raw_input_contents.push(raw);
    }

    fn good() -> ModelInferRequest {
        let mut r = ModelInferRequest {
            model_name: "frame_engine".into(),
            ..Default::default()
        };
        tensor(
            &mut r,
            "image",
            "BYTES",
            &[1],
            pack(&[b"\xff\xd8".as_slice()]),
        );
        tensor(&mut r, "plan", "BYTES", &[1], pack(&[PLAN.as_bytes()]));
        r
    }

    fn refusal(request: &ModelInferRequest) -> String {
        parse_request(request).expect_err("refused").to_string()
    }

    #[test]
    fn a_good_request_is_read() {
        let parsed = parse_request(&good()).expect("good");
        assert_eq!(
            parsed.image,
            ImageInput::Bytes(Bytes::from_static(b"\xff\xd8"))
        );
        assert!(parsed.plan.runs("detect"));
        assert!(parsed.boxes.is_none());
    }

    #[test]
    fn each_request_rule_is_its_own_refusal_with_the_contracts_text() {
        let mut counts = good();
        counts.raw_input_contents.pop();
        assert_eq!(
            refusal(&counts),
            "every input must travel in raw_input_contents"
        );

        let mut unknown = good();
        tensor(
            &mut unknown,
            "extra",
            "BYTES",
            &[1],
            pack(&[b"x".as_slice()]),
        );
        assert_eq!(refusal(&unknown), "unknown input 'extra'");

        let mut missing = ModelInferRequest::default();
        tensor(
            &mut missing,
            "plan",
            "BYTES",
            &[1],
            pack(&[PLAN.as_bytes()]),
        );
        assert_eq!(refusal(&missing), "missing input 'image' or 'source'");

        let mut no_plan = ModelInferRequest::default();
        tensor(
            &mut no_plan,
            "image",
            "BYTES",
            &[1],
            pack(&[b"x".as_slice()]),
        );
        assert_eq!(refusal(&no_plan), "missing input 'plan'");

        let mut shape = good();
        shape.inputs[0].shape[0] = 2;
        assert_eq!(refusal(&shape), "image must be BYTES [1]");

        let mut datatype = good();
        datatype.inputs[1].datatype = "FP32".into();
        assert_eq!(refusal(&datatype), "plan must be BYTES [1]");

        let mut elements = good();
        elements.raw_input_contents[0] = pack(&[b"a".as_slice(), b"b".as_slice()]);
        assert_eq!(refusal(&elements), "image must hold one element");

        let mut none = good();
        none.raw_input_contents[0] = Bytes::new();
        assert_eq!(refusal(&none), "image must hold one element");

        let mut prefix = good();
        prefix.raw_input_contents[0] = Bytes::from_static(&[1, 0]);
        assert_eq!(refusal(&prefix), "BYTES tensor ends inside a length prefix");

        let mut overrun = good();
        let mut raw = 9u32.to_le_bytes().to_vec();
        raw.extend_from_slice(b"ab");
        overrun.raw_input_contents[0] = Bytes::from(raw);
        assert_eq!(refusal(&overrun), "BYTES tensor element runs past the end");

        let mut datatype = good();
        tensor(
            &mut datatype,
            "boxes",
            "FP32",
            &[1, 4],
            Bytes::from(vec![0u8; 16]),
        );
        assert_eq!(refusal(&datatype), "boxes must be FP64 [n, 4]");

        let mut width = good();
        tensor(
            &mut width,
            "boxes",
            "FP64",
            &[1, 3],
            Bytes::from(vec![0u8; 24]),
        );
        assert_eq!(refusal(&width), "boxes must be FP64 [n, 4]");

        let mut size = good();
        tensor(
            &mut size,
            "boxes",
            "FP64",
            &[2, 4],
            Bytes::from(vec![0u8; 32]),
        );
        assert_eq!(refusal(&size), "boxes byte size does not match its shape");

        let mut negative = good();
        tensor(&mut negative, "boxes", "FP64", &[-1, 4], Bytes::new());
        assert_eq!(
            refusal(&negative),
            "boxes byte size does not match its shape"
        );
    }

    #[test]
    fn the_first_fault_in_the_python_order_wins() {
        // the count is checked before an unknown name, an unknown name before a missing input,
        // the image's shape before the plan's text, and the request before the plan
        let mut both = ModelInferRequest::default();
        for name in [
            "z_extra", "m_extra", "q_extra", "k_extra", "a_extra", "t_extra", "c_extra",
        ] {
            tensor(&mut both, name, "BYTES", &[1], Bytes::new());
        }
        assert_eq!(
            refusal(&both),
            "unknown input 'a_extra'",
            "the smallest name is reported"
        );
        let mut bad = good();
        bad.inputs[0].shape[0] = 3;
        bad.raw_input_contents[1] = pack(&[b"{".as_slice()]);
        assert_eq!(refusal(&bad), "image must be BYTES [1]");
        let mut plan = good();
        plan.raw_input_contents[1] = pack(&[b"{".as_slice()]);
        assert!(refusal(&plan).starts_with("bad_json"));
    }

    #[test]
    fn a_plan_fault_comes_with_its_name_first() {
        let mut r = good();
        r.raw_input_contents[1] = pack(&[br#"{"version": 2}"#.as_slice()]);
        assert_eq!(refusal(&r), "bad_version: 2");
    }

    #[test]
    fn stored_boxes_are_little_endian_float64_rows_and_zero_rows_are_allowed() {
        let mut r = good();
        r.raw_input_contents[1] = pack(&[
            br#"{"version": 1, "branches": ["embed_frame"], "embed": {"model": "pe", "size": 336}, "crops": "inline"}"#
                .as_slice(),
        ]);
        let mut raw = Vec::new();
        for v in [0.1f64, 0.2, 0.3, 0.4] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        tensor(&mut r, "boxes", "FP64", &[1, 4], Bytes::from(raw));
        assert_eq!(
            parse_request(&r).expect("boxes").boxes,
            Some(vec![[0.1, 0.2, 0.3, 0.4]])
        );
        let n = r.inputs.len() - 1;
        r.inputs[n].shape = vec![0, 4];
        r.raw_input_contents[n] = Bytes::new();
        assert_eq!(parse_request(&r).expect("zero rows").boxes, Some(vec![]));
    }

    #[test]
    fn a_name_given_twice_keeps_the_last_tensor() {
        let mut r = good();
        tensor(
            &mut r,
            "image",
            "BYTES",
            &[1],
            pack(&[b"second".as_slice()]),
        );
        assert_eq!(
            parse_request(&r).expect("two images").image,
            ImageInput::Bytes(Bytes::from_static(b"second"))
        );
    }

    const HASH: &str = "8e0124d8a93f5a1b7c6d2e3f40516273";

    fn source_doc(key: &str) -> Bytes {
        pack(&[format!(r#"{{"key": "{key}", "size": 7, "imohash": "{HASH}"}}"#).as_bytes()])
    }

    fn keyed() -> ModelInferRequest {
        let mut r = good();
        r.inputs.remove(0);
        r.raw_input_contents.remove(0);
        tensor(&mut r, "source", "BYTES", &[1], source_doc("source/a.jpg"));
        r
    }

    #[test]
    fn a_source_stands_in_for_the_image() {
        let parsed = parse_request(&keyed()).expect("keyed");
        let ImageInput::File(file) = parsed.image else {
            panic!("a file, not bytes");
        };
        assert_eq!((file.key(), file.claim().size), ("source/a.jpg", 7));
        assert!(parsed.plan.runs("detect"));
    }

    #[test]
    fn both_and_neither_of_image_and_source_are_each_their_own_refusal() {
        let mut both = good();
        tensor(
            &mut both,
            "source",
            "BYTES",
            &[1],
            source_doc("source/a.jpg"),
        );
        assert_eq!(refusal(&both), "image and source are exclusive: send one");
        let mut neither = keyed();
        neither.inputs.remove(neither.inputs.len() - 1);
        neither.raw_input_contents.pop();
        assert_eq!(refusal(&neither), "missing input 'image' or 'source'");
    }

    #[test]
    fn the_source_is_checked_like_the_image_and_its_document_like_the_plan() {
        let mut shape = keyed();
        shape.inputs[1].shape[0] = 2;
        assert_eq!(refusal(&shape), "source must be BYTES [1]");
        let mut elements = keyed();
        elements.raw_input_contents[1] = pack(&[b"a".as_slice(), b"b".as_slice()]);
        assert_eq!(refusal(&elements), "source must hold one element");
        let mut bad_key = keyed();
        bad_key.raw_input_contents[1] = source_doc("source/../a.jpg");
        assert_eq!(refusal(&bad_key), "bad_key: 'source/../a.jpg'");
        let mut root = keyed();
        root.raw_input_contents[1] = source_doc("projects/1/a");
        assert_eq!(refusal(&root), "unknown_root: 'projects/1/a'");
        // the plan is read first: a request with a bad plan and a bad source names the plan
        let mut both_bad = bad_key;
        both_bad.raw_input_contents[0] = pack(&[b"{".as_slice()]);
        assert!(refusal(&both_bad).starts_with("bad_json"));
    }
}
