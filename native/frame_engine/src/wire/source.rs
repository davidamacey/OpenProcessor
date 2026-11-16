//! The `source` input of the read-by-key extension (`read_by_key` in
//! `contracts/json/frame_engine_v1.json`): one JSON object naming a file and the claim made for
//! it, `{"key": "landing/...", "size": N, "imohash": "<32 hex>"}`. Every refusal is
//! `INVALID_ARGUMENT` with `<fault>: <repr of the subject>` as its details, the first fault in
//! the contract's order winning; `contracts/json/frame_engine_source_cases.json` holds this parser
//! to the cases of `scripts/codegen/frame_engine_source_cases.py`.

use op_local_file::imohash::ImoHash;
use op_local_file::{FileClaim, FileRef, KeyError, validate_key};

use super::json::{self, Json};
use super::request::RequestError;
use crate::pyfmt::repr_str;

/// The only fields a source document has, in the order they are required and checked.
const FIELDS: [&str; 3] = ["key", "size", "imohash"];
/// The largest size a claim may state: the largest integer every JSON reader holds exactly.
const MAX_SIZE: i64 = (1 << 53) - 1;

fn fault(name: &str, subject: &str) -> RequestError {
    RequestError(format!("{name}: {}", repr_str(subject)))
}

fn field<'a>(fields: &'a [(String, Json)], name: &str) -> &'a Json {
    // `parse_source` checked that every name is present before it reads any
    fields
        .iter()
        .find(|(n, _)| n == name)
        .map_or(&Json::Null, |(_, v)| v)
}

/// The file and claim a `source` element names.
///
/// # Errors
///
/// [`RequestError`] with the fault named first: `bad_source_json`, `bad_source_field`, `bad_key`,
/// `unknown_root` or `bad_claim`.
pub fn parse_source(raw: &[u8]) -> Result<FileRef, RequestError> {
    let document = json::parse(raw).map_err(|e| RequestError(format!("bad_source_json: {e}")))?;
    let Json::Object(fields) = document else {
        return Err(RequestError(
            "bad_source_json: the source must be a JSON object".into(),
        ));
    };
    if let Some((unknown, _)) = fields.iter().find(|(n, _)| !FIELDS.contains(&n.as_str())) {
        return Err(fault("bad_source_field", unknown));
    }
    if let Some(missing) = FIELDS.iter().find(|n| !fields.iter().any(|(f, _)| f == *n)) {
        return Err(fault("bad_source_field", missing));
    }
    let text = |name: &str| match field(&fields, name) {
        Json::Str(value) => Ok(value.as_str()),
        _ => Err(fault("bad_source_field", name)),
    };
    let key = text("key")?;
    if !matches!(field(&fields, "size"), Json::Int(_)) {
        return Err(fault("bad_source_field", "size"));
    }
    let hash = text("imohash")?;
    match validate_key(key) {
        Err(KeyError::Invalid { .. }) => return Err(fault("bad_key", key)),
        Err(_) => return Err(fault("unknown_root", key)),
        Ok(root) if !root.is_local() => return Err(fault("unknown_root", key)),
        Ok(_) => {}
    }
    let size = field(&fields, "size")
        .int_in(0, MAX_SIZE)
        .and_then(|s| u64::try_from(s).ok())
        .ok_or_else(|| fault("bad_claim", "size"))?;
    let imohash = ImoHash::from_hex(hash).ok_or_else(|| fault("bad_claim", "imohash"))?;
    FileRef::new(key, FileClaim { size, imohash }).map_err(|_| fault("bad_key", key))
}
