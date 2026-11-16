//! The plan text as a JSON value, parsed as Python's `json.loads` does in the wire module's
//! `parse_plan`, plus the three refusals that module adds: `NaN`, `Infinity` and `-Infinity`
//! (no JSON, RFC 8259 section 6; `serde_json` refuses them) and a repeated key in one object
//! (RFC 8259 section 4 calls the receivers' behaviour unpredictable; Python keeps the last).
//!
//! Numbers keep their kind: an integer literal stays an integer however large (Python's `int`),
//! one with a fraction or an exponent is a float (`1e0` is `1.0`, so it is not the version `1`).
//! Objects keep their key order, which decides which of two faults is reported first.
//!
//! Python also reads UTF-16 and UTF-32 text (`json.detect_encoding`) and lone surrogate escapes;
//! only UTF-8, with or without its byte order mark, is read here, and any other text is
//! `bad_json`. A client never sends them: the API writes `json.dumps` of ASCII.

use std::collections::HashSet;
use std::fmt;

use serde::de::{DeserializeSeed, Deserializer, Error as _, MapAccess, SeqAccess, Visitor};

use crate::pyfmt::{repr_float, repr_str};

/// A parsed plan document; integers keep their digits so Python's `repr` can be reproduced.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// JSON `null`.
    Null,
    /// JSON `true` or `false`.
    Bool(bool),
    /// The integer's digits (a sign when negative), as Python's `repr` writes it.
    Int(String),
    /// A JSON number written with a fraction or exponent.
    Float(f64),
    /// A JSON string.
    Str(String),
    /// A JSON array.
    List(Vec<Json>),
    /// A JSON object as key and value pairs in document order; keys are unique.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Python's `repr` of the value, as the plan faults print it (`bad_version: '1'`).
    pub fn repr(&self) -> String {
        match self {
            Self::Null => "None".into(),
            Self::Bool(true) => "True".into(),
            Self::Bool(false) => "False".into(),
            Self::Int(digits) => digits.clone(),
            Self::Float(x) => repr_float(*x),
            Self::Str(s) => repr_str(s),
            Self::List(items) => {
                let inner: Vec<String> = items.iter().map(Self::repr).collect();
                format!("[{}]", inner.join(", "))
            }
            Self::Object(pairs) => {
                let inner: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| format!("{}: {}", repr_str(k), v.repr()))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
        }
    }

    /// Python's `str` of the value: a string as it is, anything else as its `repr`.
    pub fn text(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            other => other.repr(),
        }
    }

    /// The integer in `[lo, hi]`, never a bool or a float.
    pub fn int_in(&self, lo: i64, hi: i64) -> Option<i64> {
        let Self::Int(digits) = self else {
            return None;
        };
        let value: i64 = digits.parse().ok()?;
        (lo..=hi).contains(&value).then_some(value)
    }

    /// An integer or a float, as a float, in `[lo, hi]`.
    pub fn number_in(&self, lo: f64, hi: f64) -> Option<f64> {
        let value = match self {
            Self::Int(digits) => digits.parse::<f64>().ok()?,
            Self::Float(x) => *x,
            _ => return None,
        };
        (lo <= value && value <= hi).then_some(value)
    }
}

/// A text that is no JSON document the engine accepts; the words are `serde_json`'s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError(pub String);

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Reads the document in `bytes`.
///
/// # Errors
///
/// The bytes are not UTF-8, are not valid JSON, or an object repeats a key.
pub fn parse(bytes: &[u8]) -> Result<Json, JsonError> {
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let text = std::str::from_utf8(bytes).map_err(|e| JsonError(e.to_string()))?;
    let mut checked = serde_json::Deserializer::from_str(text);
    Unique
        .deserialize(&mut checked)
        .map_err(|e| JsonError(e.to_string()))?;
    checked.end().map_err(|e| JsonError(e.to_string()))?;
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| JsonError(e.to_string()))?;
    Ok(convert(&value))
}

fn convert(value: &serde_json::Value) -> Json {
    use serde_json::Value;
    match value {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Number(n) => {
            let text = n.to_string();
            if text.contains(['.', 'e', 'E']) {
                Json::Float(text.parse().unwrap_or(f64::NAN))
            } else if text == "-0" {
                Json::Int("0".into())
            } else {
                Json::Int(text)
            }
        }
        Value::String(s) => Json::Str(s.clone()),
        Value::Array(items) => Json::List(items.iter().map(convert).collect()),
        Value::Object(map) => {
            Json::Object(map.iter().map(|(k, v)| (k.clone(), convert(v))).collect())
        }
    }
}

/// Walks a document and fails on a repeated key. With `arbitrary_precision` a number arrives as
/// a one-entry map; that is a map without a repeat, so it passes.
struct Unique;

impl<'de> DeserializeSeed<'de> for Unique {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Unique {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, _: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, _: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E>(self, _: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while seq.next_element_seed(Unique)?.is_some() {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom(format!(
                    "duplicate key {}",
                    repr_str(&key)
                )));
            }
            map.next_value_seed(Unique)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(text: &str) -> Json {
        parse(text.as_bytes()).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    #[test]
    fn numbers_keep_their_kind_and_their_size() {
        assert_eq!(ok("1"), Json::Int("1".into()));
        assert_eq!(ok("-0"), Json::Int("0".into()));
        assert_eq!(ok("1.0"), Json::Float(1.0));
        assert_eq!(ok("1e0"), Json::Float(1.0));
        assert_eq!(ok("1E+2"), Json::Float(100.0));
        assert_eq!(
            ok("123456789012345678901234567890"),
            Json::Int("123456789012345678901234567890".into())
        );
        assert_eq!(ok("true"), Json::Bool(true));
        assert_eq!(ok("1").int_in(1, 1), Some(1));
        assert_eq!(ok("true").int_in(0, 5), None);
        assert_eq!(ok("1.0").int_in(0, 5), None);
        assert_eq!(ok("99999999999999999999").int_in(0, i64::MAX), None);
        assert_eq!(ok("1").number_in(0.0, 1.0), Some(1.0));
        assert_eq!(ok("1.5").number_in(0.0, 1.0), None);
        assert_eq!(ok("false").number_in(0.0, 1.0), None);
    }

    #[test]
    fn objects_keep_the_order_they_were_written_in() {
        let Json::Object(pairs) = ok(r#"{"b": 1, "a": 2, "c": {"z": null}}"#) else {
            panic!("an object");
        };
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c"]);
    }

    #[test]
    fn what_python_refuses_or_the_wire_module_refuses_is_refused() {
        for text in [
            "",
            "{",
            "NaN",
            "Infinity",
            "-Infinity",
            r#"{"a": NaN}"#,
            r#"{"a": 1, "a": 1}"#,
            r#"{"a": 1, "b": {"c": 1, "c": 2}}"#,
            r#"[{"a": 1, "a": 2}]"#,
            "{\"a\": \"\u{1}\"}",
            "1 2",
            "[1,]",
            "01",
        ] {
            assert!(parse(text.as_bytes()).is_err(), "{text:?} is refused");
        }
        assert!(parse(&[0xFF, 0xFE]).is_err(), "bytes that are no UTF-8");
        assert!(
            parse(br#"{"a": {"a": 1}, "b": {"a": 2}}"#).is_ok(),
            "one key in two objects"
        );
    }

    #[test]
    fn a_byte_order_mark_is_skipped_as_python_does_for_bytes() {
        let mut text = vec![0xEF, 0xBB, 0xBF];
        text.extend_from_slice(b"[1]");
        assert_eq!(parse(&text), Ok(Json::List(vec![Json::Int("1".into())])));
    }

    #[test]
    fn repr_is_python_repr() {
        assert_eq!(ok("null").repr(), "None");
        assert_eq!(ok("true").repr(), "True");
        assert_eq!(ok("1e0").repr(), "1.0");
        assert_eq!(ok(r#""1""#).repr(), "'1'");
        assert_eq!(
            ok(r#"{"a": [1, "x", null]}"#).repr(),
            "{'a': [1, 'x', None]}"
        );
        assert_eq!(ok(r#""x""#).text(), "x");
        assert_eq!(ok("3").text(), "3");
    }
}
