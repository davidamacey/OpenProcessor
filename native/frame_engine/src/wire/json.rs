//! The plan text as a JSON value, parsed as Python's `json.loads` does in the wire module's
//! `parse_plan`, plus the three refusals that module adds: `NaN`, `Infinity` and `-Infinity`
//! (no JSON, RFC 8259 section 6; `serde_json` refuses them) and a repeated key in one object
//! (RFC 8259 section 4 calls the receivers' behaviour unpredictable; Python keeps the last).
//!
//! Numbers keep their kind: an integer literal stays an integer however large (Python's `int`),
//! one with a fraction or an exponent is a float (`1e0` is `1.0`, so it is not the version `1`).
//! Objects keep their key order, which decides which of two faults is reported first.
//!
//! The reader is this module's own, over text already known to be UTF-8, because `serde_json`
//! hands a number to a visitor as `u64`, `i64` or `f64` unless its `arbitrary_precision`
//! feature is on, and that feature breaks internally tagged enums and `#[serde(flatten)]`
//! for every crate of the workspace (Cargo unions features across one build; the test in
//! `op_config/tests/workspace_features.rs` guards it). `serde_json` still decodes each string
//! literal and words the fault in a text the reader refuses, so the messages are as ever.
//!
//! Python also reads UTF-16 and UTF-32 text (`json.detect_encoding`) and lone surrogate escapes;
//! only UTF-8, with or without its byte order mark, is read here, and any other text is
//! `bad_json`. A client never sends them: the API writes `json.dumps` of ASCII.

use std::collections::HashSet;
use std::fmt;
use std::ops::Range;

use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};

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
    let mut reader = Reader {
        text,
        at: 0,
        depth: 0,
        overflows: Vec::new(),
    };
    match reader.document() {
        Ok(value) => Ok(value),
        Err(Stop::Duplicate(message)) => Err(JsonError(message)),
        Err(Stop::Invalid) => Err(JsonError(syntax_error(text, &reader.overflows))),
    }
}

/// `serde_json`'s words for the fault in a text the reader refused. Without `arbitrary_precision`
/// it reads a number into an `f64` and calls one that overflows a fault (`1e400` is `inf` to
/// Python and to the reader), which would hide the fault the reader found. So each overflowing
/// number the reader passed is overwritten by a `0` and spaces of the same length: no offset moves
/// and the words are those of the text as written.
fn syntax_error(text: &str, overflows: &[Range<usize>]) -> String {
    let mut bytes = text.as_bytes().to_vec();
    for span in overflows {
        bytes[span.clone()].fill(b' ');
        bytes[span.start] = b'0';
    }
    let text = String::from_utf8(bytes).unwrap_or_else(|_| text.to_owned());
    match serde_json::from_str::<Walk>(&text) {
        Err(e) => e.to_string(),
        Ok(Walk) => "the JSON reader and serde_json disagree about this text".into(),
    }
}

/// Walks a document and keeps nothing: it is `serde_json`'s reading of the grammar.
struct Walk;

impl<'de> Deserialize<'de> for Walk {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(Self)
    }
}

impl<'de> Visitor<'de> for Walk {
    type Value = Self;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, _: bool) -> Result<Self, E> {
        Ok(Self)
    }

    fn visit_i64<E>(self, _: i64) -> Result<Self, E> {
        Ok(Self)
    }

    fn visit_u64<E>(self, _: u64) -> Result<Self, E> {
        Ok(Self)
    }

    fn visit_f64<E>(self, _: f64) -> Result<Self, E> {
        Ok(Self)
    }

    fn visit_str<E>(self, _: &str) -> Result<Self, E> {
        Ok(Self)
    }

    fn visit_unit<E>(self) -> Result<Self, E> {
        Ok(Self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self, A::Error> {
        while seq.next_element::<Self>()?.is_some() {}
        Ok(Self)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self, A::Error> {
        while map.next_key::<String>()?.is_some() {
            map.next_value::<Self>()?;
        }
        Ok(Self)
    }
}

/// Why the reader stopped.
enum Stop {
    /// A key the object already has, with the message ready.
    Duplicate(String),
    /// The text breaks the grammar; `syntax_error` words it.
    Invalid,
}

/// `serde_json` reads 127 nested arrays or objects and refuses the 128th
/// (`recursion limit exceeded`); the reader stops at the same depth, so a text too deep is
/// worded by `serde_json` and a repeated key before it is reported first, as ever.
const MAX_DEPTH: usize = 127;

/// A reader of one JSON document over UTF-8 text. It accepts what `serde_json` accepts and
/// nothing else (RFC 8259 and no more: no comments, no trailing commas, no `NaN`), keeps number
/// text as written and objects in source order, and stops at the first repeated key.
struct Reader<'a> {
    text: &'a str,
    at: usize,
    depth: usize,
    /// Where the numbers that overflow `f64` stand, for `syntax_error`.
    overflows: Vec<Range<usize>>,
}

impl Reader<'_> {
    fn document(&mut self) -> Result<Json, Stop> {
        let value = self.value()?;
        self.skip_space();
        if self.at == self.text.len() {
            Ok(value)
        } else {
            Err(Stop::Invalid)
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\t' | b'\r')) {
            self.at += 1;
        }
    }

    /// Takes `byte` after any white space, or fails.
    fn expect(&mut self, byte: u8) -> Result<(), Stop> {
        self.skip_space();
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(Stop::Invalid)
        }
    }

    fn value(&mut self) -> Result<Json, Stop> {
        self.skip_space();
        match self.peek() {
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'n') => self.word("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(Stop::Invalid),
        }
    }

    fn nested(&mut self, read: fn(&mut Self) -> Result<Json, Stop>) -> Result<Json, Stop> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(Stop::Invalid);
        }
        let value = read(self);
        self.depth -= 1;
        value
    }

    fn word(&mut self, word: &str, value: Json) -> Result<Json, Stop> {
        if self.text[self.at..].starts_with(word) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(Stop::Invalid)
        }
    }

    fn array(&mut self) -> Result<Json, Stop> {
        self.at += 1;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::List(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::List(items));
                }
                _ => return Err(Stop::Invalid),
            }
        }
    }

    fn object(&mut self) -> Result<Json, Stop> {
        self.at += 1;
        let mut pairs: Vec<(String, Json)> = Vec::new();
        let mut seen = HashSet::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Object(pairs));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.skip_space();
            if !seen.insert(key.clone()) {
                return Err(Stop::Duplicate(self.duplicate_message(&key)));
            }
            self.expect(b':')?;
            pairs.push((key, self.value()?));
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Object(pairs));
                }
                _ => return Err(Stop::Invalid),
            }
        }
    }

    /// The words `serde_json` adds to a custom error, where its reader stands when the error
    /// comes out: past the key, the white space after it and, because it closes the object it was
    /// reading before it reports, a `}` that follows. The place is a line and the count of bytes
    /// since the last line break.
    fn duplicate_message(&self, key: &str) -> String {
        let read = self.at + usize::from(self.peek() == Some(b'}'));
        let before = &self.text.as_bytes()[..read];
        let line = 1 + before.iter().filter(|b| **b == b'\n').count();
        let column = before.iter().rev().take_while(|b| **b != b'\n').count();
        format!(
            "duplicate key {} at line {line} column {column}",
            repr_str(key)
        )
    }

    /// A string literal, decoded by `serde_json` (its escapes, surrogate pairs and refusals).
    fn string(&mut self) -> Result<String, Stop> {
        if self.peek() != Some(b'"') {
            return Err(Stop::Invalid);
        }
        let bytes = self.text.as_bytes();
        let mut end = self.at + 1;
        loop {
            match bytes.get(end) {
                None | Some(0..=0x1F) => return Err(Stop::Invalid),
                Some(b'"') => break,
                Some(b'\\') => end += 2,
                Some(_) => end += 1,
            }
        }
        let literal = &self.text[self.at..=end];
        let decoded = serde_json::from_str(literal).map_err(|_| Stop::Invalid)?;
        self.at = end + 1;
        Ok(decoded)
    }

    /// A number as RFC 8259 writes it: an integer stays its digits (any size), one with a
    /// fraction or an exponent is a float, and `-0` is the integer `0` as Python's is.
    fn number(&mut self) -> Result<Json, Stop> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            Some(b'0') => self.at += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(Stop::Invalid),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            self.at += 1;
            self.some_digits()?;
            is_float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            self.some_digits()?;
            is_float = true;
        }
        let text = &self.text[start..self.at];
        if is_float || text.len() > 308 {
            let value: f64 = text.parse().unwrap_or(f64::NAN);
            if value.is_infinite() {
                self.overflows.push(start..self.at);
            }
            if is_float {
                return Ok(Json::Float(value));
            }
        }
        Ok(if text == "-0" {
            Json::Int("0".into())
        } else {
            Json::Int(text.into())
        })
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.at += 1;
        }
    }

    fn some_digits(&mut self) -> Result<(), Stop> {
        let from = self.at;
        self.digits();
        if self.at == from {
            Err(Stop::Invalid)
        } else {
            Ok(())
        }
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

    /// Every row was produced by the reader as it stood before the `serde_json` features
    /// `arbitrary_precision` and `preserve_order` were dropped: the input bytes and the `Debug`
    /// text of what `parse` returned, error words included. The refusals that Python's
    /// `json.loads` does not make are named in the module comment.
    const EDGE_CASES: &[(&[u8], &str)] = &[
        (
            b"[18446744073709551616]",
            r##"Ok(List([Int("18446744073709551616")]))"##,
        ),
        (
            b"[-9223372036854775809]",
            r##"Ok(List([Int("-9223372036854775809")]))"##,
        ),
        (
            b"18446744073709551615",
            r##"Ok(Int("18446744073709551615"))"##,
        ),
        (
            b"-9223372036854775808",
            r##"Ok(Int("-9223372036854775808"))"##,
        ),
        (
            b"[123456789012345678901234567890123456789012345678901234567890]",
            r##"Ok(List([Int("123456789012345678901234567890123456789012345678901234567890")]))"##,
        ),
        (b"1e400", r##"Ok(Float(inf))"##),
        (b"-1e400", r##"Ok(Float(-inf))"##),
        (b"1e-400", r##"Ok(Float(0.0))"##),
        (b"-0", r##"Ok(Int("0"))"##),
        (b"-0.0", r##"Ok(Float(-0.0))"##),
        (b"-0e0", r##"Ok(Float(-0.0))"##),
        (b"1.0", r##"Ok(Float(1.0))"##),
        (b"1", r##"Ok(Int("1"))"##),
        (b"1E5", r##"Ok(Float(100000.0))"##),
        (b"1e+5", r##"Ok(Float(100000.0))"##),
        (b"1.5e-3", r##"Ok(Float(0.0015))"##),
        (b"0.1", r##"Ok(Float(0.1))"##),
        (
            b"01",
            r##"Err(JsonError("invalid number at line 1 column 2"))"##,
        ),
        (
            b"-01",
            r##"Err(JsonError("invalid number at line 1 column 3"))"##,
        ),
        (
            b"[00.5]",
            r##"Err(JsonError("invalid number at line 1 column 3"))"##,
        ),
        (
            b"+1",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (
            b".5",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (
            b"1.",
            r##"Err(JsonError("EOF while parsing a value at line 1 column 2"))"##,
        ),
        (
            b"-",
            r##"Err(JsonError("EOF while parsing a value at line 1 column 1"))"##,
        ),
        (
            b"1e",
            r##"Err(JsonError("EOF while parsing a value at line 1 column 2"))"##,
        ),
        (
            b"0x10",
            r##"Err(JsonError("trailing characters at line 1 column 2"))"##,
        ),
        (
            b"{\"a\": {\"b\": [1, {\"c\": 1,\n  \"d\": 2, \"c\": 3}]}}",
            r##"Err(JsonError("duplicate key 'c' at line 2 column 13"))"##,
        ),
        (
            b"{\"a\": 1, \"a\": 2}",
            r##"Err(JsonError("duplicate key 'a' at line 1 column 12"))"##,
        ),
        (
            b"{\"\xc3\xa9\": 1, \"\xc3\xa9\": 2}",
            r##"Err(JsonError("duplicate key 'é' at line 1 column 14"))"##,
        ),
        (
            b"{\"a\": 1, \"a\": 2}",
            r##"Err(JsonError("duplicate key 'a' at line 1 column 12"))"##,
        ),
        (
            b"{\"x\": {\"k\": 1, \"k\": 2}, \"y\": {\"z\": 1, \"z\": 2}}",
            r##"Err(JsonError("duplicate key 'k' at line 1 column 18"))"##,
        ),
        (
            b"{\"z\": 1, \"a\": 2, \"m\": 3}",
            r##"Ok(Object([("z", Int("1")), ("a", Int("2")), ("m", Int("3"))]))"##,
        ),
        (
            b"\"\\u00e9\\u20ac\\n\\t\\\"\\\\\\/\"",
            r##"Ok(Str("é€\n\t\"\\/"))"##,
        ),
        (b"\"\\ud83d\\ude00\"", r##"Ok(Str("😀"))"##),
        (
            b"\"\\ud83d\"",
            r##"Err(JsonError("unexpected end of hex escape at line 1 column 8"))"##,
        ),
        (
            b"\"\\ude00\"",
            r##"Err(JsonError("lone leading surrogate in hex escape at line 1 column 7"))"##,
        ),
        (
            b"\"\\ud83dx\"",
            r##"Err(JsonError("unexpected end of hex escape at line 1 column 8"))"##,
        ),
        (
            b"\"\\u12g4\"",
            r##"Err(JsonError("invalid escape at line 1 column 7"))"##,
        ),
        (
            b"\"\\x\"",
            r##"Err(JsonError("invalid escape at line 1 column 3"))"##,
        ),
        (
            b"\"\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80\"",
            r##"Ok(Str("é€😀"))"##,
        ),
        (b"{\"A\\n\": 1}", r##"Ok(Object([("A\n", Int("1"))]))"##),
        (
            b"[1,]",
            r##"Err(JsonError("trailing comma at line 1 column 4"))"##,
        ),
        (
            b"{\"a\": 1,}",
            r##"Err(JsonError("trailing comma at line 1 column 9"))"##,
        ),
        (
            b"[,1]",
            r##"Err(JsonError("expected value at line 1 column 2"))"##,
        ),
        (
            b"NaN",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (
            b"Infinity",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (
            b"-Infinity",
            r##"Err(JsonError("invalid number at line 1 column 2"))"##,
        ),
        (
            b"[NaN]",
            r##"Err(JsonError("expected value at line 1 column 2"))"##,
        ),
        (
            b"",
            r##"Err(JsonError("EOF while parsing a value at line 1 column 0"))"##,
        ),
        (
            b"  \n ",
            r##"Err(JsonError("EOF while parsing a value at line 2 column 1"))"##,
        ),
        (
            b"[1] x",
            r##"Err(JsonError("trailing characters at line 1 column 5"))"##,
        ),
        (
            b"{} {}",
            r##"Err(JsonError("trailing characters at line 1 column 4"))"##,
        ),
        (b" [1] \n\t\r", r##"Ok(List([Int("1")]))"##),
        (b"\xef\xbb\xbf[1]", r##"Ok(List([Int("1")]))"##),
        (
            b"\xef\xbb\xbf",
            r##"Err(JsonError("EOF while parsing a value at line 1 column 0"))"##,
        ),
        (
            b"\xef\xbb\xbf\xef\xbb\xbf[1]",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (
            b"[\"\xff\"]",
            r##"Err(JsonError("invalid utf-8 sequence of 1 bytes from index 2"))"##,
        ),
        (
            b"\"a\x01b\"",
            r##"Err(JsonError("control character (\\u0000-\\u001F) found while parsing a string at line 1 column 3"))"##,
        ),
        (
            b"\"a\nb\"",
            r##"Err(JsonError("control character (\\u0000-\\u001F) found while parsing a string at line 2 column 0"))"##,
        ),
        (
            b"\"abc",
            r##"Err(JsonError("EOF while parsing a string at line 1 column 4"))"##,
        ),
        (
            b"[1, 2",
            r##"Err(JsonError("EOF while parsing a list at line 1 column 5"))"##,
        ),
        (
            b"{\"a\": 1",
            r##"Err(JsonError("EOF while parsing an object at line 1 column 7"))"##,
        ),
        (
            b"{1: 2}",
            r##"Err(JsonError("key must be a string at line 1 column 2"))"##,
        ),
        (
            b"{'a': 1}",
            r##"Err(JsonError("key must be a string at line 1 column 2"))"##,
        ),
        (
            b"{\"a\" 1}",
            r##"Err(JsonError("expected `:` at line 1 column 6"))"##,
        ),
        (b"true", r##"Ok(Bool(true))"##),
        (
            b"tru",
            r##"Err(JsonError("EOF while parsing a value at line 1 column 3"))"##,
        ),
        (
            b"True",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (b" null ", r##"Ok(Null)"##),
        (b"[]", r##"Ok(List([]))"##),
        (b"{}", r##"Ok(Object([]))"##),
        (b"{\"\": 1}", r##"Ok(Object([("", Int("1"))]))"##),
        (
            b"[1,\n 2,\n  x]",
            r##"Err(JsonError("expected value at line 3 column 3"))"##,
        ),
        (
            b"\x0c[1]",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (
            b"\xc2\xa0[1]",
            r##"Err(JsonError("expected value at line 1 column 1"))"##,
        ),
        (
            b"[1x]",
            r##"Err(JsonError("expected `,` or `]` at line 1 column 3"))"##,
        ),
        (
            b"0.1000000000000000055511151231257827",
            r##"Ok(Float(0.1))"##,
        ),
        (b"1e99999999999999999999", r##"Ok(Float(inf))"##),
        (b"1e-99999999999999999999", r##"Ok(Float(0.0))"##),
        (
            b"{\"true\": true}",
            r##"Ok(Object([("true", Bool(true))]))"##,
        ),
        (
            b"[{\"a\": 1, \"a\": 2}, x]",
            r##"Err(JsonError("duplicate key 'a' at line 1 column 13"))"##,
        ),
        (
            b"{\"a\": 1, \"a\": 2} x",
            r##"Err(JsonError("duplicate key 'a' at line 1 column 12"))"##,
        ),
        (
            b"{\"a\": 1, \"a\": 2,}",
            r##"Err(JsonError("duplicate key 'a' at line 1 column 12"))"##,
        ),
        (
            b"[x, {\"a\": 1, \"a\": 2}]",
            r##"Err(JsonError("expected value at line 1 column 2"))"##,
        ),
        (
            b"{\"a\": 1, \"\\u0061\": 2}",
            r##"Err(JsonError("duplicate key 'a' at line 1 column 17"))"##,
        ),
        (
            b"{\n  \"k\": 1,\n\n    \"k\": [2]}",
            r##"Err(JsonError("duplicate key 'k' at line 4 column 7"))"##,
        ),
        (
            b"{\"\xf0\x9f\x98\x80\": 1,\n \"\xf0\x9f\x98\x80\": 2}",
            r##"Err(JsonError("duplicate key '😀' at line 2 column 7"))"##,
        ),
        (
            b"[{\"a\": 1}, {\"a\": 2}]",
            r##"Ok(List([Object([("a", Int("1"))]), Object([("a", Int("2"))])]))"##,
        ),
        (
            b"{\"a\": 1, \"a\": \"x\"}",
            r##"Err(JsonError("duplicate key 'a' at line 1 column 12"))"##,
        ),
        (
            b"{\"1\": 1, \"1.0\": 2}",
            r##"Ok(Object([("1", Int("1")), ("1.0", Int("2"))]))"##,
        ),
        (
            b"\"a\tb\"",
            r##"Err(JsonError("control character (\\u0000-\\u001F) found while parsing a string at line 1 column 3"))"##,
        ),
        (b"\"\\u0000\"", r##"Ok(Str("\0"))"##),
        (
            b"\"\\ud83d\\u0041\"",
            r##"Err(JsonError("lone leading surrogate in hex escape at line 1 column 13"))"##,
        ),
        (
            b"[1 2]",
            r##"Err(JsonError("expected `,` or `]` at line 1 column 4"))"##,
        ),
        (
            b"{\"a\": 1 \"b\": 2}",
            r##"Err(JsonError("expected `,` or `}` at line 1 column 9"))"##,
        ),
        (
            b"{\"a\": }",
            r##"Err(JsonError("expected value at line 1 column 7"))"##,
        ),
        (
            b"[1,,2]",
            r##"Err(JsonError("expected value at line 1 column 4"))"##,
        ),
        (
            b"{,}",
            r##"Err(JsonError("key must be a string at line 1 column 2"))"##,
        ),
        (
            b"[truex]",
            r##"Err(JsonError("expected `,` or `]` at line 1 column 6"))"##,
        ),
        (
            b"nul",
            r##"Err(JsonError("EOF while parsing a value at line 1 column 3"))"##,
        ),
        (
            b"[-Infinity]",
            r##"Err(JsonError("invalid number at line 1 column 3"))"##,
        ),
        (b"1E400", r##"Ok(Float(inf))"##),
        (
            b"[1e+0, 1E-0, 2e00]",
            r##"Ok(List([Float(1.0), Float(1.0), Float(2.0)]))"##,
        ),
        (
            b"[-0.5, 0.0, -1]",
            r##"Ok(List([Float(-0.5), Float(0.0), Int("-1")]))"##,
        ),
        (b"\xef\xbb\xbf  {}", r##"Ok(Object([]))"##),
        (
            b" {\t\"a\"\r:\n[ 1 , 2 ] } ",
            r##"Ok(Object([("a", List([Int("1"), Int("2")]))]))"##,
        ),
        (
            b"{\"a\": 1, \"A\": 2}",
            r##"Ok(Object([("a", Int("1")), ("A", Int("2"))]))"##,
        ),
        (
            br#"{"a": 1, "a" 2}"#,
            r##"Err(JsonError("duplicate key 'a' at line 1 column 13"))"##,
        ),
        (
            br#"{"a": 1, "a"}"#,
            r##"Err(JsonError("duplicate key 'a' at line 1 column 13"))"##,
        ),
        (
            br#"{"a": 1, "a"   : 2}"#,
            r##"Err(JsonError("duplicate key 'a' at line 1 column 15"))"##,
        ),
        (
            br#"{"a": 1, "a",}"#,
            r##"Err(JsonError("duplicate key 'a' at line 1 column 12"))"##,
        ),
        (
            b"{\"a\": 1, \"a\"\n\n}",
            r##"Err(JsonError("duplicate key 'a' at line 3 column 1"))"##,
        ),
        (
            b"[1e400, x]",
            r##"Err(JsonError("expected value at line 1 column 9"))"##,
        ),
        (
            br#"{"a": 1e400, "a": 1e400}"#,
            r##"Err(JsonError("duplicate key 'a' at line 1 column 16"))"##,
        ),
        (
            b"[1e400 x]",
            r##"Err(JsonError("expected `,` or `]` at line 1 column 8"))"##,
        ),
        (
            b"[1e400.5]",
            r##"Err(JsonError("expected `,` or `]` at line 1 column 7"))"##,
        ),
        (
            b"[-1e400-]",
            r##"Err(JsonError("expected `,` or `]` at line 1 column 8"))"##,
        ),
        (
            b"[2e400, 01]",
            r##"Err(JsonError("invalid number at line 1 column 10"))"##,
        ),
    ];

    #[test]
    fn the_edge_cases_read_as_they_always_did() {
        for (input, expected) in EDGE_CASES {
            let got = format!("{:?}", parse(input));
            assert_eq!(&got, expected, "input {:?}", String::from_utf8_lossy(input));
        }
    }

    #[test]
    fn nesting_is_read_to_a_depth_of_127_and_refused_beyond() {
        let nested = |depth: usize| [vec![b'['; depth], vec![b']'; depth]].concat();
        let mut value = ok("[]");
        for _ in 1..127 {
            value = Json::List(vec![value]);
        }
        assert_eq!(parse(&nested(127)), Ok(value));
        for (input, error) in [
            (nested(128), "recursion limit exceeded at line 1 column 128"),
            (nested(129), "recursion limit exceeded at line 1 column 128"),
            (
                vec![b'['; 200],
                "recursion limit exceeded at line 1 column 128",
            ),
        ] {
            assert_eq!(parse(&input), Err(JsonError(error.into())));
        }
        let objects = [br#"{"a":"#.repeat(130), b"1".to_vec(), b"}".repeat(130)].concat();
        assert_eq!(
            parse(&objects),
            Err(JsonError(
                "recursion limit exceeded at line 1 column 636".into()
            ))
        );
        let mut deep = vec![b'['; 126];
        deep.extend_from_slice(b"1e400");
        deep.extend_from_slice(&[b']'; 126]);
        assert!(parse(&deep).is_ok(), "an overflowing float at depth 126");
    }
}
