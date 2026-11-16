//! JSON text as Python's `json.dumps(value, sort_keys=True)` writes it: `", "` and `": "`
//! separators, ASCII only (`ensure_ascii`), floats as `repr` spells them. The map is sorted
//! because `serde_json` is built without `preserve_order` (a `BTreeMap`); this writer only fixes
//! the spelling, through the documented `serde_json::ser::Formatter` hooks
//! (<https://docs.rs/serde_json/1.0.154/serde_json/ser/trait.Formatter.html>).
//!
//! The engine parses the plan, it does not hash or cache by its text, so the spelling only has to
//! read back to the same plan; matching the Python client byte for byte keeps one spelling on
//! the wire and makes a captured request comparable with a Python one.

use std::fmt::Write as _;
use std::io;

use serde::Serialize;
use serde_json::Serializer;
use serde_json::ser::Formatter;

/// Python's separators, escapes and float spelling.
struct PythonFormat;

impl Formatter for PythonFormat {
    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(float_repr(value).as_bytes())
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }

    /// The runs `serde_json` did not escape: printable ASCII stays, everything else (DEL and
    /// every non-ASCII character, as UTF-16 code units) is `\uXXXX`.
    fn write_string_fragment<W>(&mut self, writer: &mut W, fragment: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        let mut text = String::with_capacity(fragment.len());
        for c in fragment.chars() {
            if (' '..='~').contains(&c) {
                text.push(c);
            } else {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    // writing to a String cannot fail
                    let _ = write!(text, "\\u{unit:04x}");
                }
            }
        }
        writer.write_all(text.as_bytes())
    }
}

/// `value` as Python's `json.dumps(value, sort_keys=True)` writes it (the map keys are sorted
/// by the value's own `BTreeMap`).
///
/// # Errors
///
/// Returns the `serde_json` error when `value` cannot be serialised (a map key that is not a
/// string, a non-finite float is written as `null` by `serde_json` and never reaches here).
pub fn dumps<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let mut out = Vec::new();
    value.serialize(&mut Serializer::with_formatter(&mut out, PythonFormat))?;
    // serde_json writes UTF-8 only, and this formatter writes ASCII
    String::from_utf8(out)
        .map_err(|e| <serde_json::Error as serde::ser::Error>::custom(format!("not UTF-8: {e}")))
}

/// `value` as `d.ddde-5` with the fewest digits that read back to `value`, and of those the one
/// nearest to `value`, ties to the even digit, as Python's `repr` (David Gay's `dtoa`, mode 0)
/// picks it. Rust's `{:e}` finds the length; its fixed-precision form is correctly rounded, so
/// reformatting at that length settles a tie the way Python does (the shortest-digit search alone
/// rounds a tie up: `704758667114124.25` is `...124.3` there and `...124.2` in Python).
fn shortest_scientific(value: f64) -> String {
    let shortest = format!("{value:e}");
    let digits = shortest.split_once('e').map_or(&shortest[..], |(m, _)| m);
    let places = digits
        .chars()
        .filter(char::is_ascii_digit)
        .count()
        .saturating_sub(1);
    format!("{value:.places$e}")
}

/// Python's `repr(float)`: the shortest digits that read back to the same value; positional
/// notation from `1e-4` up to below `1e16`, `d.ddde+XX` outside, and `.0` on a whole number.
#[must_use]
pub fn float_repr(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }
    let sign = if value < 0.0 { "-" } else { "" };
    let scientific = shortest_scientific(value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    // value = 0.DIGITS * 10^point
    let point = exponent + 1;
    if point <= -4 || point > 16 {
        let (head, tail) = digits.split_at(1);
        let fraction = if tail.is_empty() {
            String::new()
        } else {
            format!(".{tail}")
        };
        let sign_e = if exponent < 0 { '-' } else { '+' };
        return format!("{sign}{head}{fraction}e{sign_e}{:02}", exponent.abs());
    }
    let len = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let body = if point <= 0 {
        format!(
            "0.{}{digits}",
            "0".repeat(usize::try_from(-point).unwrap_or(0))
        )
    } else if point >= len {
        format!(
            "{digits}{}.0",
            "0".repeat(usize::try_from(point - len).unwrap_or(0))
        )
    } else {
        let (whole, fraction) = digits.split_at(usize::try_from(point).unwrap_or(0));
        format!("{whole}.{fraction}")
    };
    format!("{sign}{body}")
}
