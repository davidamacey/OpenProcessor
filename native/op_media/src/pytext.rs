//! The few text rules of the Python capture mapping (`src/domain/capture_metadata.py`) that the
//! Rust standard library spells differently: what `str.strip()` removes, which characters
//! `str.isprintable()` keeps, the decimal digits `\d` and `int()` accept, the strings `float()`
//! accepts, and `repr(float)`. Each is checked against Python through the generated capture cases.
//!
//! Python 3.12 answers from Unicode 15.0 and `unicode-general-category` 1.1.0 from Unicode 16.0, so
//! a character assigned only in version 16 is printable here and not there; no camera writes one.

use unicode_general_category::{GeneralCategory, get_general_category};

/// `str.strip()`: Unicode White_Space plus the four information separators U+001C to U+001F.
pub(crate) fn strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// `str.isprintable()` for one character: not a control, format, surrogate, private-use,
/// unassigned or separator character, except the ASCII space.
pub(crate) fn is_printable(c: char) -> bool {
    c == ' '
        || !matches!(
            get_general_category(c),
            GeneralCategory::Control
                | GeneralCategory::Format
                | GeneralCategory::Surrogate
                | GeneralCategory::PrivateUse
                | GeneralCategory::Unassigned
                | GeneralCategory::LineSeparator
                | GeneralCategory::ParagraphSeparator
                | GeneralCategory::SpaceSeparator
        )
}

/// The value of a decimal digit of any script (Unicode category Nd), as Python's `\d` matches it
/// and `int()` reads it. Every Nd range is a run of ten characters from zero to nine, so the value
/// is the distance from the start of the run of adjacent Nd characters, modulo ten.
pub(crate) fn digit_value(c: char) -> Option<u32> {
    if get_general_category(c) != GeneralCategory::DecimalNumber {
        return None;
    }
    let mut start = c as u32;
    while let Some(before) = char::from_u32(start - 1)
        && get_general_category(before) == GeneralCategory::DecimalNumber
    {
        start -= 1;
    }
    Some((c as u32 - start) % 10)
}

/// `float(text)`: whitespace is stripped, an underscore may sit between two digits, digits of
/// other scripts count, and `inf`, `infinity` and `nan` are numbers (callers then drop them as not
/// finite, as Python's does). `None` where Python raises `ValueError`.
pub(crate) fn parse_float(text: &str) -> Option<f64> {
    let stripped = strip(text);
    let mut ascii = String::with_capacity(stripped.len());
    let mut previous: Option<char> = None;
    let mut chars = stripped.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '_' {
            let after = chars.peek().copied();
            if !previous.is_some_and(|p| digit_value(p).is_some())
                || !after.is_some_and(|n| digit_value(n).is_some())
            {
                return None;
            }
        } else if let Some(d) = digit_value(c) {
            ascii.push(char::from_digit(d, 10)?);
        } else {
            ascii.push(c);
        }
        previous = Some(c);
    }
    ascii.parse().ok()
}

/// `repr(value)` of a Python float: the shortest text that reads back to the same number, in
/// positional notation for exponents from -4 to 15 and in `e` notation outside it.
pub(crate) fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.into();
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (-4..16).contains(&exponent) {
        return format!("{sign}{}", positional(&digits, exponent));
    }
    let tail = digits.get(1..).filter(|t| !t.is_empty());
    let head = &digits[..1];
    let fraction = tail.map_or(String::new(), |t| format!(".{t}"));
    format!(
        "{sign}{head}{fraction}e{}{:02}",
        exponent_sign(exponent),
        exponent.abs()
    )
}

fn exponent_sign(exponent: i32) -> char {
    if exponent < 0 { '-' } else { '+' }
}

fn positional(digits: &str, exponent: i32) -> String {
    let point = exponent + 1;
    if point <= 0 {
        return format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize));
    }
    let point = point as usize;
    if digits.len() <= point {
        format!("{digits}{}.0", "0".repeat(point - digits.len()))
    } else {
        format!("{}.{}", &digits[..point], &digits[point..])
    }
}

/// Python's `round(value, ndigits)`: the exactly rounded decimal, ties to even, read back.
pub(crate) fn round_to(value: f64, decimals: usize) -> f64 {
    format!("{value:.decimals$}").parse().unwrap_or(value)
}
