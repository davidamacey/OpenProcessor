//! Python's text for values, so a message the native engine sends or logs reads as the Python
//! engine's did: `repr()` of a string and the `g` format of a float (Python's format
//! specification mini-language: <https://docs.python.org/3/library/string.html#formatspec>).

/// `repr(s)` for a `str`: single quotes unless the text holds a single and no double quote.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() => {
                let n = u32::from(c);
                if n <= 0xff {
                    out.push_str(&format!("\\x{n:02x}"));
                } else if n <= 0xffff {
                    out.push_str(&format!("\\u{n:04x}"));
                } else {
                    out.push_str(&format!("\\U{n:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr(x)` for a `float`: the shortest text that reads back as `x`, in Python's layout
/// (`1.0`, `0.0001`, `1e-05`, `1e+16`, `inf`, `nan`).
pub fn repr_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's `{:e}` is the shortest round-trip digits: "d.ddde<exp>".
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        format!("{sign}{}", place_point(&digits, exp, true))
    } else {
        let head = &digits[..1];
        let tail = &digits[1..];
        let frac = if tail.is_empty() {
            String::new()
        } else {
            format!(".{tail}")
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{head}{frac}e{esign}{:02}", exp.abs())
    }
}

/// `format(x, 'g')`: six significant digits, trailing zeros dropped, exponent form when the
/// exponent is below -4 or at least 6.
pub fn format_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let sci = format!("{x:.5e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    if (-4..6).contains(&exp) {
        format!("{sign}{}", place_point(digits, exp, false))
    } else {
        let frac = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{}{frac}e{esign}{:02}", &digits[..1], exp.abs())
    }
}

/// `digits` read as `d.ddd` times ten to `exp`, written out without an exponent; `keep_point`
/// ends a whole number in `.0` (a float's `repr`).
fn place_point(digits: &str, exp: i32, keep_point: bool) -> String {
    let len = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    if exp < 0 {
        let zeros = "0".repeat(usize::try_from(-exp - 1).unwrap_or(0));
        format!("0.{zeros}{digits}")
    } else if exp + 1 >= len {
        let zeros = "0".repeat(usize::try_from(exp + 1 - len).unwrap_or(0));
        let dot = if keep_point { ".0" } else { "" };
        format!("{digits}{zeros}{dot}")
    } else {
        let at = usize::try_from(exp + 1).unwrap_or(0);
        format!("{}.{}", &digits[..at], &digits[at..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_quoted_as_python_does() {
        assert_eq!(repr_str("abc"), "'abc'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a\"b'c"), "'a\"b\\'c'");
        assert_eq!(repr_str("a\\b\n\t\u{1}"), "'a\\\\b\\n\\t\\x01'");
        assert_eq!(repr_str("caf\u{e9}"), "'caf\u{e9}'");
        assert_eq!(repr_str(""), "''");
    }

    #[test]
    fn floats_are_written_as_python_repr_does() {
        let cases = [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (-2.5, "-2.5"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (123456.789, "123456.789"),
            (0.1, "0.1"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (-0.0, "-0.0"),
        ];
        for (x, want) in cases {
            assert_eq!(repr_float(x), want, "{x:e}");
        }
        assert_eq!(repr_float(f64::NAN), "nan");
    }

    #[test]
    fn the_g_format_keeps_six_digits_and_drops_trailing_zeros() {
        let cases = [
            (60.0, "60"),
            (0.3, "0.3"),
            (16384.0, "16384"),
            (262_144.0, "262144"),
            (100_000.0, "100000"),
            (1_000_000.0, "1e+06"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (2.5, "2.5"),
            (0.0, "0"),
        ];
        for (x, want) in cases {
            assert_eq!(format_g(x), want, "{x:e}");
        }
    }
}
