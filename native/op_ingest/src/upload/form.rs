//! The pure rules of the upload form, ported from `src/routers/curation/ingest_upload.py` and
//! `src/services/curation/upload_ingest.py`: which identifier a photo gets, which extension it is
//! judged by, and the per-photo refusals with their messages.

use serde_json::Value;
use unicode_general_category::{GeneralCategory, get_general_category};

use super::refusal::Refusal;

/// `ERROR_KIND_EMPTY` in `src/services/curation/ingest_models.py`.
pub const ERROR_KIND_EMPTY: &str = "empty";
/// `ERROR_KIND_UNSUPPORTED_TYPE` in `src/services/curation/ingest_models.py`.
pub const ERROR_KIND_UNSUPPORTED_TYPE: &str = "unsupported_type";

/// Python's `str.strip()` strips whitespace as `str.isspace` knows it: Unicode white space and
/// the four information separators.
fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Whether `raw` is what Python's route treats as no `image_paths` at all.
#[must_use]
pub fn is_blank(raw: &str) -> bool {
    raw.chars().all(is_python_space)
}

/// The identifiers of `image_paths`, one per photo (`_parse_upload_paths`).
///
/// # Errors
///
/// [`Refusal::PathsNotJson`] or [`Refusal::PathsNotList`], 422 with Python's messages.
pub fn parse_image_paths(raw: &str) -> Result<Vec<String>, Refusal> {
    let value: Value =
        serde_json::from_str(raw).map_err(|e| Refusal::PathsNotJson(e.to_string()))?;
    let Value::Array(items) = value else {
        return Err(Refusal::PathsNotList);
    };
    items
        .into_iter()
        .map(|item| match item {
            Value::String(s) if !s.is_empty() => Ok(s),
            _ => Err(Refusal::PathsNotList),
        })
        .collect()
}

/// The identifier of photo `index` when `image_paths` was not sent: its file name, or
/// `upload_<index>` when it has none.
#[must_use]
pub fn default_identifier(filename: Option<&str>, index: usize) -> String {
    match filename {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => format!("upload_{index}"),
    }
}

/// `Path(candidate).suffix.lower()` for a POSIX path: the last component after `.` and empty
/// components are dropped, and a leading or trailing dot makes no suffix.
#[must_use]
pub fn path_suffix(candidate: &str) -> String {
    let name = candidate
        .split('/')
        .rfind(|part| !part.is_empty() && *part != ".")
        .unwrap_or("");
    match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => name[i..].to_lowercase(),
        _ => String::new(),
    }
}

/// The extension a photo is stored and judged by: the identifier's suffix, else the part's file
/// name's, else `.jpg` (`_upload_extension`).
#[must_use]
pub fn upload_extension(identifier: &str, filename: Option<&str>) -> String {
    [identifier, filename.unwrap_or("")]
        .into_iter()
        .map(path_suffix)
        .find(|suffix| !suffix.is_empty())
        .unwrap_or_else(|| ".jpg".to_string())
}

/// Whether `extension` is one of the accepted ones.
#[must_use]
pub fn is_accepted(extension: &str, accepted: &[String]) -> bool {
    accepted.iter().any(|a| a == extension)
}

/// The message of a photo refused for its extension.
#[must_use]
pub fn unsupported_message(extension: &str, accepted: &[String]) -> String {
    format!(
        "{} is not an accepted extension ({})",
        python_repr(extension),
        accepted.join(", ")
    )
}

/// Python's `repr` of a string, as `{ext!r}` writes it into the message.
#[must_use]
pub fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c => push_escape(&mut out, c),
        }
    }
    out.push(quote);
    out
}

/// `str.isprintable` for one character: not in the categories Cc, Cf, Cs, Co, Cn, Zl, Zp or Zs,
/// except the plain space.
fn is_printable(c: char) -> bool {
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

fn push_escape(out: &mut String, c: char) {
    let code = u32::from(c);
    let escaped = if code <= 0xff {
        format!("\\x{code:02x}")
    } else if code <= 0xffff {
        format!("\\u{code:04x}")
    } else {
        format!("\\U{code:08x}")
    };
    out.push_str(&escaped);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_follows_pathlib() {
        assert_eq!(path_suffix("a/b.JPG"), ".jpg");
        assert_eq!(path_suffix(".jpg"), "");
        assert_eq!(path_suffix("a."), "");
        assert_eq!(path_suffix("dir.gif/"), ".gif");
        assert_eq!(path_suffix("dir.gif/."), ".gif");
        assert_eq!(path_suffix("a.tar.gif"), ".gif");
        assert_eq!(path_suffix(".."), "");
        assert_eq!(path_suffix(""), "");
        assert_eq!(path_suffix("c:\\pics\\a.gif"), ".gif");
    }

    #[test]
    fn the_extension_falls_back_from_the_identifier_to_the_file_name_to_jpg() {
        assert_eq!(upload_extension("x/a.webp", Some("a.jpg")), ".webp");
        assert_eq!(upload_extension("x/a", Some("a.gif")), ".gif");
        assert_eq!(upload_extension("x/a", None), ".jpg");
    }

    #[test]
    fn repr_picks_quotes_and_escapes_like_python() {
        assert_eq!(python_repr(".gif"), "'.gif'");
        assert_eq!(python_repr(".j'pg"), "\".j'pg\"");
        assert_eq!(python_repr(".j'p\"g"), "'.j\\'p\"g'");
        assert_eq!(python_repr(".\u{1}"), "'.\\x01'");
        assert_eq!(python_repr(".\u{a0}x"), "'.\\xa0x'");
        assert_eq!(python_repr(".\u{e9}\u{4e2d}"), "'.\u{e9}\u{4e2d}'");
        assert_eq!(python_repr(".\u{1f600}"), "'.\u{1f600}'");
        assert_eq!(python_repr("\\"), "'\\\\'");
    }

    #[test]
    fn blank_follows_python_strip() {
        assert!(is_blank(""));
        assert!(is_blank(" \t\n\u{2003}\u{1f}"));
        assert!(!is_blank(" a "));
    }
}
