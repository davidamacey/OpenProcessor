//! The capture fields of one photo: when and with what it was taken, and where. A port of
//! `src/domain/capture_metadata.py` and `src/domain/location_policy.py`, held to them by the cases
//! of `contracts/json/capture_cases.json` (written by `scripts/codegen/capture_cases.py`, which
//! runs the Python functions over hand-written tag sets and over ExifTool outputs recorded from the
//! committed test images).
//!
//! [`parse_capture`] takes the tag map the ExifTool read of one file returns (`-j -G -n`: numeric
//! values, group-prefixed keys; see <https://exiftool.org/exiftool_pod.html>) and returns the
//! capture fields. A missing or garbled tag gives `None` for that field and never an error; a
//! malformed capture date sets [`ExifStatus::Malformed`].

mod clock;
mod location;

use serde_json::{Map, Value};

use crate::pytext::{float_repr, is_printable, strip};

pub use location::{LocationPolicy, UnknownLocationPolicy, apply_location_policy, coarse_location};

/// The parser stamp of the fields written today; a file stamped with another is read again.
pub const PARSER_VERSION: &str = "capture@1";

/// The tags a read asks ExifTool for, in the keys `-G` prints them under.
pub const EXIFTOOL_TAGS: [&str; 20] = [
    "EXIF:DateTimeOriginal",
    "EXIF:OffsetTimeOriginal",
    "EXIF:SubSecTimeOriginal",
    "EXIF:CreateDate",
    "EXIF:OffsetTimeDigitized",
    "EXIF:SubSecTimeDigitized",
    "EXIF:Make",
    "EXIF:Model",
    "EXIF:LensModel",
    "EXIF:FocalLength",
    "EXIF:FocalLengthIn35mmFormat",
    "EXIF:Orientation",
    "EXIF:GPSDateStamp",
    "EXIF:GPSTimeStamp",
    "EXIF:GPSStatus",
    "EXIF:GPSHPositioningError",
    "EXIF:GPSImgDirection",
    "Composite:GPSLatitude",
    "Composite:GPSLongitude",
    "Composite:GPSAltitude",
];

/// Set by ExifTool when it reports an error for the file itself (unreadable or corrupt).
pub const FILE_ERROR_KEY: &str = "ExifTool:Error";

/// The longest camera make, model or lens text kept, in characters.
const TEXT_MAX: usize = 128;

/// What the read found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExifStatus {
    /// Some capture tag was read.
    Ok,
    /// The file has no capture tag.
    Absent,
    /// The capture date is garbled, or ExifTool reported an error for the file.
    Malformed,
    /// The read itself failed (an ExifTool outage); nothing is known.
    Unknown,
}

/// Which clock the capture time comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeSource {
    /// `DateTimeOriginal`.
    ExifOriginal,
    /// `CreateDate`, the digitising time.
    ExifDigitized,
    /// The GPS UTC time, when the file has no camera clock.
    GpsUtc,
    /// No capture time.
    #[default]
    None,
}

/// Where the UTC offset comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffsetSource {
    /// `OffsetTimeOriginal` or `OffsetTimeDigitized`.
    ExifOffset,
    /// The camera clock against the GPS clock, rounded to 15 minutes.
    GpsUtcDelta,
    /// No offset.
    #[default]
    None,
}

/// How exact the capture time is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePrecision {
    /// The file has sub-second digits.
    Subsecond,
    /// Whole seconds.
    Second,
}

/// How exact a stored position is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocationPrecision {
    /// As the file has it.
    Exact,
    /// Rounded by the location policy.
    Coarse,
}

/// A valid GPS fix, signed decimal degrees (WGS 84).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureLocation {
    /// Degrees north, -90 to 90.
    pub latitude: f64,
    /// Degrees east, -180 to 180.
    pub longitude: f64,
    /// Metres above sea level.
    pub altitude_m: Option<f64>,
    /// Horizontal positioning error in metres, not negative.
    pub accuracy_m: Option<f64>,
    /// Direction the camera faced, 0 up to 360 degrees.
    pub heading_deg: Option<f64>,
    /// Whether the policy rounded it.
    pub precision: LocationPrecision,
}

/// Everything capture-related read from one file; every field may be absent.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureMetadata {
    /// What the read found.
    pub exif_status: ExifStatus,
    /// The parser stamp; `None` after an outage, so the capture is read again.
    pub parser: Option<&'static str>,
    /// ISO 8601 wall-clock time without a zone, with `.` and up to six sub-second digits.
    pub taken_local: Option<String>,
    /// The same shape, set only when an offset is known.
    pub taken_utc: Option<String>,
    /// Minutes east of UTC, -720 to 840.
    pub offset_minutes: Option<i64>,
    /// Which clock the time is from.
    pub time_source: TimeSource,
    /// Where the offset is from.
    pub offset_source: OffsetSource,
    /// Whole seconds or sub-seconds.
    pub time_precision: Option<TimePrecision>,
    /// Camera make.
    pub camera_make: Option<String>,
    /// Camera model.
    pub camera_model: Option<String>,
    /// Lens model.
    pub lens_model: Option<String>,
    /// Focal length in millimetres, above zero.
    pub focal_length_mm: Option<f64>,
    /// 35 mm equivalent focal length, above zero.
    pub focal_length_35mm: Option<f64>,
    /// EXIF orientation, 1 to 8.
    pub orientation: Option<u8>,
    /// The GPS fix, after the location policy if one was applied.
    pub location: Option<CaptureLocation>,
    /// A valid GPS fix was in the file; the location policy does not change it.
    pub has_location: bool,
}

/// The capture of a file whose read failed (an ExifTool outage): nothing known and no parser
/// stamp, so the capture is read again later.
#[must_use]
pub fn unknown_capture() -> CaptureMetadata {
    CaptureMetadata {
        parser: None,
        ..empty(ExifStatus::Unknown)
    }
}

fn empty(exif_status: ExifStatus) -> CaptureMetadata {
    CaptureMetadata {
        exif_status,
        parser: Some(PARSER_VERSION),
        taken_local: None,
        taken_utc: None,
        offset_minutes: None,
        time_source: TimeSource::None,
        offset_source: OffsetSource::None,
        time_precision: None,
        camera_make: None,
        camera_model: None,
        lens_model: None,
        focal_length_mm: None,
        focal_length_35mm: None,
        orientation: None,
        location: None,
        has_location: false,
    }
}

/// A printable, trimmed text of at most [`TEXT_MAX`] characters from a string or a number.
fn text(value: Option<&Value>) -> Option<String> {
    let raw = match value? {
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => float_repr(n.as_f64()?),
        _ => return None,
    };
    let printable: String = raw.chars().filter(|c| is_printable(*c)).collect();
    let cleaned: String = strip(&printable).chars().take(TEXT_MAX).collect();
    (!cleaned.is_empty()).then_some(cleaned)
}

fn positive(value: Option<&Value>) -> Option<f64> {
    value.and_then(location::number).filter(|n| *n > 0.0)
}

/// An orientation from 1 to 8; a value such as `6.0` or `"6"` counts, `6.5` does not.
fn orientation(value: Option<&Value>) -> Option<u8> {
    let number = value.and_then(location::number)?;
    (number.fract() == 0.0 && (1.0..=8.0).contains(&number)).then_some(number as u8)
}

/// The capture fields in `tags`, the tag map of one ExifTool read.
#[must_use]
pub fn parse_capture(tags: &Map<String, Value>) -> CaptureMetadata {
    let any_wanted = EXIFTOOL_TAGS.iter().any(|key| tags.contains_key(*key));
    let unreadable = tags.contains_key(FILE_ERROR_KEY);
    if !any_wanted {
        return empty(if unreadable {
            ExifStatus::Malformed
        } else {
            ExifStatus::Absent
        });
    }
    let when = clock::parse_time(tags);
    let location = location::parse_location(tags);
    CaptureMetadata {
        exif_status: if when.malformed || unreadable {
            ExifStatus::Malformed
        } else {
            ExifStatus::Ok
        },
        taken_local: when.local,
        taken_utc: when.utc,
        offset_minutes: when.offset_minutes,
        time_source: when.source,
        offset_source: when.offset_source,
        time_precision: when.precision,
        camera_make: text(tags.get("EXIF:Make")),
        camera_model: text(tags.get("EXIF:Model")),
        lens_model: text(tags.get("EXIF:LensModel")),
        focal_length_mm: positive(tags.get("EXIF:FocalLength")),
        focal_length_35mm: positive(tags.get("EXIF:FocalLengthIn35mmFormat")),
        orientation: orientation(tags.get("EXIF:Orientation")),
        has_location: location.is_some(),
        location,
        ..empty(ExifStatus::Ok)
    }
}
