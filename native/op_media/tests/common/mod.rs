//! Shared by the capture tests: the JSON form of the capture fields, the shape Python's
//! `dataclasses.asdict` gives them in `contracts/json/capture_cases.json`.

use op_media::capture::{
    CaptureLocation, CaptureMetadata, ExifStatus, LocationPrecision, OffsetSource, TimePrecision,
    TimeSource,
};
use serde_json::{Value, json};

fn float(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |v| json!(v))
}

fn location(fix: &CaptureLocation) -> Value {
    json!({
        "latitude": fix.latitude,
        "longitude": fix.longitude,
        "altitude_m": float(fix.altitude_m),
        "accuracy_m": float(fix.accuracy_m),
        "heading_deg": float(fix.heading_deg),
        "precision": match fix.precision {
            LocationPrecision::Exact => "exact",
            LocationPrecision::Coarse => "coarse",
        },
    })
}

/// The capture fields as the cases file writes them.
pub fn capture_json(meta: &CaptureMetadata) -> Value {
    json!({
        "exif_status": match meta.exif_status {
            ExifStatus::Ok => "ok",
            ExifStatus::Absent => "absent",
            ExifStatus::Malformed => "malformed",
            ExifStatus::Unknown => "unknown",
        },
        "parser": meta.parser,
        "taken_local": meta.taken_local,
        "taken_utc": meta.taken_utc,
        "offset_minutes": meta.offset_minutes,
        "time_source": match meta.time_source {
            TimeSource::ExifOriginal => "exif_original",
            TimeSource::ExifDigitized => "exif_digitized",
            TimeSource::GpsUtc => "gps_utc",
            TimeSource::None => "none",
        },
        "offset_source": match meta.offset_source {
            OffsetSource::ExifOffset => "exif_offset",
            OffsetSource::GpsUtcDelta => "gps_utc_delta",
            OffsetSource::None => "none",
        },
        "time_precision": meta.time_precision.map(|p| match p {
            TimePrecision::Subsecond => "subsecond",
            TimePrecision::Second => "second",
        }),
        "camera_make": meta.camera_make,
        "camera_model": meta.camera_model,
        "lens_model": meta.lens_model,
        "focal_length_mm": float(meta.focal_length_mm),
        "focal_length_35mm": float(meta.focal_length_35mm),
        "orientation": meta.orientation,
        "location": meta.location.as_ref().map_or(Value::Null, location),
        "has_location": meta.has_location,
    })
}
