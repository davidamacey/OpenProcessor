//! The GPS fix of a file (`_parse_location`) and the project's location policy
//! (`src/domain/location_policy.py`).

use std::fmt;
use std::str::FromStr;

use serde_json::{Map, Value};

use super::{CaptureLocation, CaptureMetadata, LocationPrecision};
use crate::pytext::{parse_float, round_to, strip};

/// Decimals kept by `store_coarse`: about one kilometre.
const COARSE_DECIMALS: usize = 2;

/// A number from a JSON number or a numeric string; not a boolean, not `NaN` or infinite.
pub(super) fn number(value: &Value) -> Option<f64> {
    let parsed = match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => parse_float(s),
        _ => None,
    };
    parsed.filter(|n| n.is_finite())
}

/// The fix in the requested tags: none when the status is void, a coordinate is missing or out of
/// range, or the position is exactly (0, 0).
pub(super) fn parse_location(tags: &Map<String, Value>) -> Option<CaptureLocation> {
    let void = tags
        .get("EXIF:GPSStatus")
        .and_then(Value::as_str)
        .is_some_and(|status| strip(status).eq_ignore_ascii_case("v"));
    if void {
        return None;
    }
    let latitude = tags.get("Composite:GPSLatitude").and_then(number)?;
    let longitude = tags.get("Composite:GPSLongitude").and_then(number)?;
    if !(-90.0..=90.0).contains(&latitude) || !(-180.0..=180.0).contains(&longitude) {
        return None;
    }
    if latitude == 0.0 && longitude == 0.0 {
        return None;
    }
    let optional = |key: &str| tags.get(key).and_then(number);
    Some(CaptureLocation {
        latitude,
        longitude,
        altitude_m: optional("Composite:GPSAltitude"),
        accuracy_m: optional("EXIF:GPSHPositioningError").filter(|a| *a >= 0.0),
        heading_deg: optional("EXIF:GPSImgDirection").filter(|h| (0.0..360.0).contains(h)),
        precision: LocationPrecision::Exact,
    })
}

/// How a project lets a photo's coordinates be stored and served.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocationPolicy {
    /// Keep no coordinate.
    Drop,
    /// Keep latitude and longitude rounded to 0.01 degree and nothing finer.
    StoreCoarse,
    /// Keep the fix as parsed (the default of a project).
    #[default]
    Store,
}

/// The text is not one of the three policies: an unknown policy fails closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLocationPolicy(String);

impl fmt::Display for UnknownLocationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "location_policy must be one of ('drop', 'store_coarse', 'store'), not {:?}",
            self.0
        )
    }
}

impl std::error::Error for UnknownLocationPolicy {}

impl FromStr for LocationPolicy {
    type Err = UnknownLocationPolicy;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "drop" => Ok(Self::Drop),
            "store_coarse" => Ok(Self::StoreCoarse),
            "store" => Ok(Self::Store),
            other => Err(UnknownLocationPolicy(other.to_owned())),
        }
    }
}

/// `location` rounded to two decimals, without altitude, accuracy or heading; `None` when the
/// rounding lands on (0, 0), which is no location.
#[must_use]
pub fn coarse_location(location: &CaptureLocation) -> Option<CaptureLocation> {
    let latitude = round_to(location.latitude, COARSE_DECIMALS);
    let longitude = round_to(location.longitude, COARSE_DECIMALS);
    if latitude == 0.0 && longitude == 0.0 {
        return None;
    }
    Some(CaptureLocation {
        latitude,
        longitude,
        altitude_m: None,
        accuracy_m: None,
        heading_deg: None,
        precision: LocationPrecision::Coarse,
    })
}

/// `meta` as the project's `policy` lets it be stored or served. `has_location` is not changed:
/// it says a valid fix was in the file, and the policy only decides whether the fix is kept.
#[must_use]
pub fn apply_location_policy(mut meta: CaptureMetadata, policy: LocationPolicy) -> CaptureMetadata {
    meta.location = match (policy, meta.location.take()) {
        (LocationPolicy::Store, kept) => kept,
        (LocationPolicy::Drop, _) | (_, None) => None,
        (LocationPolicy::StoreCoarse, Some(fix)) => coarse_location(&fix),
    };
    meta
}
