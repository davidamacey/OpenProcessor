//! Capture time: which clock of the file is the capture time, its UTC offset and its sub-seconds
//! (`_parse_time` and its helpers in `src/domain/capture_metadata.py`).

use serde_json::{Map, Value};

use super::{OffsetSource, TimePrecision, TimeSource};
use crate::pytext::{digit_value, strip};

const MIN_OFFSET_MIN: i64 = -12 * 60;
const MAX_OFFSET_MIN: i64 = 14 * 60;
const SECONDS_PER_DAY: i64 = 86_400;
/// Days from 0000-03-01, the origin of the civil-date algorithm, to 1970-01-01.
const DAYS_TO_UNIX_EPOCH: i64 = 719_468;
const QUARTER_HOUR_S: f64 = 15.0 * 60.0;

/// The clocks, in the order the capture time is taken from: the original time, else the digitised
/// time (`EXIF:CreateDate`), each with the tags of its own offset and sub-seconds.
const CLOCKS: [(TimeSource, &str, &str, &str); 2] = [
    (
        TimeSource::ExifOriginal,
        "EXIF:DateTimeOriginal",
        "EXIF:OffsetTimeOriginal",
        "EXIF:SubSecTimeOriginal",
    ),
    (
        TimeSource::ExifDigitized,
        "EXIF:CreateDate",
        "EXIF:OffsetTimeDigitized",
        "EXIF:SubSecTimeDigitized",
    ),
];

/// What the time tags say; the default is "no usable clock".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct Clock {
    pub(super) local: Option<String>,
    pub(super) utc: Option<String>,
    pub(super) offset_minutes: Option<i64>,
    pub(super) source: TimeSource,
    pub(super) offset_source: OffsetSource,
    pub(super) precision: Option<TimePrecision>,
    pub(super) malformed: bool,
}

/// A second of the proleptic Gregorian calendar between years 1 and 9999 (Python's `datetime`
/// range), as seconds since 1970-01-01 and read as wall-clock time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Moment(i64);

/// The digits after the decimal point, as written (one to nine characters). They stay characters
/// because Python slices the original text, whatever script the digits are in.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SubSec(String);

impl Moment {
    fn new(year: i64, month: i64, day: i64, hour: i64, minute: i64, second: i64) -> Option<Self> {
        let valid = (1..=9999).contains(&year)
            && (1..=12).contains(&month)
            && (1..=days_in_month(year, month)).contains(&day)
            && (0..24).contains(&hour)
            && (0..60).contains(&minute)
            && (0..60).contains(&second);
        valid.then(|| {
            let days = days_from_civil(year, month, day);
            Self(days * SECONDS_PER_DAY + hour * 3600 + minute * 60 + second)
        })
    }

    /// Python's `moment - timedelta(minutes=...)` raises `OverflowError` outside years 1 to 9999;
    /// here that is `None`, so the file keeps its other fields.
    fn minus_minutes(self, minutes: i64) -> Option<Self> {
        let moved = Self(self.0 - minutes * 60);
        let (year, ..) = civil(moved.0);
        (1..=9999).contains(&year).then_some(moved)
    }

    /// `YYYY-MM-DDTHH:MM:SS`, then `.` and the first six sub-second characters when there are
    /// any. The year is not zero-padded: Python's `strftime("%Y")` calls glibc's, which is not.
    fn iso(self, subsec: Option<&SubSec>) -> String {
        let (year, month, day, hour, minute, second) = civil(self.0);
        let base = format!("{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}");
        match subsec {
            Some(s) => format!("{base}.{}", s.0.chars().take(6).collect::<String>()),
            None => base,
        }
    }
}

const fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

const fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01: Howard Hinnant's `days_from_civil`
/// (<https://howardhinnant.github.io/date_algorithms.html#days_from_civil>).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * shifted_month + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - DAYS_TO_UNIX_EPOCH
}

/// The year, month, day, hour, minute and second of a second count (Hinnant's `civil_from_days`).
fn civil(seconds: i64) -> (i64, i64, i64, i64, i64, i64) {
    let z = seconds.div_euclid(SECONDS_PER_DAY) + DAYS_TO_UNIX_EPOCH;
    let rest = seconds.rem_euclid(SECONDS_PER_DAY);
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day, rest / 3600, rest % 3600 / 60, rest % 60)
}

/// A cursor over the characters of a stripped string, for the fixed shapes Python matches with
/// regular expressions whose `\d` is any Unicode decimal digit.
struct Scan {
    chars: Vec<char>,
    at: usize,
}

impl Scan {
    fn new(text: &str) -> Self {
        Self {
            chars: strip(text).chars().collect(),
            at: 0,
        }
    }

    /// Exactly `count` digits, as a number.
    fn digits(&mut self, count: usize) -> Option<i64> {
        let slice = self.chars.get(self.at..self.at + count)?;
        let mut value = 0;
        for c in slice {
            value = value * 10 + i64::from(digit_value(*c)?);
        }
        self.at += count;
        Some(value)
    }

    fn take(&mut self, allowed: &[char]) -> Option<char> {
        let c = *self.chars.get(self.at)?;
        allowed.contains(&c).then(|| {
            self.at += 1;
            c
        })
    }

    fn at_end(&self) -> bool {
        self.at == self.chars.len()
    }

    fn rest_is_digits(&self) -> bool {
        let rest = &self.chars[self.at..];
        !rest.is_empty() && rest.iter().all(|c| digit_value(*c).is_some())
    }
}

/// `YYYY:MM:DD` then `HH:MM:SS` after a space or a `T`.
fn local_moment(value: &Value) -> Option<Moment> {
    let mut scan = Scan::new(value.as_str()?);
    let year = scan.digits(4)?;
    scan.take(&[':'])?;
    let month = scan.digits(2)?;
    scan.take(&[':'])?;
    let day = scan.digits(2)?;
    scan.take(&[' ', 'T'])?;
    let hour = scan.digits(2)?;
    scan.take(&[':'])?;
    let minute = scan.digits(2)?;
    scan.take(&[':'])?;
    let second = scan.digits(2)?;
    scan.at_end().then_some(())?;
    Moment::new(year, month, day, hour, minute, second)
}

/// `+HH:MM` or `-HH:MM` within -12:00 to +14:00, as minutes.
fn offset_minutes(value: &Value) -> Option<i64> {
    let mut scan = Scan::new(value.as_str()?);
    let sign = scan.take(&['+', '-'])?;
    let hours = scan.digits(2)?;
    scan.take(&[':'])?;
    let minutes = scan.digits(2)?;
    scan.at_end().then_some(())?;
    if minutes > 59 {
        return None;
    }
    let total = if sign == '-' {
        -(hours * 60 + minutes)
    } else {
        hours * 60 + minutes
    };
    (MIN_OFFSET_MIN..=MAX_OFFSET_MIN)
        .contains(&total)
        .then_some(total)
}

/// One to nine digits, from a string or an integer.
fn subsecond(value: &Value) -> Option<SubSec> {
    let text = match value {
        Value::String(s) => strip(s).to_owned(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        _ => return None,
    };
    let count = text.chars().count();
    ((1..=9).contains(&count) && text.chars().all(|c| digit_value(c).is_some()))
        .then_some(SubSec(text))
}

/// The GPS date and time stamps (`YYYY:MM:DD`, `HH:MM:SS` with optional fraction) as UTC.
fn gps_utc(tags: &Map<String, Value>) -> Option<Moment> {
    let mut date = Scan::new(tags.get("EXIF:GPSDateStamp")?.as_str()?);
    let year = date.digits(4)?;
    date.take(&[':'])?;
    let month = date.digits(2)?;
    date.take(&[':'])?;
    let day = date.digits(2)?;
    date.at_end().then_some(())?;
    let mut time = Scan::new(tags.get("EXIF:GPSTimeStamp")?.as_str()?);
    let hour = time.digits(2)?;
    time.take(&[':'])?;
    let minute = time.digits(2)?;
    time.take(&[':'])?;
    let second = time.digits(2)?;
    if !time.at_end() {
        time.take(&['.'])?;
        time.rest_is_digits().then_some(())?;
    }
    Moment::new(year, month, day, hour, minute, second)
}

/// The offset implied by the camera clock and the GPS clock, rounded to a quarter hour (ties to
/// even, as Python's `round`), when it is a possible one.
fn gps_offset(local: Moment, gps: Moment) -> Option<i64> {
    let quarters = (local.0 - gps.0) as f64 / QUARTER_HOUR_S;
    let minutes = quarters.round_ties_even() as i64 * 15;
    (MIN_OFFSET_MIN..=MAX_OFFSET_MIN)
        .contains(&minutes)
        .then_some(minutes)
}

/// The capture time of the requested tags `tags`.
pub(super) fn parse_time(tags: &Map<String, Value>) -> Clock {
    let gps = gps_utc(tags);
    for (source, date_key, offset_key, subsec_key) in CLOCKS {
        let Some(raw) = tags.get(date_key) else {
            continue;
        };
        let Some(moment) = local_moment(raw) else {
            return Clock {
                malformed: true,
                ..Clock::default()
            };
        };
        let subsec = tags.get(subsec_key).and_then(subsecond);
        let mut offset = tags.get(offset_key).and_then(offset_minutes);
        let mut offset_source = if offset.is_some() {
            OffsetSource::ExifOffset
        } else {
            OffsetSource::None
        };
        if offset.is_none()
            && let Some(delta) = gps.and_then(|g| gps_offset(moment, g))
        {
            offset = Some(delta);
            offset_source = OffsetSource::GpsUtcDelta;
        }
        let utc = offset
            .and_then(|m| moment.minus_minutes(m))
            .map(|m| m.iso(subsec.as_ref()));
        return Clock {
            local: Some(moment.iso(subsec.as_ref())),
            utc,
            offset_minutes: offset,
            source,
            offset_source,
            precision: Some(if subsec.is_some() {
                TimePrecision::Subsecond
            } else {
                TimePrecision::Second
            }),
            malformed: false,
        };
    }
    gps.map_or_else(Clock::default, |g| Clock {
        utc: Some(g.iso(None)),
        source: TimeSource::GpsUtc,
        precision: Some(TimePrecision::Second),
        ..Clock::default()
    })
}
