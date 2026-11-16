//! Every setting of the frame engine, read once from `OP_FRAME_ENGINE_*` (`env.template`): the
//! names, defaults, ranges and refusals of `frame_engine/settings.py`, which stays the reference.
//! A value that is set but malformed or out of range refuses the start, as there.
//!
//! What differs from the Python engine, each by plan 17:
//!
//! * `OP_FRAME_ENGINE_LANES` defaults to the physical cores available to the process
//!   (section 2.1; [`crate::cores`]), not 16.
//! * `OP_FRAME_ENGINE_SWITCH_INTERVAL_MS` means nothing here: a start with it set is refused
//!   (section 1.1).

use std::fmt;

use crate::cores;
use crate::pyfmt::{format_g, repr_str};

/// The text of a refusal, naming the variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsError(String);

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SettingsError {}

fn refuse<T>(text: String) -> Result<T, SettingsError> {
    Err(SettingsError(text))
}

/// `OP_FRAME_ENGINE_TRANSPORT`: `cuda` (the default) or `body`.
fn read_transport(env: &Env<'_>) -> Result<Transport, SettingsError> {
    match env.raw("OP_FRAME_ENGINE_TRANSPORT").as_deref() {
        None | Some("cuda") => Ok(Transport::Cuda),
        Some("body") => Ok(Transport::Body),
        Some(other) => refuse(format!(
            "OP_FRAME_ENGINE_TRANSPORT={}: one of cuda, body",
            repr_str(other)
        )),
    }
}

/// How the model inputs reach the model server (`OP_FRAME_ENGINE_TRANSPORT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// `cuda`: slices of a registered CUDA shared-memory region on the card.
    Cuda,
    /// `body`: copied to the host and sent in the request body (plan 17 section 1.1, the Python
    /// `BodyTransport`), for a model server that cannot import the region.
    Body,
}

/// Every setting of one engine process, read from `OP_FRAME_ENGINE_*`.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// `host:port` of the model server's gRPC endpoint, `OP_FRAME_ENGINE_TRITON_URL`.
    pub triton_url: String,
    /// How inputs reach the model server, `OP_FRAME_ENGINE_TRANSPORT`.
    pub transport: Transport,
    /// Size of the one CUDA shared-memory region, MiB.
    pub region_mib: u32,
    /// Names this process's region with the card: one engine slot per card by default.
    pub instance: String,
    /// The card's index as the model server numbers it, used when registering the region, `OP_FRAME_ENGINE_TRITON_DEVICE_ID`.
    pub triton_device_id: u32,
    /// gRPC channels to the model server, `OP_FRAME_ENGINE_TRITON_CHANNELS`.
    pub triton_channels: u32,
    /// Deadline of one model-server call, also capped by what is left of the request.
    pub triton_timeout_seconds: f64,
    /// Every request is answered by then (`DEADLINE_EXCEEDED` if the photo is still running).
    pub request_deadline_seconds: f64,
    /// Photos in work at once past admission, each holding a frame and region slices, `OP_FRAME_ENGINE_WORKERS`.
    pub workers: u32,
    /// Decode lanes, `OP_FRAME_ENGINE_LANES`; defaults to the physical cores available to the process.
    pub lanes: u32,
    /// Finish-pool threads that copy and JPEG-encode crops, `OP_FRAME_ENGINE_ENCODE_THREADS`.
    pub encode_threads: u32,
    /// Requests admitted at once (running or queued); above it: overload.
    pub max_in_flight: u32,
    /// Device memory the frame pool may hold, MiB, `OP_FRAME_ENGINE_POOL_MIB`.
    pub pool_mib: u32,
    /// Decoded-frame bytes photos may reserve together, MiB, `OP_FRAME_ENGINE_BUDGET_MIB`; at most the pool.
    pub budget_mib: u32,
    /// How long a photo waits for the frame budget before an overload, seconds, `OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS`.
    pub budget_wait_seconds: f64,
    /// How long a photo waits for room in the region before an overload, seconds, `OP_FRAME_ENGINE_SLAB_WAIT_SECONDS`.
    pub slab_wait_seconds: f64,
    /// Cached profile-to-sRGB colour tables on the card, `OP_FRAME_ENGINE_COLOUR_TABLES`.
    pub colour_tables: u32,
    /// Images below this many megapixels decode on the host, `OP_FRAME_ENGINE_MIN_MEGAPIXELS`.
    pub min_megapixels: f64,
    /// Images above this many megapixels are refused as `huge`, `OP_FRAME_ENGINE_MAX_MEGAPIXELS`.
    pub max_megapixels: f64,
    /// JPEG quality, 1 to 100, of the encoded crops, `OP_FRAME_ENGINE_JPEG_QUALITY`.
    pub jpeg_quality: u32,
    /// Long-side cap of the crop JPEGs in pixels (0: the crop's own size).
    pub crop_max_side: u32,
    /// Whether kernel stages wait for their stream so stage seconds are exact, `OP_FRAME_ENGINE_STAGE_SYNC`.
    pub stage_sync: bool,
    /// TCP port of the gRPC service, `OP_FRAME_ENGINE_PORT`.
    pub port: u16,
    /// TCP port of the metrics endpoint, `OP_FRAME_ENGINE_METRICS_PORT`.
    pub metrics_port: u16,
    /// Largest gRPC message in either direction, MiB, `OP_FRAME_ENGINE_MAX_MESSAGE_MIB`.
    pub max_message_mib: u32,
    /// Seconds in-flight photos get to finish on shutdown, `OP_FRAME_ENGINE_DRAIN_SECONDS`.
    pub drain_seconds: f64,
    /// Seconds between the engine's readiness checks of the model server, `OP_FRAME_ENGINE_HEALTH_SECONDS`.
    pub health_seconds: f64,
    /// `OP_FRAME_ENGINE_EXPECT_BUS_ID`: the PCI bus id of the card this engine must run on; the
    /// start is refused on any other (unset: no check).
    pub expect_bus_id: Option<String>,
}

/// What an unset variable means, apart from `lanes` (derived from the cores).
impl Default for Settings {
    fn default() -> Self {
        Self::with_lanes(cores::physical_cores())
    }
}

impl Settings {
    fn with_lanes(lanes: usize) -> Self {
        Self {
            triton_url: "triton-server:8001".into(),
            transport: Transport::Cuda,
            region_mib: 1024,
            instance: "fe0".into(),
            triton_device_id: 0,
            triton_channels: 4,
            triton_timeout_seconds: 15.0,
            request_deadline_seconds: 60.0,
            workers: 16,
            lanes: u32::try_from(lanes).unwrap_or(256).clamp(1, 256),
            encode_threads: 8,
            max_in_flight: 128,
            pool_mib: 4096,
            budget_mib: 2048,
            budget_wait_seconds: 4.0,
            slab_wait_seconds: 4.0,
            colour_tables: 2,
            min_megapixels: 0.0,
            max_megapixels: 100.0,
            jpeg_quality: 90,
            crop_max_side: 0,
            stage_sync: false,
            port: 8001,
            metrics_port: 8002,
            max_message_mib: 256,
            drain_seconds: 30.0,
            health_seconds: 2.0,
            expect_bus_id: None,
        }
    }

    /// The settings of this process's environment.
    ///
    /// # Errors
    ///
    /// A variable is malformed or out of range, a retired or unbuilt option is set, or the settings contradict each other (budget above pool, model-call timeout above the request deadline).
    pub fn from_env() -> Result<Self, SettingsError> {
        Self::from_lookup(&|name| std::env::var(name).ok())
    }

    /// The settings read through `lookup` (a variable's text, or `None` when unset).
    ///
    /// # Errors
    ///
    /// A variable is malformed or out of range, a retired or unbuilt option is set, or the settings contradict each other (budget above pool, model-call timeout above the request deadline).
    pub fn from_lookup(lookup: &dyn Fn(&str) -> Option<String>) -> Result<Self, SettingsError> {
        Self::read(lookup, cores::physical_cores())
    }

    fn read(
        lookup: &dyn Fn(&str) -> Option<String>,
        default_lanes: usize,
    ) -> Result<Self, SettingsError> {
        let env = Env(lookup);
        let d = Self::with_lanes(default_lanes);
        let (instance, expect_bus_id) = Self::read_identity(&env, d.instance.clone())?;
        let s = Self {
            triton_url: env
                .raw("OP_FRAME_ENGINE_TRITON_URL")
                .unwrap_or(d.triton_url),
            transport: read_transport(&env)?,
            region_mib: env.int("OP_FRAME_ENGINE_REGION_MIB", d.region_mib, 16, 16_384)?,
            instance,
            triton_device_id: env.int(
                "OP_FRAME_ENGINE_TRITON_DEVICE_ID",
                d.triton_device_id,
                0,
                63,
            )?,
            triton_channels: env.int(
                "OP_FRAME_ENGINE_TRITON_CHANNELS",
                d.triton_channels,
                1,
                64,
            )?,
            triton_timeout_seconds: env.float(
                "OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS",
                d.triton_timeout_seconds,
                1.0,
                600.0,
            )?,
            request_deadline_seconds: env.float(
                "OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS",
                d.request_deadline_seconds,
                1.0,
                600.0,
            )?,
            workers: env.int("OP_FRAME_ENGINE_WORKERS", d.workers, 1, 256)?,
            lanes: env.int("OP_FRAME_ENGINE_LANES", d.lanes, 1, 256)?,
            encode_threads: env.int("OP_FRAME_ENGINE_ENCODE_THREADS", d.encode_threads, 1, 256)?,
            max_in_flight: env.int("OP_FRAME_ENGINE_MAX_IN_FLIGHT", d.max_in_flight, 1, 100_000)?,
            pool_mib: env.int("OP_FRAME_ENGINE_POOL_MIB", d.pool_mib, 64, 262_144)?,
            budget_mib: env.int("OP_FRAME_ENGINE_BUDGET_MIB", d.budget_mib, 64, 262_144)?,
            budget_wait_seconds: env.float(
                "OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS",
                d.budget_wait_seconds,
                0.0,
                600.0,
            )?,
            slab_wait_seconds: env.float(
                "OP_FRAME_ENGINE_SLAB_WAIT_SECONDS",
                d.slab_wait_seconds,
                0.0,
                600.0,
            )?,
            colour_tables: env.int("OP_FRAME_ENGINE_COLOUR_TABLES", d.colour_tables, 1, 64)?,
            min_megapixels: env.float(
                "OP_FRAME_ENGINE_MIN_MEGAPIXELS",
                d.min_megapixels,
                0.0,
                10_000.0,
            )?,
            max_megapixels: env.float(
                "OP_FRAME_ENGINE_MAX_MEGAPIXELS",
                d.max_megapixels,
                0.0,
                10_000.0,
            )?,
            jpeg_quality: env.int("OP_FRAME_ENGINE_JPEG_QUALITY", d.jpeg_quality, 1, 100)?,
            crop_max_side: env.int("OP_FRAME_ENGINE_CROP_MAX_SIDE", d.crop_max_side, 0, 65_535)?,
            stage_sync: env.boolean("OP_FRAME_ENGINE_STAGE_SYNC", d.stage_sync)?,
            port: env.int("OP_FRAME_ENGINE_PORT", d.port, 1, 65_535)?,
            metrics_port: env.int("OP_FRAME_ENGINE_METRICS_PORT", d.metrics_port, 1, 65_535)?,
            max_message_mib: env.int(
                "OP_FRAME_ENGINE_MAX_MESSAGE_MIB",
                d.max_message_mib,
                1,
                2047,
            )?,
            drain_seconds: env.float(
                "OP_FRAME_ENGINE_DRAIN_SECONDS",
                d.drain_seconds,
                0.0,
                3600.0,
            )?,
            health_seconds: env.float(
                "OP_FRAME_ENGINE_HEALTH_SECONDS",
                d.health_seconds,
                0.1,
                600.0,
            )?,
            expect_bus_id,
        };
        s.check_relations()?;
        Ok(s)
    }

    /// The settings that name something rather than size it, and the ones this engine refuses
    /// because a feature is not built: the instance name and the expected bus id.
    fn read_identity(
        env: &Env<'_>,
        default_instance: String,
    ) -> Result<(String, Option<String>), SettingsError> {
        if env.raw("OP_FRAME_ENGINE_SWITCH_INTERVAL_MS").is_some() {
            return refuse(
                "OP_FRAME_ENGINE_SWITCH_INTERVAL_MS is set: it tunes Python's interpreter lock \
                 and means nothing to the native engine; unset it"
                    .into(),
            );
        }
        let instance = env
            .raw("OP_FRAME_ENGINE_INSTANCE")
            .unwrap_or(default_instance);
        let stripped: String = instance.replace('-', "");
        if stripped.is_empty() || !stripped.chars().all(char::is_alphanumeric) {
            return refuse(format!(
                "OP_FRAME_ENGINE_INSTANCE={}: letters, digits and dashes only",
                repr_str(&instance)
            ));
        }
        let expect_bus_id = env.raw("OP_FRAME_ENGINE_EXPECT_BUS_ID");
        if let Some(text) = &expect_bus_id
            && parse_bus_id(text).is_none()
        {
            return refuse(format!(
                "OP_FRAME_ENGINE_EXPECT_BUS_ID={}: a PCI bus id such as 0000:65:00.0",
                repr_str(text)
            ));
        }
        Ok((instance, expect_bus_id))
    }

    /// Settings that must agree with each other.
    fn check_relations(&self) -> Result<(), SettingsError> {
        let s = self;
        if s.budget_mib > s.pool_mib {
            return refuse(format!(
                "OP_FRAME_ENGINE_BUDGET_MIB {} exceeds OP_FRAME_ENGINE_POOL_MIB {}",
                s.budget_mib, s.pool_mib
            ));
        }
        if s.triton_timeout_seconds > s.request_deadline_seconds {
            return refuse(format!(
                "OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS {} exceeds \
                 OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS {}",
                format_g(s.triton_timeout_seconds),
                format_g(s.request_deadline_seconds)
            ));
        }
        Ok(())
    }
}

/// A PCI bus id (`domain:bus:device.function`, hex, as CUDA's `0000:65:00.0` and nvidia-smi's
/// `00000000:65:00.0` write it) as numbers, so the two spellings of one card compare equal.
pub fn parse_bus_id(text: &str) -> Option<(u32, u8, u8, u8)> {
    let (domain, rest) = text.trim().split_once(':')?;
    let (bus, rest) = rest.split_once(':')?;
    let (device, function) = rest.split_once('.')?;
    Some((
        u32::from_str_radix(domain, 16).ok()?,
        u8::from_str_radix(bus, 16).ok()?,
        u8::from_str_radix(device, 16).ok()?,
        u8::from_str_radix(function, 16).ok()?,
    ))
}

struct Env<'a>(&'a dyn Fn(&str) -> Option<String>);

impl Env<'_> {
    /// The variable's text without surrounding whitespace; unset and empty are the same.
    fn raw(&self, name: &str) -> Option<String> {
        let value = (self.0)(name)?;
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_owned())
    }

    fn int<T: TryFrom<i64>>(
        &self,
        name: &str,
        default: T,
        lo: i64,
        hi: i64,
    ) -> Result<T, SettingsError> {
        let Some(raw) = self.raw(name) else {
            return Ok(default);
        };
        let Some(value) = parse_py_int(&raw) else {
            return refuse(format!("{name}={}: not a number", repr_str(&raw)));
        };
        if value < i128::from(lo) || value > i128::from(hi) {
            return refuse(format!(
                "{name}={}: outside [{}, {}]",
                repr_str(&raw),
                format_g(lo as f64),
                format_g(hi as f64)
            ));
        }
        let value = i64::try_from(value).expect("a value within [lo, hi] fits i64");
        Ok(T::try_from(value).unwrap_or_else(|_| unreachable!("the range fits the field")))
    }

    fn float(&self, name: &str, default: f64, lo: f64, hi: f64) -> Result<f64, SettingsError> {
        let Some(raw) = self.raw(name) else {
            return Ok(default);
        };
        let Some(value) = parse_py_float(&raw) else {
            return refuse(format!("{name}={}: not a number", repr_str(&raw)));
        };
        // written as Python writes it: a NaN fails both comparisons and is refused
        if !(lo <= value && value <= hi) {
            return refuse(format!(
                "{name}={}: outside [{}, {}]",
                repr_str(&raw),
                format_g(lo),
                format_g(hi)
            ));
        }
        Ok(value)
    }

    fn boolean(&self, name: &str, default: bool) -> Result<bool, SettingsError> {
        let Some(raw) = self.raw(name) else {
            return Ok(default);
        };
        match raw.to_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => refuse(format!("{name}={}: not a boolean", repr_str(&raw))),
        }
    }
}

/// Python's rule for `_` in a number: only between two digits.
fn without_separators(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if *c == '_' {
            let before = i.checked_sub(1).and_then(|j| chars.get(j));
            let after = chars.get(i + 1);
            if !before.is_some_and(char::is_ascii_digit) || !after.is_some_and(char::is_ascii_digit)
            {
                return None;
            }
        }
    }
    Some(chars.into_iter().filter(|c| *c != '_').collect())
}

/// `int(text)` as Python reads a decimal integer: a sign, digits, single `_` between digits.
fn parse_py_int(text: &str) -> Option<i128> {
    let clean = without_separators(text)?;
    let digits = clean.strip_prefix(['+', '-']).unwrap_or(&clean);
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    // a value past i128 is far outside every range: report it as out of range, not as text
    Some(clean.parse::<i128>().unwrap_or(if clean.starts_with('-') {
        i128::MIN
    } else {
        i128::MAX
    }))
}

/// `float(text)`: Rust's grammar (`inf`, `nan`, exponents) plus Python's `_` between digits.
fn parse_py_float(text: &str) -> Option<f64> {
    without_separators(text)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn read(vars: &[(&str, &str)]) -> Result<Settings, SettingsError> {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        Settings::read(&|name| map.get(name).cloned(), 12)
    }

    fn refusal(vars: &[(&str, &str)]) -> String {
        read(vars).expect_err("the start is refused").to_string()
    }

    #[test]
    fn unset_variables_give_the_python_defaults_and_the_cores_for_lanes() {
        let s = read(&[]).expect("defaults");
        let want = Settings {
            lanes: 12,
            ..Settings::with_lanes(12)
        };
        assert_eq!(s, want);
        assert_eq!(
            (
                s.triton_url.as_str(),
                s.region_mib,
                s.instance.as_str(),
                s.workers
            ),
            ("triton-server:8001", 1024, "fe0", 16)
        );
        assert_eq!(
            (s.pool_mib, s.budget_mib, s.max_in_flight, s.encode_threads),
            (4096, 2048, 128, 8)
        );
        assert_eq!((s.budget_wait_seconds, s.slab_wait_seconds), (4.0, 4.0));
        assert_eq!(
            (s.triton_timeout_seconds, s.request_deadline_seconds),
            (15.0, 60.0)
        );
        assert_eq!(
            (s.port, s.metrics_port, s.max_message_mib),
            (8001, 8002, 256)
        );
        assert_eq!((s.drain_seconds, s.health_seconds), (30.0, 2.0));
        assert_eq!(
            (s.jpeg_quality, s.colour_tables, s.triton_channels),
            (90, 2, 4)
        );
        assert_eq!(
            (s.min_megapixels, s.max_megapixels, s.stage_sync),
            (0.0, 100.0, false)
        );
        // four budget waits (one try and the API's three retries) fit the old 20 s wait
        assert!(4.0 * s.budget_wait_seconds <= 20.0);
    }

    #[test]
    fn values_are_read_with_python_number_syntax() {
        let s = read(&[
            ("OP_FRAME_ENGINE_WORKERS", " 8 "),
            ("OP_FRAME_ENGINE_LANES", "+3"),
            ("OP_FRAME_ENGINE_MAX_IN_FLIGHT", "1_000"),
            ("OP_FRAME_ENGINE_STAGE_SYNC", "YES"),
            ("OP_FRAME_ENGINE_TRITON_URL", "triton-gpu2:8001"),
            ("OP_FRAME_ENGINE_COLOUR_TABLES", "5"),
            ("OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS", "1e1"),
            ("OP_FRAME_ENGINE_HEALTH_SECONDS", ""),
        ])
        .expect("values");
        assert_eq!((s.workers, s.lanes, s.max_in_flight), (8, 3, 1000));
        assert!(s.stage_sync);
        assert_eq!(
            (s.triton_url.as_str(), s.colour_tables),
            ("triton-gpu2:8001", 5)
        );
        assert_eq!(s.budget_wait_seconds, 10.0);
        assert_eq!(s.health_seconds, 2.0, "an empty variable is an unset one");
    }

    #[test]
    fn a_bad_value_refuses_the_start_and_names_the_variable() {
        let cases: [(&str, &str); 22] = [
            ("TRANSPORT", "system"),
            ("WORKERS", "0"),
            ("WORKERS", "many"),
            ("WORKERS", "1__0"),
            ("WORKERS", "_1"),
            ("WORKERS", "1.5"),
            ("LANES", "257"),
            ("REGION_MIB", "8"),
            ("REGION_MIB", "16385"),
            ("STAGE_SYNC", "maybe"),
            ("INSTANCE", "fe 0"),
            ("INSTANCE", "-"),
            ("BUDGET_MIB", "8192"),
            ("TRITON_TIMEOUT_SECONDS", "90"),
            ("REQUEST_DEADLINE_SECONDS", "0.5"),
            ("REQUEST_DEADLINE_SECONDS", "nan"),
            ("COLOUR_TABLES", "0"),
            ("COLOUR_TABLES", "65"),
            ("MAX_MESSAGE_MIB", "2048"),
            ("PORT", "65536"),
            ("HEALTH_SECONDS", "0.05"),
            ("MAX_MEGAPIXELS", "inf"),
        ];
        for (name, value) in cases {
            let full = format!("OP_FRAME_ENGINE_{name}");
            let text = refusal(&[(full.as_str(), value)]);
            assert!(text.contains(&full), "{name}={value}: {text}");
        }
    }

    #[test]
    fn every_range_of_the_python_settings_is_the_range_here() {
        // name, low, high: `frame_engine/settings.py` `from_env`
        let ranges: [(&str, i64, i64); 21] = [
            ("REGION_MIB", 16, 16_384),
            ("TRITON_DEVICE_ID", 0, 63),
            ("TRITON_CHANNELS", 1, 64),
            ("TRITON_TIMEOUT_SECONDS", 1, 60),
            ("REQUEST_DEADLINE_SECONDS", 1, 600),
            ("WORKERS", 1, 256),
            ("LANES", 1, 256),
            ("ENCODE_THREADS", 1, 256),
            ("MAX_IN_FLIGHT", 1, 100_000),
            ("POOL_MIB", 64, 262_144),
            ("BUDGET_MIB", 64, 4096),
            ("BUDGET_WAIT_SECONDS", 0, 600),
            ("SLAB_WAIT_SECONDS", 0, 600),
            ("COLOUR_TABLES", 1, 64),
            ("JPEG_QUALITY", 1, 100),
            ("CROP_MAX_SIDE", 0, 65_535),
            ("PORT", 1, 65_535),
            ("METRICS_PORT", 1, 65_535),
            ("MAX_MESSAGE_MIB", 1, 2047),
            ("DRAIN_SECONDS", 0, 3600),
            ("MAX_MEGAPIXELS", 0, 10_000),
        ];
        let companions = [
            ("OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS", "1"),
            ("OP_FRAME_ENGINE_BUDGET_MIB", "64"),
        ];
        for (name, lo, hi) in ranges {
            let full = format!("OP_FRAME_ENGINE_{name}");
            for ok in [lo, hi] {
                let value = ok.to_string();
                // a deadline below the call timeout and a pool below the budget are refused:
                // the companions come first so the setting under test wins
                assert!(
                    read(&[companions[0], companions[1], (full.as_str(), &value)]).is_ok(),
                    "{name}={value} is in range"
                );
            }
            for bad in [lo - 1, hi + 1] {
                let value = bad.to_string();
                assert!(
                    read(&[(full.as_str(), &value)]).is_err(),
                    "{name}={value} is out of range"
                );
            }
        }
        // the ranges whose ceiling is another setting's value are checked above at their own
        // ceiling (TRITON_TIMEOUT_SECONDS 60 = the request deadline, BUDGET_MIB 4096 = the pool)
        assert!(read(&[("OP_FRAME_ENGINE_MIN_MEGAPIXELS", "10000")]).is_ok());
        assert!(read(&[("OP_FRAME_ENGINE_MIN_MEGAPIXELS", "10001")]).is_err());
        assert!(read(&[("OP_FRAME_ENGINE_HEALTH_SECONDS", "0.1")]).is_ok());
        assert!(read(&[("OP_FRAME_ENGINE_HEALTH_SECONDS", "600.1")]).is_err());
    }

    #[test]
    fn a_refusal_reads_as_the_python_engines_does() {
        assert_eq!(
            refusal(&[("OP_FRAME_ENGINE_WORKERS", "0")]),
            "OP_FRAME_ENGINE_WORKERS='0': outside [1, 256]"
        );
        assert_eq!(
            refusal(&[("OP_FRAME_ENGINE_WORKERS", "many")]),
            "OP_FRAME_ENGINE_WORKERS='many': not a number"
        );
        assert_eq!(
            refusal(&[("OP_FRAME_ENGINE_BUDGET_MIB", "8192")]),
            "OP_FRAME_ENGINE_BUDGET_MIB 8192 exceeds OP_FRAME_ENGINE_POOL_MIB 4096"
        );
        assert_eq!(
            refusal(&[("OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS", "90")]),
            "OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS 90 exceeds \
             OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS 60"
        );
        assert_eq!(
            refusal(&[("OP_FRAME_ENGINE_STAGE_SYNC", "maybe")]),
            "OP_FRAME_ENGINE_STAGE_SYNC='maybe': not a boolean"
        );
    }

    #[test]
    fn the_expected_card_is_read_as_numbers_and_a_malformed_one_refuses_the_start() {
        assert_eq!(parse_bus_id("0000:65:00.0"), Some((0, 0x65, 0, 0)));
        assert_eq!(
            parse_bus_id("00000000:65:00.0"),
            parse_bus_id("0000:65:00.0")
        );
        assert_eq!(parse_bus_id("0000:B3:00.1"), Some((0, 0xb3, 0, 1)));
        assert_ne!(parse_bus_id("0000:17:00.0"), parse_bus_id("0000:65:00.0"));
        assert!(parse_bus_id("GPU-2").is_none());
        let s = read(&[("OP_FRAME_ENGINE_EXPECT_BUS_ID", "00000000:65:00.0")]).expect("read");
        assert_eq!(s.expect_bus_id.as_deref(), Some("00000000:65:00.0"));
        assert_eq!(read(&[]).expect("read").expect_bus_id, None);
        assert!(refusal(&[("OP_FRAME_ENGINE_EXPECT_BUS_ID", "card two")]).contains("PCI bus id"));
    }

    #[test]
    fn the_settings_the_native_engine_does_not_take_refuse_the_start() {
        let text = refusal(&[("OP_FRAME_ENGINE_SWITCH_INTERVAL_MS", "0.1")]);
        assert!(
            text.contains("OP_FRAME_ENGINE_SWITCH_INTERVAL_MS"),
            "{text}"
        );
        let text = refusal(&[("OP_FRAME_ENGINE_SWITCH_INTERVAL_MS", "0")]);
        assert!(
            text.contains("SWITCH_INTERVAL"),
            "even 0 is a setting that means nothing: {text}"
        );
        assert_eq!(
            read(&[("OP_FRAME_ENGINE_CROP_MAX_SIDE", "512")]).map(|s| s.crop_max_side),
            Ok(512)
        );
    }

    #[test]
    fn the_transport_is_cuda_by_default_and_body_when_asked() {
        assert_eq!(read(&[]).map(|s| s.transport), Ok(Transport::Cuda));
        assert_eq!(
            read(&[("OP_FRAME_ENGINE_TRANSPORT", "body")]).map(|s| s.transport),
            Ok(Transport::Body)
        );
        assert_eq!(
            refusal(&[("OP_FRAME_ENGINE_TRANSPORT", "shm")]),
            "OP_FRAME_ENGINE_TRANSPORT='shm': one of cuda, body"
        );
    }

    #[test]
    fn python_number_syntax_is_followed_exactly() {
        assert_eq!(parse_py_int("1_000"), Some(1000));
        assert_eq!(parse_py_int("-7"), Some(-7));
        assert_eq!(parse_py_int("1_"), None);
        assert_eq!(parse_py_int("1e3"), None);
        assert_eq!(parse_py_int("0x10"), None);
        assert_eq!(parse_py_float("1_0.5"), Some(10.5));
        assert_eq!(parse_py_float("1e-2"), Some(0.01));
        assert!(parse_py_float("nan").is_some_and(f64::is_nan));
        assert_eq!(parse_py_float("1._5"), None);
    }
}
