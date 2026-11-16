//! Facts about the host a row was measured on, read from `/proc`: the 1-minute load at the
//! window's edges, the CPU model and its physical cores, the clock. Pure parsers take text so
//! tests need no particular machine.

use std::collections::BTreeSet;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// The 1-minute load average from the text of `/proc/loadavg`.
pub fn parse_load1(text: &str) -> Option<f64> {
    text.split_whitespace().next()?.parse().ok()
}

/// The 1-minute load average now; `None` when `/proc/loadavg` cannot be read or parsed.
pub fn load1() -> Option<f64> {
    parse_load1(&fs::read_to_string("/proc/loadavg").ok()?)
}

/// The CPU the host has.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CpuInfo {
    /// The CPU model name of the first logical CPU.
    pub model: String,
    /// Logical CPUs (hardware threads) the host has.
    pub logical_cpus: usize,
    /// Distinct (package, core) pairs: what the plan calls the physical cores.
    pub physical_cores: usize,
}

/// `/proc/cpuinfo` text: blocks separated by blank lines, one per logical CPU.
pub fn parse_cpuinfo(text: &str) -> CpuInfo {
    let mut model = String::new();
    let mut logical = 0usize;
    let mut cores = BTreeSet::new();
    for block in text.split("\n\n") {
        let field = |key: &str| {
            block.lines().find_map(|l| {
                let (k, v) = l.split_once(':')?;
                (k.trim() == key).then(|| v.trim().to_string())
            })
        };
        let Some(processor) = field("processor") else {
            continue;
        };
        let _ = processor;
        logical += 1;
        if model.is_empty() {
            model = field("model name").unwrap_or_default();
        }
        if let (Some(package), Some(core)) = (field("physical id"), field("core id")) {
            cores.insert((package, core));
        }
    }
    CpuInfo {
        model,
        logical_cpus: logical,
        physical_cores: if cores.is_empty() {
            logical
        } else {
            cores.len()
        },
    }
}

/// The host's CPU, read from `/proc/cpuinfo`; empty fields when it cannot be read.
pub fn cpuinfo() -> CpuInfo {
    parse_cpuinfo(&fs::read_to_string("/proc/cpuinfo").unwrap_or_default())
}

/// The host name; empty when it cannot be read.
pub fn hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Milliseconds since the Unix epoch.
pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// `2026-11-15T13:04:05.123Z` for a Unix time in milliseconds (the civil-from-days algorithm of
/// <https://howardhinnant.github.io/date_algorithms.html>, valid for all dates after 1970).
pub fn iso_utc(ms: u64) -> String {
    let secs = ms / 1000;
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        ms % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_is_the_first_field() {
        assert_eq!(parse_load1("3.25 4.10 5.00 2/900 12345\n"), Some(3.25));
        assert_eq!(parse_load1(""), None);
        assert_eq!(parse_load1("x"), None);
    }

    #[test]
    fn physical_cores_are_distinct_package_and_core_pairs() {
        // two packages of two cores, each core with two hyperthreads: 8 logical, 4 physical
        let mut text = String::new();
        let mut n = 0;
        for package in 0..2 {
            for core in 0..2 {
                for _ in 0..2 {
                    text.push_str(&format!(
                        "processor\t: {n}\nmodel name\t: Test CPU @ 2.5GHz\nphysical id\t: {package}\ncore id\t\t: {core}\n\n"
                    ));
                    n += 1;
                }
            }
        }
        let info = parse_cpuinfo(&text);
        assert_eq!(info.model, "Test CPU @ 2.5GHz");
        assert_eq!((info.logical_cpus, info.physical_cores), (8, 4));
    }

    #[test]
    fn a_cpuinfo_without_topology_counts_logical_cpus() {
        let info =
            parse_cpuinfo("processor : 0\nmodel name : X\n\nprocessor : 1\nmodel name : X\n");
        assert_eq!((info.logical_cpus, info.physical_cores), (2, 2));
    }

    #[test]
    fn times_are_rendered_in_utc() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00.000Z");
        // 2026-11-15 12:30:45.678 UTC: 20_407 days after the epoch
        assert_eq!(iso_utc(1_794_745_845_678), "2026-11-15T12:30:45.678Z");
        // a leap day
        assert_eq!(iso_utc(1_709_164_800_000), "2024-02-29T00:00:00.000Z");
    }
}
