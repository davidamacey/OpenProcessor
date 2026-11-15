//! The measured engine's CPU from its cgroup (version 2): `cpu.stat` read at the window's two
//! edges. The path is given on the command line (the container's cgroup directory on the host,
//! for example `/sys/fs/cgroup/system.slice/docker-<id>.scope/cpu.stat`), so the row names
//! exactly which process tree was charged (https://docs.kernel.org/admin-guide/cgroup-v2.html,
//! "cpu.stat": `usage_usec`, `user_usec`, `system_usec` always; the throttling fields only with
//! the cpu controller; every duration in microseconds).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// One read of `cpu.stat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuStat {
    pub usage_usec: u64,
    pub user_usec: u64,
    pub system_usec: u64,
    pub nr_throttled: u64,
    pub throttled_usec: u64,
}

/// A `cpu.stat` that does not hold the three always-present fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatError(pub String);

impl std::fmt::Display for StatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StatError {}

/// Parse the flat `key value` lines; unknown keys are ignored, the throttling keys default to 0
/// when the cpu controller is off.
pub fn parse(text: &str) -> Result<CpuStat, StatError> {
    let mut stat = CpuStat::default();
    let mut seen = [false; 3];
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(key), Some(value)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Ok(value) = value.parse::<u64>() else {
            return Err(StatError(format!(
                "cpu.stat: {key} is not a number: {value:?}"
            )));
        };
        match key {
            "usage_usec" => (stat.usage_usec, seen[0]) = (value, true),
            "user_usec" => (stat.user_usec, seen[1]) = (value, true),
            "system_usec" => (stat.system_usec, seen[2]) = (value, true),
            "nr_throttled" => stat.nr_throttled = value,
            "throttled_usec" => stat.throttled_usec = value,
            _ => {}
        }
    }
    if seen.iter().all(|s| *s) {
        Ok(stat)
    } else {
        Err(StatError(
            "cpu.stat lacks usage_usec, user_usec or system_usec (not a cgroup v2 file?)".into(),
        ))
    }
}

/// The CPU an engine used inside a window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CpuWindow {
    pub usage_usec: u64,
    pub user_usec: u64,
    pub system_usec: u64,
    pub nr_throttled: u64,
    /// Cores busy on average: CPU time over the window's wall time.
    pub cores_busy: f64,
    /// CPU milliseconds per photo completed in the window.
    pub cpu_ms_per_photo: f64,
}

/// The window between two reads; `None` when a counter went backwards (the cgroup was recreated,
/// so the engine restarted inside the window).
pub fn window(before: &CpuStat, after: &CpuStat, seconds: f64, photos: usize) -> Option<CpuWindow> {
    let usage = after.usage_usec.checked_sub(before.usage_usec)?;
    let user = after.user_usec.checked_sub(before.user_usec)?;
    let system = after.system_usec.checked_sub(before.system_usec)?;
    let throttled = after.nr_throttled.checked_sub(before.nr_throttled)?;
    Some(CpuWindow {
        usage_usec: usage,
        user_usec: user,
        system_usec: system,
        nr_throttled: throttled,
        cores_busy: if seconds > 0.0 {
            usage as f64 / 1e6 / seconds
        } else {
            0.0
        },
        cpu_ms_per_photo: if photos > 0 {
            usage as f64 / 1e3 / photos as f64
        } else {
            0.0
        },
    })
}

/// A cgroup's `cpu.stat` file, and the memory files beside it.
#[derive(Debug, Clone)]
pub struct Cgroup {
    cpu_stat: PathBuf,
}

/// The cgroup's memory counters at the end of the window (bytes). `memory.peak` is the maximum
/// since the cgroup was created, so the hold records a fresh container per hold or says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MemoryRead {
    pub current_bytes: Option<u64>,
    pub peak_bytes: Option<u64>,
}

impl Cgroup {
    pub fn new(cpu_stat: PathBuf) -> Cgroup {
        Cgroup { cpu_stat }
    }

    pub fn path(&self) -> &Path {
        &self.cpu_stat
    }

    pub fn read_cpu(&self) -> io::Result<CpuStat> {
        parse(&fs::read_to_string(&self.cpu_stat)?)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn read_memory(&self) -> MemoryRead {
        let beside = |name: &str| {
            let dir = self.cpu_stat.parent()?;
            fs::read_to_string(dir.join(name)).ok()?.trim().parse().ok()
        };
        MemoryRead {
            current_bytes: beside("memory.current"),
            peak_bytes: beside("memory.peak"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WITH_CONTROLLER: &str = "usage_usec 9000000\nuser_usec 6000000\nsystem_usec 3000000\n\
        nr_periods 10\nnr_throttled 2\nthrottled_usec 700\nnr_bursts 0\nburst_usec 0\n";

    #[test]
    fn the_three_always_present_fields_and_the_throttling_ones_are_read() {
        let s = parse(WITH_CONTROLLER).unwrap();
        assert_eq!(
            s,
            CpuStat {
                usage_usec: 9_000_000,
                user_usec: 6_000_000,
                system_usec: 3_000_000,
                nr_throttled: 2,
                throttled_usec: 700
            }
        );
    }

    #[test]
    fn a_file_without_the_controller_still_parses() {
        let s = parse("usage_usec 5\nuser_usec 3\nsystem_usec 2\n").unwrap();
        assert_eq!((s.usage_usec, s.nr_throttled), (5, 0));
    }

    #[test]
    fn a_file_that_is_not_cpu_stat_is_refused() {
        assert!(parse("").is_err());
        assert!(parse("user_usec 1\nsystem_usec 1\n").is_err());
        assert!(parse("usage_usec x\nuser_usec 1\nsystem_usec 1\n").is_err());
    }

    #[test]
    fn a_window_charges_cpu_time_to_photos_and_to_wall_time() {
        let a = CpuStat {
            usage_usec: 1_000_000,
            user_usec: 800_000,
            system_usec: 200_000,
            nr_throttled: 1,
            throttled_usec: 0,
        };
        let b = CpuStat {
            usage_usec: 11_000_000,
            user_usec: 8_000_000,
            system_usec: 3_000_000,
            nr_throttled: 4,
            throttled_usec: 0,
        };
        // 10 CPU seconds in 20 s of wall time and 500 photos: half a core, 20 ms a photo
        let w = window(&a, &b, 20.0, 500).unwrap();
        assert!((w.cores_busy - 0.5).abs() < 1e-9);
        assert!((w.cpu_ms_per_photo - 20.0).abs() < 1e-9);
        assert_eq!(
            (w.user_usec, w.system_usec, w.nr_throttled),
            (7_200_000, 2_800_000, 3)
        );
        assert_eq!(
            window(&b, &a, 20.0, 500),
            None,
            "a recreated cgroup is not a window"
        );
    }

    #[test]
    fn memory_files_beside_the_stat_file_are_read_when_present() {
        let dir = std::env::temp_dir().join(format!("op-fe-loadgen-cg-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cpu.stat"), WITH_CONTROLLER).unwrap();
        fs::write(dir.join("memory.current"), "123\n").unwrap();
        let cg = Cgroup::new(dir.join("cpu.stat"));
        assert_eq!(cg.read_cpu().unwrap().usage_usec, 9_000_000);
        let m = cg.read_memory();
        assert_eq!((m.current_bytes, m.peak_bytes), (Some(123), None));
        fs::remove_dir_all(&dir).unwrap();
    }
}
