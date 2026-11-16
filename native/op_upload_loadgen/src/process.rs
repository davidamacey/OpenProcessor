//! This process's own CPU, so a row shows whether the load generator itself was near its limit.
//!
//! Read from `/proc/self/stat` (<https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html>):
//! `utime` and `stime`, fields 14 and 15, in clock ticks. The tick is `USER_HZ`, 100 on every
//! Linux architecture the kernel's user ABI exports (`sysconf(_SC_CLK_TCK)`), so no libc call is
//! needed.

use std::fs;

use serde::Serialize;

const USER_HZ: f64 = 100.0;

/// Cumulative CPU seconds of this process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuSeconds {
    /// User time in clock ticks.
    pub utime: u64,
    /// System time in clock ticks.
    pub stime: u64,
}

/// Parse the text of `/proc/<pid>/stat`: the command name sits in parentheses and may hold spaces
/// and parentheses, so the fields are counted from the last `)`.
pub fn parse_stat(text: &str) -> Option<CpuSeconds> {
    let rest = text.get(text.rfind(')')? + 1..)?;
    // after the command, fields start at number 3 (state): utime is field 14, stime 15
    let mut fields = rest.split_whitespace();
    let utime = fields.nth(11)?.parse().ok()?;
    let stime = fields.next()?.parse().ok()?;
    Some(CpuSeconds { utime, stime })
}

/// This process's cumulative CPU now; `None` when `/proc/self/stat` cannot be read or parsed.
pub fn read() -> Option<CpuSeconds> {
    parse_stat(&fs::read_to_string("/proc/self/stat").ok()?)
}

/// This process's CPU over the window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ProcessWindow {
    /// User plus system CPU seconds used in the window.
    pub cpu_seconds: f64,
    /// Cores busy on average: `cpu_seconds` over the window's length.
    pub cores_busy: f64,
}

/// The CPU between two reads over `seconds`; `None` when a counter went backwards or the window
/// has no length.
pub fn window(before: CpuSeconds, after: CpuSeconds, seconds: f64) -> Option<ProcessWindow> {
    let ticks = after
        .utime
        .checked_sub(before.utime)?
        .checked_add(after.stime.checked_sub(before.stime)?)?;
    if seconds <= 0.0 {
        return None;
    }
    let cpu_seconds = ticks as f64 / USER_HZ;
    Some(ProcessWindow {
        cpu_seconds,
        cores_busy: cpu_seconds / seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_times_are_read_after_a_command_with_spaces_and_parentheses() {
        let text = "123 (op upload) (x)) S 1 2 3 4 5 6 7 8 9 10 250 50 0 0 20 0 8 0 100 200 300\n";
        assert_eq!(
            parse_stat(text),
            Some(CpuSeconds {
                utime: 250,
                stime: 50
            })
        );
        assert_eq!(parse_stat("garbage"), None);
        assert_eq!(parse_stat("1 (a) S 1 2"), None);
    }

    #[test]
    fn cores_busy_is_cpu_over_the_window() {
        let w = window(
            CpuSeconds {
                utime: 100,
                stime: 20,
            },
            CpuSeconds {
                utime: 500,
                stime: 120,
            },
            2.0,
        )
        .unwrap();
        assert_eq!((w.cpu_seconds, w.cores_busy), (5.0, 2.5));
        assert_eq!(
            window(
                CpuSeconds { utime: 5, stime: 0 },
                CpuSeconds { utime: 4, stime: 0 },
                1.0
            ),
            None
        );
    }
}
