//! The physical cores this process may use: the default of `OP_FRAME_ENGINE_LANES` (plan 17
//! section 2.1). Decode lanes scale with physical cores and fall off past them (the throwaway
//! pipeline's knee sat at the 24 physical cores of its host), so hyperthreads are not counted.
//!
//! A core is a distinct (package, core) pair among the CPUs of the process's affinity mask
//! (`Cpus_allowed_list` of `/proc/self/status`, proc(5)); the count is capped by what
//! `std::thread::available_parallelism` reports, which honours the same mask and a cgroup
//! `cpu.max` quota (std documentation, "Platform-specific behavior").

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// The CPU numbers of a kernel list such as `0-3,8,10-11` (cpuset(7) "List format"); `None` for
/// text that is not one.
pub fn parse_cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in text.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi): (usize, usize) = (lo.trim().parse().ok()?, hi.trim().parse().ok()?);
                if hi < lo {
                    return None;
                }
                out.extend(lo..=hi);
            }
            None => out.push(part.trim().parse().ok()?),
        }
    }
    Some(out)
}

/// The number of distinct (package, core) pairs among `cpus`, from the sysfs topology below
/// `sys_cpu` (`/sys/devices/system/cpu`); `None` when no listed CPU has a readable topology.
pub fn count_physical(sys_cpu: &Path, cpus: &[usize]) -> Option<usize> {
    let read = |cpu: usize, file: &str| -> Option<String> {
        let path = sys_cpu.join(format!("cpu{cpu}/topology/{file}"));
        fs::read_to_string(path).ok().map(|t| t.trim().to_owned())
    };
    let pairs: BTreeSet<(String, String)> = cpus
        .iter()
        .filter_map(|&cpu| Some((read(cpu, "physical_package_id")?, read(cpu, "core_id")?)))
        .collect();
    (!pairs.is_empty()).then_some(pairs.len())
}

/// The CPU list of the affinity mask in the text of `/proc/self/status`.
fn allowed_cpus(status: &str) -> Option<Vec<usize>> {
    let line = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?;
    parse_cpu_list(line)
}

/// Physical cores available to this process, at least one.
pub fn physical_cores() -> usize {
    let usable = std::thread::available_parallelism().map_or(1, usize::from);
    let counted = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| allowed_cpus(&s))
        .and_then(|cpus| count_physical(Path::new("/sys/devices/system/cpu"), &cpus));
    counted
        .map_or(usable, |physical| physical.min(usable))
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_are_read_as_the_kernel_writes_them() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list("3-1"), None);
        assert_eq!(parse_cpu_list("a"), None);
    }

    fn topology(root: &Path, cpu: usize, package: &str, core: &str) {
        let dir = root.join(format!("cpu{cpu}/topology"));
        fs::create_dir_all(&dir).expect("create topology dir");
        fs::write(dir.join("physical_package_id"), format!("{package}\n")).expect("write package");
        fs::write(dir.join("core_id"), format!("{core}\n")).expect("write core");
    }

    #[test]
    fn hyperthread_siblings_and_other_packages_are_counted_as_the_hardware_has_them() {
        let dir = std::env::temp_dir().join(format!("fe-cores-{}", std::process::id()));
        // two packages, two cores each, two threads per core; core ids repeat across packages
        for (cpu, package, core) in [
            (0, "0", "0"),
            (1, "0", "1"),
            (2, "1", "0"),
            (3, "1", "1"),
            (4, "0", "0"),
            (5, "0", "1"),
            (6, "1", "0"),
            (7, "1", "1"),
        ] {
            topology(&dir, cpu, package, core);
        }
        let all: Vec<usize> = (0..8).collect();
        assert_eq!(count_physical(&dir, &all), Some(4));
        assert_eq!(count_physical(&dir, &[0, 4]), Some(1));
        assert_eq!(count_physical(&dir, &[0, 2]), Some(2));
        assert_eq!(count_physical(&dir, &[99]), None);
        fs::remove_dir_all(&dir).expect("remove fixture");
    }

    #[test]
    fn the_affinity_list_comes_from_the_status_file() {
        let status = "Name:\tx\nCpus_allowed:\tff\nCpus_allowed_list:\t0-3\nMems_allowed:\t1\n";
        assert_eq!(allowed_cpus(status), Some(vec![0, 1, 2, 3]));
        assert_eq!(allowed_cpus("Name:\tx\n"), None);
    }

    #[test]
    fn this_host_has_at_least_one_core_and_no_more_than_it_may_run_on() {
        let cores = physical_cores();
        assert!(cores >= 1);
        assert!(cores <= std::thread::available_parallelism().map_or(1, usize::from));
    }
}
