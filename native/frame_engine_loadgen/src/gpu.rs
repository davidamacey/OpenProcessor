//! The card's busy share and per-process device memory, sampled from NVML every 50 ms
//! (the interval the bake-off's sampler and the throwaway pipeline used). The device is opened by
//! its PCI bus id, so a row can only describe the card it names. The window arithmetic is pure
//! and tested with a fake source; the NVML source is the thin shell around it
//! (<https://docs.rs/nvml-wrapper/0.13.0/nvml_wrapper/struct.Device.html>).
//!
//! `utilization_rates().gpu` is NVML's "percent of time over the past sample period during which
//! one or more kernels was executing": the mean of the samples inside the window is the card's
//! busy share. Process memory is NVML's per-process figure, by host process id: the report maps
//! the id to its container through `/proc/<pid>/cgroup`, recorded beside it.

use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::UsedGpuMemory;
use serde::Serialize;

/// The sampling interval.
pub const SAMPLE_INTERVAL: Duration = Duration::from_millis(50);
const MIB: u64 = 1 << 20;

/// One process's device memory at one sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessMem {
    /// The host process id NVML reports.
    pub pid: u32,
    /// Device memory the process holds, in bytes.
    pub used_bytes: u64,
}

/// One read of the card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reading {
    /// Share of the last sample period in which a kernel ran, in percent (0 to 100).
    pub busy_pct: u32,
    /// Device memory in use on the whole card, in bytes.
    pub mem_used_bytes: u64,
    /// Per-process device memory of the compute processes on the card.
    pub processes: Vec<ProcessMem>,
}

/// Where readings come from.
pub trait Source: Send + 'static {
    /// Read the card once.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the reason when the card cannot be read.
    fn read(&mut self) -> Result<Reading, String>;
}

/// What identifies the card and its link, recorded in every row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Identity {
    /// The card's product name.
    pub name: String,
    /// The PCI bus id as NVML reports it.
    pub bus_id: String,
    /// The driver version; empty when NVML does not give it.
    pub driver_version: String,
    /// Current PCIe link generation; null when NVML does not give it.
    pub pcie_link_gen: Option<u32>,
    /// Current PCIe link width in lanes; null when NVML does not give it.
    pub pcie_link_width: Option<u32>,
}

/// One process's peak device memory inside a window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessPeak {
    /// The host process id NVML reports.
    pub pid: u32,
    /// Largest device memory the process held in any sample of the window, in MiB.
    pub peak_mib: u64,
}

/// What the card did inside a window.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GpuWindow {
    /// Sampling interval, in milliseconds.
    pub interval_ms: u64,
    /// Samples that fell inside the window.
    pub samples: usize,
    /// Mean of the samples' busy share, in percent.
    pub busy_pct_mean: f64,
    /// Largest busy share of any sample, in percent.
    pub busy_pct_max: u32,
    /// Card memory in use before the first window sample, in the sampler's first read (the idle
    /// baseline when the sampler starts before any load).
    pub mem_used_mib_idle: u64,
    /// Largest card memory in use of any sample in the window, in MiB.
    pub mem_used_mib_peak: u64,
    /// Peak per process, largest first.
    pub processes: Vec<ProcessPeak>,
}

#[derive(Debug, Clone)]
struct Sample {
    at: Instant,
    reading: Reading,
}

/// The window `[from, to]` of `samples`, which are in time order; `None` when no sample falls in
/// it.
fn summarise(samples: &[Sample], from: Instant, to: Instant) -> Option<GpuWindow> {
    let inside: Vec<&Sample> = samples
        .iter()
        .filter(|s| s.at >= from && s.at <= to)
        .collect();
    if inside.is_empty() {
        return None;
    }
    let mut peaks: BTreeMap<u32, u64> = BTreeMap::new();
    for s in &inside {
        for p in &s.reading.processes {
            let peak = peaks.entry(p.pid).or_default();
            *peak = (*peak).max(p.used_bytes);
        }
    }
    let mut processes: Vec<ProcessPeak> = peaks
        .into_iter()
        .map(|(pid, bytes)| ProcessPeak {
            pid,
            peak_mib: bytes / MIB,
        })
        .collect();
    processes.sort_by(|a, b| b.peak_mib.cmp(&a.peak_mib).then(a.pid.cmp(&b.pid)));
    Some(GpuWindow {
        interval_ms: u64::try_from(SAMPLE_INTERVAL.as_millis()).unwrap_or(50),
        samples: inside.len(),
        busy_pct_mean: inside
            .iter()
            .map(|s| f64::from(s.reading.busy_pct))
            .sum::<f64>()
            / inside.len() as f64,
        busy_pct_max: inside.iter().map(|s| s.reading.busy_pct).max().unwrap_or(0),
        mem_used_mib_idle: samples
            .first()
            .map_or(0, |s| s.reading.mem_used_bytes / MIB),
        mem_used_mib_peak: inside
            .iter()
            .map(|s| s.reading.mem_used_bytes)
            .max()
            .unwrap_or(0)
            / MIB,
        processes,
    })
}

/// A background thread reading a [`Source`] every [`SAMPLE_INTERVAL`].
pub struct Sampler {
    samples: Arc<Mutex<Vec<Sample>>>,
    stop: Arc<AtomicBool>,
    errors: Arc<Mutex<Vec<String>>>,
    handle: Option<JoinHandle<()>>,
}

impl Sampler {
    /// Start the thread that reads `source` every `interval`; the thread stops when the sampler is dropped. A failed read is kept for `read_errors`.
    pub fn start<S: Source>(mut source: S, interval: Duration) -> Sampler {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let handle = {
            let (samples, stop, errors) = (samples.clone(), stop.clone(), errors.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    match source.read() {
                        Ok(reading) => lock(&samples).push(Sample {
                            at: Instant::now(),
                            reading,
                        }),
                        Err(e) => lock(&errors).push(e),
                    }
                    std::thread::sleep(interval);
                }
            })
        };
        Sampler {
            samples,
            stop,
            errors,
            handle: Some(handle),
        }
    }

    /// The window between two instants taken with [`Instant::now`].
    pub fn window(&self, from: Instant, to: Instant) -> Option<GpuWindow> {
        summarise(&lock(&self.samples), from, to)
    }

    /// Reads that failed so far; a row with any says so.
    pub fn read_errors(&self) -> Vec<String> {
        lock(&self.errors).clone()
    }
}

impl Drop for Sampler {
    /// Stop the thread and wait for it (it sleeps at most one interval).
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// NVML as a [`Source`] for one card.
pub struct NvmlSource {
    nvml: Nvml,
    bus_id: String,
}

impl NvmlSource {
    /// Open NVML (`libnvidia-ml.so.1`, loaded at run time) and the card at `bus_id`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when NVML cannot be loaded or initialised, no card has `bus_id`, or the card's name or PCI information cannot be read.
    pub fn open(bus_id: &str) -> Result<(NvmlSource, Identity), String> {
        let nvml = Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init()
            .map_err(|e| format!("NVML: {e}"))?;
        let identity = {
            let device = nvml
                .device_by_pci_bus_id(bus_id)
                .map_err(|e| format!("NVML: no card at bus id {bus_id}: {e}"))?;
            Identity {
                name: device.name().map_err(|e| format!("NVML name: {e}"))?,
                bus_id: device
                    .pci_info()
                    .map_err(|e| format!("NVML pci info: {e}"))?
                    .bus_id,
                driver_version: nvml.sys_driver_version().unwrap_or_default(),
                pcie_link_gen: device.current_pcie_link_gen().ok(),
                pcie_link_width: device.current_pcie_link_width().ok(),
            }
        };
        Ok((
            NvmlSource {
                nvml,
                bus_id: bus_id.to_string(),
            },
            identity,
        ))
    }
}

impl Source for NvmlSource {
    fn read(&mut self) -> Result<Reading, String> {
        let device = self
            .nvml
            .device_by_pci_bus_id(self.bus_id.as_str())
            .map_err(|e| e.to_string())?;
        let busy = device.utilization_rates().map_err(|e| e.to_string())?.gpu;
        let memory = device.memory_info().map_err(|e| e.to_string())?.used;
        let processes = device
            .running_compute_processes()
            .map_err(|e| e.to_string())?
            .into_iter()
            .filter_map(|p| match p.used_gpu_memory {
                UsedGpuMemory::Used(bytes) => Some(ProcessMem {
                    pid: p.pid,
                    used_bytes: bytes,
                }),
                UsedGpuMemory::Unavailable => None,
            })
            .collect();
        Ok(Reading {
            busy_pct: busy,
            mem_used_bytes: memory,
            processes,
        })
    }
}

/// A process's name and cgroup as the host sees them, so a row maps NVML's host process id to a
/// container. Both are empty when the process is gone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessLabel {
    /// The host process id NVML reports.
    pub pid: u32,
    /// The process's `comm` name; empty when the process is gone.
    pub command: String,
    /// The first line of the process's `/proc/<pid>/cgroup`; empty when the process is gone.
    pub cgroup: String,
}

/// The command and cgroup of host process `pid`, read from `/proc`.
pub fn label(pid: u32) -> ProcessLabel {
    let read = |file: &str| fs::read_to_string(format!("/proc/{pid}/{file}")).unwrap_or_default();
    ProcessLabel {
        pid,
        command: read("comm").trim().to_string(),
        cgroup: read("cgroup")
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(busy: u32, mem_mib: u64, procs: &[(u32, u64)]) -> Reading {
        Reading {
            busy_pct: busy,
            mem_used_bytes: mem_mib * MIB,
            processes: procs
                .iter()
                .map(|(pid, mib)| ProcessMem {
                    pid: *pid,
                    used_bytes: mib * MIB,
                })
                .collect(),
        }
    }

    fn at(base: Instant, ms: u64, r: Reading) -> Sample {
        Sample {
            at: base + Duration::from_millis(ms),
            reading: r,
        }
    }

    #[test]
    fn only_samples_inside_the_window_are_averaged_and_the_idle_baseline_is_the_first_read() {
        let t = Instant::now();
        let samples = vec![
            at(t, 0, reading(0, 500, &[(10, 300)])),
            at(t, 50, reading(10, 600, &[(10, 300)])),
            at(t, 100, reading(80, 4000, &[(10, 300), (20, 3000)])),
            at(t, 150, reading(100, 4200, &[(10, 320), (20, 3100)])),
            at(t, 200, reading(90, 4100, &[(10, 310), (20, 3050)])),
            at(t, 250, reading(0, 600, &[(10, 300)])),
        ];
        let w = summarise(
            &samples,
            t + Duration::from_millis(100),
            t + Duration::from_millis(200),
        )
        .unwrap();
        assert_eq!(w.samples, 3);
        assert!((w.busy_pct_mean - 90.0).abs() < 1e-9);
        assert_eq!(w.busy_pct_max, 100);
        assert_eq!(w.mem_used_mib_idle, 500);
        assert_eq!(w.mem_used_mib_peak, 4200);
        assert_eq!(
            w.processes,
            [
                ProcessPeak {
                    pid: 20,
                    peak_mib: 3100
                },
                ProcessPeak {
                    pid: 10,
                    peak_mib: 320
                }
            ]
        );
    }

    #[test]
    fn a_window_without_samples_is_none() {
        let t = Instant::now();
        let samples = vec![at(t, 0, reading(1, 1, &[]))];
        assert_eq!(
            summarise(
                &samples,
                t + Duration::from_secs(1),
                t + Duration::from_secs(2)
            ),
            None
        );
    }

    struct Counting(u32);

    impl Source for Counting {
        fn read(&mut self) -> Result<Reading, String> {
            self.0 += 1;
            if self.0.is_multiple_of(4) {
                return Err("transient".into());
            }
            Ok(reading(self.0, 1, &[]))
        }
    }

    #[test]
    fn the_sampler_reads_on_its_own_thread_and_keeps_its_errors() {
        let start = Instant::now();
        let sampler = Sampler::start(Counting(0), Duration::from_millis(2));
        std::thread::sleep(Duration::from_millis(80));
        let w = sampler.window(start, Instant::now()).unwrap();
        assert!(w.samples >= 5, "{} samples", w.samples);
        assert!(!sampler.read_errors().is_empty());
    }
}
