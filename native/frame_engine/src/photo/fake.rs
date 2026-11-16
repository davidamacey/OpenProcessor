//! A device on host threads for the photo tests: `gpu_frame`'s own worker pool (so cancellation
//! at the hand-off and panic isolation are the real ones), `gpu_frame`'s slab over offsets (so the
//! region's bounded waits are the real ones), frames that count themselves, model calls that
//! answer from a script and log what they were sent.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use gpu_frame::slab::Slab;
use gpu_frame::{
    CallError, CancelToken, JobError, JobHandle, OutputTensor, SliceError, WorkerPool,
};
use tokio::time::Instant;

use super::device::{DecodeJob, Decoded, Device, JobResult, LaunchJob, Launched, ModelCall};
use crate::branches::crops::CropJpeg;
use crate::fault::GpuFault;

/// What the fake does, set by each test.
#[derive(Default)]
pub struct Script {
    /// The detector's rows (`x1, y1, x2, y2, score, class` in canvas pixels).
    pub detections: Vec<[f32; 6]>,
    /// Calls to these models fail so.
    pub fail: HashMap<String, CallError>,
    /// The decode refuses with this status, or fails with this fault.
    pub refuse: Option<&'static str>,
    pub decode_fault: Option<GpuFault>,
    pub launch_fault: Option<GpuFault>,
    pub panic_in_decode: bool,
    /// Model calls answer after this long.
    pub call_hold: Option<Duration>,
    /// The decode job blocks until this is set.
    pub decode_gate: Option<Arc<AtomicBool>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Decode {
        bytes: Vec<usize>,
        fills: Vec<Option<u8>>,
    },
    Launch {
        writes: usize,
        blur: usize,
    },
    Crop([i64; 4]),
    /// A colour table uploaded (its bytes), and a decode that gathers through one.
    ColourTable(usize),
    Gather,
    Call {
        model: String,
        shape: [i64; 4],
        deadline: Instant,
        region_in_use: usize,
    },
}

pub struct Slice {
    slab: Arc<Slab>,
    off: usize,
    len: usize,
}

impl Drop for Slice {
    fn drop(&mut self) {
        self.slab.give(self.off, self.len);
    }
}

/// A colour table that only knows its size.
struct HostTable(usize);

impl gpu_frame::DeviceRange for HostTable {
    fn device_addr(&self) -> u64 {
        0
    }

    fn byte_len(&self) -> usize {
        self.0
    }
}

pub struct Frame {
    alive: Arc<AtomicUsize>,
    _hold: Box<dyn Send + Sync>,
}

impl Drop for Frame {
    fn drop(&mut self) {
        self.alive.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct Fake {
    pub slab: Arc<Slab>,
    pool: WorkerPool<()>,
    pub script: Mutex<Script>,
    pub log: Mutex<Vec<Event>>,
    pub frames: Arc<AtomicUsize>,
    pub lost: AtomicBool,
    pub wakes: AtomicUsize,
    /// Vector width of both encoders.
    pub dim: usize,
}

impl Fake {
    pub fn new(region: usize, threads: usize) -> Arc<Self> {
        Arc::new(Self {
            slab: Arc::new(Slab::new(region).expect("a region of whole 256-byte blocks")),
            pool: WorkerPool::new("fake", threads, |_| Ok::<(), String>(())).expect("pool"),
            script: Mutex::new(Script::default()),
            log: Mutex::new(Vec::new()),
            frames: Arc::new(AtomicUsize::new(0)),
            lost: AtomicBool::new(false),
            wakes: AtomicUsize::new(0),
            dim: 8,
        })
    }

    /// A decoded frame with no budget behind it, counted like a decoded one.
    pub fn frame(&self) -> Arc<Frame> {
        self.frames.fetch_add(1, Ordering::SeqCst);
        Arc::new(Frame {
            alive: Arc::clone(&self.frames),
            _hold: Box::new(()),
        })
    }

    pub fn set(&self, f: impl FnOnce(&mut Script)) {
        f(&mut self.script.lock().expect("script"));
    }

    pub fn events(&self) -> Vec<Event> {
        self.log.lock().expect("log").clone()
    }

    pub fn calls(&self) -> Vec<(String, [i64; 4])> {
        self.events()
            .into_iter()
            .filter_map(|e| match e {
                Event::Call { model, shape, .. } => Some((model, shape)),
                _ => None,
            })
            .collect()
    }

    fn note(&self, e: Event) {
        self.log.lock().expect("log").push(e);
    }

    fn submit<T: Send + 'static>(
        &self,
        cancel: &CancelToken,
        job: impl FnOnce() -> T + Send + 'static,
    ) -> Result<JobHandle<T>, JobError> {
        self.pool.submit(cancel, move |_: &mut ()| job())
    }
}

fn tensor(name: &str, shape: Vec<i64>, values: &[f32]) -> OutputTensor {
    OutputTensor {
        name: name.into(),
        datatype: "FP32".into(),
        shape,
        data: Bytes::from(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
    }
}

impl Device for Fake {
    type Frame = Frame;
    type Input = Slice;

    fn lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    async fn slice(&self, bytes: usize, wait: Duration) -> Result<Slice, SliceError> {
        let off = self.slab.take(bytes, wait).await?;
        Ok(Slice {
            slab: Arc::clone(&self.slab),
            off,
            len: bytes,
        })
    }

    fn decode(
        &self,
        job: DecodeJob<Slice>,
        cancel: &CancelToken,
    ) -> JobResult<Decoded<Frame, Slice>> {
        let script = self.script.lock().expect("script");
        let (refuse, fault, panic, gate) = (
            script.refuse,
            script.decode_fault.clone(),
            script.panic_in_decode,
            script.decode_gate.clone(),
        );
        drop(script);
        if job.colour.is_some() {
            self.note(Event::Gather);
        }
        self.note(Event::Decode {
            bytes: job.writes.iter().map(|w| w.input.len).collect(),
            fills: job.writes.iter().map(|w| w.fill).collect(),
        });
        let alive = Arc::clone(&self.frames);
        self.submit(cancel, move || {
            if let Some(gate) = gate {
                while !gate.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            assert!(!panic, "a decode blew up");
            if let Some(f) = fault {
                return Err(f);
            }
            let mut stages = vec![("decode", 0.001)];
            stages.extend(job.writes.iter().map(|w| (w.stage, 0.001)));
            if let Some(status) = refuse {
                return Ok(Decoded::Refused { status, stages });
            }
            alive.fetch_add(1, Ordering::SeqCst);
            Ok(Decoded::Frame {
                frame: Arc::new(Frame {
                    alive,
                    _hold: job.hold,
                }),
                inputs: job.writes.into_iter().map(|w| w.input).collect(),
                stages,
            })
        })
    }

    fn launch(
        &self,
        job: LaunchJob<Frame, Slice>,
        cancel: &CancelToken,
    ) -> JobResult<Launched<Slice>> {
        let fault = self.script.lock().expect("script").launch_fault.clone();
        self.note(Event::Launch {
            writes: job.writes.len(),
            blur: job.blur.len(),
        });
        self.submit(cancel, move || {
            let _frame = job.frame;
            if let Some(f) = fault {
                return Err(f);
            }
            let mut stages: Vec<(&'static str, f64)> =
                job.writes.iter().map(|w| (w.stage, 0.001)).collect();
            if !job.blur.is_empty() {
                stages.push(("blur", 0.001));
            }
            Ok(Launched {
                blur: job.blur.iter().map(|[x1, _, x2, _]| [*x1, *x2]).collect(),
                inputs: job.writes.into_iter().map(|w| w.input).collect(),
                stages,
            })
        })
    }

    fn crop_jpeg(
        &self,
        frame: Arc<Frame>,
        rect: [i64; 4],
        cancel: &CancelToken,
    ) -> JobResult<CropJpeg> {
        self.note(Event::Crop(rect));
        self.submit(cancel, move || {
            let _frame = frame;
            Ok(CropJpeg {
                data: Bytes::from(format!("jpeg {rect:?}")),
                copy_seconds: 0.001,
                encode_seconds: 0.002,
            })
        })
    }

    fn host<T: Send + 'static>(
        &self,
        f: impl FnOnce() -> T + Send + 'static,
        cancel: &CancelToken,
    ) -> JobResult<T> {
        self.submit(cancel, move || Ok(f()))
    }

    fn colour_table(&self, host: &[u8]) -> Result<crate::colour::Table, GpuFault> {
        self.note(Event::ColourTable(host.len()));
        Ok(Arc::new(HostTable(host.len())))
    }

    fn infer(
        self: Arc<Self>,
        call: ModelCall<Slice>,
    ) -> impl Future<Output = (Result<Vec<OutputTensor>, CallError>, Slice)> + Send + 'static {
        self.note(Event::Call {
            model: call.model.clone(),
            shape: call.input.shape,
            deadline: call.deadline,
            region_in_use: self.slab.usage().in_use,
        });
        async move {
            let (fail, hold, rows) = {
                let script = self.script.lock().expect("script");
                (
                    script.fail.get(&call.model).cloned(),
                    script.call_hold,
                    script.detections.clone(),
                )
            };
            if let Some(hold) = hold {
                tokio::time::sleep(hold).await;
            }
            if let Some(e) = fail {
                return (Err(e), call.slice);
            }
            let out = if call.outputs == ["output0"] {
                let mut values: Vec<f32> = rows.iter().flatten().copied().collect();
                values.resize(300 * 6, 0.0);
                tensor("output0", vec![1, 300, 6], &values)
            } else {
                let n = usize::try_from(call.input.shape[0]).unwrap_or(0);
                let values: Vec<f32> = (0..n * self.dim).map(|v| v as f32).collect();
                tensor("image_embeddings", vec![n as i64, self.dim as i64], &values)
            };
            (Ok(vec![out]), call.slice)
        }
    }

    fn call_failed(&self, error: &CallError) {
        if error.suspects_server() {
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }
}
