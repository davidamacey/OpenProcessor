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

use super::device::{
    DecodeJob, Decoded, Device, JobResult, LaunchJob, Launched, ModelCall, WarpKind,
};
use crate::branches::crops::CropJpeg;
use crate::branches::derivatives::{Rendered, RenditionJob};
use crate::branches::faces::scrfd::{FPN_STRIDES, OUTPUT_NAMES};
use crate::fault::GpuFault;
use crate::photo::clock::{DERIVATIVE_JPEG, DERIVATIVE_RENDER};

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
    /// The rendition job fails with this fault (an encode that failed).
    pub encode_fault: Option<GpuFault>,
    pub panic_in_decode: bool,
    /// Model calls answer after this long.
    pub call_hold: Option<Duration>,
    /// The decode job blocks until this is set.
    pub decode_gate: Option<Arc<AtomicBool>>,
    /// Face detector anchors that fire in the first input of every detector call: stride-8
    /// anchor, score, distances (in strides) and landmark offsets (in strides).
    pub face_fires: Vec<(usize, f32, [f32; 4], [f32; 10])>,
    /// Bright bars `x0, x1, y0, y1, value` in the text detector's probability map.
    pub text_bars: Vec<(usize, usize, usize, usize, f32)>,
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
    /// A rendition job: the size and whether a resample makes it, per rendition.
    Renditions(Vec<((u32, u32), bool)>),
    /// A launch's warps: the faces of each.
    Warps(Vec<usize>),
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

/// The text detector's probability map `[1, 1, h, w]` with `bars` lit.
pub fn text_map(h: usize, w: usize, bars: &[(usize, usize, usize, usize, f32)]) -> OutputTensor {
    let mut pred = vec![0f32; h * w];
    for &(x0, x1, y0, y1, v) in bars {
        for y in y0..y1.min(h) {
            for x in x0..x1.min(w) {
                pred[y * w + x] = v;
            }
        }
    }
    tensor("fetch_name_0", vec![1, 1, h as i64, w as i64], &pred)
}

/// The recogniser's answer for `n` lines: line `k` reads the dictionary's class `k % 5 + 1`.
fn rec_answer(n: usize) -> Vec<OutputTensor> {
    let steps = 3;
    let indices: Vec<i32> = (0..n).flat_map(|k| [(k % 5 + 1) as i32, 0, 0]).collect();
    let scores = vec![0.875f32; n * steps];
    vec![
        OutputTensor {
            name: "indices".into(),
            datatype: "INT32".into(),
            shape: vec![n as i64, steps as i64],
            data: Bytes::from(
                indices
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        },
        tensor("scores", vec![n as i64, steps as i64], &scores),
    ]
}

/// The nine SCRFD outputs for `batch` inputs of `net`: zeros, except `fires` at stride 8 of the
/// first input.
fn scrfd_answer(
    batch: usize,
    net: usize,
    fires: &[(usize, f32, [f32; 4], [f32; 10])],
) -> Vec<OutputTensor> {
    OUTPUT_NAMES
        .iter()
        .enumerate()
        .map(|(k, name)| {
            let stride = FPN_STRIDES[k % 3];
            let anchors = (net / stride) * (net / stride) * 2;
            let d = [1, 4, 10][k / 3];
            let mut v = vec![0f32; batch * anchors * d];
            for (anchor, score, ltrb, kps) in fires.iter().filter(|_| stride == 8) {
                let at = anchor * d;
                match k / 3 {
                    0 => v[at] = *score,
                    1 => v[at..at + 4].copy_from_slice(ltrb),
                    _ => v[at..at + 10].copy_from_slice(kps),
                }
            }
            tensor(name, vec![batch as i64, anchors as i64, d as i64], &v)
        })
        .collect()
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
        if !job.warps.is_empty() {
            let faces = job.warps.iter().map(|w| match &w.kind {
                WarpKind::Faces(f) => f.inv.len(),
                WarpKind::Text(t) => t.lines(),
            });
            self.note(Event::Warps(faces.collect()));
        }
        self.submit(cancel, move || {
            let _frame = job.frame;
            if let Some(f) = fault {
                return Err(f);
            }
            let mut stages: Vec<(&'static str, f64)> =
                job.writes.iter().map(|w| (w.stage, 0.001)).collect();
            stages.extend(job.warps.iter().map(|w| (w.stage, 0.001)));
            if !job.blur.is_empty() {
                stages.push((job.blur_stage, 0.001));
            }
            Ok(Launched {
                blur: job.blur.iter().map(|[x1, _, x2, _]| [*x1, *x2]).collect(),
                inputs: job.writes.into_iter().map(|w| w.input).collect(),
                warped: job.warps.into_iter().map(|w| w.input).collect(),
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

    fn renditions(&self, job: RenditionJob<Frame>, cancel: &CancelToken) -> JobResult<Rendered> {
        let fault = self.script.lock().expect("script").encode_fault.clone();
        self.note(Event::Renditions(
            job.renditions
                .iter()
                .map(|r| (r.size, r.resample.is_some()))
                .collect(),
        ));
        self.submit(cancel, move || {
            let _frame = job.frame;
            if let Some(f) = fault {
                return Err(f);
            }
            Ok(Rendered {
                jpegs: job
                    .renditions
                    .iter()
                    .map(|r| Bytes::from(format!("jpeg {:?} q{}", r.size, r.quality)))
                    .collect(),
                stages: vec![(DERIVATIVE_RENDER, 0.001), (DERIVATIVE_JPEG, 0.002)],
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
            let (fail, hold, rows, fires, bars) = {
                let script = self.script.lock().expect("script");
                (
                    script.fail.get(&call.model).cloned(),
                    script.call_hold,
                    script.detections.clone(),
                    script.face_fires.clone(),
                    script.text_bars.clone(),
                )
            };
            if let Some(hold) = hold {
                tokio::time::sleep(hold).await;
            }
            if let Some(e) = fail {
                return (Err(e), call.slice);
            }
            let n = usize::try_from(call.input.shape[0]).unwrap_or(0);
            if call.outputs.first() == Some(&"score_8") {
                let net = usize::try_from(call.input.shape[2]).unwrap_or(0);
                return (Ok(scrfd_answer(n, net, &fires)), call.slice);
            }
            let [_, _, ih, iw] = call.input.shape.map(|d| usize::try_from(d).unwrap_or(0));
            if call.outputs == ["fetch_name_0"] {
                return (Ok(vec![text_map(ih, iw, &bars)]), call.slice);
            }
            if call.outputs == ["indices", "scores"] {
                return (Ok(rec_answer(n)), call.slice);
            }
            if call.outputs == ["output"] {
                let values: Vec<f32> = (0..n * self.dim).map(|v| (v + 1) as f32).collect();
                let out = tensor("output", vec![n as i64, self.dim as i64], &values);
                return (Ok(vec![out]), call.slice);
            }
            let out = if call.outputs == ["output0"] {
                let mut values: Vec<f32> = rows.iter().flatten().copied().collect();
                values.resize(300 * 6, 0.0);
                tensor("output0", vec![1, 300, 6], &values)
            } else {
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
