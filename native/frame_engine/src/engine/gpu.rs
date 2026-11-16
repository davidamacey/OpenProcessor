//! The card as the photo flow uses it (`photo::device::Device` over `gpu_frame`): decode lanes,
//! the launch pool, the finish pool, the region and the model server.
//!
//! Who runs where (plan 17 section 2.1): a decode lane runs nvJPEG, the orientation, the round-1
//! fills and resamples and one wait for its stream; a launch thread, with its own stream, runs the
//! crop squares and the blur sums and one wait; a finish thread copies a crop's pixels to its
//! pinned staging (one rectangle copy) and encodes the JPEG. The network runtime only awaits.
//!
//! Every stream that touched a frame or a slice is waited for before either is let go
//! (`engine.py`'s rule made a guard): each job starts a [`WaitOnDrop`] after it took ownership of
//! its slices and its hold on the frame, so on success, on an error and on a panic alike the
//! stream is waited for before they drop. A frame's buffer is freed in order on the lane stream it
//! was allocated on.

use std::sync::Arc;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use gpu_frame::{
    CallError, CancelToken, Context, DecodeError, DeviceBuf, DeviceRange, FramePool, InferCall,
    InputTensor, Lane, Lanes, ModelCalls, ModelServer, OutputTensor, SliceError, Stream,
    WorkerPool,
};
use tokio::sync::Notify;

use crate::branches::InputSpec;
use crate::branches::crops::{self, CropJpeg};
use crate::fault::GpuFault;
use crate::kernels::{FrameRef, PrepKernels, Staging, WEIGHT_CACHE_BYTES, WeightCache, WeightSet};
use crate::photo::clock::{DECODE, JobClock, SYNCED};
use crate::photo::device::{
    DecodeJob, Decoded, Device, JobResult, LaunchJob, Launched, ModelCall, Write,
};

/// A decoded, oriented frame: interleaved RGB on the card, and the budget it holds.
pub struct Frame {
    buf: DeviceBuf,
    width: u32,
    height: u32,
    // after `buf`: the bytes are freed (in order on their stream) before the budget is credited
    _hold: Box<dyn Send + Sync>,
}

impl DeviceRange for Frame {
    fn device_addr(&self) -> u64 {
        self.buf.device_addr()
    }

    fn byte_len(&self) -> usize {
        self.buf.byte_len()
    }
}

impl FrameRef for Frame {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

/// The two weight sets of each resample, per write.
type SetsPerWrite = Vec<Vec<[Arc<WeightSet<DeviceBuf>>; 2]>>;

/// Waits for a stream when dropped.
struct WaitOnDrop(Arc<Stream>);

impl Drop for WaitOnDrop {
    fn drop(&mut self) {
        let _ = self.0.synchronize();
    }
}

/// What every pool thread reads.
struct Shared {
    frames: FramePool,
    kernels: PrepKernels,
    /// One cache per lane, locked only by that lane's thread (lanes own their state in gpu_frame).
    lane_weights: Vec<Mutex<WeightCache<DeviceBuf>>>,
    stage_sync: bool,
}

struct LaunchThread {
    stream: Arc<Stream>,
    weights: WeightCache<DeviceBuf>,
    staging: Staging,
}

struct FinishThread {
    stream: Arc<Stream>,
    staging: Staging,
    encoder: turbojpeg::Compressor,
    max_side: u32,
}

/// How the pools are sized and what they do.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub lanes: usize,
    pub launch_threads: usize,
    pub finish_threads: usize,
    pub jpeg_quality: u32,
    /// `OP_FRAME_ENGINE_CROP_MAX_SIDE`: a crop's long side above it is shrunk before the encode.
    pub crop_max_side: u32,
    pub stage_sync: bool,
}

pub struct Gpu {
    ctx: Arc<Context>,
    shared: Arc<Shared>,
    lanes: Lanes,
    launch: WorkerPool<LaunchThread>,
    finish: WorkerPool<FinishThread>,
    pub(super) server: Arc<ModelServer>,
    pub(super) wake: Notify,
}

fn pool_failed<E: std::fmt::Display>(what: &str, e: E) -> GpuFault {
    GpuFault::Failed(format!("{what}: {e}"))
}

impl Gpu {
    /// The lanes, the pools and the kernels of the context's card; `server` holds the region.
    pub fn start(
        ctx: &Arc<Context>,
        frames: FramePool,
        server: ModelServer,
        cfg: &GpuConfig,
    ) -> Result<Self, GpuFault> {
        let host = Arc::new(gpu_frame::HostCount::default());
        let lanes = Lanes::new(&frames, &host, cfg.lanes, None)
            .map_err(|e| pool_failed("decode lanes", e))?;
        let kernels = PrepKernels::load(&gpu_frame::Kernels::new(ctx)?)?;
        let shared = Arc::new(Shared {
            frames: frames.clone(),
            kernels,
            lane_weights: (0..lanes.count())
                .map(|_| Mutex::new(WeightCache::new(WEIGHT_CACHE_BYTES)))
                .collect(),
            stage_sync: cfg.stage_sync,
        });
        let launch = {
            let ctx = Arc::clone(ctx);
            WorkerPool::new("fe-launch", cfg.launch_threads, move |_| {
                ctx.bind()?;
                Ok::<_, gpu_frame::CudaError>(LaunchThread {
                    stream: Stream::new(&ctx)?,
                    weights: WeightCache::new(WEIGHT_CACHE_BYTES),
                    staging: Staging::new(&ctx),
                })
            })
            .map_err(|e| pool_failed("launch pool", e))?
        };
        let finish = {
            let ctx = Arc::clone(ctx);
            let quality = cfg.jpeg_quality;
            let max_side = cfg.crop_max_side;
            WorkerPool::new("fe-finish", cfg.finish_threads, move |_| {
                ctx.bind().map_err(|e| e.to_string())?;
                Ok::<_, String>(FinishThread {
                    stream: Stream::new(&ctx).map_err(|e| e.to_string())?,
                    staging: Staging::new(&ctx),
                    encoder: crops::encoder(quality)?,
                    max_side,
                })
            })
            .map_err(|e| pool_failed("finish pool", e))?
        };
        Ok(Self {
            ctx: Arc::clone(ctx),
            shared,
            lanes,
            launch,
            finish,
            server: Arc::new(server),
            wake: Notify::new(),
        })
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }

    pub fn lane_count(&self) -> usize {
        self.lanes.count()
    }

    pub fn nvjpeg_version(&self) -> (i32, i32, i32) {
        self.lanes.library_version()
    }

    /// Stop the pools (each finishes the job it holds; every wait in one is bounded), then hand
    /// back the model server, which holds the region.
    pub fn into_server(self) -> Arc<ModelServer> {
        let Self {
            server,
            lanes,
            launch,
            finish,
            shared,
            ..
        } = self;
        drop(launch);
        drop(finish);
        drop(lanes);
        drop(shared);
        server
    }
}

/// `prep_budget.is_memory_error`, then the context: a decode failure is the photo's (`decoder`)
/// only while the context still works (`engine._check_context`; the lane waits for its stream
/// after an nvJPEG failure and reports a lost context instead).
fn decode_refusal(e: DecodeError) -> Result<&'static str, GpuFault> {
    if e.is_context_lost() {
        return Err(GpuFault::ContextLost(e.to_string()));
    }
    if e.is_out_of_memory() {
        return Ok("gpu_memory");
    }
    match e {
        DecodeError::Library(m) => Err(GpuFault::Failed(format!("nvJPEG: {m}"))),
        DecodeError::Job(j) => Err(GpuFault::Failed(j.to_string())),
        _ => Ok("decoder"),
    }
}

/// The fills and resamples of `writes`, each timed as its stage (waited for at its end under
/// `OP_FRAME_ENGINE_STAGE_SYNC`).
fn write_inputs<I: DeviceRange>(
    shared: &Shared,
    stream: &Arc<Stream>,
    frame: &dyn FrameRef,
    writes: &[Write<I>],
    sets: &SetsPerWrite,
    clock: &mut JobClock,
) -> Result<(), GpuFault> {
    for (w, sets) in writes.iter().zip(sets) {
        clock.time(w.stage, || -> Result<(), GpuFault> {
            if let Some(fill) = w.fill {
                stream.memset(&w.input, 0, w.input.byte_len(), fill)?;
            }
            for (r, set) in w.resamples.iter().zip(sets) {
                shared
                    .kernels
                    .resample(&shared.frames, stream, frame, &w.input, r, set)?;
            }
            if shared.stage_sync && SYNCED.contains(&w.stage) {
                stream.synchronize()?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// Every weight set the writes need, fetched before the job queues anything else.
fn weight_sets<I>(
    shared: &Shared,
    cache: &mut WeightCache<DeviceBuf>,
    stream: &Arc<Stream>,
    writes: &[Write<I>],
) -> Result<SetsPerWrite, GpuFault> {
    writes
        .iter()
        .map(|w| {
            w.resamples
                .iter()
                .map(|r| shared.kernels.weights(cache, &shared.frames, stream, r))
                .collect()
        })
        .collect()
}

fn decode_on_lane(
    shared: &Shared,
    lane: &mut Lane,
    job: DecodeJob<gpu_frame::Slice>,
) -> Result<Decoded<Frame, gpu_frame::Slice>, GpuFault> {
    let DecodeJob {
        image,
        orientation,
        size,
        writes,
        hold,
        request,
    } = job;
    let stream = Arc::clone(lane.stream());
    // declared after what it guards, so it waits before they drop, on every path
    let _wait = WaitOnDrop(Arc::clone(&stream));
    let mut clock = JobClock::new(request);
    let sets = {
        let mut cache = shared.lane_weights[lane.index()]
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        weight_sets(shared, &mut cache, &stream, &writes)?
    };
    let decoded = clock.time(
        DECODE,
        || -> Result<Result<(DeviceBuf, u32, u32), &'static str>, GpuFault> {
            let d = match lane.decode(image) {
                Ok(d) => d,
                Err(e) => return decode_refusal(e).map(Err),
            };
            let (w, h) = (d.frame.width(), d.frame.height());
            let frame = match shared.kernels.orient(
                &shared.frames,
                &stream,
                d.frame.buffer(),
                (w, h),
                orientation,
            )? {
                Some(oriented) => oriented,
                None => (d.frame.into_buffer(), w, h),
            };
            if (frame.1, frame.2) != size {
                return Ok(Err("decoder"));
            }
            Ok(Ok(frame))
        },
    )?;
    let (buf, width, height) = match decoded {
        Ok(frame) => frame,
        Err(status) => {
            lane.synchronize().map_err(|e| match decode_refusal(e) {
                Err(fault) => fault,
                Ok(_) => GpuFault::Failed("the lane's stream wait failed".into()),
            })?;
            return Ok(Decoded::Refused {
                status,
                stages: clock.stages,
            });
        }
    };
    let frame = Frame {
        buf,
        width,
        height,
        _hold: hold,
    };
    write_inputs(shared, &stream, &frame, &writes, &sets, &mut clock)?;
    lane.synchronize().map_err(|e| match e {
        DecodeError::Cuda(c) => GpuFault::from(c),
        other => GpuFault::Failed(other.to_string()),
    })?;
    Ok(Decoded::Frame {
        frame: Arc::new(frame),
        inputs: writes.into_iter().map(|w| w.input).collect(),
        stages: clock.stages,
    })
}

fn launch_on_thread(
    shared: &Shared,
    t: &mut LaunchThread,
    job: LaunchJob<Frame, gpu_frame::Slice>,
) -> Result<Launched<gpu_frame::Slice>, GpuFault> {
    let LaunchJob {
        frame,
        writes,
        blur,
        request,
    } = job;
    let _wait = WaitOnDrop(Arc::clone(&t.stream));
    let mut clock = JobClock::new(request);
    let sets = weight_sets(shared, &mut t.weights, &t.stream, &writes)?;
    write_inputs(shared, &t.stream, &*frame, &writes, &sets, &mut clock)?;
    let sums = if blur.is_empty() {
        t.stream.synchronize()?;
        Vec::new()
    } else {
        // the sums are read back after a wait for the stream, which covers the squares too
        clock.time(crate::photo::clock::BLUR, || {
            shared
                .kernels
                .lap_sums(&shared.frames, &t.stream, &mut t.staging, &*frame, &blur)
        })?
    };
    Ok(Launched {
        inputs: writes.into_iter().map(|w| w.input).collect(),
        blur: sums,
        stages: clock.stages,
    })
}

fn crop_on_thread(
    t: &mut FinishThread,
    frame: Arc<Frame>,
    rect: [i64; 4],
) -> Result<CropJpeg, GpuFault> {
    let [x1, y1, x2, y2] = rect.map(|v| usize::try_from(v).unwrap_or(0));
    let (width, height) = (x2.saturating_sub(x1), y2.saturating_sub(y1));
    if width == 0 || height == 0 {
        return Err(GpuFault::Failed(format!(
            "an empty crop rectangle {rect:?}"
        )));
    }
    let pitch = frame.width() as usize * 3;
    let FinishThread {
        stream,
        staging,
        encoder,
        max_side,
    } = t;
    let started = Instant::now();
    let pinned = staging.at_least(width * height * 3)?;
    let pixels = stream
        .copy_rect_to_pinned(
            &*frame,
            y1 * pitch + x1 * 3,
            pitch,
            width * 3,
            height,
            pinned,
        )?
        .wait()?;
    let copied = started.elapsed().as_secs_f64();
    // the copy was waited for: the frame may go
    drop(frame);
    let started = Instant::now();
    let jpeg = crops::encode_capped(encoder, pixels, (width, height), *max_side)
        .map_err(GpuFault::Failed)?;
    Ok(CropJpeg {
        data: bytes::Bytes::from(jpeg),
        copy_seconds: copied,
        encode_seconds: started.elapsed().as_secs_f64(),
    })
}

impl Device for Gpu {
    type Frame = Frame;
    type Input = gpu_frame::Slice;

    fn lost(&self) -> bool {
        self.ctx.is_lost()
    }

    async fn slice(&self, bytes: usize, wait: Duration) -> Result<gpu_frame::Slice, SliceError> {
        self.server.region().slice(bytes, wait).await
    }

    fn decode(
        &self,
        job: DecodeJob<gpu_frame::Slice>,
        cancel: &CancelToken,
    ) -> JobResult<Decoded<Frame, gpu_frame::Slice>> {
        let shared = Arc::clone(&self.shared);
        self.lanes
            .run(cancel, move |lane| decode_on_lane(&shared, lane, job))
    }

    fn launch(
        &self,
        job: LaunchJob<Frame, gpu_frame::Slice>,
        cancel: &CancelToken,
    ) -> JobResult<Launched<gpu_frame::Slice>> {
        let shared = Arc::clone(&self.shared);
        self.launch.submit(cancel, move |t: &mut LaunchThread| {
            launch_on_thread(&shared, t, job)
        })
    }

    fn crop_jpeg(
        &self,
        frame: Arc<Frame>,
        rect: [i64; 4],
        cancel: &CancelToken,
    ) -> JobResult<CropJpeg> {
        self.finish.submit(cancel, move |t: &mut FinishThread| {
            crop_on_thread(t, frame, rect)
        })
    }

    async fn infer(
        self: Arc<Self>,
        call: ModelCall<gpu_frame::Slice>,
    ) -> (Result<Vec<OutputTensor>, CallError>, gpu_frame::Slice) {
        let ModelCall {
            model,
            input,
            slice,
            outputs,
            deadline,
        } = call;
        let answer = {
            let inputs = [InputTensor {
                name: input.name,
                datatype: InputSpec::DATATYPE,
                shape: &input.shape,
                slice: &slice,
            }];
            self.server
                .infer(InferCall {
                    model: &model,
                    inputs: &inputs,
                    outputs: &outputs,
                    deadline,
                })
                .await
        };
        (answer, slice)
    }

    fn call_failed(&self, error: &CallError) {
        if error.suspects_server() {
            self.wake.notify_one();
        }
    }
}
