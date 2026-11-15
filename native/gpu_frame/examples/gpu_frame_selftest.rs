//! A self-test of `gpu_frame` on a real GPU, for a coordinator's hold (lanes never run GPU code).
//! One line per check, `PASS` or `FAIL`; the exit status is non-zero when any check failed.
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=2 gpu_frame_selftest --lanes 8 --jpeg a.jpg --jpeg b.jpg \
//!     --kernel native/frame_engine/kernels/prep.cu --kernel-function resample_h \
//!     [--nvjpeg-dir DIR] [--server 127.0.0.1:4601 --prefix op-fe-selftest-] \
//!     [--model NAME --input images --datatype UINT8 --shape 1,3,640,640 --output output0]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use gpu_frame::model_server::RegistrationCause;
use gpu_frame::{
    CancelToken, Context, DeviceCap, DeviceRange, FramePool, HostCount, InferCall, InputTensor,
    KernelArg, Kernels, Lanes, LaunchConfig, ModelCalls, ModelServer, ModelServerConfig, PinnedBuf,
    Region, Stream,
};

const PROBE: &str = r#"extern "C" __global__ void gpu_frame_probe(unsigned char* dst, int n, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (unsigned char)(value + i % 7);
}"#;

struct Args {
    device: usize,
    lanes: usize,
    jpegs: Vec<PathBuf>,
    kernel: Option<PathBuf>,
    kernel_function: String,
    nvjpeg_dir: Option<PathBuf>,
    server: Option<String>,
    prefix: String,
    model: Option<String>,
    input: String,
    datatype: String,
    shape: Vec<i64>,
    output: String,
    region_mib: usize,
    cap_mib: usize,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        device: 0,
        lanes: 4,
        jpegs: Vec::new(),
        kernel: None,
        kernel_function: "resample_h".into(),
        nvjpeg_dir: None,
        server: None,
        prefix: "op-fe-selftest-".into(),
        model: None,
        input: "images".into(),
        datatype: "UINT8".into(),
        shape: vec![1, 3, 640, 640],
        output: "output0".into(),
        region_mib: 64,
        cap_mib: 4096,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        let number = |v: String| v.parse::<usize>().map_err(|e| format!("{v}: {e}"));
        match flag.as_str() {
            "--device" => a.device = number(value()?)?,
            "--lanes" => a.lanes = number(value()?)?,
            "--jpeg" => a.jpegs.push(value()?.into()),
            "--kernel" => a.kernel = Some(value()?.into()),
            "--kernel-function" => a.kernel_function = value()?,
            "--nvjpeg-dir" => a.nvjpeg_dir = Some(value()?.into()),
            "--server" => a.server = Some(value()?),
            "--prefix" => a.prefix = value()?,
            "--model" => a.model = Some(value()?),
            "--input" => a.input = value()?,
            "--datatype" => a.datatype = value()?,
            "--shape" => {
                a.shape = value()?
                    .split(',')
                    .map(|d| {
                        d.trim()
                            .parse::<i64>()
                            .map_err(|e| format!("shape {d}: {e}"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "--output" => a.output = value()?,
            "--region-mib" => a.region_mib = number(value()?)?,
            "--cap-mib" => a.cap_mib = number(value()?)?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

#[derive(Default)]
struct Report {
    failed: usize,
}

impl Report {
    fn line(&mut self, name: &str, outcome: Result<String, String>) -> bool {
        match outcome {
            Ok(detail) => {
                println!("PASS {name}: {detail}");
                true
            }
            Err(e) => {
                println!("FAIL {name}: {e}");
                self.failed += 1;
                false
            }
        }
    }
}

/// Width, height, components, backend, pixel checksum of one decode.
type Seen = (u32, u32, u32, &'static str, u64);

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    })
}

fn read_back(
    ctx: &Arc<Context>,
    stream: &Stream,
    src: &dyn DeviceRange,
) -> Result<Vec<u8>, String> {
    let mut host = PinnedBuf::new(ctx, src.byte_len()).map_err(|e| e.to_string())?;
    let pending = stream
        .copy_to_pinned(src, 0, &mut host, src.byte_len())
        .map_err(|e| e.to_string())?;
    pending
        .wait()
        .map(<[u8]>::to_vec)
        .map_err(|e| e.to_string())
}

fn datatype_bytes(datatype: &str) -> Option<usize> {
    match datatype {
        "UINT8" | "INT8" | "BOOL" => Some(1),
        "FP16" | "BF16" | "INT16" | "UINT16" => Some(2),
        "FP32" | "INT32" | "UINT32" => Some(4),
        "FP64" | "INT64" | "UINT64" => Some(8),
        _ => None,
    }
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gpu_frame_selftest: {e}");
            return ExitCode::from(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gpu_frame_selftest: tokio runtime: {e}");
            return ExitCode::from(2);
        }
    };
    let started = Instant::now();
    let failed = runtime.block_on(run(args));
    println!(
        "{} check(s) failed in {:.1} s",
        failed,
        started.elapsed().as_secs_f64()
    );
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

async fn run(args: Args) -> usize {
    let mut r = Report::default();
    let ctx = match Context::new(args.device) {
        Ok(c) => c,
        Err(e) => {
            r.line("context", Err(e.to_string()));
            return r.failed;
        }
    };
    r.line(
        "context",
        (|| {
            let name = ctx.device_name().map_err(|e| e.to_string())?;
            let cc = ctx.compute_capability().map_err(|e| e.to_string())?;
            let bus = ctx.pci_bus_id().map_err(|e| e.to_string())?;
            Ok(format!(
                "device {} {name}, sm_{}{}, {bus}",
                args.device, cc.0, cc.1
            ))
        })(),
    );
    let (free_before, total) = ctx.memory_info().unwrap_or((0, 0));
    let cap = DeviceCap::new(args.cap_mib << 20);
    let host = Arc::new(HostCount::default());
    let frames = match FramePool::new(&ctx, cap.clone()) {
        Ok(f) => f,
        Err(e) => {
            r.line("frame pool", Err(e.to_string()));
            return r.failed;
        }
    };
    decode_checks(&mut r, &args, &ctx, &frames, &host).await;
    kernel_checks(&mut r, &args, &ctx, &frames);
    region_checks(&mut r, &args, &ctx).await;
    let (free_after, _) = ctx.memory_info().unwrap_or((0, 0));
    r.line(
        "memory",
        Ok(format!(
            "cap peak {} MiB of {}, pinned host peak {} MiB, device used by all processes {} MiB before and {} MiB after; per process: {}",
            cap.peak() >> 20,
            args.cap_mib,
            host.peak() >> 20,
            (total - free_before) >> 20,
            (total - free_after) >> 20,
            nvidia_smi_for_this_process()
        )),
    );
    drop(frames);
    r.line(
        "counts return to zero",
        if cap.in_use() == 0 && host.in_use() == 0 {
            Ok("device cap 0 bytes, pinned host 0 bytes".into())
        } else {
            Err(format!(
                "device cap {} bytes, pinned host {} bytes still counted",
                cap.in_use(),
                host.in_use()
            ))
        },
    );
    r.line(
        "context not lost",
        if ctx.is_lost() {
            Err("a sticky error was seen".into())
        } else {
            Ok("ok".into())
        },
    );
    r.failed
}

async fn decode_checks(
    r: &mut Report,
    args: &Args,
    ctx: &Arc<Context>,
    frames: &FramePool,
    host: &Arc<HostCount>,
) {
    let lanes = match Lanes::new(frames, host, args.lanes, args.nvjpeg_dir.as_deref()) {
        Ok(l) => l,
        Err(e) => {
            r.line("lanes", Err(e.to_string()));
            return;
        }
    };
    let v = lanes.library_version();
    r.line(
        "lanes",
        Ok(format!(
            "{} lanes, nvJPEG {}.{}.{}",
            lanes.count(),
            v.0,
            v.1,
            v.2
        )),
    );
    let mut jobs = Vec::new();
    for path in &args.jpegs {
        let data = match std::fs::read(path) {
            Ok(d) => Bytes::from(d),
            Err(e) => {
                r.line(&format!("decode {}", path.display()), Err(e.to_string()));
                continue;
            }
        };
        // Each file once per lane, all queued at once, so every lane decodes concurrently.
        for _ in 0..args.lanes {
            match lanes.decode(data.clone(), &CancelToken::new()) {
                Ok(h) => jobs.push((path.clone(), h)),
                Err(e) => {
                    r.line("decode submit", Err(e.to_string()));
                }
            }
        }
    }
    let t0 = Instant::now();
    let mut seen: Vec<(PathBuf, Result<Seen, String>)> = Vec::new();
    for (path, job) in jobs {
        let outcome = match job.wait().await {
            Ok(Ok(d)) => {
                let stream = d.frame.buffer().stream().clone();
                read_back(ctx, &stream, &d.frame).map(|px| {
                    (
                        d.frame.width(),
                        d.frame.height(),
                        d.components,
                        d.backend.name(),
                        fnv1a(&px),
                    )
                })
            }
            Ok(Err(e)) => Err(e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        seen.push((path, outcome));
    }
    let n = seen.len();
    for path in &args.jpegs {
        let runs: Vec<&Result<_, String>> = seen
            .iter()
            .filter(|(p, _)| p == path)
            .map(|(_, o)| o)
            .collect();
        let name = format!("decode {}", path.display());
        let outcome = match runs.first() {
            None => Err("not decoded".to_string()),
            Some(Err(e)) => Err(e.clone()),
            Some(Ok(first)) => {
                if runs.iter().all(|o| matches!(o, Ok(x) if x == first)) {
                    Ok(format!(
                        "{}x{} components {} backend {} pixel fnv1a {:016x}, identical on {} decodes",
                        first.0,
                        first.1,
                        first.2,
                        first.3,
                        first.4,
                        runs.len()
                    ))
                } else {
                    Err(format!("the {} decodes disagree: {runs:?}", runs.len()))
                }
            }
        };
        r.line(&name, outcome);
    }
    r.line(
        "decode throughput",
        Ok(format!(
            "{n} decodes and read-backs in {:.2} s",
            t0.elapsed().as_secs_f64()
        )),
    );
}

fn kernel_checks(r: &mut Report, args: &Args, ctx: &Arc<Context>, frames: &FramePool) {
    let kernels = match Kernels::new(ctx) {
        Ok(k) => k,
        Err(e) => {
            r.line("nvrtc", Err(e.to_string()));
            return;
        }
    };
    r.line("nvrtc options", Ok(kernels.options().join(" ")));
    if let Some(path) = &args.kernel {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let outcome = std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|src| {
                let t0 = Instant::now();
                let m = kernels.load(&name, &src).map_err(|e| e.to_string())?;
                m.function(&args.kernel_function)
                    .map_err(|e| e.to_string())?;
                Ok(format!(
                    "{name} compiled and loaded in {:.2} s; {} resolved",
                    t0.elapsed().as_secs_f64(),
                    args.kernel_function
                ))
            });
        r.line("kernel file", outcome);
    }
    let outcome = (|| {
        let stream = Stream::new(ctx).map_err(|e| e.to_string())?;
        let n = 4096usize;
        let buf = frames.alloc(n, &stream).map_err(|e| e.to_string())?;
        let module = kernels
            .load("gpu_frame_probe.cu", PROBE)
            .map_err(|e| e.to_string())?;
        let cfg = LaunchConfig {
            grid: (n.div_ceil(256) as u32, 1, 1),
            block: (256, 1, 1),
            shared_bytes: 0,
        };
        let args = [
            KernelArg::U64(buf.device_addr()),
            KernelArg::I32(n as i32),
            KernelArg::I32(100),
        ];
        module
            .launch("gpu_frame_probe", &cfg, &stream, &args)
            .map_err(|e| e.to_string())?;
        let back = read_back(ctx, &stream, &buf)?;
        let wrong = back
            .iter()
            .enumerate()
            .filter(|(i, b)| **b as usize != 100 + i % 7)
            .count();
        if wrong == 0 {
            Ok(format!("{n} bytes written as expected"))
        } else {
            Err(format!("{wrong} of {n} bytes wrong"))
        }
    })();
    r.line("kernel launch", outcome);
}

async fn region_checks(r: &mut Report, args: &Args, ctx: &Arc<Context>) {
    let region = match Region::new(ctx, args.region_mib << 20) {
        Ok(reg) => reg,
        Err(e) => {
            r.line("region", Err(e.to_string()));
            return;
        }
    };
    let exported = region.raw_handle().iter().any(|b| *b != 0);
    r.line(
        "region",
        if exported {
            Ok(format!("{} MiB exported", args.region_mib))
        } else {
            Err("empty IPC handle".into())
        },
    );
    let Some(url) = args.server.clone() else {
        println!("SKIP model server: no --server given");
        return;
    };
    let pid = std::process::id();
    let cfg = |name: String, prefix: String| ModelServerConfig {
        url: url.clone(),
        channels: 4,
        call_timeout: Duration::from_secs(10),
        region_prefix: prefix,
        region_name: name,
        device_id: 0,
    };
    // A predecessor that dies without unregistering: its region stays registered (and its
    // memory to process exit), so the next start must purge it.
    let dead = Region::new(ctx, 1 << 20)
        .map_err(|e| e.to_string())
        .and_then(|reg| {
            ModelServer::new(
                cfg(format!("{}dead-{pid}", args.prefix), args.prefix.clone()),
                reg,
            )
            .map_err(|e| e.to_string())
        });
    match dead {
        Ok(dead) => {
            let h = dead.check().await;
            r.line(
                "predecessor registered",
                if h.ready() {
                    Ok("ok".into())
                } else {
                    Err(format!("{h:?}"))
                },
            );
            std::mem::forget(dead);
        }
        Err(e) => {
            r.line("predecessor registered", Err(e));
        }
    }
    let name = format!("{}live-{pid}", args.prefix);
    let server = match ModelServer::new(cfg(name.clone(), args.prefix.clone()), region.clone()) {
        Ok(s) => s,
        Err(e) => {
            r.line("model server", Err(e.to_string()));
            return;
        }
    };
    let h = server.check().await;
    r.line(
        "purge and register",
        if h.ready() && h.purged >= 1 {
            Ok(format!(
                "purged {} leftover region(s), registered {name}",
                h.purged
            ))
        } else {
            Err(format!("{h:?}"))
        },
    );
    if let Some(model) = &args.model {
        knock_out(ctx, &cfg, &name).await;
        r.line(
            "infer (re-registers a region the server lost)",
            infer_once(args, ctx, &server, model).await,
        );
        let n = server.registrations(RegistrationCause::RegionUnknown);
        r.line(
            "re-register on region unknown",
            if n == 1 {
                Ok("1 registration".into())
            } else {
                Err(format!("{n}"))
            },
        );
    }
    knock_out(ctx, &cfg, &name).await;
    let h = server.check().await;
    let n = server.registrations(RegistrationCause::ServerLostRegion);
    r.line(
        "re-register on the health tick",
        if h.ready() && n == 1 {
            Ok(format!("{h:?}"))
        } else {
            Err(format!("{h:?}, {n} registrations"))
        },
    );
    let freed = server.close().await;
    r.line(
        "unregister then free",
        if freed {
            Ok("unregistered; freed with the last handle".into())
        } else {
            Err("unregister failed".into())
        },
    );
    let is_registered = region.is_registered();
    drop(region);
    r.line(
        "region released",
        if is_registered {
            Err("still flagged registered".into())
        } else {
            Ok("ok".into())
        },
    );
}

/// Make the server forget the live region as a restart would: another engine whose prefix is
/// the live region's full name purges it.
async fn knock_out(
    ctx: &Arc<Context>,
    cfg: &impl Fn(String, String) -> ModelServerConfig,
    live: &str,
) {
    if let Ok(reg) = Region::new(ctx, 1 << 20)
        && let Ok(purger) = ModelServer::new(cfg(format!("{live}+purger"), live.to_string()), reg)
    {
        let _ = purger.check().await;
        let _ = purger.close().await;
    }
}

async fn infer_once(
    args: &Args,
    ctx: &Arc<Context>,
    server: &ModelServer,
    model: &str,
) -> Result<String, String> {
    let elems: i64 = args.shape.iter().product();
    let width =
        datatype_bytes(&args.datatype).ok_or_else(|| format!("datatype {}", args.datatype))?;
    let bytes = usize::try_from(elems).map_err(|e| e.to_string())? * width;
    let slice = server
        .region()
        .slice(bytes, Duration::from_secs(1))
        .await
        .map_err(|e| e.to_string())?;
    let stream = Stream::new(ctx).map_err(|e| e.to_string())?;
    stream
        .memset(&slice, 0, bytes, 114)
        .map_err(|e| e.to_string())?;
    stream.synchronize().map_err(|e| e.to_string())?;
    let inputs = [InputTensor {
        name: &args.input,
        datatype: &args.datatype,
        shape: &args.shape,
        slice: &slice,
    }];
    let outputs = [args.output.as_str()];
    let call = InferCall {
        model,
        inputs: &inputs,
        outputs: &outputs,
        deadline: tokio::time::Instant::now() + Duration::from_secs(30),
    };
    let t0 = Instant::now();
    let out = server.infer(call).await.map_err(|e| e.to_string())?;
    let o = out.first().ok_or("no output")?;
    Ok(format!(
        "{model}: {} {} {:?}, {} bytes in {:.1} ms",
        o.name,
        o.datatype,
        o.shape,
        o.data.len(),
        t0.elapsed().as_secs_f64() * 1e3
    ))
}

fn nvidia_smi_for_this_process() -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else {
        return "nvidia-smi not available".into();
    };
    let pid = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| {
            let mut parts = l.split(',').map(str::trim);
            (parts.next() == Some(pid.as_str()))
                .then(|| format!("{} MiB (NVML via nvidia-smi)", parts.next().unwrap_or("?")))
        })
        .unwrap_or_else(|| {
            "this process not listed by nvidia-smi (a container hides host pids)".into()
        })
}
