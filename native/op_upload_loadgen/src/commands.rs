//! The command line, and the wiring of the list, the pool and the sources a row reads.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use frame_engine_loadgen::cgroup::Cgroup;
use frame_engine_loadgen::host;
use frame_engine_loadgen::list::{PhotoList, parse_paths, sha256_hex};
use hyper::Uri;

use crate::closed_loop::LoopConfig;
use crate::measure::{NamedCgroup, Setup, UploadSettings, measure, upload_evidence};
use crate::multipart::{FieldOrder, Fields, Photo};
use crate::pack::pack;
use crate::send::{Sender, pooled_client};

/// The arguments: one row per run.
#[derive(Debug, Parser)]
#[command(
    version,
    about = "Sends browser-shaped multipart uploads in a closed loop and times them"
)]
pub struct Args {
    /// The server's address, `http://host:port` (plain HTTP only), with an optional path prefix.
    #[arg(long)]
    pub base_url: String,
    /// The project the photos are uploaded to.
    #[arg(long)]
    pub project: String,
    /// The photo list: one path per line, every file read into memory before the clock starts.
    #[arg(long)]
    pub list: PathBuf,
    /// Requests in flight (also the connection pool's size).
    #[arg(long)]
    pub in_flight: usize,
    /// Seconds of load before the window opens; those requests are not measured.
    #[arg(long, default_value_t = 5.0)]
    pub warmup_s: f64,
    /// Seconds of the measured window.
    #[arg(long, default_value_t = 20.0)]
    pub window_s: f64,
    /// The most photos in one request (the server's `upload_max_images_per_request`).
    #[arg(long, default_value_t = 128)]
    pub max_images: usize,
    /// The most photo bytes in one request (the server's `upload_max_bytes_per_request`).
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    pub max_bytes: u64,
    /// The order of the form's fields.
    #[arg(long, value_enum, default_value_t = FieldOrder::Browser)]
    pub field_order: FieldOrder,
    /// The `source` field.
    #[arg(long, default_value = "upload")]
    pub source: String,
    /// The `run_id` field; by default `loadgen-<start time in Unix ms>`.
    #[arg(long)]
    pub run_id: Option<String>,
    /// The largest frame of a photo part, in KiB.
    #[arg(long, default_value_t = 64)]
    pub chunk_kib: usize,
    /// One request's time limit, body and answer included.
    #[arg(long, default_value_t = 600.0)]
    pub timeout_s: f64,
    /// Name of this arm, recorded in the row.
    #[arg(long, default_value = "")]
    pub label: String,
    /// Append the row to this file as one JSON line (it is also printed).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Write the per-request, per-part stamps here (JSON lines, replaced).
    #[arg(long)]
    pub stamps_out: PathBuf,
    /// A container to read CPU from: `NAME=PATH` to its cgroup v2 `cpu.stat`. Repeat for each.
    #[arg(long = "cpu-stat", value_name = "NAME=PATH", value_parser = parse_named_cgroup)]
    pub cpu_stat: Vec<(String, PathBuf)>,
    /// Text recorded as given in the row: the server's build, image digests, settings.
    #[arg(long)]
    pub server_note: Option<String>,
    /// Exit 0 even when requests or photos failed (fault runs); the row still counts them.
    #[arg(long)]
    pub allow_failures: bool,
}

fn parse_named_cgroup(text: &str) -> Result<(String, PathBuf), String> {
    match text.split_once('=') {
        Some((name, path)) if !name.is_empty() && !path.is_empty() => {
            Ok((name.to_string(), PathBuf::from(path)))
        }
        _ => Err("expected NAME=PATH".to_string()),
    }
}

/// The upload route's URI for a base address and a project; plain `http` only.
///
/// # Errors
///
/// Returns `Err` naming the argument when the address is not `http://host[:port][/prefix]` or the
/// project is not made of letters, digits, `-`, `_` and `.`.
pub fn upload_uri(base_url: &str, project: &str) -> Result<Uri, String> {
    let base: Uri = base_url
        .parse()
        .map_err(|e| format!("--base-url {base_url:?}: {e}"))?;
    if base.scheme_str() != Some("http") {
        return Err(format!(
            "--base-url {base_url:?}: only http:// is supported (terminate TLS in front, or use the loopback)"
        ));
    }
    let authority = base
        .authority()
        .ok_or_else(|| format!("--base-url {base_url:?}: no host"))?;
    if project.is_empty()
        || !project
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(format!(
            "--project {project:?}: letters, digits, '-', '_' and '.' only"
        ));
    }
    let prefix = base.path().trim_end_matches('/');
    format!("http://{authority}{prefix}/curation/projects/{project}/ingest/upload")
        .parse()
        .map_err(|e| format!("--base-url/--project: {e}"))
}

fn positive(name: &str, value: f64) -> Result<Duration, String> {
    if value.is_finite() && value > 0.0 {
        Ok(Duration::from_secs_f64(value))
    } else {
        Err(format!("{name} must be a positive number of seconds"))
    }
}

/// The list's photos with their identifiers; the list file is read again and must still hash to
/// what the list's evidence says.
fn photos_of(list: &PhotoList) -> Result<Vec<Photo>, String> {
    let raw = fs::read(&list.path).map_err(|e| format!("{}: {e}", list.path.display()))?;
    if sha256_hex(&raw) != list.sha256 {
        return Err(format!(
            "{}: changed while it was being read",
            list.path.display()
        ));
    }
    let text = String::from_utf8(raw).map_err(|e| e.to_string())?;
    Ok(parse_paths(&text)
        .iter()
        .zip(&list.photos)
        .map(|(path, data)| Photo {
            identifier: path.display().to_string(),
            filename: path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            data: data.clone(),
        })
        .collect())
}

fn append_line(path: &Path, line: &str) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))
}

/// Run one measurement and print its row.
///
/// # Errors
///
/// Returns `Err` with a message naming the argument or file at fault: a bad argument, an
/// unreadable list, a photo over the byte cap, or an unwritable output.
pub async fn run(args: Args, argv: Vec<String>) -> Result<ExitCode, String> {
    if args.in_flight == 0 || args.chunk_kib == 0 || args.max_bytes == 0 {
        return Err("--in-flight, --chunk-kib and --max-bytes must be at least 1".into());
    }
    let (warmup, window) = (
        Duration::try_from_secs_f64(args.warmup_s).map_err(|e| format!("--warmup-s: {e}"))?,
        positive("--window-s", args.window_s)?,
    );
    let timeout = positive("--timeout-s", args.timeout_s)?;
    let uri = upload_uri(&args.base_url, &args.project)?;
    let list = PhotoList::load(&args.list).map_err(|e| e.to_string())?;
    let photos = photos_of(&list)?;
    let sizes: Vec<u64> = photos.iter().map(|p| p.data.len() as u64).collect();
    let packs = pack(&sizes, args.max_images, args.max_bytes).map_err(|e| e.to_string())?;
    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| format!("loadgen-{}", host::unix_ms()));
    let settings = UploadSettings {
        url: &uri.to_string(),
        order: args.field_order,
        source: &args.source,
        run_id: &run_id,
        max_images: args.max_images,
        max_bytes: args.max_bytes,
        chunk_bytes: args.chunk_kib * 1024,
        timeout,
    };
    let setup = Setup {
        label: args.label.clone(),
        command: argv,
        list,
        upload: upload_evidence(&settings, args.in_flight, &packs),
        config: LoopConfig {
            in_flight: args.in_flight,
            warmup,
            window,
        },
        stamps_path: args.stamps_out.clone(),
        cgroups: args
            .cpu_stat
            .iter()
            .map(|(name, path)| NamedCgroup {
                name: name.clone(),
                cgroup: Cgroup::new(path.clone()),
            })
            .collect(),
        server_note: args.server_note.clone(),
    };
    let sender = Arc::new(Sender {
        client: pooled_client(args.in_flight),
        uri,
        timeout,
        order: args.field_order,
        fields: Fields {
            source: args.source.clone(),
            run_id: run_id.clone(),
        },
        chunk: settings.chunk_bytes,
        photos: Arc::new(photos),
        packs,
        t0: Instant::now(),
    });
    let row = measure(setup, sender).await?;
    let line = serde_json::to_string(&row).map_err(|e| e.to_string())?;
    println!("{line}");
    if let Some(out) = &args.out {
        append_line(out, &line)?;
    }
    let failed = row.summary.failed_requests() + row.summary.failed_photos();
    Ok(
        if args.allow_failures || (failed == 0 && row.summary.requests_ok > 0) {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_route_is_built_from_the_base_and_the_project() {
        assert_eq!(
            upload_uri("http://127.0.0.1:4603", "demo")
                .unwrap()
                .to_string(),
            "http://127.0.0.1:4603/curation/projects/demo/ingest/upload"
        );
        assert_eq!(
            upload_uri("http://h:80/api/", "p-1.x").unwrap().to_string(),
            "http://h:80/api/curation/projects/p-1.x/ingest/upload"
        );
    }

    #[test]
    fn tls_a_bad_address_and_a_bad_project_are_refused_by_name() {
        assert!(
            upload_uri("https://h", "p")
                .unwrap_err()
                .contains("--base-url")
        );
        assert!(upload_uri("h:80", "p").unwrap_err().contains("--base-url"));
        assert!(
            upload_uri("http://h", "a/b")
                .unwrap_err()
                .contains("--project")
        );
        assert!(
            upload_uri("http://h", "")
                .unwrap_err()
                .contains("--project")
        );
    }

    #[test]
    fn a_named_cgroup_needs_both_halves() {
        assert!(parse_named_cgroup("api=/sys/fs/cgroup/x/cpu.stat").is_ok());
        assert!(parse_named_cgroup("api").is_err());
        assert!(parse_named_cgroup("=x").is_err());
        assert!(parse_named_cgroup("a=").is_err());
    }
}
