//! The colour step's caches on the host: one build per profile under many photos at once, the
//! kind memo, eviction, the bounded wait and a failed build. The tables are Little CMS's real
//! ones (the fixtures of `tests/fixtures/icc/`); the upload is a host copy.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use gpu_frame::DeviceRange;
use prometheus_client::metrics::gauge::Gauge;

use super::*;

const P3: &[u8] = include_bytes!("../../../../tests/fixtures/icc/DisplayP3-v4.icc");
const ADOBE: &[u8] = include_bytes!("../../../../tests/fixtures/icc/AdobeCompat-v4.icc");
const COLORD_SRGB: &[u8] = include_bytes!("../../../../tests/fixtures/icc/colord-sRGB.icc");

struct HostTable(Vec<u8>);

impl DeviceRange for HostTable {
    fn device_addr(&self) -> u64 {
        self.0.as_ptr() as u64
    }

    fn byte_len(&self) -> usize {
        self.0.len()
    }
}

/// Builds on their own threads, counted; uploads counted and copied.
struct Bed {
    builds: Arc<AtomicUsize>,
    uploads: Arc<AtomicUsize>,
    upload: Upload,
}

impl Bed {
    fn new() -> Self {
        let uploads = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&uploads);
        let upload: Upload = Arc::new(move |host: &[u8]| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(HostTable(host.to_vec())) as Table)
        });
        Self {
            builds: Arc::new(AtomicUsize::new(0)),
            uploads,
            upload,
        }
    }

    fn spawn(&self) -> impl Fn(Box<dyn FnOnce() + Send>) -> Result<(), GpuFault> + use<> {
        let builds = Arc::clone(&self.builds);
        move |job| {
            builds.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(job);
            Ok(())
        }
    }

    async fn ask(&self, step: &ColourStep, icc: &[u8]) -> Result<Colour, ColourError> {
        step.lookup(Some(icc), 3, &self.spawn(), &self.upload).await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn many_photos_with_one_new_profile_share_one_build() {
    let step = Arc::new(ColourStep::new(2, Gauge::default()));
    let bed = Arc::new(Bed::new());
    let mut photos = Vec::new();
    for _ in 0..32 {
        let (step, bed) = (Arc::clone(&step), Arc::clone(&bed));
        photos.push(tokio::spawn(async move { bed.ask(&step, P3).await }));
    }
    let mut tables = Vec::new();
    for p in photos {
        match p.await.expect("task") {
            Ok(Colour::Convert(t)) => tables.push(t),
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(
        bed.builds.load(Ordering::SeqCst),
        1,
        "one build for 32 photos"
    );
    assert_eq!(bed.uploads.load(Ordering::SeqCst), 1);
    assert!(tables.iter().all(|t| Arc::ptr_eq(t, &tables[0])));
    assert_eq!(tables[0].byte_len(), TABLE_BYTES);
    assert_eq!(step.table_bytes(), TABLE_BYTES);
}

#[tokio::test]
async fn untagged_grey_and_srgb_class_photos_never_wait_twice() {
    let step = ColourStep::new(2, Gauge::default());
    let bed = Bed::new();
    let spawn = bed.spawn();
    assert!(matches!(
        step.lookup(None, 3, &spawn, &bed.upload).await,
        Ok(Colour::Srgb)
    ));
    assert!(matches!(
        step.lookup(Some(P3), 1, &spawn, &bed.upload).await,
        Ok(Colour::Ignored)
    ));
    for _ in 0..3 {
        assert!(matches!(
            bed.ask(&step, COLORD_SRGB).await,
            Ok(Colour::Srgb)
        ));
    }
    assert!(matches!(
        bed.ask(&step, b"not a profile").await,
        Ok(Colour::Ignored)
    ));
    assert_eq!(
        bed.builds.load(Ordering::SeqCst),
        2,
        "each kind once, then the memo"
    );
    assert_eq!(bed.uploads.load(Ordering::SeqCst), 0, "no table for either");
}

#[tokio::test]
async fn an_evicted_table_is_built_again_when_its_profile_returns() {
    let gauge = Gauge::default();
    let step = ColourStep::new(1, gauge.clone());
    let bed = Bed::new();
    for icc in [P3, ADOBE, P3] {
        assert!(matches!(bed.ask(&step, icc).await, Ok(Colour::Convert(_))));
    }
    assert_eq!(bed.builds.load(Ordering::SeqCst), 3);
    assert_eq!(gauge.get(), i64::try_from(TABLE_BYTES).expect("fits"));
}

#[tokio::test]
async fn a_build_that_does_not_end_in_time_is_an_overload_with_cause_colour() {
    let step = ColourStep::with_wait(2, Gauge::default(), Duration::from_millis(50));
    let bed = Bed::new();
    let (held_tx, held_rx) = mpsc::channel::<Box<dyn FnOnce() + Send>>();
    let held_tx = std::sync::Mutex::new(held_tx);
    let hold = move |job| {
        held_tx.lock().expect("lock").send(job).expect("held");
        Ok(())
    };
    match step.lookup(Some(P3), 3, &hold, &bed.upload).await {
        Err(ColourError::Overload(o)) => assert_eq!(o.cause, "colour"),
        other => panic!("{other:?}"),
    }
    // the late build still lands in the cache: the next photo needs no build of its own
    held_rx.recv().expect("the job")();
    assert!(matches!(bed.ask(&step, P3).await, Ok(Colour::Convert(_))));
    assert_eq!(bed.builds.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_failed_build_fails_its_photo_and_is_not_remembered() {
    let step = ColourStep::new(2, Gauge::default());
    let bed = Bed::new();
    let failing: Upload = Arc::new(|_: &[u8]| Err(GpuFault::OutOfMemory("cap".into())));
    match step.lookup(Some(P3), 3, &bed.spawn(), &failing).await {
        Err(ColourError::Fault(GpuFault::OutOfMemory(_))) => {}
        other => panic!("{other:?}"),
    }
    assert!(matches!(bed.ask(&step, P3).await, Ok(Colour::Convert(_))));
    assert_eq!(bed.builds.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_build_the_pool_drops_unrun_answers_its_waiters_and_is_tried_again() {
    let step = ColourStep::new(2, Gauge::default());
    let bed = Bed::new();
    let dropping = |job: Box<dyn FnOnce() + Send>| {
        drop(job);
        Ok(())
    };
    match step.lookup(Some(P3), 3, &dropping, &bed.upload).await {
        Err(ColourError::Fault(GpuFault::Failed(m))) => assert!(m.contains("did not finish")),
        other => panic!("{other:?}"),
    }
    assert!(matches!(bed.ask(&step, P3).await, Ok(Colour::Convert(_))));
    assert_eq!(bed.builds.load(Ordering::SeqCst), 1);
}
