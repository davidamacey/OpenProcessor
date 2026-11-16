//! Waiting out a database outage for a fenced write.
//!
//! A database that restarts or fails over answers with errors until it is back. A control-plane
//! write waits for it for as long as the holder may still write (until the fence margin before the
//! lease can expire), instead of failing a job that would have finished. The wait doubles from half
//! a second to five.

use std::future::Future;
use std::time::Duration;

use tokio::time::Instant;

use crate::store::StoreError;

const FIRST_WAIT: Duration = Duration::from_millis(500);
const LONGEST_WAIT: Duration = Duration::from_secs(5);

/// Run `call` until it succeeds, fails with an error that is not an outage, or `budget` would be
/// exceeded by the next wait.
pub(crate) async fn through_outage<T, F, Fut>(
    budget: Duration,
    mut call: F,
) -> Result<T, StoreError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, StoreError>>,
{
    let started = Instant::now();
    let mut wait = FIRST_WAIT;
    loop {
        match call().await {
            Err(error) if error.is_unavailable() && started.elapsed() + wait < budget => {
                tracing::warn!(%error, "the database is unavailable; trying again");
                tokio::time::sleep(wait).await;
                wait = (wait * 2).min(LONGEST_WAIT);
            }
            other => return other,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn outage() -> StoreError {
        StoreError::Other("down".into())
    }

    #[tokio::test(start_paused = true)]
    async fn an_outage_is_waited_out_with_doubling_waits() {
        let calls = AtomicUsize::new(0);
        let started = Instant::now();
        let value = through_outage(Duration::from_secs(60), || async {
            if calls.fetch_add(1, Ordering::Relaxed) < 3 {
                Err(outage())
            } else {
                Ok(7)
            }
        })
        .await
        .unwrap();
        assert_eq!(value, 7);
        assert_eq!(calls.load(Ordering::Relaxed), 4);
        assert_eq!(started.elapsed(), Duration::from_millis(500 + 1000 + 2000));
    }

    #[tokio::test(start_paused = true)]
    async fn the_wait_is_capped_and_the_budget_ends_the_waiting_with_the_last_error() {
        let calls = AtomicUsize::new(0);
        let started = Instant::now();
        let error = through_outage(Duration::from_secs(20), || async {
            calls.fetch_add(1, Ordering::Relaxed);
            Err::<(), _>(outage())
        })
        .await
        .unwrap_err();
        assert!(error.is_unavailable());
        // 0.5 + 1 + 2 + 4 + 5 + 5 = 17.5 s of waiting; the next 5 s would pass 20 s
        assert_eq!(started.elapsed(), Duration::from_millis(17_500));
        assert_eq!(calls.load(Ordering::Relaxed), 7);
    }

    #[tokio::test(start_paused = true)]
    async fn a_refusal_that_is_not_an_outage_is_returned_at_once() {
        let calls = AtomicUsize::new(0);
        let refused = through_outage(Duration::from_secs(60), || async {
            calls.fetch_add(1, Ordering::Relaxed);
            Err::<(), _>(StoreError::Db(op_db::DbError::Finished { what: "x" }))
        })
        .await;
        assert!(refused.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
