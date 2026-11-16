//! The `LISTEN` connection: one dedicated, reconnecting connection per process, outside the pool.
//!
//! tokio-postgres delivers notifications only through the connection task the caller drives
//! (<https://docs.rs/tokio-postgres/0.7.18/tokio_postgres/enum.AsyncMessage.html>,
//! `AsyncMessage::Notification`), so the pool cannot carry them, and a transaction pooler cannot
//! either: the listener connects to the direct host. PostgreSQL delivers a notification only after
//! the notifying transaction commits and not to a session that was not listening
//! (<https://www.postgresql.org/docs/18/sql-notify.html>), so a notification is only a wake-up:
//! whoever listens also polls, and after every reconnect polls at once ([`ListenerEvent::Reconnected`]).
//!
//! The reconnect loop is generic over a [`ListenConnector`], so its behaviour is tested with a fake
//! connector and a paused clock; [`PgConnector`] is the real one.

use std::future::{Future, poll_fn};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_postgres::AsyncMessage;

use crate::error::DbError;
use crate::settings::DbSettings;

/// One `NOTIFY` as received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notification {
    /// The channel it was sent on.
    pub channel: String,
    /// Its payload (empty when the sender gave none).
    pub payload: String,
}

/// What the listener tells its consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListenerEvent {
    /// A notification arrived.
    Notification(Notification),
    /// The connection was lost and made again. A notification sent while it was down is gone:
    /// the consumer must poll its table now.
    Reconnected,
}

/// How long to wait between connection attempts: doubles from `initial` up to `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backoff {
    /// The first wait, and the wait after a connection that worked.
    pub initial: Duration,
    /// The longest wait.
    pub max: Duration,
}

impl Backoff {
    /// Half a second, doubling to 30 s.
    pub const DEFAULT: Self = Self {
        initial: Duration::from_millis(500),
        max: Duration::from_secs(30),
    };

    fn next(self, current: Duration) -> Duration {
        (current * 2).min(self.max)
    }
}

/// One live listening connection.
pub trait ListenSession: Send + 'static {
    /// The error of a failed connection.
    type Error: std::error::Error + Send + 'static;

    /// The next notification; `Ok(None)` when the connection closed.
    ///
    /// # Errors
    ///
    /// When the connection broke.
    fn next_notification(
        &mut self,
    ) -> impl Future<Output = Result<Option<Notification>, Self::Error>> + Send;
}

/// Makes listening connections.
pub trait ListenConnector: Send + 'static {
    /// The session it makes.
    type Session: ListenSession<Error = Self::Error>;
    /// Why a connection could not be made.
    type Error: std::error::Error + Send + 'static;

    /// Connect and issue `LISTEN`; a returned session is already listening.
    ///
    /// # Errors
    ///
    /// When the connection or the `LISTEN` fails.
    fn connect(&mut self) -> impl Future<Output = Result<Self::Session, Self::Error>> + Send;
}

/// The consumer's end: a task that keeps a listening connection alive and forwards what it hears.
///
/// The channel to the consumer is bounded (back pressure); dropping the `Listener` stops the task.
#[derive(Debug)]
pub struct Listener {
    events: mpsc::Receiver<ListenerEvent>,
    task: JoinHandle<()>,
}

const EVENT_QUEUE: usize = 64;

/// Where a consumer gets wake-ups from: a [`Listener`], or a plain channel in a test.
pub trait ListenEvents: Send {
    /// The next event; `None` when the source has ended.
    fn recv(&mut self) -> impl Future<Output = Option<ListenerEvent>> + Send;
}

impl ListenEvents for Listener {
    async fn recv(&mut self) -> Option<ListenerEvent> {
        self.events.recv().await
    }
}

impl ListenEvents for mpsc::Receiver<ListenerEvent> {
    async fn recv(&mut self) -> Option<ListenerEvent> {
        mpsc::Receiver::recv(self).await
    }
}

impl Listener {
    /// Start listening through `connector`, reconnecting with `backoff`.
    #[must_use]
    pub fn spawn<C: ListenConnector>(connector: C, backoff: Backoff) -> Self {
        let (tx, events) = mpsc::channel(EVENT_QUEUE);
        let task = tokio::spawn(run(connector, backoff, tx));
        Self { events, task }
    }

    /// Listen on the given channels of the database `settings` name, on a dedicated direct
    /// connection.
    #[must_use]
    pub fn spawn_pg(
        settings: &DbSettings,
        application_name: &str,
        channels: &'static [&'static str],
    ) -> Self {
        Self::spawn(
            PgConnector::new(settings, application_name, channels),
            Backoff::DEFAULT,
        )
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn run<C: ListenConnector>(
    mut connector: C,
    backoff: Backoff,
    events: mpsc::Sender<ListenerEvent>,
) {
    let mut delay = backoff.initial;
    let mut connected_before = false;
    loop {
        match connector.connect().await {
            Ok(mut session) => {
                delay = backoff.initial;
                if connected_before && events.send(ListenerEvent::Reconnected).await.is_err() {
                    return;
                }
                connected_before = true;
                loop {
                    match session.next_notification().await {
                        Ok(Some(n)) => {
                            if events.send(ListenerEvent::Notification(n)).await.is_err() {
                                return;
                            }
                        }
                        Ok(None) => {
                            tracing::warn!("the LISTEN connection closed; reconnecting");
                            break;
                        }
                        Err(error) => {
                            tracing::warn!(%error, "the LISTEN connection broke; reconnecting");
                            break;
                        }
                    }
                }
            }
            Err(error) => tracing::warn!(%error, "cannot open the LISTEN connection"),
        }
        tokio::time::sleep(delay).await;
        delay = backoff.next(delay);
    }
}

/// The real connector: a direct tokio-postgres connection that stays out of the pool.
#[derive(Debug)]
pub struct PgConnector {
    settings: DbSettings,
    application_name: String,
    channels: &'static [&'static str],
}

impl PgConnector {
    /// A connector for the channels of the database `settings` name, on a direct connection.
    #[must_use]
    pub fn new(
        settings: &DbSettings,
        application_name: &str,
        channels: &'static [&'static str],
    ) -> Self {
        Self {
            settings: settings.clone(),
            application_name: application_name.to_owned(),
            channels,
        }
    }
}

/// A live `LISTEN` connection of [`PgConnector`].
#[derive(Debug)]
pub struct PgSession {
    notifications: mpsc::Receiver<Notification>,
    _client: tokio_postgres::Client,
    driver: JoinHandle<()>,
}

impl Drop for PgSession {
    fn drop(&mut self) {
        self.driver.abort();
    }
}

/// How long the `LISTEN` statements may take.
const LISTEN_TIMEOUT: Duration = Duration::from_secs(10);
/// Notifications the driver keeps for a slow consumer. A full queue drops the newest: a
/// notification is a wake-up, and the consumer polls anyway.
const DRIVER_QUEUE: usize = 64;

fn listen_sql(channels: &[&str]) -> Result<String, &'static str> {
    let mut sql = String::new();
    for channel in channels {
        let valid = !channel.is_empty()
            && channel
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !valid {
            return Err("a channel name is lowercase ASCII letters, digits and underscores");
        }
        sql.push_str("LISTEN ");
        sql.push_str(channel);
        sql.push_str(";\n");
    }
    Ok(sql)
}

impl ListenConnector for PgConnector {
    type Session = PgSession;
    type Error = DbError;

    async fn connect(&mut self) -> Result<PgSession, DbError> {
        let sql = listen_sql(self.channels).map_err(|why| {
            DbError::Settings(crate::settings::SettingsError::Invalid {
                key: "listen channels",
                reason: why.to_owned(),
            })
        })?;
        let config = self
            .settings
            .pg_config(&self.application_name, true, false)?;
        let (client, mut connection) = config
            .connect(self.settings.tls_connector()?)
            .await
            .map_err(|e| DbError::call("listen.connect", e))?;
        let (tx, notifications) = mpsc::channel(DRIVER_QUEUE);
        let driver = tokio::spawn(async move {
            loop {
                match poll_fn(|cx| connection.poll_message(cx)).await {
                    Some(Ok(AsyncMessage::Notification(n))) => {
                        let _ = tx.try_send(Notification {
                            channel: n.channel().to_owned(),
                            payload: n.payload().to_owned(),
                        });
                    }
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        tracing::warn!(%error, "the LISTEN connection failed");
                        return;
                    }
                    None => return,
                }
            }
        });
        let session = PgSession {
            notifications,
            _client: client,
            driver,
        };
        // The driver task must be running for the statements to be answered.
        match tokio::time::timeout(LISTEN_TIMEOUT, session._client.batch_execute(&sql)).await {
            Ok(Ok(())) => Ok(session),
            Ok(Err(e)) => Err(DbError::call("listen.listen", e)),
            Err(_) => Err(DbError::CallTimeout {
                what: "listen.listen",
                seconds: LISTEN_TIMEOUT.as_secs(),
            }),
        }
    }
}

impl ListenSession for PgSession {
    type Error = DbError;

    async fn next_notification(&mut self) -> Result<Option<Notification>, DbError> {
        Ok(self.notifications.recv().await)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::fmt;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Debug)]
    struct Down;
    impl fmt::Display for Down {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("down")
        }
    }
    impl std::error::Error for Down {}

    type Script = VecDeque<Result<Option<Notification>, Down>>;

    struct FakeSession(Script);
    impl ListenSession for FakeSession {
        type Error = Down;
        async fn next_notification(&mut self) -> Result<Option<Notification>, Down> {
            match self.0.pop_front() {
                Some(step) => step,
                None => std::future::pending().await,
            }
        }
    }

    /// Each connect attempt pops one answer: a script for a session, or a refusal.
    struct FakeConnector {
        attempts: Arc<Mutex<Vec<tokio::time::Instant>>>,
        answers: VecDeque<Option<Script>>,
    }
    impl ListenConnector for FakeConnector {
        type Session = FakeSession;
        type Error = Down;
        async fn connect(&mut self) -> Result<FakeSession, Down> {
            self.attempts
                .lock()
                .unwrap()
                .push(tokio::time::Instant::now());
            match self.answers.pop_front() {
                Some(Some(script)) => Ok(FakeSession(script)),
                Some(None) => Err(Down),
                None => std::future::pending().await,
            }
        }
    }

    fn note(payload: &str) -> Result<Option<Notification>, Down> {
        Ok(Some(Notification {
            channel: "op_jobs".into(),
            payload: payload.into(),
        }))
    }

    fn fake(
        answers: Vec<Option<Script>>,
    ) -> (FakeConnector, Arc<Mutex<Vec<tokio::time::Instant>>>) {
        let attempts = Arc::new(Mutex::new(Vec::new()));
        let connector = FakeConnector {
            attempts: attempts.clone(),
            answers: answers.into(),
        };
        (connector, attempts)
    }

    const BACKOFF: Backoff = Backoff {
        initial: Duration::from_secs(1),
        max: Duration::from_secs(4),
    };

    #[tokio::test(start_paused = true)]
    async fn notifications_are_forwarded_and_the_first_connect_is_not_a_reconnect() {
        let (connector, _) = fake(vec![Some(VecDeque::from([note("a"), note("b")]))]);
        let mut listener = Listener::spawn(connector, BACKOFF);
        for want in ["a", "b"] {
            match listener.recv().await {
                Some(ListenerEvent::Notification(n)) => assert_eq!(n.payload, want),
                other => panic!("expected a notification, got {other:?}"),
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn after_a_lost_connection_it_reconnects_and_tells_the_consumer_to_poll_at_once() {
        let (connector, attempts) = fake(vec![
            Some(VecDeque::from([note("a"), Ok(None)])),
            Some(VecDeque::from([note("b")])),
        ]);
        let mut listener = Listener::spawn(connector, BACKOFF);
        assert!(matches!(
            listener.recv().await,
            Some(ListenerEvent::Notification(_))
        ));
        assert_eq!(listener.recv().await, Some(ListenerEvent::Reconnected));
        match listener.recv().await {
            Some(ListenerEvent::Notification(n)) => assert_eq!(n.payload, "b"),
            other => panic!("expected the notification after the reconnect, got {other:?}"),
        }
        assert_eq!(attempts.lock().unwrap().len(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn a_broken_connection_is_replaced_like_a_closed_one() {
        let (connector, _) = fake(vec![
            Some(VecDeque::from([Err(Down)])),
            Some(VecDeque::from([note("c")])),
        ]);
        let mut listener = Listener::spawn(connector, BACKOFF);
        assert_eq!(listener.recv().await, Some(ListenerEvent::Reconnected));
        assert!(matches!(
            listener.recv().await,
            Some(ListenerEvent::Notification(_))
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn refused_connections_back_off_up_to_the_bound_and_a_success_resets_the_wait() {
        let (connector, attempts) = fake(vec![
            None,
            None,
            None,
            None,
            Some(VecDeque::from([Ok(None)])),
            Some(VecDeque::from([note("z")])),
        ]);
        let mut listener = Listener::spawn(connector, BACKOFF);
        // the first session that worked is the first connect, not a reconnect; the second is
        assert_eq!(listener.recv().await, Some(ListenerEvent::Reconnected));
        assert!(matches!(
            listener.recv().await,
            Some(ListenerEvent::Notification(_))
        ));
        let times = attempts.lock().unwrap().clone();
        let gaps: Vec<u64> = times.windows(2).map(|w| (w[1] - w[0]).as_secs()).collect();
        // refused four times: waits 1, 2, 4, 4 (bounded); the session that worked then closed:
        // reconnect after the reset wait of 1 s.
        assert_eq!(gaps, vec![1, 2, 4, 4, 1]);
    }

    #[test]
    fn the_listen_statement_is_built_only_from_plain_channel_names() {
        assert_eq!(
            listen_sql(&["op_jobs", "op_events"]).unwrap(),
            "LISTEN op_jobs;\nLISTEN op_events;\n"
        );
        assert!(listen_sql(&["op_jobs; DROP TABLE jobs"]).is_err());
        assert!(listen_sql(&[""]).is_err());
    }
}
