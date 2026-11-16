//! The settings of the two clients, read from the environment through `op_config::Env`.

use std::time::Duration;

use op_config::{Env, EnvError};

use crate::error::ConfigError;
use crate::retry::DEADLINE;

/// The engine's address when `OP_FRAME_ENGINE_URL` is not set (`DEFAULT_URL`).
pub const DEFAULT_ENGINE_URL: &str = "frame-engine:8001";
/// The model server's address when `TRITON_URL` is not set (`Settings.triton_url`).
pub const DEFAULT_MODEL_SERVER_URL: &str = "triton-server:8001";
/// One photo's whole pipeline plus the engine's waits: the deadline of one call
/// (`TIMEOUT_SECONDS`).
pub const CALL_TIMEOUT: Duration = Duration::from_secs(120);
/// How often an HTTP/2 ping goes out on a connection, also while no call is in flight
/// (`ENGINE_KEEPALIVE_TIME_MS`).
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
/// How long a ping may go unanswered before the connection is dropped
/// (`ENGINE_KEEPALIVE_TIMEOUT_MS`).
pub const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a dial may take.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn address(env: &Env, key: &str, default: &str) -> Result<String, EnvError> {
    let value: String = env.get(key, default.to_owned())?;
    // a blank value is no value, as `env_str` reads it
    Ok(if value.trim().is_empty() {
        default.to_owned()
    } else {
        value.trim().to_owned()
    })
}

/// How to reach the engine and how much to ask of it at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineSettings {
    /// `host:port` or a URI.
    pub url: String,
    /// HTTP/2 connections to the engine, each its own TCP connection.
    pub channels: usize,
    /// Photos in the engine's hands at once from this process; the next waits for a permit.
    pub max_in_flight: usize,
    /// How long one call may take ([`CALL_TIMEOUT`]).
    pub call_timeout: Duration,
    /// How long a photo waits for the engine across retries ([`DEADLINE`]).
    pub deadline: Duration,
}

impl EngineSettings {
    /// `OP_FRAME_ENGINE_URL` (default [`DEFAULT_ENGINE_URL`]) with the pool sizes the process
    /// takes as its share of the deployment (the API divides the totals among its processes).
    ///
    /// # Errors
    ///
    /// [`EnvError`] naming the key when the value is not Unicode.
    pub fn from_env(env: &Env, channels: usize, max_in_flight: usize) -> Result<Self, EnvError> {
        Ok(Self {
            url: address(env, "OP_FRAME_ENGINE_URL", DEFAULT_ENGINE_URL)?,
            channels,
            max_in_flight,
            call_timeout: CALL_TIMEOUT,
            deadline: DEADLINE,
        })
    }

    pub(crate) fn check(&self) -> Result<(), ConfigError> {
        if self.channels == 0 {
            return Err(ConfigError::Empty("channels"));
        }
        if self.max_in_flight == 0 {
            return Err(ConfigError::Empty("max_in_flight"));
        }
        Ok(())
    }
}

/// Where the model server is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelServerSettings {
    /// `host:port` or a URI.
    pub url: String,
}

impl ModelServerSettings {
    /// `TRITON_URL` (default [`DEFAULT_MODEL_SERVER_URL`]).
    ///
    /// # Errors
    ///
    /// [`EnvError`] naming the key when the value is not Unicode.
    pub fn from_env(env: &Env) -> Result<Self, EnvError> {
        Ok(Self {
            url: address(env, "TRITON_URL", DEFAULT_MODEL_SERVER_URL)?,
        })
    }
}
