//! The calls the readiness check makes on the model server (Triton, explicit model control):
//! `ModelReady` and `RepositoryModelLoad` (`AsyncTritonPool.is_model_ready` and `load_model`).

use std::error::Error;
use std::fmt;
use std::time::Duration;

use tonic::{Code, Request};

use crate::channel::{Client, model_server_channel};
use crate::error::{ConfigError, code_name};
use crate::proto::{ModelReadyRequest, RepositoryModelLoadRequest};
use crate::settings::ModelServerSettings;

/// A readiness answer is a flag read: it gets seconds, not a photo's deadline.
pub const READY_TIMEOUT: Duration = Duration::from_secs(5);
/// Loading a TensorRT engine into the server can take most of a minute.
pub const LOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// The model server refused or could not take a load request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadError {
    /// The model.
    pub model: String,
    /// The status of the call (`DEADLINE_EXCEEDED` when it ran out of time).
    pub code: Code,
    /// The status's message.
    pub message: String,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "load_model({:?}) failed ({}): {}",
            self.model,
            code_name(self.code),
            self.message
        )
    }
}

impl Error for LoadError {}

/// A connection to the model server.
#[derive(Clone)]
pub struct ModelServer {
    client: Client,
}

impl ModelServer {
    /// A client that dials on first use.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Url`] when the address is not a URI.
    ///
    /// # Panics
    ///
    /// Panics when called outside a tokio runtime, which the channel's worker needs.
    pub fn connect_lazy(settings: &ModelServerSettings) -> Result<Self, ConfigError> {
        Ok(Self {
            client: model_server_channel(&settings.url)?,
        })
    }

    /// Whether the server says `model` is ready. Any failure to ask (unreachable, unknown model,
    /// no answer in [`READY_TIMEOUT`]) is `false`, with the reason logged: "cannot tell" is "not
    /// ready" to a start-up gate.
    pub async fn is_model_ready(&self, model: &str) -> bool {
        let mut client = self.client.clone();
        let request = ModelReadyRequest {
            name: model.to_owned(),
            version: String::new(),
        };
        match tokio::time::timeout(READY_TIMEOUT, client.model_ready(Request::new(request))).await {
            Ok(Ok(answer)) => answer.into_inner().ready,
            Ok(Err(status)) => {
                tracing::warn!(model, error = %status, "is_model_ready failed");
                false
            }
            Err(_) => {
                tracing::warn!(model, "is_model_ready timed out");
                false
            }
        }
    }

    /// Asks the server to load `model` (explicit model control).
    ///
    /// # Errors
    ///
    /// [`LoadError`] when the server refuses, cannot be reached or does not answer in
    /// [`LOAD_TIMEOUT`].
    pub async fn load_model(&self, model: &str) -> Result<(), LoadError> {
        let mut client = self.client.clone();
        let request = RepositoryModelLoadRequest {
            repository_name: String::new(),
            model_name: model.to_owned(),
            parameters: std::collections::HashMap::new(),
        };
        let failed = |code, message: String| LoadError {
            model: model.to_owned(),
            code,
            message,
        };
        match tokio::time::timeout(
            LOAD_TIMEOUT,
            client.repository_model_load(Request::new(request)),
        )
        .await
        {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(status)) => Err(failed(status.code(), status.message().to_owned())),
            Err(_) => Err(failed(
                Code::DeadlineExceeded,
                format!("no answer in {LOAD_TIMEOUT:?}"),
            )),
        }
    }
}
