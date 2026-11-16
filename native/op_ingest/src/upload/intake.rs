//! One request's pass over its form: the fields, the photos, and the checks that need the whole
//! form. The streaming of one photo into the landing zone is in `part.rs`.
//!
//! Python reads the whole request and then validates; here the request is validated while it
//! streams, and the answer a client sees is the same. A fault that Python reports only after
//! parsing, and that more than one condition can cause (too many photos, a bad `image_paths`, too
//! many bytes), is not answered at the first sign: the rest of the body is read and dropped, with
//! nothing landed, so the message can name the real count and the precedence is Python's. Faults
//! that stop the request at once are the ones Python also raises first or that no later part can
//! change: a file over the cap, a body that is not a form, a full disk, a part that cannot be
//! admitted.

use std::sync::Arc;

use axum::extract::Multipart;
use axum::extract::multipart::Field;
use op_storage::keys::ProjectId;
use op_storage::landing::LandedPhoto;
use serde_json::Value;
use uuid::Uuid;

use super::UploadState;
use super::cleanup::Scope;
use super::errors::body_refusal;
use super::form::{
    default_identifier, is_accepted, is_blank, parse_image_paths, unsupported_message,
    upload_extension,
};
use super::handoff::{self, Handoff};
use super::refusal::{self, Refusal};
use super::sink::{PartSink, ReceivedPart, UploadEnd};
use super::wire::{BatchResponse, IngestImageResponse, merge};
use crate::admission::WorkPermit;

/// Starlette's limits on a form (`max_files`, `max_fields`, `max_part_size`).
const MAX_FILES: usize = 1000;
const MAX_FIELDS: usize = 1000;
pub(super) const MAX_TEXT_BYTES: usize = 1024 * 1024;

/// The text fields of the form, in the order FastAPI validates them.
const TEXT_FIELDS: [&str; 3] = ["image_paths", "source", "run_id"];

/// What became of one `images` file part.
pub(super) enum Slot {
    /// No bytes.
    Empty,
    /// Judged by its extension and refused; the file is gone.
    Refused { extension: String },
    /// Landed, waiting for the identifiers that decide its extension.
    Held { photo: LandedPhoto, size: u64 },
    /// Handed to the sink.
    Handed,
    /// Not landed: the request is already doomed and the part was read and dropped.
    Skipped,
}

pub(super) struct Image {
    pub(super) filename: Option<String>,
    pub(super) slot: Slot,
}

/// The form fields seen so far; a repeated field replaces the earlier one, as in Starlette.
#[derive(Default)]
struct Fields {
    paths: Option<Result<Vec<String>, Refusal>>,
    source: Option<String>,
    run_id: Option<String>,
}

pub(super) struct Intake<S: PartSink> {
    pub(super) state: Arc<UploadState<S>>,
    pub(super) project: ProjectId,
    pub(super) upload_id: Uuid,
    pub(super) content_length: Option<u64>,
    pub(super) scope: Scope<S>,
    pub(super) images: Vec<Image>,
    pub(super) total: u64,
    pub(super) total_exceeded: bool,
    fields: Fields,
    pub(super) ordinal: usize,
    wrong_images: Vec<Value>,
    wrong_fields: [bool; 3],
    pub(super) holding: bool,
    files_seen: usize,
    fields_seen: usize,
    handoff: Option<Handoff>,
    handed: usize,
}

impl<S: PartSink> Intake<S> {
    pub(super) fn new(
        state: Arc<UploadState<S>>,
        project: ProjectId,
        upload_id: Uuid,
        content_length: Option<u64>,
    ) -> Self {
        let scope = Scope::new(
            state.landing.clone(),
            Arc::clone(&state.sink),
            state.cleanup.clone(),
            upload_id,
        );
        Self {
            state,
            project,
            upload_id,
            content_length,
            scope,
            images: Vec::new(),
            total: 0,
            total_exceeded: false,
            fields: Fields::default(),
            ordinal: 0,
            wrong_images: Vec::new(),
            wrong_fields: [false; 3],
            holding: false,
            files_seen: 0,
            fields_seen: 0,
            handoff: None,
            handed: 0,
        }
    }

    /// Read the form and, when it passes, return the answer. On any refusal the request's files
    /// are removed and the sink is told before this returns.
    pub(super) async fn run(mut self, mut multipart: Multipart) -> Result<BatchResponse, Refusal> {
        match self.process(&mut multipart).await {
            Ok(answer) => {
                self.scope.complete().await;
                Ok(answer)
            }
            Err(refusal) => {
                if let Some(handoff) = self.handoff.take() {
                    handoff.cancel();
                }
                self.scope.abort().await;
                Err(refusal)
            }
        }
    }

    async fn process(&mut self, multipart: &mut Multipart) -> Result<BatchResponse, Refusal> {
        loop {
            let next = multipart
                .next_field()
                .await
                .map_err(|e| body_refusal(&e, &self.state.settings))?;
            let Some(mut field) = next else { break };
            self.field(&mut field).await?;
        }
        self.finish().await
    }

    async fn field(&mut self, field: &mut Field<'_>) -> Result<(), Refusal> {
        let is_file = field.file_name().is_some();
        self.count_part(is_file)?;
        let name = field.name().map(str::to_string);
        match (name.as_deref(), is_file) {
            (Some("images"), true) => self.image(field).await,
            (Some("images"), false) => {
                let text = self.read_text(field).await?;
                self.wrong_images
                    .push(refusal::images_not_a_file(self.ordinal, &text));
                self.ordinal += 1;
                Ok(())
            }
            (Some(known), true) if TEXT_FIELDS.contains(&known) => {
                self.drain(field).await?;
                if let Some(at) = TEXT_FIELDS.iter().position(|f| *f == known) {
                    self.wrong_fields[at] = true;
                }
                Ok(())
            }
            (Some(known), false) if TEXT_FIELDS.contains(&known) => {
                let text = self.read_text(field).await?;
                self.text_field(known, text);
                Ok(())
            }
            (_, true) => self.drain(field).await.map(drop),
            (_, false) => self.read_text(field).await.map(drop),
        }
    }

    fn count_part(&mut self, is_file: bool) -> Result<(), Refusal> {
        let (seen, max, what) = if is_file {
            (&mut self.files_seen, MAX_FILES, "files")
        } else {
            (&mut self.fields_seen, MAX_FIELDS, "fields")
        };
        *seen += 1;
        if *seen > max {
            return Err(Refusal::Malformed(format!(
                "Too many {what}. Maximum number of {what} is {max}."
            )));
        }
        Ok(())
    }

    fn text_field(&mut self, name: &str, text: String) {
        match name {
            "image_paths" => {
                self.fields.paths = (!is_blank(&text)).then(|| parse_image_paths(&text));
            }
            "source" => self.fields.source = (!text.is_empty()).then_some(text),
            _ => self.fields.run_id = (!text.is_empty()).then_some(text),
        }
    }

    /// Whether the request can no longer succeed, so the next photo is read and dropped.
    pub(super) fn doomed(&self) -> bool {
        self.total_exceeded
            || self.images.len() >= self.state.settings.max_images
            || match &self.fields.paths {
                Some(Err(_)) => true,
                Some(Ok(paths)) => paths.len() <= self.images.len(),
                None => false,
            }
    }

    /// The identifier of photo `index` when `image_paths` is known and covers it.
    pub(super) fn known_identifier(&self, index: usize) -> Option<String> {
        match &self.fields.paths {
            Some(Ok(paths)) => paths.get(index).cloned(),
            _ => None,
        }
    }

    fn identifier(&self, index: usize) -> String {
        self.known_identifier(index)
            .unwrap_or_else(|| default_identifier(self.images[index].filename.as_deref(), index))
    }

    /// Judge a landed photo by its extension: refuse it, or hand it over.
    pub(super) async fn settle(
        &mut self,
        index: usize,
        identifier: String,
        photo: LandedPhoto,
        size: u64,
        permit: WorkPermit,
    ) -> Result<(), Refusal> {
        let Some(extension) = self.judge(index, &identifier, &photo).await else {
            return Ok(());
        };
        self.total += size;
        if self.total > self.state.settings.max_total_bytes {
            self.total_exceeded = true;
            self.remove(&photo).await;
            self.images[index].slot = Slot::Skipped;
            return Ok(());
        }
        self.hand_over(index, identifier, extension, photo, permit)
            .await
    }

    /// The extension of an accepted photo; `None` when it is refused for its extension (its file
    /// is removed and its slot says so).
    async fn judge(
        &mut self,
        index: usize,
        identifier: &str,
        photo: &LandedPhoto,
    ) -> Option<String> {
        let extension = upload_extension(identifier, self.images[index].filename.as_deref());
        if is_accepted(&extension, &self.state.settings.accepted_extensions) {
            return Some(extension);
        }
        self.remove(photo).await;
        self.images[index].slot = Slot::Refused { extension };
        None
    }

    async fn hand_over(
        &mut self,
        index: usize,
        identifier: String,
        extension: String,
        photo: LandedPhoto,
        permit: WorkPermit,
    ) -> Result<(), Refusal> {
        let part = ReceivedPart {
            upload_id: self.upload_id,
            project: self.project,
            index,
            identifier,
            extension,
            filename: self.images[index].filename.clone(),
            photo,
            permit,
        };
        self.send(part).await?;
        self.images[index].slot = Slot::Handed;
        Ok(())
    }

    pub(super) async fn remove(&self, photo: &LandedPhoto) {
        if let Err(e) = self.state.landing.remove(photo).await {
            tracing::warn!(error = %e, "a refused photo's landing file was not removed");
        }
    }

    async fn send(&mut self, part: ReceivedPart) -> Result<(), Refusal> {
        self.scope.mark_handed();
        let state = &self.state;
        let handoff = self.handoff.get_or_insert_with(|| {
            Handoff::start(
                Arc::clone(&state.sink),
                state.settings.handoff_queue,
                &state.cleanup,
            )
        });
        if handoff.send(part).await {
            self.handed += 1;
            return Ok(());
        }
        match self.handoff.take() {
            Some(handoff) => handoff.close().await.and(Err(Refusal::Internal)),
            None => Err(Refusal::Internal),
        }
    }

    /// The end of the body: the checks that need the whole form, in Python's order, then the
    /// photos that were waiting for their identifiers, then the sink's commit.
    async fn finish(&mut self) -> Result<BatchResponse, Refusal> {
        self.check_form()?;
        self.resolve_held().await?;
        let early = self.early_failures();
        if let Some(handoff) = self.handoff.take() {
            handoff.close().await?;
        }
        let batch = if self.handed > 0 {
            let end = UploadEnd {
                upload_id: self.upload_id,
                project: self.project,
                source: self.fields.source.take().unwrap_or_else(|| "upload".into()),
                run_id: self.fields.run_id.take(),
            };
            Some(handoff::finish(&self.state.sink, end).await?)
        } else {
            None
        };
        Ok(merge(early, batch))
    }

    /// FastAPI's validation, then the route's checks: photo count, `image_paths`, bytes.
    fn check_form(&mut self) -> Result<(), Refusal> {
        let mut invalid = Vec::new();
        if self.ordinal == 0 {
            invalid.push(refusal::missing_entry("images"));
        }
        invalid.append(&mut self.wrong_images);
        invalid.extend(
            TEXT_FIELDS
                .iter()
                .zip(self.wrong_fields)
                .filter(|(_, wrong)| *wrong)
                .map(|(name, _)| refusal::text_sent_as_file(name)),
        );
        if !invalid.is_empty() {
            return Err(Refusal::Validation(invalid));
        }
        let max = self.state.settings.max_images;
        if self.images.len() > max {
            return Err(Refusal::TooManyImages {
                sent: self.images.len(),
                limit: max,
            });
        }
        match self.fields.paths.take() {
            Some(Err(refusal)) => return Err(refusal),
            Some(Ok(paths)) if paths.len() != self.images.len() => {
                return Err(Refusal::PathsMismatch {
                    paths: paths.len(),
                    images: self.images.len(),
                });
            }
            other => self.fields.paths = other,
        }
        if self.total_exceeded {
            return Err(self.total_refusal());
        }
        Ok(())
    }

    fn total_refusal(&self) -> Refusal {
        Refusal::TotalTooLarge {
            limit: self.state.settings.max_total_bytes,
        }
    }

    /// Judge the photos that waited for their identifiers, check the total, and only then hand
    /// the accepted ones over, in photo order: a request that is going to be refused for its
    /// bytes never starts work in the sink.
    async fn resolve_held(&mut self) -> Result<(), Refusal> {
        let mut ready = Vec::new();
        for index in 0..self.images.len() {
            if !matches!(self.images[index].slot, Slot::Held { .. }) {
                continue;
            }
            let Slot::Held { photo, size } =
                std::mem::replace(&mut self.images[index].slot, Slot::Skipped)
            else {
                continue;
            };
            let identifier = self.identifier(index);
            let Some(extension) = self.judge(index, &identifier, &photo).await else {
                continue;
            };
            self.total += size;
            ready.push((index, identifier, extension, photo, size));
        }
        if self.total > self.state.settings.max_total_bytes {
            return Err(self.total_refusal());
        }
        for (index, identifier, extension, photo, size) in ready {
            let mut permit = self.state.admission.admit_photo().await?;
            self.state
                .admission
                .reserve_bytes(&mut permit, size)
                .await?;
            self.hand_over(index, identifier, extension, photo, permit)
                .await?;
        }
        Ok(())
    }

    /// The photos refused before the pipeline, in photo order, with their identifiers.
    fn early_failures(&self) -> Vec<IngestImageResponse> {
        let accepted = &self.state.settings.accepted_extensions;
        self.images
            .iter()
            .enumerate()
            .filter_map(|(index, image)| match &image.slot {
                Slot::Empty => Some(IngestImageResponse::empty_upload(&self.identifier(index))),
                Slot::Refused { extension } => Some(IngestImageResponse::unsupported(
                    &self.identifier(index),
                    unsupported_message(extension, accepted),
                )),
                Slot::Held { .. } | Slot::Handed | Slot::Skipped => None,
            })
            .collect()
    }
}
