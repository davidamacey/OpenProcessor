//! Streaming one part of the form: a photo into the landing zone, a text field into a bounded
//! string, any other file into nothing.

use axum::extract::multipart::Field;
use op_storage::landing::LandedPhoto;

use super::errors::{body_refusal, storage_refusal};
use super::intake::{Image, Intake, MAX_TEXT_BYTES, Slot};
use super::refusal::Refusal;
use super::sink::PartSink;
use super::writer::{PartPipe, Stop};
use crate::admission::WorkPermit;

impl<S: PartSink> Intake<S> {
    /// One `images` file part.
    pub(super) async fn image(&mut self, field: &mut Field<'_>) -> Result<(), Refusal> {
        self.ordinal += 1;
        let index = self.images.len();
        let filename = field.file_name().map(str::to_string);
        let skipped = self.doomed();
        self.images.push(Image {
            filename,
            slot: Slot::Skipped,
        });
        if skipped {
            self.drain(field).await?;
            return Ok(());
        }
        let Some((photo, size, permit)) = self.land(field, index).await? else {
            self.images[index].slot = Slot::Empty;
            return Ok(());
        };
        match self.known_identifier(index) {
            Some(identifier) if !self.holding => {
                self.settle(index, identifier, photo, size, permit).await
            }
            _ => {
                // nothing is in work while the photo waits for its identifier, so it holds no share
                drop(permit);
                self.holding = true;
                self.images[index].slot = Slot::Held { photo, size };
                Ok(())
            }
        }
    }

    /// Land the part under a photo permit and the disk guard. `None` for a part without bytes;
    /// otherwise the permit comes back, shrunk to the photo's size, to travel with it.
    async fn land(
        &mut self,
        field: &mut Field<'_>,
        index: usize,
    ) -> Result<Option<(LandedPhoto, u64, WorkPermit)>, Refusal> {
        let mut permit = self.state.admission.admit_photo().await?;
        self.state.disk.check().await?;
        let number = u32::try_from(index).map_err(|_| Refusal::Internal)?;
        let writer = self.state.landing.begin_part(self.upload_id, number);
        let pipe = PartPipe::start(self.state.as_ref(), self.project, self.upload_id, writer);
        match self.pump(field, &pipe, &mut permit).await {
            Ok(size) => {
                permit.settle(size);
                let photo = pipe
                    .end()
                    .await
                    .map_err(|e| storage_refusal(&e, self.state.disk.volume()))?;
                if let Some(photo) = &photo {
                    self.scope.track(photo.clone());
                }
                Ok(photo.map(|p| (p, size, permit)))
            }
            Err(Stop::Refused(refusal)) => Err(refusal),
            Err(Stop::WriterGone) => Err(match pipe.end().await {
                Err(e) => storage_refusal(&e, self.state.disk.volume()),
                Ok(_) => Refusal::Internal,
            }),
        }
    }

    /// Chunks from the body to the writer task, the cap checked on the running size. When this
    /// returns an error, `pipe` is dropped by the caller and the writer task discards the part.
    async fn pump(
        &self,
        field: &mut Field<'_>,
        pipe: &PartPipe,
        permit: &mut WorkPermit,
    ) -> Result<u64, Stop> {
        let settings = &self.state.settings;
        let reserve = self
            .content_length
            .map_or(settings.max_file_bytes, |n| n.min(settings.max_file_bytes));
        let mut size = 0_u64;
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|e| Stop::Refused(body_refusal(&e, settings)))?
        {
            if chunk.is_empty() {
                continue;
            }
            if size == 0 {
                self.state
                    .admission
                    .reserve_bytes(permit, reserve)
                    .await
                    .map_err(|busy| Stop::Refused(busy.into()))?;
            }
            size += chunk.len() as u64;
            if size > settings.max_file_bytes {
                return Err(Stop::Refused(Refusal::FileTooLarge {
                    max_file_size_mb: settings.max_file_mb(),
                }));
            }
            pipe.send(chunk).await?;
        }
        Ok(size)
    }

    /// Read a part to its end and drop it, counting its bytes against the file cap. Returns the
    /// size.
    pub(super) async fn drain(&self, field: &mut Field<'_>) -> Result<u64, Refusal> {
        let settings = &self.state.settings;
        let mut size = 0_u64;
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|e| body_refusal(&e, settings))?
        {
            size += chunk.len() as u64;
            if size > settings.max_file_bytes {
                return Err(Refusal::FileTooLarge {
                    max_file_size_mb: settings.max_file_mb(),
                });
            }
        }
        Ok(size)
    }

    /// A text field, at most as large as Starlette allows a non-file part.
    pub(super) async fn read_text(&self, field: &mut Field<'_>) -> Result<String, Refusal> {
        let settings = &self.state.settings;
        let mut bytes = Vec::new();
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|e| body_refusal(&e, settings))?
        {
            if bytes.len() + chunk.len() > MAX_TEXT_BYTES {
                return Err(Refusal::Malformed(format!(
                    "Part exceeded maximum size of {}KB.",
                    MAX_TEXT_BYTES / 1024
                )));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
