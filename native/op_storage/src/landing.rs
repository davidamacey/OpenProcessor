//! The landing zone: where an upload's bytes arrive on local fast disk (plan 18 section 4.2).
//!
//! A part is written to `<landing>/<upload_id>/<index>.part` while its SHA-256 is computed over the
//! same bytes. Only [`PartWriter::finish`] makes it a [`LandedPart`], and only
//! [`Landing::name_by_content`] gives it the name `landing/<project>/<sha256>.<upload_id>.<index>`,
//! so a half written part is never visible under a name that carries its hash. The name carries
//! the part it came from, so a landing file has exactly one owner: two parts with equal bytes get
//! two files and [`Landing::remove`] only ever deletes its caller's file. Equal bytes meet only in
//! the content-addressed object store and in the database's content claim.
//! A part whose upload is dropped half way stays under its upload folder until the reconcile tick
//! removes stale landing files.
//!
//! File and hashing work does not block the executor. `tokio::fs` runs each file call on tokio's
//! blocking pool (<https://docs.rs/tokio/1.54.1/tokio/fs/index.html>), so the file is wrapped in a
//! [`BufWriter`] and a hop to that pool is paid per 256 KiB, not per network chunk. The hash is
//! computed by [`Sha256Stream::update_cooperative`], which yields to the executor between 64 KiB
//! slices (why: its documentation).

use std::path::{Path, PathBuf};

use bytes::Bytes;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use uuid::Uuid;

use crate::error::StorageError;
use crate::hash::Sha256Stream;
use crate::keys::{ContentHash, ProjectId, StorageKey};
use crate::local::LocalRoots;

/// Bytes buffered before one write reaches the blocking pool.
const WRITE_BUFFER: usize = 256 * 1024;

/// The landing zone under one canonical root.
#[derive(Debug, Clone)]
pub struct Landing {
    root: PathBuf,
}

/// A part that is completely written and flushed, and not yet named by its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandedPart {
    /// The upload the part belongs to.
    pub upload_id: Uuid,
    /// The part's index in that upload.
    pub index: u32,
    /// Where the part is, inside the landing root.
    pub path: PathBuf,
    /// Its size in bytes.
    pub size: u64,
    /// The SHA-256 of its bytes, computed while they were written.
    pub sha256: ContentHash,
}

/// A photo under its own landing name, `landing/<project>/<sha256>.<upload_id>.<index>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandedPhoto {
    /// The landing key.
    pub key: StorageKey,
    /// Where the file is, inside the landing root.
    pub path: PathBuf,
    /// Its size in bytes.
    pub size: u64,
    /// Its SHA-256.
    pub sha256: ContentHash,
}

impl Landing {
    /// The landing zone of `roots`.
    #[must_use]
    pub fn new(roots: &LocalRoots) -> Self {
        Self {
            root: roots.landing.clone(),
        }
    }

    /// The canonical landing root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Start part `index` of upload `upload_id`. The file is created by the first
    /// [`PartWriter::write`] (or by [`PartWriter::finish`] for an empty part), and creating it
    /// fails if it exists, so two writers never share a part.
    #[must_use]
    pub fn begin_part(&self, upload_id: Uuid, index: u32) -> PartWriter {
        PartWriter {
            upload_id,
            index,
            path: self
                .root
                .join(upload_id.to_string())
                .join(format!("{index}.part")),
            file: None,
            hash: Sha256Stream::new(),
        }
    }

    /// Give a finished part its name `<sha256>.<upload_id>.<index>` under its project, moving it
    /// inside the landing root without replacing anything: the new name is made with `link(2)`,
    /// which fails with `EEXIST` rather than overwrite (<https://man7.org/linux/man-pages/man2/link.2.html>),
    /// and the part's own name is then removed, so the inode is kept. A name that is already taken
    /// is an error, never a silent overwrite. `object_store`'s local `PutMode::Create` links
    /// the same way, so the landing volume needs hard links like the store does.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutsideRoot`] when the part is not inside the landing root,
    /// [`StorageError::Io`] when the folder cannot be made, the name is taken or the move fails.
    pub async fn name_by_content(
        &self,
        project: ProjectId,
        part: LandedPart,
    ) -> Result<LandedPhoto, StorageError> {
        self.check_inside(&part.path)?;
        let key = StorageKey::landing(project, &part.sha256, part.upload_id, part.index);
        let dir = self.root.join(project.to_string());
        let path = dir.join(format!("{}.{}.{}", part.sha256, part.upload_id, part.index));
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| StorageError::io("create folder", &dir, e))?;
        tokio::fs::hard_link(&part.path, &path)
            .await
            .map_err(|e| StorageError::io("link", &path, e))?;
        if let Err(e) = tokio::fs::remove_file(&part.path).await {
            let _ = tokio::fs::remove_file(&path).await;
            return Err(StorageError::io("remove", &part.path, e));
        }
        Ok(LandedPhoto {
            key,
            path,
            size: part.size,
            sha256: part.sha256,
        })
    }

    /// Delete a landed photo: its caller's own file, which no other task shares. A file that is
    /// already gone is not an error.
    ///
    /// # Errors
    ///
    /// [`StorageError::OutsideRoot`] when the photo's path is not inside the landing root,
    /// [`StorageError::Io`] when the delete fails for another reason.
    pub async fn remove(&self, photo: &LandedPhoto) -> Result<(), StorageError> {
        self.check_inside(&photo.path)?;
        match tokio::fs::remove_file(&photo.path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StorageError::io("remove", &photo.path, e)),
        }
    }

    /// The free fraction (0.0 to 1.0) of the volume holding the landing root, as the disk guard
    /// measures it: available blocks over total blocks (`statvfs`, the call behind Python's
    /// `shutil.disk_usage`). Runs on the blocking pool.
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] when the volume cannot be measured.
    pub async fn free_ratio(&self) -> Result<f64, StorageError> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || {
            let stat = rustix::fs::statvfs(&root)
                .map_err(|e| StorageError::io("statvfs", &root, e.into()))?;
            #[expect(
                clippy::cast_precision_loss,
                reason = "block counts of a volume are far below 2^52, and a ratio needs no more"
            )]
            let ratio = if stat.f_blocks == 0 {
                0.0
            } else {
                stat.f_bavail as f64 / stat.f_blocks as f64
            };
            Ok(ratio)
        })
        .await
        .map_err(|e| StorageError::io("statvfs", &self.root, e.into()))?
    }

    fn check_inside(&self, path: &Path) -> Result<(), StorageError> {
        let clean = path
            .components()
            .all(|c| !matches!(c, std::path::Component::ParentDir));
        if clean && path.starts_with(&self.root) {
            Ok(())
        } else {
            Err(StorageError::OutsideRoot {
                path: path.to_path_buf(),
            })
        }
    }
}

/// Writes one part and hashes the same bytes. After an error from [`PartWriter::write`] the part
/// is unusable: call [`PartWriter::discard`].
#[derive(Debug)]
pub struct PartWriter {
    upload_id: Uuid,
    index: u32,
    path: PathBuf,
    file: Option<BufWriter<File>>,
    hash: Sha256Stream,
}

impl PartWriter {
    /// Append `chunk` to the part and to the hash.
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] when the file cannot be created or written (a full disk).
    pub async fn write(&mut self, chunk: &Bytes) -> Result<(), StorageError> {
        self.hash.update_cooperative(chunk).await;
        let path = self.path.clone();
        let file = self.open().await?;
        file.write_all(chunk)
            .await
            .map_err(|e| StorageError::io("write", &path, e))
    }

    /// Flush the part and return it with its size and SHA-256. Nothing is visible under a content
    /// name until [`Landing::name_by_content`].
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] when the file cannot be created or flushed.
    pub async fn finish(mut self) -> Result<LandedPart, StorageError> {
        let path = self.path.clone();
        let file = self.open().await?;
        file.flush()
            .await
            .map_err(|e| StorageError::io("flush", &path, e))?;
        Ok(LandedPart {
            upload_id: self.upload_id,
            index: self.index,
            path,
            size: self.hash.len(),
            sha256: self.hash.finish(),
        })
    }

    /// Throw the part away. A part that was never created is not an error.
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] when the file exists and cannot be deleted.
    pub async fn discard(self) -> Result<(), StorageError> {
        drop(self.file);
        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StorageError::io("remove", &self.path, e)),
        }
    }

    async fn open(&mut self) -> Result<&mut BufWriter<File>, StorageError> {
        if self.file.is_none() {
            if let Some(dir) = self.path.parent() {
                tokio::fs::create_dir_all(dir)
                    .await
                    .map_err(|e| StorageError::io("create folder", dir, e))?;
            }
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.path)
                .await
                .map_err(|e| StorageError::io("create", &self.path, e))?;
            self.file = Some(BufWriter::with_capacity(WRITE_BUFFER, file));
        }
        self.file.as_mut().ok_or_else(|| StorageError::OutsideRoot {
            path: self.path.clone(),
        })
    }
}
