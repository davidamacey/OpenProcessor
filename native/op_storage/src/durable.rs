//! The durable object store and the mover (plan 18 sections 2.2 and 4.2 step 4).
//!
//! One interface, `object_store` 0.14.2, for a local directory and for S3: the same code runs on
//! both, and the same tests run on `InMemory`, a temporary directory and (owed to the sweep) an
//! S3-compatible server. <https://docs.rs/object_store/0.14.2/object_store/>
//!
//! S3 is built from the crate's `aws-base`, `reqwest` and `ring` features, not its `aws` feature:
//! `aws` pulls `aws-lc-rs` and with it `aws-lc-sys`, whose OpenSSL licence is not on
//! `native/deny.toml`'s allow list. TLS is rustls with the ring provider (reqwest's
//! `rustls-no-provider` feature, which verifies against the platform's root certificates), so no
//! system OpenSSL is linked. The provider is installed once per process by
//! [`op_config::install_tls_provider`], which `op_db` uses too.

use std::path::Path as FsPath;
use std::sync::Arc;

use object_store::aws::{AmazonS3Builder, Checksum, S3ConditionalPut};
use object_store::local::LocalFileSystem;
use object_store::{
    ObjectStore, ObjectStoreExt, PutMode, PutPayload, WriteMultipart, path::Path as StorePath,
};
use op_config::Env;
use tokio::io::AsyncReadExt;

use crate::error::StorageError;
use crate::keys::StorageKey;
use crate::landing::LandedPhoto;
use crate::local::LocalRoots;

/// The setting naming the durable store: `file:///absolute/dir` or `s3://bucket`.
pub const STORAGE_URL_KEY: &str = "OP_STORAGE_URL";
/// The setting naming an S3-compatible server (empty or unset: Amazon S3).
pub const S3_ENDPOINT_KEY: &str = "OP_STORAGE_S3_ENDPOINT";
/// The setting that allows plain HTTP to that server (a local test server).
pub const S3_ALLOW_HTTP_KEY: &str = "OP_STORAGE_S3_ALLOW_HTTP";

/// Up to this size a photo is written with one `put_opts(PutMode::Create)`.
const SINGLE_PUT_MAX: u64 = 8 * 1024 * 1024;
/// Part size of a multipart upload (S3 wants at least 5 MiB per part but the last).
const MULTIPART_CHUNK: usize = 8 * 1024 * 1024;
/// Parts in flight per upload, so one upload holds at most `MULTIPART_CHUNK` times this in memory.
const MULTIPART_PARTS_IN_FLIGHT: usize = 4;
/// Bytes read from the landing file per read while streaming a multipart upload.
const READ_BLOCK: usize = 1024 * 1024;

/// Where the landing zone is relative to the store, which decides how a photo is moved in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// A local directory whose `landing/` folder is the landing root: the move is a link inside
    /// one store on one file system, no copy.
    LocalWithLanding,
    /// Any other store (a local directory elsewhere, S3): the photo is streamed from its file.
    Streamed,
}

/// A durable object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableObject {
    /// The content key of the original.
    pub key: StorageKey,
    /// Its size, as the store reports it.
    pub size: u64,
    /// `true` when this call wrote it, `false` when the same bytes were already there.
    pub created: bool,
}

/// The durable object store.
#[derive(Debug, Clone)]
pub struct Durable {
    store: Arc<dyn ObjectStore>,
    backend: Backend,
    remote: bool,
}

enum Outcome {
    Created,
    AlreadyThere,
}

impl Durable {
    /// Build the store named by `OP_STORAGE_URL`: `file:///dir` (the directory must exist) or
    /// `s3://bucket`. For S3 the endpoint and plain-HTTP switch come from `OP_STORAGE_S3_ENDPOINT`
    /// and `OP_STORAGE_S3_ALLOW_HTTP`, and credentials and region from the standard `AWS_*`
    /// variables (`AmazonS3Builder::from_env`); none is logged. Uploads carry a SHA-256 checksum
    /// and `PutMode::Create` uses S3's conditional put (`If-None-Match`).
    ///
    /// # Errors
    ///
    /// [`StorageError::Setting`] for a missing or unparsable setting, [`StorageError::Config`]
    /// for a URL that is neither form, [`StorageError::Store`] when the store cannot be built.
    pub fn from_settings(env: &Env, roots: &LocalRoots) -> Result<Self, StorageError> {
        let url: String = env
            .require(STORAGE_URL_KEY)
            .map_err(StorageError::Setting)?;
        if let Some(dir) = url.strip_prefix("file://") {
            Self::local(dir, roots)
        } else if let Some(bucket) = url.strip_prefix("s3://") {
            Self::s3(env, bucket)
        } else {
            Err(config("expected file:///absolute/dir or s3://bucket"))
        }
    }

    fn local(dir: &str, roots: &LocalRoots) -> Result<Self, StorageError> {
        if !dir.starts_with('/') {
            return Err(config("a file:// URL needs an absolute path"));
        }
        let root =
            std::fs::canonicalize(dir).map_err(|e| config(&format!("{dir} is not usable: {e}")))?;
        let store = LocalFileSystem::new_with_prefix(&root)
            .map_err(|e| config(&e.to_string()))?
            .with_fsync(true);
        let backend = if root.join("landing") == roots.landing {
            Backend::LocalWithLanding
        } else {
            Backend::Streamed
        };
        Ok(Self::from_store(Arc::new(store), backend, false))
    }

    fn s3(env: &Env, bucket: &str) -> Result<Self, StorageError> {
        if bucket.is_empty() || bucket.contains('/') {
            return Err(config("expected s3://bucket with no path"));
        }
        let endpoint: String = env
            .get(S3_ENDPOINT_KEY, String::new())
            .map_err(StorageError::Setting)?;
        let allow_http: bool = env
            .get(S3_ALLOW_HTTP_KEY, false)
            .map_err(StorageError::Setting)?;
        let mut builder = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .with_checksum_algorithm(Checksum::SHA256)
            .with_conditional_put(S3ConditionalPut::ETagMatch)
            .with_allow_http(allow_http);
        if !endpoint.is_empty() {
            builder = builder.with_endpoint(endpoint);
        }
        op_config::install_tls_provider();
        let store = builder.build().map_err(|e| config(&e.to_string()))?;
        Ok(Self::from_store(Arc::new(store), Backend::Streamed, true))
    }

    /// Wrap a store that is already built (tests, and callers that build their own). `remote` says
    /// whether it is a network store, which decides whether derived files get a local cache.
    #[must_use]
    pub fn from_store(store: Arc<dyn ObjectStore>, backend: Backend, remote: bool) -> Self {
        Self {
            store,
            backend,
            remote,
        }
    }

    /// The store, for the derived, erase and reconcile modules.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// Whether the store is a network store.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        self.remote
    }

    /// Whether an object exists under `key`.
    ///
    /// # Errors
    ///
    /// [`StorageError::Store`] when the store fails for a reason other than "not found".
    pub async fn exists(&self, key: &StorageKey) -> Result<bool, StorageError> {
        match self.store.head(key.as_path()).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(StorageError::store("head", key, e)),
        }
    }

    /// Store a landed photo under its content key `projects/<id>/originals/<xx>/<sha256>` and
    /// return only when the object exists in the store with the right size, so the record can say
    /// "stored" afterwards and never before.
    ///
    /// Nothing is ever overwritten. A photo up to 8 MiB is written with `put_opts` and
    /// `PutMode::Create`; a bigger one is streamed in 8 MiB parts with `put_multipart_opts`, after
    /// a `head` that stops the upload when the content is already there. A multipart upload cannot
    /// be create-only, so on S3 two writers of the same content may both upload: that is harmless
    /// because the key is the content hash (equal bytes), and the size check after the upload and
    /// the SHA-256 checksum S3 verifies on upload are the guards. If the content is already
    /// there, its size is checked and `created` is `false`; the caller removes its own landing
    /// file either way. Equal size is taken as equal bytes because the key is the SHA-256 the
    /// landing file was hashed with.
    ///
    /// On a local store whose `landing/` folder is the landing root ([`Backend::LocalWithLanding`])
    /// the photo is not copied: `rename_if_not_exists` hard-links it to its content key and
    /// removes the landing name, so the inode is kept. The landing file's data is `fsync`ed first
    /// (the store assumes a source object is already durable), and the store, built with
    /// `with_fsync(true)`, `fsync`s the destination's directory after the link, the directories
    /// it had to create, and the source's directory after the removal. The `fsync(2)` man page
    /// is why: "calling fsync() does not necessarily ensure that the entry in the directory
    /// containing the file has also reached disk. For that an explicit fsync() on a file
    /// descriptor for the directory is also needed" (<https://man7.org/linux/man-pages/man2/fsync.2.html>).
    ///
    /// Guaranteed when this returns `Ok` on a local store: the object's bytes and its directory
    /// entry have been handed to stable storage with `fsync`, so it survives a crash or a power
    /// loss as far as the file system and the device honour `fsync`. On S3 it is the service's
    /// own durability: the object was acknowledged and `head` reports its size.
    ///
    /// # Errors
    ///
    /// [`StorageError::SizeMismatch`] when the stored object, or the landing file, does not have
    /// `photo.size` bytes; [`StorageError::Io`] when the landing file cannot be read or synced;
    /// [`StorageError::Store`] when the store fails.
    pub async fn make_durable(&self, photo: &LandedPhoto) -> Result<DurableObject, StorageError> {
        let key = StorageKey::original(photo.project, &photo.sha256);
        let outcome = match self.backend {
            Backend::LocalWithLanding => self.link_into_store(photo, &key).await?,
            Backend::Streamed => self.put_from_file(photo, &key).await?,
        };
        let stored = self
            .store
            .head(key.as_path())
            .await
            .map_err(|e| StorageError::store("head", &key, e))?;
        if stored.size != photo.size {
            return Err(StorageError::SizeMismatch {
                key: key.to_string(),
                expected: photo.size,
                found: stored.size,
            });
        }
        Ok(DurableObject {
            key,
            size: stored.size,
            created: matches!(outcome, Outcome::Created),
        })
    }

    async fn link_into_store(
        &self,
        photo: &LandedPhoto,
        key: &StorageKey,
    ) -> Result<Outcome, StorageError> {
        sync_file(&photo.path).await?;
        match self
            .store
            .rename_if_not_exists(photo.key.as_path(), key.as_path())
            .await
        {
            Ok(()) => Ok(Outcome::Created),
            Err(object_store::Error::AlreadyExists { .. }) => Ok(Outcome::AlreadyThere),
            Err(e) => Err(StorageError::store("link", key, e)),
        }
    }

    async fn put_from_file(
        &self,
        photo: &LandedPhoto,
        key: &StorageKey,
    ) -> Result<Outcome, StorageError> {
        if photo.size <= SINGLE_PUT_MAX {
            let bytes = tokio::fs::read(&photo.path)
                .await
                .map_err(|e| StorageError::io("read", &photo.path, e))?;
            if bytes.len() as u64 != photo.size {
                return Err(StorageError::SizeMismatch {
                    key: photo.key.to_string(),
                    expected: photo.size,
                    found: bytes.len() as u64,
                });
            }
            let put = self
                .store
                .put_opts(
                    key.as_path(),
                    PutPayload::from(bytes),
                    PutMode::Create.into(),
                )
                .await;
            return match put {
                Ok(_) => Ok(Outcome::Created),
                Err(object_store::Error::AlreadyExists { .. }) => Ok(Outcome::AlreadyThere),
                Err(e) => Err(StorageError::store("put", key, e)),
            };
        }
        if self.exists(key).await? {
            return Ok(Outcome::AlreadyThere);
        }
        self.upload_multipart(&photo.path, key.as_path())
            .await
            .map_err(|e| StorageError::store("multipart put", key, e))?;
        Ok(Outcome::Created)
    }

    async fn upload_multipart(
        &self,
        path: &FsPath,
        location: &StorePath,
    ) -> Result<(), object_store::Error> {
        let generic = |e: std::io::Error| object_store::Error::Generic {
            store: "landing file",
            source: Box::new(e),
        };
        let mut file = tokio::fs::File::open(path).await.map_err(generic)?;
        let upload = self
            .store
            .put_multipart_opts(location, object_store::PutMultipartOptions::default())
            .await?;
        let mut writer = WriteMultipart::new_with_chunk_size(upload, MULTIPART_CHUNK);
        let mut block = vec![0_u8; READ_BLOCK];
        loop {
            let n = match file.read(&mut block).await {
                Ok(n) => n,
                Err(e) => {
                    let _ = writer.abort().await;
                    return Err(generic(e));
                }
            };
            if n == 0 {
                break;
            }
            writer.write(&block[..n]);
            if let Err(e) = writer.wait_for_capacity(MULTIPART_PARTS_IN_FLIGHT).await {
                let _ = writer.abort().await;
                return Err(e);
            }
        }
        writer.finish().await.map(|_| ())
    }
}

async fn sync_file(path: &FsPath) -> Result<(), StorageError> {
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|e| StorageError::io("open", path, e))?;
    file.sync_all()
        .await
        .map_err(|e| StorageError::io("fsync", path, e))
}

fn config(reason: &str) -> StorageError {
    StorageError::Config {
        key: STORAGE_URL_KEY,
        reason: reason.to_owned(),
    }
}
