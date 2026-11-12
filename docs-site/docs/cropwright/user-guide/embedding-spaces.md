---
sidebar_position: 8
title: Embedding spaces
---

# Embedding spaces (`/p/<project>/settings/embedding-spaces`)

A space is one set of vectors for a target made by one encoder version. The page lists the
project's spaces per target, the encoders the deployment can serve, and builds, promotes and
retires spaces. The routes and refusals are in the
[Embedding spaces guide](../../guides/embedding-spaces.mdx).

## The page

- **Spaces.** One table per target that has spaces: the encoder and size, the role (primary,
  optional, candidate), the status, when it became ready and how many documents hold a vector.
- **Encoders.** The encoders this deployment can serve, with their sizes and licence. One that
  cannot be used now says why.
- **Register a space** picks an encoder and a target (and a smaller size when the encoder offers
  one) and starts the build. Pressing it twice does not start two builds.
- **Build progress.** A build started from this page shows its job status and the documents done,
  failed and skipped, as the server reports them, until the job ends. A build started somewhere
  else shows its status and vector count; restart it here to follow it.
- **Restart or backfill** starts a failed or cancelled build again, or embeds the documents a
  ready space lacks.
- **Cancel build** stops a build started here. Without "abandon" the space stays building and can
  be restarted as a backfill; with it, the space is marked failed and its partial vectors are cleared.
- **Promote** makes a space the primary. The server refuses until the build is complete
  ("not ready") and until an evaluation against the current primary exists. After that second
  refusal only, the dialog offers **Promote without an evaluation**, which waives just that.
- **Retire** removes a space that is not the primary, and queues the run that clears its vectors.

The page shows no comparison between spaces: it shows only what the server serves, and the
server serves none.
