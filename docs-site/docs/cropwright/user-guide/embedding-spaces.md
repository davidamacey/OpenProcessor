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
- **Build progress.** A build that is running shows its job status and the documents done, failed
  and skipped, as the server reports them, until the job ends, whether it was started on this
  page, from the API or from another tab. A space whose last build has ended shows that run's
  status and mode.
- **Restart or backfill** starts a failed or cancelled build again, or embeds the documents a
  ready space lacks.
- **Cancel build** stops a running build. Without "abandon" the space stays building and can
  be restarted as a backfill; with it, the space is marked failed and its partial vectors are cleared.
- **Evaluation** shows the evaluation runs the server has recorded for the space: the verdict,
  the protocol, who ran it and when, the metrics and the sample as stored, and which run compares
  it with the current primary (the one a promote asks for). When none has been recorded it says
  so.
- **Promote** makes a space the primary and shows the same evaluation. The server refuses until the build is complete
  ("not ready") and until an evaluation against the current primary exists. After that second
  refusal only, the dialog offers **Promote without an evaluation**, which waives just that.
- **Retire** removes a space that is not the primary, and queues the run that clears its vectors.

The page computes no comparison between spaces: the evaluation is what the server recorded.
Nothing records evaluations yet (the evaluate route is not built), so until it is the list is empty
and a promote needs **Promote without an evaluation**.
