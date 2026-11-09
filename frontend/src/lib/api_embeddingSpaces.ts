/** `{prefix}/embedding/spaces`: the spaces a project holds. Types in `$lib/types_embeddingSpaces`. */
import { apiFetch, qs, scoped } from '$lib/api';
import type {
  EmbeddingSpaceListResponse,
  EmbeddingSpaceView,
  EmbeddingTarget,
} from '$lib/types_embeddingSpaces';

/** `GET /embedding/spaces`; `target` narrows to one target (the server default is all). */
export async function getEmbeddingSpaces(
  target?: EmbeddingTarget,
  signal?: AbortSignal,
): Promise<EmbeddingSpaceView[]> {
  const res = await apiFetch<EmbeddingSpaceListResponse>(
    `${scoped()}/embedding/spaces${qs({ target })}`,
    {},
    signal,
  );
  return res.spaces;
}
