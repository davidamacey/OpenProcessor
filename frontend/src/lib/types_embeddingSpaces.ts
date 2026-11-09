/**
 * Embedding spaces (`{prefix}/embedding/spaces`): the vector spaces a project
 * holds per target. Search takes a ready space key (`space`) and more keys to
 * fuse (`fuse`); the response echoes both. Pinned to the OpenAPI by
 * `contract/searchSpaceContract.test.ts`.
 */

export const EMBEDDING_TARGETS = ['item', 'box', 'frame', 'item_text'] as const;
export type EmbeddingTarget = (typeof EMBEDDING_TARGETS)[number];

export const EMBEDDING_SPACE_ROLES = ['primary', 'optional', 'candidate'] as const;
export type EmbeddingSpaceRole = (typeof EMBEDDING_SPACE_ROLES)[number];

export const EMBEDDING_SPACE_STATUSES = [
  'pending',
  'building',
  'ready',
  'retiring',
  'retired',
  'failed',
] as const;
export type EmbeddingSpaceStatus = (typeof EMBEDDING_SPACE_STATUSES)[number];

export interface EmbeddingSpaceView {
  space_key: string;
  target: EmbeddingTarget;
  model_key: string;
  model_version: number;
  dim: number;
  role: EmbeddingSpaceRole;
  status: EmbeddingSpaceStatus;
  ready_at: string | null;
  docs_with_vector: number | null;
}

export interface EmbeddingSpaceListResponse {
  spaces: EmbeddingSpaceView[];
}
