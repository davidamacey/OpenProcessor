/**
 * Global vitest setup. Seeds `scoped()`'s active-project prefix to
 * `API_PREFIX` before any test runs — P1 projects cutover made `scoped()`
 * throw `ProjectNotSelectedError` until `setScopedPrefix()` has been
 * called (normally by `projectsStore.load()` at app boot). The vast
 * majority of existing tests exercise scoped call sites directly and
 * don't care about project bootstrap at all, so this keeps them
 * byte-identical to pre-cutover behavior; a test that specifically
 * exercises project switching calls `setScopedPrefix()`/`projectsStore`
 * itself.
 */
import { API_PREFIX, setScopedPrefix } from '$lib/api';

setScopedPrefix(API_PREFIX);
