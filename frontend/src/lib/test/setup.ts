/**
 * Global vitest setup. Selects a test project before any test runs —
 * `scoped()` throws `ProjectNotSelectedError` and `projectHref()` refuses
 * to build a link until a project is active (normally the `/p/[project]`
 * layout's `projectsStore.select()`). The test project's `prefix` is
 * `API_PREFIX` itself, so every existing scoped-call test keeps asserting
 * against `API_PREFIX` literals unchanged, and its slug is `default`, so
 * project links read `/p/default/...`. A test that exercises project
 * switching drives `projectsStore`/`setScopedPrefix()` itself.
 */
import { API_PREFIX } from '$lib/api';
import { projectsStore } from '$stores/projects.svelte';
import { testProject } from './fixtures/projects';

projectsStore.select(
  testProject({ slug: 'default', prefix: API_PREFIX, is_default: true }),
);
