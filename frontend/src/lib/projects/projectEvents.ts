/**
 * What the root layout does with a global `project.*` event: re-read the
 * served project list (its `paused`, status and counts), and for a pause
 * or resume also re-read that project's `GET {prefix}/pause` — the fuller
 * answer (who holds the pause and why) — so another tab's action shows
 * here without a reload.
 */
import type { ProjectEvent } from '$lib/sse';
import { projectPauseStore } from '$stores/projectPause.svelte';
import { projectsStore } from '$stores/projects.svelte';

export async function handleProjectEvent(event: ProjectEvent): Promise<void> {
  await projectsStore.refresh();
  if (event.type !== 'project.paused' && event.type !== 'project.resumed') return;
  const target = projectsStore.list.find((p) => p.slug === event.target);
  if (target) await projectPauseStore.load(target);
}
