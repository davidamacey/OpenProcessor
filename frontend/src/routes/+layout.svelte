<script lang="ts">
  /**
   * Root layout: only what every route shares, project-scoped or not —
   * global CSS, the toast host, the blocking error for an unreachable
   * project list, and the global `project.*` event stream. The app shell
   * (top bar, nav, sidebar, every project-scoped store) lives in
   * `p/[project]/+layout.svelte`; `/projects` renders its own header.
   */
  import '@fontsource-variable/inter';
  import '@fontsource-variable/jetbrains-mono';
  import '../app.css';
  import { updated } from '$app/state';
  import type { Snippet } from 'svelte';
  import Toast from '$components/Toast.svelte';
  import { subscribeGlobalEvents } from '$lib/sse';
  import { handleProjectEvent } from '$lib/projects/projectEvents';

  interface Props {
    children?: Snippet;
    data: { apiBase: string; projectsError?: string | null };
  }
  let { children, data }: Props = $props();

  // The global `project.*` stream keeps the switcher's list fresh: any
  // project event (created, archived, deleted, paused, ...) re-reads the
  // served list. A pause or resume also re-reads that project's
  // `GET {prefix}/pause` (the fuller answer: who holds the pause and why),
  // so another tab's action shows here. Never opened on a blocking
  // projectsError.
  $effect(() => {
    if (data.projectsError) return;
    const sub = subscribeGlobalEvents({
      onEvent: (event) => void handleProjectEvent(event),
    });
    return () => sub.close();
  });
</script>

{#if data.projectsError}
  <!-- No scoped call can succeed without a project, so a failed
       `GET {globalApi()}/projects` is a full blocking error state — never
       a half-rendered app with every scoped request throwing
       ProjectNotSelectedError. -->
  <div
    class="flex h-screen flex-col items-center justify-center gap-3 bg-surface-0 px-6 text-center text-fg-1"
    data-testid="projects-blocking-error"
  >
    <p class="text-title font-semibold">Can't reach the backend's project list</p>
    <p class="max-w-md text-body text-fg-2">{data.projectsError}</p>
    <button
      type="button"
      class="rounded border border-line-strong px-3 py-1.5 text-body hover:border-fg-3"
      onclick={() => window.location.reload()}
    >
      Retry
    </button>
  </div>
{:else}
  {@render children?.()}
  {#if updated.current}
    <div
      class="fixed right-3 bottom-3 z-80 flex items-center gap-3 rounded border border-line-strong bg-surface-1 px-3 py-2 text-body text-fg-1"
      role="status"
      data-testid="new-version-banner"
    >
      <span>A new version is available.</span>
      <button
        type="button"
        class="rounded border border-line-strong px-2 py-0.5 hover:border-fg-3"
        onclick={() => window.location.reload()}
      >
        Reload
      </button>
    </div>
  {/if}
  <Toast />
{/if}
