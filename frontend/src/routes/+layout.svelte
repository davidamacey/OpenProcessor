<script lang="ts">
  /**
   * Root layout: only what every route shares, project-scoped or not —
   * global CSS, the toast host, the blocking error for an unreachable
   * project list, and the global `project.*` event stream. The app shell
   * (top bar, nav, sidebar, every project-scoped store) lives in
   * `p/[project]/+layout.svelte`; `/projects` renders its own header.
   */
  import '../app.css';
  import type { Snippet } from 'svelte';
  import Toast from '$components/Toast.svelte';
  import { subscribeGlobalEvents } from '$lib/sse';
  import { projectsStore } from '$stores/projects.svelte';

  interface Props {
    children?: Snippet;
    data: { apiBase: string; projectsError?: string | null };
  }
  let { children, data }: Props = $props();

  // The global `project.*` stream keeps the switcher's list fresh: any
  // project event (created, archived, deleted, ...) re-reads the served
  // list. Never opened on a blocking projectsError.
  $effect(() => {
    if (data.projectsError) return;
    const sub = subscribeGlobalEvents({
      onEvent: () => void projectsStore.refresh(),
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
    class="flex h-screen flex-col items-center justify-center gap-3 bg-zinc-950 px-6 text-center text-zinc-100"
    data-testid="projects-blocking-error"
  >
    <p class="text-lg font-semibold">Can't reach the backend's project list</p>
    <p class="max-w-md text-sm text-zinc-400">{data.projectsError}</p>
    <button
      type="button"
      class="rounded border border-zinc-700 px-3 py-1.5 text-sm hover:border-zinc-500"
      onclick={() => window.location.reload()}
    >
      Retry
    </button>
  </div>
{:else}
  {@render children?.()}
  <Toast />
{/if}
