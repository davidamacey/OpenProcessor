<script lang="ts">
  import '../app.css';
  import type { Snippet } from 'svelte';
  import { page } from '$app/state';
  import ClassSidebar from '$components/ClassSidebar.svelte';
  import ShortcutOverlay from '$components/ShortcutOverlay.svelte';
  import Toast from '$components/Toast.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { healthStore } from '$stores/health.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';

  interface Props {
    children?: Snippet;
    data: { apiBase: string };
  }
  let { children, data }: Props = $props();

  // Acquire singleton-store subscriptions for the lifetime of the layout.
  $effect(() => {
    const releaseHealth = healthStore.acquire();
    const releaseClasses = classesStore.acquire();
    return () => {
      releaseHealth();
      releaseClasses();
    };
  });

  const path = $derived(page.url.pathname);
  const showSidebar = $derived(path === '/clusters' || path.startsWith('/clusters/'));

  // Class filter applied to the cluster grid. Page reads from URL (?class=ID).
  const selectedClassId = $derived.by(() => {
    const v = page.url.searchParams.get('class');
    return v == null ? null : Number.isFinite(+v) ? +v : null;
  });

  function selectClass(cls: { id: number } | null): void {
    const url = new URL(page.url);
    if (cls) url.searchParams.set('class', String(cls.id));
    else url.searchParams.delete('class');
    history.pushState({}, '', url);
    // SvelteKit listens to popstate but not pushState; force a goto.
    void (async () => {
      const { goto } = await import('$app/navigation');
      void goto(url.pathname + url.search, { replaceState: false, keepFocus: true });
    })();
  }

  // Build crumbs from the path.
  const crumbs = $derived.by(() => {
    const parts = path.split('/').filter(Boolean);
    if (parts.length === 0) return [{ label: 'Dashboard', href: '/' }];
    const out: Array<{ label: string; href: string }> = [{ label: 'Home', href: '/' }];
    let acc = '';
    for (const p of parts) {
      acc += '/' + p;
      out.push({ label: decodeURIComponent(p), href: acc });
    }
    return out;
  });

  const dotClass = $derived(healthStore.ok ? 'bg-green-500' : 'bg-red-500');
  const dotTitle = $derived(
    healthStore.ok
      ? `openprocessor OK (last checked ${healthStore.lastChecked ? new Date(healthStore.lastChecked).toLocaleTimeString() : '—'})`
      : `openprocessor unavailable: ${healthStore.error ?? 'no response'}`,
  );
</script>

<div class="flex h-screen flex-col bg-zinc-950 text-zinc-100">
  <!-- Top bar -->
  <header
    class="flex h-12 shrink-0 items-center gap-4 border-b border-zinc-800 bg-zinc-950 px-4"
  >
    <a href="/" class="flex items-center gap-2 text-sm font-semibold tracking-tight">
      <span class="rounded bg-blue-600 px-1.5 py-0.5 font-mono text-xs text-white">KB</span>
      legacy Labeler
    </a>

    <nav class="flex items-center gap-1 text-sm" aria-label="Breadcrumb">
      {#each crumbs as c, i (c.href)}
        {#if i > 0}
          <span class="text-zinc-600">/</span>
        {/if}
        <a
          href={c.href}
          class="rounded px-1.5 py-0.5 text-zinc-300 hover:bg-zinc-900 hover:text-white"
          aria-current={i === crumbs.length - 1 ? 'page' : undefined}
        >
          {c.label}
        </a>
      {/each}
    </nav>

    <span class="grow"></span>

    <nav class="flex items-center gap-3 text-sm text-zinc-300">
      <a href="/clusters" class="hover:text-white">Clusters</a>
      <a href="/review" class="hover:text-white">Review</a>
      <a href="/classes" class="hover:text-white">Classes</a>
      <a href="/export" class="hover:text-white">Export</a>
      <a href="/models" class="hover:text-white">Models</a>
    </nav>

    <button
      type="button"
      class="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-xs text-zinc-300 hover:bg-zinc-800"
      onclick={() => keyboardStore.toggleOverlay()}
      title="Keyboard shortcuts (~)"
    >
      ?
    </button>

    <span
      class="flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-300"
      title={dotTitle}
    >
      <span class="h-2 w-2 rounded-full {dotClass}"></span>
      <span class="font-mono">{healthStore.ok ? 'API OK' : 'API down'}</span>
    </span>

    <span class="font-mono text-xs text-zinc-500">{data.apiBase}</span>
  </header>

  <!-- Content -->
  <div class="flex min-h-0 flex-1">
    {#if showSidebar}
      <ClassSidebar selectedId={selectedClassId} onselect={selectClass} />
    {/if}
    <main class="min-h-0 flex-1 overflow-auto">
      {@render children?.()}
    </main>
  </div>
</div>

<ShortcutOverlay />
<Toast />
