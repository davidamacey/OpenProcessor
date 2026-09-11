<script lang="ts">
  import '../app.css';
  import type { Snippet } from 'svelte';
  import { page } from '$app/state';
  import ClassSidebar from '$components/ClassSidebar.svelte';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
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

  // Global hotkey listener — class.hotkey_letter bindings are honored
  // across every page by routing through dropOnClassStore. The active
  // page registers its own dispatch handler (cluster page bulk-labels
  // selected crops; review page assigns the current crop). Skips when
  // a modal/input is focused so typing doesn't accidentally trigger an
  // assignment.
  $effect(() => {
    function isTextInputActive(): boolean {
      const el = document.activeElement;
      if (!el) return false;
      const tag = el.tagName.toLowerCase();
      return (
        tag === 'input' ||
        tag === 'textarea' ||
        tag === 'select' ||
        (el as HTMLElement).isContentEditable === true
      );
    }
    function onKeydown(e: KeyboardEvent): void {
      // Shift is guarded like the other modifiers: hotkey letters are
      // single lowercase chars by design, and this listener runs in
      // parallel with the keyboardStore dispatcher (preventDefault does
      // not stop the other listener). Without the guard, Shift+N would
      // both flag-for-new-class AND assign whichever class is bound to
      // 'n' — two actions from one keypress.
      if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
      if (isTextInputActive()) return;
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      const cls = classesStore.classes.find(
        (c) => !c.deprecated && (c.hotkey_letter ?? '').toLowerCase() === key,
      );
      if (cls) {
        e.preventDefault();
        // Keyboard hotkey path carries no dragged-crop context — the
        // page-level handler falls back to its `selected` set when
        // droppedIds is empty (see dropOnClass.svelte.ts).
        void dropOnClassStore.dispatch(cls, []);
      }
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  });

  const path = $derived(page.url.pathname);
  const showSidebar = $derived(path === '/clusters' || path.startsWith('/clusters/'));

  // Class filter applied to the cluster grid. Page reads from URL (?class=ID).
  const selectedClassId = $derived.by(() => {
    const v = page.url.searchParams.get('class');
    return v == null ? null : Number.isFinite(+v) ? +v : null;
  });

  function selectClass(cls: { id: number } | null): void {
    void (async () => {
      const { goto } = await import('$app/navigation');
      // In the legacy ensemble, cluster_id == class_id, so clicking a
      // class in the sidebar navigates straight to that class's cluster
      // instead of filtering the current page. Behaves the same on /classes
      // since /clusters/{id} is the canonical view.
      if (cls) {
        // license_plate is not a cluster — plates are sub-bboxes on
        // vehicle crops (plate_bbox_norm). Route to the gallery branch
        // backed by /curation/plates so the operator sees every plate-bearing
        // crop, not just the 1-2 rows whose PRIMARY class is license_plate.
        const lpClass = classesStore.classes.find(
          (c) => (c.name ?? '').toLowerCase() === 'license_plate',
        );
        if (lpClass && cls.id === lpClass.id) {
          void goto(`/clusters?class=${cls.id}`, {
            replaceState: false,
            keepFocus: true,
          });
          return;
        }
        void goto(`/clusters/${cls.id}`, {
          replaceState: false,
          keepFocus: true,
        });
        return;
      }
      const url = new URL(page.url);
      url.searchParams.delete('class');
      history.pushState({}, '', url);
      void goto(url.pathname + url.search, {
        replaceState: false,
        keepFocus: true,
      });
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
      <span class="rounded bg-blue-600 px-1.5 py-0.5 font-mono text-xs text-white"
        >KB</span
      >
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
      <a href="/dashboard" class="hover:text-white">Dashboard</a>
      <a href="/clusters" class="hover:text-white">Clusters</a>
      <a href="/review" class="hover:text-white">Review</a>
      <a href="/classes" class="hover:text-white">Classes</a>
      <a href="/export" class="hover:text-white">Export</a>
      <a href="/models" class="hover:text-white">Models</a>
      <a href="/train" class="hover:text-white">Train</a>
      <a href="/bakeoff" class="hover:text-white">Bake-off</a>
    </nav>

    <button
      type="button"
      class="flex items-center gap-1.5 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-white"
      onclick={() => keyboardStore.toggleOverlay()}
      title="Keyboard shortcuts (~)"
    >
      <span class="font-mono">?</span>
      <span class="hidden sm:inline">shortcuts</span>
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
      <ClassSidebar
        selectedId={selectedClassId}
        onselect={selectClass}
        ondrop={(cls, ids) => void dropOnClassStore.dispatch(cls, ids)}
      />
    {/if}
    <main class="min-h-0 flex-1 overflow-auto">
      {@render children?.()}
    </main>
  </div>
</div>

<ShortcutOverlay />
<Toast />
