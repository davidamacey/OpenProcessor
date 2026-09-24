<script lang="ts">
  import '../app.css';
  import type { Snippet } from 'svelte';
  import { untrack } from 'svelte';
  import { page } from '$app/state';
  import AboutModal from '$components/AboutModal.svelte';
  import ClassSidebar from '$components/ClassSidebar.svelte';
  import {
    slotForClassName,
    slotRegistryWarnings,
  } from '$lib/annotations/registeredSlots';
  import { bakeoffAvailability } from '$lib/bakeoffAvailability.svelte';
  import { isPickerHiddenClass } from '$lib/classVisibility';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import ShortcutOverlay from '$components/ShortcutOverlay.svelte';
  import Toast from '$components/Toast.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';
  import { healthStore } from '$stores/health.svelte';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    children?: Snippet;
    data: { apiBase: string };
  }
  let { children, data }: Props = $props();

  // Wordmark/badge are env-configurable so a rebrand (or a white-label
  // deployment) doesn't require another hardcoded string — same
  // PUBLIC_* convention as PUBLIC_TRITON_API_URL in src/lib/api.ts.
  const appName =
    (import.meta.env?.PUBLIC_APP_NAME as string | undefined) || 'Cropwright';
  const appBadge = (import.meta.env?.PUBLIC_APP_BADGE as string | undefined) || 'CW';
  let aboutOpen = $state<boolean>(false);

  // Acquire singleton-store subscriptions for the lifetime of the layout.
  $effect(() => {
    const releaseHealth = healthStore.acquire();
    const releaseClasses = classesStore.acquire();
    void classSourcesStore.init();
    void regionStatusesStore.init();
    return () => {
      releaseHealth();
      releaseClasses();
    };
  });

  // One-shot, never-rejecting probe deciding whether the /bakeoff nav
  // link renders at all — see bakeoffAvailability.svelte.ts's doc comment
  // for why a probe is safe here (idempotent read, unambiguous 404 vs.
  // "no runs yet") and why this whole mechanism is provisional.
  $effect(() => {
    void bakeoffAvailability.init();
  });

  // Tier-2 deployment-profile problems are the operator's to fix and are
  // invisible otherwise — the app has already degraded silently to the
  // built-in slots by the time this runs. One toast, not one per warning:
  // a broken file typically produces several and they are all the same
  // action item ("go fix annotation-profiles.json"). Full detail is on the
  // console via loadDeploymentProfiles().
  //
  // MUST be wrapped in `untrack()`: `toastStore.error()` reads
  // `this.toasts` (to spread it) before writing it
  // (`src/lib/stores/toast.svelte.ts`'s `push()`), and `slotRegistryWarnings`
  // is a plain, non-reactive `let` this effect otherwise reads with no
  // tracked dependency at all. Without `untrack`, the read of
  // `toastStore.toasts` inside `push()` makes THIS effect depend on
  // `toastStore.toasts` — so the very toast this effect pushes
  // immediately re-triggers it, which pushes another toast, forever.
  // `untrack` keeps the intended "runs exactly once, on mount" semantics
  // this effect's own doc comment (and the tier-2 plan's §4.6) describe.
  $effect(() => {
    if (slotRegistryWarnings.length === 0) return;
    untrack(() => {
      toastStore.error(
        `Deployment annotation profile: ${slotRegistryWarnings.length} problem(s) — see the browser console. Using built-in slots.`,
      );
    });
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
        (c) =>
          !c.deprecated &&
          !isPickerHiddenClass(c.name) &&
          (c.hotkey_letter ?? '').toLowerCase() === key,
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
      // For `cluster_kind === 'class'` clusters the backend guarantees
      // cluster_id == class_id (see `ClusterKind` in src/lib/types.ts), so
      // clicking a class in the sidebar navigates straight to that class's
      // cluster instead of filtering the current page. Behaves the same on
      // /classes since /clusters/{id} is the canonical view.
      if (cls) {
        // A class bound to a slot (e.g. license_plate) isn't a cluster —
        // plates are sub-bboxes on vehicle crops (region_bbox_norm). Route
        // to the gallery branch backed by {API_PREFIX}/regions so the operator sees
        // every slot-bearing crop, not just the 1-2 rows whose PRIMARY
        // class matches the slot's bound class name. Driven by
        // registeredSlots (P2.10) instead of a hardcoded license_plate
        // string literal so a new registered slot gets this routing for free.
        const clsName = classesStore.classes.find((c) => c.id === cls.id)?.name;
        if (slotForClassName(clsName) != null) {
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
    // No crumb at all on the root or /dashboard routes — both are "home",
    // already labeled by the title/nav; a redundant lowercase "dashboard"
    // breadcrumb next to them added nothing.
    if (parts.length === 0 || path === '/dashboard') return [];
    // No leading "Home" crumb either — the top-right nav's own
    // "Dashboard" link already covers that, and having both was redundant.
    const out: Array<{ label: string; href: string }> = [];
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
      ? `API OK at ${data.apiBase || window.location.host} (last checked ${healthStore.lastChecked ? new Date(healthStore.lastChecked).toLocaleTimeString() : '—'})`
      : `API unavailable at ${data.apiBase || window.location.host}: ${healthStore.error ?? 'no response'}`,
  );
</script>

<div class="flex h-screen flex-col bg-zinc-950 text-zinc-100">
  <!-- Top bar -->
  <header
    class="flex h-12 shrink-0 items-center gap-4 border-b border-zinc-800 bg-zinc-950 px-4"
  >
    <div class="flex items-center gap-2 text-sm font-semibold tracking-tight">
      <button
        type="button"
        class="flex shrink-0 items-center justify-center rounded border border-zinc-700 transition-transform duration-150 hover:scale-110 hover:border-zinc-500"
        onclick={() => (aboutOpen = true)}
        aria-label="About {appName}"
        title="About {appName}"
      >
        <svg viewBox="0 0 128 128" class="h-6 w-6" role="img" aria-label={appBadge}>
          <rect width="128" height="128" rx="24" fill="#09090b" />
          <rect x="26" y="70" width="30" height="30" rx="5" fill="#3f3f46" />
          <rect x="60" y="70" width="30" height="30" rx="5" fill="#3f3f46" />
          <rect x="26" y="34" width="30" height="30" rx="5" fill="#60a5fa" />
          <rect x="60" y="34" width="30" height="30" rx="5" fill="#f59e0b" />
          <path
            d="M33 49 l6 6 l12 -12"
            fill="none"
            stroke="#09090b"
            stroke-width="4"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </button>
      <a href="/dashboard" class="hover:text-white">{appName}</a>
    </div>

    <AboutModal open={aboutOpen} onclose={() => (aboutOpen = false)} {appName} />

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
      {#if bakeoffAvailability.available !== false}
        <a href="/bakeoff" class="hover:text-white">Bake-off</a>
      {/if}
      <a href="/settings" class="hover:text-white">Settings</a>
    </nav>

    <span
      class="chip gap-1.5 rounded-full border-zinc-700 bg-zinc-900 text-zinc-300"
      title={dotTitle}
    >
      <span class="h-2 w-2 rounded-full {dotClass}"></span>
      <span class="font-mono">{healthStore.ok ? 'API OK' : 'API down'}</span>
    </span>
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
