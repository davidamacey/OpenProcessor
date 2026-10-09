<script lang="ts">
  /**
   * The project-scoped app shell: top bar (project switcher, breadcrumb,
   * primary nav, API chip), the class sidebar, and every project-scoped
   * store subscription. Rendered only once `+layout.ts` resolved the
   * `/p/<slug>` segment to a selectable served project; otherwise the
   * "not available" page renders instead and no scoped call fires.
   */
  import type { Snippet } from 'svelte';
  import { untrack } from 'svelte';
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import ProjectSwitcher from '$components/ProjectSwitcher.svelte';
  import ProjectUnavailable from '$components/ProjectUnavailable.svelte';
  import ResourcesMenu from '$components/ResourcesMenu.svelte';
  import { projectHref, sectionOf } from '$lib/projectPaths';
  import type { ProjectResolution } from '$stores/projects.svelte';
  import AboutModal from '$components/AboutModal.svelte';
  import ClassSidebar from '$components/ClassSidebar.svelte';
  import ScrollStrip from '$components/ScrollStrip.svelte';
  import {
    slotForClassName,
    slotRegistryWarnings,
  } from '$lib/annotations/registeredSlots';
  import { isPickerHiddenClass } from '$lib/classVisibility';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import ShortcutOverlay from '$components/ShortcutOverlay.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { keymapAvailability, loadKeymap } from '$stores/keymap.svelte';
  import { subscribeCurationEvents } from '$lib/sse';
  import { projectsStore } from '$stores/projects.svelte';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
  import { healthStore, healthChip, HEALTH_CHIP_TEXT } from '$stores/health.svelte';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    children?: Snippet;
    data: {
      apiBase: string;
      projectsError?: string | null;
      resolution: ProjectResolution | null;
    };
  }
  let { children, data }: Props = $props();

  /** The active project, once `+layout.ts` selected it. Every scoped
   *  effect below reads `slug`, so a project switch re-runs it against
   *  the new project (the stores themselves were reset by
   *  `projectsStore.select()`'s project-change hooks). */
  const active = $derived(data.resolution?.kind === 'ok' ? projectsStore.current : null);
  const slug = $derived(active?.slug ?? null);

  // Wordmark/badge are env-configurable so a rebrand (or a white-label
  // deployment) doesn't require another hardcoded string — same
  // PUBLIC_* convention as PUBLIC_TRITON_API_URL in src/lib/api.ts.
  const appName =
    (import.meta.env?.PUBLIC_APP_NAME as string | undefined) || 'Cropwright';
  const appBadge = (import.meta.env?.PUBLIC_APP_BADGE as string | undefined) || 'CW';
  let aboutOpen = $state<boolean>(false);

  // Acquire singleton-store subscriptions for the lifetime of the layout.
  // Skipped entirely on a blocking projectsError: every one of these
  // fires a scoped() call, which throws ProjectNotSelectedError with no
  // active project.
  $effect(() => {
    if (!slug) return;
    const releaseHealth = healthStore.acquire();
    const releaseClasses = classesStore.acquire();
    void classSourcesStore.init();
    // Region vocabularies only exist with a served region profile; without
    // one no region route is called at all.
    if (regionProfileStore.configured) {
      void regionStatusesStore.init();
      void regionVocabularyStore.init();
    }
    void reviewTabsVocabularyStore.init();
    return () => {
      releaseHealth();
      releaseClasses();
    };
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
        `Deployment annotation profile: ${slotRegistryWarnings.length} problem(s) — see the browser console.`,
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
      // Plan §5.3: a registered action shortcut beats a class hotkey
      // bound to the same key, in a context where the keymap declares
      // that context's `class_hotkeys_live`. This is defense in depth
      // behind the server-side reserved-hotkey check — it makes the
      // grandfathered collision case (a class bound to a letter before
      // it became reserved) deterministic: the action wins, never both.
      if (keyboardStore.hasActiveBinding(key)) return;
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

  // K2 (docs/design/configurable-keyboard-shortcuts-plan-2026-09-26.md
  // §5.1): the served keymap applies live, with no reload — a
  // `config.changed axis=keymap` frame refetches `GET {prefix}/keymap`
  // and swaps `keymapStore`'s document in place; every dispatch resolves
  // an action's keys through the store at keypress time, so a rebind
  // takes effect immediately. Only subscribed once the keymap route is
  // known to exist (`available !== false`) — a pre-W2b backend has no
  // `config.changed axis=keymap` event to wait for anyway.
  $effect(() => {
    if (!slug || keymapAvailability.available === false) return;
    const sub = subscribeCurationEvents({
      topic: 'config',
      onEvent: (ev) => {
        if (ev.type !== 'config.changed') return;
        if ((ev as { axis?: string }).axis !== 'keymap') return;
        void loadKeymap().then(() => {
          toastStore.info('Keyboard shortcuts updated');
        });
      },
    });
    return () => sub.close();
  });

  const path = $derived(page.url.pathname);
  /** The page section under `/p/<slug>/` (`review`, `clusters`, ...). */
  const section = $derived(sectionOf(path));

  function navCurrent(s: string): 'page' | undefined {
    return section === s ? 'page' : undefined;
  }
  function navLinkClass(s: string): string {
    return `shrink-0 hover:text-white${section === s ? ' text-white' : ''}`;
  }
  const showSidebar = $derived(section === 'clusters');

  // Class filter applied to the cluster grid. Page reads from URL (?class=ID).
  const selectedClassId = $derived.by(() => {
    const v = page.url.searchParams.get('class');
    return v == null ? null : Number.isFinite(+v) ? +v : null;
  });

  function selectClass(cls: { id: number } | null): void {
    void (async () => {
      // For `cluster_kind === 'class'` clusters the backend guarantees
      // cluster_id == class_id (see `ClusterKind` in src/lib/types.ts), so
      // clicking a class in the sidebar navigates straight to that class's
      // cluster instead of filtering the current page. Behaves the same on
      // /classes since /clusters/{id} is the canonical view.
      if (cls) {
        // A class bound to a slot isn't a cluster — its regions are
        // sub-bboxes on other items (region_bbox_norm). Route to the slot
        // gallery so the operator sees every slot-bearing crop, not just
        // the 1-2 rows whose PRIMARY class matches the slot's bound class
        // name. Driven by registeredSlots, so a new registered slot gets
        // this routing for free.
        const clsName = classesStore.classes.find((c) => c.id === cls.id)?.name;
        if (slotForClassName(clsName) != null) {
          void goto(resolve(projectHref(`/clusters?class=${cls.id}`)), {
            reset: false,
          });
          return;
        }
        void goto(resolve(projectHref(`/clusters/${cls.id}`)), {
          reset: false,
        });
        return;
      }
      const q = [...page.url.searchParams]
        .filter(([k]) => k !== 'class')
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('&');
      void goto(resolve(projectHref(`/clusters${q ? `?${q}` : ''}`)), {
        reset: false,
      });
    })();
  }

  // Crumbs from the path below `/p/<slug>`: the section, plus the
  // cluster id on `/clusters/<id>` (the only nested page).
  const crumbs = $derived.by(() => {
    // No crumb on /dashboard — it is "home", already labeled by the
    // title/nav; a redundant lowercase "dashboard" crumb added nothing.
    if (!section || section === 'dashboard') return [];
    const out: Array<{ label: string; href: ReturnType<typeof projectHref> }> = [
      { label: section, href: projectHref(`/${section}`) },
    ];
    const rest = path.split('/').filter(Boolean).slice(3);
    if (section === 'clusters' && rest[0]) {
      out.push({
        label: decodeURIComponent(rest[0]),
        href: projectHref(`/clusters/${rest[0]}`),
      });
    }
    return out;
  });

  const chipState = $derived(healthChip(healthStore.ok, healthStore.lastChecked));
  const dotClass = $derived(
    chipState === 'ok'
      ? 'bg-green-500'
      : chipState === 'down'
        ? 'bg-red-500'
        : 'bg-zinc-500',
  );
  const dotTitle = $derived(
    chipState === 'checking'
      ? `Checking API at ${data.apiBase || window.location.host}…`
      : healthStore.ok
        ? `API OK at ${data.apiBase || window.location.host} (last checked ${healthStore.lastChecked ? new Date(healthStore.lastChecked).toLocaleTimeString() : '—'})`
        : `API unavailable at ${data.apiBase || window.location.host}: ${healthStore.error ?? 'no response'}`,
  );
</script>

{#if !active}
  <!-- An unknown slug, or one the server marks not selectable: a clear
       page with a way out, and no scoped call ever fires. The root
       layout already handles an unreachable project list. -->
  {#if data.resolution && data.resolution.kind !== 'ok'}
    <ProjectUnavailable resolution={data.resolution} />
  {/if}
{:else}
  <div class="flex h-screen flex-col bg-zinc-950 text-zinc-100">
    <!-- Top bar -->
    <header
      class="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-b border-zinc-800 bg-zinc-950 px-4 py-1.5 md:h-12 md:flex-nowrap md:py-0"
    >
      <div class="flex shrink-0 items-center gap-2 text-sm font-semibold tracking-tight">
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
        <a
          href={resolve(projectHref('/dashboard'))}
          class="hidden hover:text-white sm:inline">{appName}</a
        >
      </div>

      <ProjectSwitcher />

      <AboutModal open={aboutOpen} onclose={() => (aboutOpen = false)} {appName} />

      <!-- F8 D10 / F-51: the crumb never shrinks (it truncated to "reviev" /
         "classe" at 800px); the primary nav strip to its right scrolls
         with a chevron instead. -->
      <nav
        class="hidden shrink-0 items-center gap-1 whitespace-nowrap text-sm sm:flex"
        aria-label="Breadcrumb"
        data-testid="breadcrumb"
      >
        {#each crumbs as c, i (c.href)}
          {#if i > 0}
            <span class="shrink-0 text-zinc-600">/</span>
          {/if}
          <a
            href={resolve(c.href)}
            class="shrink-0 rounded px-1.5 py-0.5 text-zinc-300 hover:bg-zinc-900 hover:text-white"
            aria-current={i === crumbs.length - 1 ? 'page' : undefined}
          >
            {c.label}
          </a>
        {/each}
      </nav>

      <span class="grow"></span>

      <!-- Narrow widths (~800px and below): this used to be a plain
         `flex` row with no shrink/overflow control, so "Bake-off"
         wrapped onto two lines and pushed the "API OK" chip past the
         viewport edge (clipped, and forcing horizontal page overflow).
         Scrolls horizontally within its own box instead of ever
         wrapping link text or growing past its flex slot. -->
      <!-- Visual audit 2026-09-24: the scrolling strip alone gave no hint
         that links sat off-screen at 800px — ScrollStrip adds a chevron
         on the side with hidden links and scrolls the current page's link
         into view. -->
      <ScrollStrip
        navLabel="Primary"
        navClass="order-last basis-full md:order-none md:basis-auto"
        activeKey={path}
        class="gap-3 text-sm text-zinc-300"
        testId="primary-nav"
      >
        <a
          href={resolve(projectHref('/dashboard'))}
          class={navLinkClass('dashboard')}
          aria-current={navCurrent('dashboard')}>Dashboard</a
        >
        <a
          href={resolve(projectHref('/ingest'))}
          class={navLinkClass('ingest')}
          aria-current={navCurrent('ingest')}>Ingest</a
        >
        <a
          href={resolve(projectHref('/clusters'))}
          class={navLinkClass('clusters')}
          aria-current={navCurrent('clusters')}>Clusters</a
        >
        <a
          href={resolve(projectHref('/review'))}
          class={navLinkClass('review')}
          aria-current={navCurrent('review')}>Review</a
        >
        <a
          href={resolve(projectHref('/audit'))}
          class={navLinkClass('audit')}
          aria-current={navCurrent('audit')}>Audit</a
        >
        <a
          href={resolve(projectHref('/classes'))}
          class={navLinkClass('classes')}
          aria-current={navCurrent('classes')}>Classes</a
        >
        <a
          href={resolve(projectHref('/export'))}
          class={navLinkClass('export')}
          aria-current={navCurrent('export')}>Export</a
        >
        <a
          href={resolve(projectHref('/models'))}
          class={navLinkClass('models')}
          aria-current={navCurrent('models')}>Models</a
        >
        <a
          href={resolve(projectHref('/train'))}
          class={navLinkClass('train')}
          aria-current={navCurrent('train')}>Train</a
        >
        <a
          href={resolve(projectHref('/bakeoff'))}
          class={navLinkClass('bakeoff')}
          aria-current={navCurrent('bakeoff')}>Bake-off</a
        >
        <a
          href={resolve(projectHref('/settings'))}
          class={navLinkClass('settings')}
          aria-current={navCurrent('settings')}>Settings</a
        >
      </ScrollStrip>

      <ResourcesMenu />

      <span
        class="chip shrink-0 gap-1.5 rounded-full border-zinc-700 bg-zinc-900 text-zinc-300"
        title={dotTitle}
      >
        <span class="h-2 w-2 rounded-full {dotClass}"></span>
        <span class="font-mono" data-testid="api-health-chip"
          >{HEALTH_CHIP_TEXT[chipState]}</span
        >
      </span>
    </header>

    <!-- Content. Keyed on the region profile's seedVersion (F-78): the
       slot registry is a plain module binding, so when a slow boot left
       the profile unknown and a later /health poll seeds it, re-mounting
       the page is what makes the region tab and other region surfaces
       appear without a reload. A normal boot seeds before first render,
       so this never re-mounts in the common case. -->
    {#if !active.writable}
      <!-- Served `writable: false` (e.g. an archived project): reads
           work, every write answers 409 server-side. -->
      <div
        class="shrink-0 border-b border-amber-900/60 bg-amber-950/40 px-4 py-1.5 text-xs text-amber-200"
        data-testid="project-read-only-banner"
      >
        {active.display_name} is {projectsStore.statusLabel(active.status).toLowerCase()} and
        read-only: changes are refused by the server.
      </div>
    {/if}

    <!-- Keyed on the project too: SvelteKit reuses a page instance across
       a `/p/a/review` -> `/p/b/review` param change, so without the slug
       in the key the previous project's page-local state (queue, cursor,
       selection) would survive a switch. -->
    {#key `${active.slug}:${regionProfileStore.seedVersion}`}
      <div class="flex min-h-0 flex-1 flex-col md:flex-row">
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
    {/key}
  </div>

  <ShortcutOverlay />
{/if}
