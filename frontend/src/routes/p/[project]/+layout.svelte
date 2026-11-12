<script lang="ts">
  /**
   * The project-scoped app shell: the top bar (`TopNav`) and every
   * project-scoped store subscription. Rendered only once `+layout.ts` resolved the
   * `/p/<slug>` segment to a selectable served project; otherwise the
   * "not available" page renders instead and no scoped call fires.
   */
  import type { Snippet } from 'svelte';
  import { untrack } from 'svelte';
  import ProjectUnavailable from '$components/ProjectUnavailable.svelte';
  import TopNav from '$components/shell/TopNav.svelte';
  import type { ProjectResolution } from '$stores/projects.svelte';
  import { slotRegistryWarnings } from '$lib/annotations/registeredSlots';
  import { isPickerHiddenClass } from '$lib/classVisibility';
  import { dropOnClassStore } from '$stores/dropOnClass.svelte';
  import ShortcutOverlay from '$components/ShortcutOverlay.svelte';
  import { classesStore } from '$stores/classes.svelte';
  import { classSourcesStore } from '$stores/classSources.svelte';
  import { keyboardStore } from '$stores/keyboard.svelte';
  import { keymapAvailability, loadKeymap } from '$stores/keymap.svelte';
  import { subscribeConfigEvents } from '$lib/sse';
  import { projectsStore } from '$stores/projects.svelte';
  import { regionProfileStore } from '$stores/regionProfile.svelte';
  import { regionStatusesStore } from '$stores/regionStatuses.svelte';
  import { regionVocabularyStore } from '$stores/regionVocabulary.svelte';
  import { reviewTabsVocabularyStore } from '$stores/reviewTabsVocabulary.svelte';
  import { healthStore } from '$stores/health.svelte';
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

  // K2 (configurable keyboard shortcuts plan
  // §5.1): the served keymap applies live, with no reload — a
  // `config.changed axis=keymap` frame refetches `GET {prefix}/keymap`
  // and swaps `keymapStore`'s document in place; every dispatch resolves
  // an action's keys through the store at keypress time, so a rebind
  // takes effect immediately. Only subscribed once the keymap route is
  // known to exist (`available !== false`) — a backend without the keymap route has no
  // `config.changed axis=keymap` event to wait for anyway.
  $effect(() => {
    if (!slug || keymapAvailability.available === false) return;
    const sub = subscribeConfigEvents(
      (ev) => {
        if (ev.type !== 'config.changed') return;
        if ((ev as { axis?: string }).axis !== 'keymap') return;
        void loadKeymap().then(() => {
          toastStore.info('Keyboard shortcuts updated');
        });
      },
      () => void loadKeymap(),
    );
    return () => sub.close();
  });
</script>

{#if !active}
  <!-- An unknown slug, or one the server marks not selectable: a clear
       page with a way out, and no scoped call ever fires. The root
       layout already handles an unreachable project list. -->
  {#if data.resolution && data.resolution.kind !== 'ok'}
    <ProjectUnavailable resolution={data.resolution} />
  {/if}
{:else}
  <div class="flex h-screen flex-col bg-surface-0 text-fg-1">
    <TopNav apiBase={data.apiBase} {appName} {appBadge} />

    <!-- Content. Keyed on the region profile's seedVersion: the
       slot registry is a plain module binding, so when a slow boot left
       the profile unknown and a later /health poll seeds it, re-mounting
       the page is what makes the region tab and other region surfaces
       appear without a reload. A normal boot seeds before first render,
       so this never re-mounts in the common case. -->
    {#if !active.writable}
      <!-- Served `writable: false` (e.g. an archived project): reads
           work, every write answers 409 server-side. -->
      <div
        class="shrink-0 border-b border-warn-line bg-warn-soft px-4 py-1.5 text-xs text-warn-text"
        data-testid="project-read-only-banner"
      >
        {active.display_name} is {projectsStore.statusLabel(active.status).toLowerCase()} and
        read-only: changes are refused by the server.
      </div>
    {/if}

    <!-- Keyed on the project too: SvelteKit reuses a page instance across
       a `/p/a/workspace` -> `/p/b/workspace` param change, so without the slug
       in the key the previous project's page-local state (queue, cursor,
       selection) would survive a switch. -->
    {#key `${active.slug}:${regionProfileStore.seedVersion}`}
      <div class="flex min-h-0 flex-1 flex-col md:flex-row">
        <main class="min-h-0 flex-1 overflow-auto">
          {@render children?.()}
        </main>
      </div>
    {/key}
  </div>

  <ShortcutOverlay />
{/if}
