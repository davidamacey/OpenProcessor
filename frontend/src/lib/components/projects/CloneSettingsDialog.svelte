<script lang="ts">
  /**
   * Copy settings from another project into this one:
   * `POST /projects/{slug}/clone_settings {from, axes, expected_revision}`.
   * The source list is the served selectable projects; the axes are the
   * served `limits.cloneable_axes`, all ticked by default. Refusals such
   * as `target_not_empty` (classes only copy into an empty project) or
   * `revision_conflict` render their served message.
   */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import type { ProjectsAdmin } from '$lib/projects/projectsAdminController.svelte';
  import type { ProjectSummary } from '$lib/types_projects';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    project: ProjectSummary | null;
    admin: ProjectsAdmin;
    onclose: () => void;
  }
  let { project, admin, onclose }: Props = $props();

  let from = $state('');
  let axes = $state<string[]>([]);
  let busy = $state(false);
  let errorText = $state<string | null>(null);

  const sources = $derived(
    admin.list.filter((p) => p.selectable && p.slug !== project?.slug),
  );
  const axisOptions = $derived(admin.limits?.cloneable_axes ?? []);

  // Reset only when the dialog opens or its target changes. The early
  // return keeps a list reload (including the one after this very submit)
  // from re-reading `axisOptions` here and wiping the operator's choices.
  let initFor: string | null = null;
  $effect(() => {
    const slug = project?.slug ?? null;
    if (slug === initFor) return;
    initFor = slug;
    if (!slug) return;
    from = '';
    axes = [...axisOptions];
    errorText = null;
  });

  function toggle(axis: string, on: boolean): void {
    axes = on
      ? [...axes.filter((a) => a !== axis), axis]
      : axes.filter((a) => a !== axis);
  }

  async function submit(): Promise<void> {
    if (!project || !from) return;
    busy = true;
    errorText = null;
    const ordered = axisOptions.filter((a) => axes.includes(a));
    const res = await admin.cloneSettings(project, from, ordered);
    busy = false;
    if (res.ok) {
      toastStore.success(`Copied settings into "${res.project.display_name}".`);
      onclose();
    } else {
      errorText = res.message;
    }
  }
</script>

{#if project}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Copy settings"
    use:focusOnMount
    use:trapFocus={{ onEscape: onclose }}
    tabindex="-1"
    data-testid="clone-settings-dialog"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">
        Copy settings into {project.display_name}
      </h3>
      <form
        onsubmit={async (e) => {
          e.preventDefault();
          await submit();
        }}
      >
        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">From project</span>
          <select
            bind:value={from}
            data-testid="clone-settings-from"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100"
          >
            <option value="" disabled>Choose a project…</option>
            {#each sources as s (s.slug)}
              <option value={s.slug}>{s.display_name} ({s.slug})</option>
            {/each}
          </select>
        </label>
        <fieldset class="mb-3 text-sm">
          <legend class="mb-1 text-xs text-zinc-400">What to copy</legend>
          {#each axisOptions as a (a)}
            <label class="flex items-center gap-2 text-zinc-300">
              <input
                type="checkbox"
                checked={axes.includes(a)}
                data-testid="clone-settings-axis-{a}"
                onchange={(e) => toggle(a, (e.currentTarget as HTMLInputElement).checked)}
              />
              <span class="font-mono text-xs">{a}</span>
            </label>
          {/each}
        </fieldset>
        {#if errorText}
          <p class="mb-3 text-xs text-red-300" data-testid="clone-settings-error">
            {errorText}
          </p>
        {/if}
        <div class="flex items-center justify-end gap-2">
          <button type="button" class="btn" onclick={onclose} disabled={busy}
            >Cancel</button
          >
          <button
            type="submit"
            class="btn btn-primary"
            data-testid="clone-settings-submit"
            disabled={busy || !from || axes.length === 0}
          >
            {busy ? 'Copying…' : 'Copy settings'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}
