<script lang="ts">
  /**
   * Rename / re-describe a project: `PATCH /projects/{slug}` with only the
   * changed fields and the served `revision` as `expected_revision`. A 409
   * `revision_conflict` (someone else changed it) shows the served message
   * and offers "Reload", which re-reads the served record — keeping what
   * the operator typed — so the next save carries the fresh revision.
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

  let base = $state<ProjectSummary | null>(null);
  let displayName = $state('');
  let description = $state('');
  let busy = $state(false);
  let errorText = $state<string | null>(null);
  let conflict = $state(false);

  $effect(() => {
    if (!project) return;
    base = project;
    displayName = project.display_name;
    description = project.description;
    errorText = null;
    conflict = false;
  });

  const changes = $derived.by(() => {
    const out: { display_name?: string; description?: string } = {};
    if (!base) return out;
    if (displayName.trim() !== base.display_name) out.display_name = displayName.trim();
    if (description.trim() !== base.description) out.description = description.trim();
    return out;
  });
  const dirty = $derived(Object.keys(changes).length > 0);

  async function submit(): Promise<void> {
    if (!base) return;
    busy = true;
    errorText = null;
    conflict = false;
    const res = await admin.edit(base, changes);
    busy = false;
    if (res.ok) {
      toastStore.success(`Saved "${res.project.display_name}".`);
      onclose();
      return;
    }
    errorText = res.message;
    conflict = res.code === 'revision_conflict';
  }

  async function reload(): Promise<void> {
    if (!base) return;
    busy = true;
    await admin.load();
    busy = false;
    const fresh = admin.find(base.slug);
    if (fresh) base = fresh;
    errorText = null;
    conflict = false;
  }
</script>

{#if project && base}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Edit project"
    use:focusOnMount
    use:trapFocus={{ onEscape: onclose }}
    tabindex="-1"
    data-testid="edit-project-dialog"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-1 text-base font-semibold">Edit project</h3>
      <p class="mb-3 font-mono text-xs text-zinc-500">{base.slug}</p>
      <form
        onsubmit={async (e) => {
          e.preventDefault();
          await submit();
        }}
      >
        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">Display name</span>
          <input
            type="text"
            bind:value={displayName}
            data-testid="edit-project-name"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">Description</span>
          <textarea
            bind:value={description}
            rows="2"
            data-testid="edit-project-description"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          ></textarea>
        </label>

        {#if errorText}
          <div class="mb-3 text-xs text-red-300" data-testid="edit-project-error">
            <p>{errorText}</p>
            {#if conflict}
              <button
                type="button"
                class="mt-1 text-blue-400 hover:underline"
                data-testid="edit-project-reload"
                onclick={() => void reload()}>Reload the latest version</button
              >
            {/if}
          </div>
        {/if}

        <div class="flex items-center justify-end gap-2">
          <button type="button" class="btn" onclick={onclose} disabled={busy}
            >Cancel</button
          >
          <button
            type="submit"
            class="btn btn-primary"
            data-testid="edit-project-submit"
            disabled={busy || !dirty || conflict}
          >
            {busy ? 'Saving…' : 'Save'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}
