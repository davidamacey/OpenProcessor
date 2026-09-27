<script lang="ts">
  /**
   * Create a project: slug, display name, description. The slug rules
   * shown are the served `limits` (pattern, length, reserved and retired
   * slugs) as hints only — the server decides, and its refusal
   * (`slug_invalid`, `slug_taken`, `slug_retired`,
   * `shard_budget_exceeded`, a pydantic 422) renders verbatim. A served
   * `capacity.status === 'blocked'` disables Create; `warn` shows the
   * served message and still allows it (the 201's `warnings` then toast).
   */
  import { focusOnMount } from '$lib/actions/focusOnMount';
  import { trapFocus } from '$lib/actions/trapFocus';
  import type { ProjectsAdmin } from '$lib/projects/projectsAdminController.svelte';
  import { toastStore } from '$stores/toast.svelte';

  interface Props {
    open: boolean;
    admin: ProjectsAdmin;
    onclose: () => void;
  }
  let { open, admin, onclose }: Props = $props();

  let slug = $state('');
  let displayName = $state('');
  let description = $state('');
  let busy = $state(false);
  let errorText = $state<string | null>(null);

  $effect(() => {
    if (!open) return;
    slug = '';
    displayName = '';
    description = '';
    errorText = null;
  });

  const limits = $derived(admin.limits);
  const capacity = $derived(admin.capacity);
  const blocked = $derived(capacity?.status === 'blocked');

  /** Served-rule hints for the typed slug; never blocks the submit. */
  const slugHint = $derived.by(() => {
    const s = slug.trim();
    if (!s || !limits) return null;
    if (limits.reserved_slugs.includes(s)) return `"${s}" is reserved.`;
    if (limits.retired_slugs?.includes(s))
      return `"${s}" belonged to a deleted project and is retired.`;
    let re: RegExp | null;
    try {
      re = new RegExp(limits.slug_pattern);
    } catch {
      re = null;
    }
    if (s.length < limits.slug_min || s.length > limits.slug_max || (re && !re.test(s))) {
      return `Doesn't match the served rule (${limits.slug_min}–${limits.slug_max} characters, ${limits.slug_pattern}).`;
    }
    return null;
  });

  async function submit(): Promise<void> {
    busy = true;
    errorText = null;
    const res = await admin.create({
      slug: slug.trim(),
      display_name: displayName.trim(),
      description: description.trim(),
    });
    busy = false;
    if (res.ok) {
      toastStore.success(`Created project "${res.project.display_name}".`);
      onclose();
    } else {
      errorText = res.message;
    }
  }
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <div
    class="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
    role="dialog"
    aria-modal="true"
    aria-label="Create project"
    use:focusOnMount
    use:trapFocus={{ onEscape: onclose }}
    tabindex="-1"
    data-testid="create-project-dialog"
    onclick={(e) => {
      if (e.target === e.currentTarget) onclose();
    }}
  >
    <div
      class="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
    >
      <h3 class="mb-3 text-base font-semibold">Create project</h3>
      <form
        onsubmit={async (e) => {
          e.preventDefault();
          await submit();
        }}
      >
        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400"
            >Slug (used in the URL, can't change)</span
          >
          <input
            type="text"
            bind:value={slug}
            required
            autocomplete="off"
            data-testid="create-project-slug"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 font-mono text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
          {#if slugHint}
            <span
              class="mt-1 block text-[11px] text-amber-300"
              data-testid="create-project-slug-hint">{slugHint}</span
            >
          {:else if limits}
            <span class="mt-1 block text-[11px] text-zinc-500"
              >{limits.slug_min}–{limits.slug_max} characters, {limits.slug_pattern}</span
            >
          {/if}
        </label>
        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">Display name</span>
          <input
            type="text"
            bind:value={displayName}
            required
            data-testid="create-project-name"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="mb-3 block text-sm">
          <span class="mb-1 block text-xs text-zinc-400">Description (optional)</span>
          <textarea
            bind:value={description}
            rows="2"
            data-testid="create-project-description"
            class="w-full rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
          ></textarea>
        </label>

        {#if capacity && capacity.status !== 'ok'}
          <p
            class="mb-3 rounded border px-2 py-1.5 text-xs {blocked
              ? 'border-red-900 bg-red-950/40 text-red-200'
              : 'border-amber-900 bg-amber-950/40 text-amber-200'}"
            data-testid="create-project-capacity"
          >
            <span class="font-semibold"
              >{capacity.labels[capacity.status] ?? capacity.status}.</span
            >
            {capacity.message}
          </p>
        {/if}

        {#if errorText}
          <p class="mb-3 text-xs text-red-300" data-testid="create-project-error">
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
            data-testid="create-project-submit"
            disabled={busy || blocked || slug.trim() === '' || displayName.trim() === ''}
          >
            {busy ? 'Creating…' : 'Create'}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}
