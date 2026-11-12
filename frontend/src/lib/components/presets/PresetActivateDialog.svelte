<script lang="ts">
  /**
   * The confirm step before activating a preset revision: what is active now (on the chosen axis) and
   * what will be, the choice of axis where the kind has several, and the served refusal (a stale
   * `expected_active` or setting revision is a 409 shown verbatim, with the active ref re-read).
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import type { PresetActive } from '$lib/presets/presetActive.svelte';
  import { activationReach } from '$lib/presets/presetKinds';

  interface Props {
    active: PresetActive;
    target: { name: string; revision: number };
    /** The editor has unsaved edits: the saved revision is what is activated. */
    unsaved?: boolean;
    onclose: () => void;
    onactivated: (target: { name: string; revision: number }) => void;
  }

  let { active, target, unsaved = false, onclose, onactivated }: Props = $props();

  const kind = $derived(active.kind);
  const single = $derived(kind.axes.length <= 1);
  let chosen = $state('');
  const axis = $derived<string | null>(
    kind.axes.length === 0 ? null : single ? kind.axes[0]!.id : chosen || null,
  );
  const needsChoice = $derived(kind.axes.length > 1 && chosen === '');
  const current = $derived(active.entries.find((e) => e.axis === axis));

  function refText(name: string | null | undefined, revision: number | null | undefined) {
    return name ? `${name} r${revision}` : 'none';
  }

  async function run(): Promise<void> {
    const done = { ...target };
    if (await active.activate(done.name, done.revision, axis)) {
      onclose();
      onactivated(done);
    }
  }
</script>

<ConfirmDialog
  title="Activate {target.name} revision {target.revision}"
  confirmLabel="Activate"
  busy={active.busy}
  confirmDisabled={needsChoice}
  onconfirm={() => void run()}
  oncancel={() => {
    active.clearAction();
    onclose();
  }}
>
  {#if kind.axes.length > 1}
    <label class="flex flex-col gap-1 text-xs text-zinc-400">
      Activate for
      <select class="input input-sm" bind:value={chosen} data-testid="activate-axis">
        <option value="" disabled>Choose a target</option>
        {#each kind.axes as a (a.id)}<option value={a.id}>{a.label}</option>{/each}
      </select>
    </label>
  {/if}
  {#if !needsChoice}
    <p data-testid="activate-from-to">
      <span class="font-mono">{refText(current?.name, current?.revision)}</span>
      →
      <strong class="font-mono">{refText(target.name, target.revision)}</strong>
    </p>
  {/if}
  <p class="text-xs text-zinc-400">{activationReach(kind)}</p>
  {#if unsaved}
    <p class="text-xs text-amber-300" data-testid="activate-unsaved-note">
      This activates the saved revision; your unsaved edits are not included.
    </p>
  {/if}
  {#if active.actionError}
    <p class="text-red-300" data-testid="activate-error">{active.actionError}</p>
  {/if}
</ConfirmDialog>
