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
  const axes = $derived(active.axes);
  const single = $derived(axes.length <= 1);
  let chosen = $state('');
  const axis = $derived<string | null>(
    axes.length === 0 ? null : single ? axes[0]!.axis : chosen || null,
  );
  const needsChoice = $derived(axes.length > 1 && chosen === '');
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
  {#if axes.length > 1}
    <label class="flex flex-col gap-1 text-xs text-fg-2">
      Activate for
      <select class="input input-sm" bind:value={chosen} data-testid="activate-axis">
        <option value="" disabled>Choose a target</option>
        {#each axes as a (a.axis)}<option value={a.axis}>{a.label}</option>{/each}
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
  <p class="text-xs text-fg-2">{activationReach(kind)}</p>
  {#if unsaved}
    <p class="text-xs text-warn" data-testid="activate-unsaved-note">
      This activates the saved revision; your unsaved edits are not included.
    </p>
  {/if}
  {#if active.actionError}
    <p class="text-danger" data-testid="activate-error">{active.actionError}</p>
  {/if}
</ConfirmDialog>
