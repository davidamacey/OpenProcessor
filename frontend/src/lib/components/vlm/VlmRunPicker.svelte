<script lang="ts">
  /**
   * Per-run VLM endpoint picker (W9; docs/design/w9-p4-w5-w10-ui-plan-2026-10-01.md
   * §3.3). A `<select>` over the served `/methods` `vlm` axis: "Project
   * default" (sends nothing) and every served entry, labelled with its
   * served endpoint status. When the chosen entry's served
   * `per_run_ack_required` is true, the served `warning` shows in a banner
   * with an "I understand" checkbox bound to `acknowledgeExternal`. The run
   * button is never gated on it: the server is the gate and its refusal is
   * shown verbatim by the caller.
   *
   * Absent, not disabled, when the axis serves nothing to pick
   * (`isVlmSelectable`); the axis comes from the cached `/methods` load,
   * no request of its own.
   */
  import { isVlmSelectable, pickableVlmEntries } from '$lib/strategies';
  import { strategiesStore } from '$stores/strategies.svelte';

  interface Props {
    /** The chosen entry id, or `null` = the project's default. */
    vlm: string | null;
    acknowledgeExternal: boolean;
    disabled?: boolean;
    /** The no-pick option's label: "Project default" on a run, "Active
     *  endpoint" on a test (both send nothing). */
    defaultLabel?: string;
    onchange: (next: { vlm: string | null; acknowledgeExternal: boolean }) => void;
  }

  let {
    vlm,
    acknowledgeExternal,
    disabled = false,
    defaultLabel = 'Project default',
    onchange,
  }: Props = $props();

  $effect(() => {
    void strategiesStore.init();
  });

  const available = $derived(isVlmSelectable(strategiesStore.methods));
  const entries = $derived(pickableVlmEntries(strategiesStore.methods.vlm));
  const chosen = $derived(entries.find((e) => e.id === vlm) ?? null);
  const needsAck = $derived(chosen?.per_run_ack_required === true);

  function pick(id: string): void {
    // A new pick starts without an acknowledgement.
    onchange({ vlm: id === '' ? null : id, acknowledgeExternal: false });
  }
</script>

{#if available}
  <div class="inline-flex flex-wrap items-center gap-1.5" data-testid="vlm-run-picker">
    <label class="flex items-center gap-1.5">
      <span class="text-zinc-500">VLM</span>
      <select
        class="select-sm"
        value={vlm ?? ''}
        {disabled}
        data-testid="vlm-run-select"
        onchange={(e) => pick((e.currentTarget as HTMLSelectElement).value)}
      >
        <option value="">{defaultLabel}</option>
        {#each entries as e (e.id)}
          <option value={e.id}
            >{e.label}{e.endpoint_status_label
              ? ` · ${e.endpoint_status_label}`
              : ''}</option
          >
        {/each}
      </select>
    </label>
    {#if needsAck}
      <div
        class="flex basis-full flex-col gap-1 rounded border border-red-500/40 bg-red-500/10 px-2 py-1 text-red-200"
        data-testid="vlm-run-ack"
      >
        {#if chosen?.warning}<p>{chosen.warning}</p>{/if}
        <label class="flex items-center gap-2">
          <input
            type="checkbox"
            checked={acknowledgeExternal}
            {disabled}
            data-testid="vlm-run-ack-checkbox"
            onchange={(e) =>
              onchange({
                vlm,
                acknowledgeExternal: (e.currentTarget as HTMLInputElement).checked,
              })}
          />
          I understand crops will be sent outside this deployment
        </label>
      </div>
    {/if}
  </div>
{/if}
