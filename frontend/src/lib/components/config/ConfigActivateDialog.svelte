<script lang="ts">
  /**
   * The confirm step before activating a config doc at a pinned revision
   * (§4.4, §7.6 item 4): what is active now → what will be, the revision's
   * served report, a note when the draft has unsaved edits, and, only after
   * a 422 whose served report says `force_allowed`, "Activate anyway"
   * (`force: true`). A refusal's served message stays in the dialog.
   *
   * `ack` (a VLM endpoint that sends crops outside the deployment): the
   * served warning in a red banner and, when `required`, an "I understand"
   * checkbox whose state becomes `acknowledge_external` (sent only when
   * checked). The checkbox also appears after the server refused with
   * `vlm_external_not_acknowledged`, whatever the list said. Packs and
   * profiles pass nothing.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import type { ConfigEditorView } from '$lib/config/configEditor.svelte';
  import type { ActiveRef, ConfigDocBase } from '$lib/types_config';
  import ConfigIssueList from './ConfigIssueList.svelte';

  interface Props {
    /** What the dialog reads: the active ref and its writes, and whether
     *  the editor has unsaved edits or a revision open. */
    ed: Pick<ConfigEditorView, 'active' | 'dirty' | 'viewing'>;
    /** The doc or viewed revision being activated. */
    target: Pick<ConfigDocBase<unknown>, 'name' | 'revision' | 'validation'>;
    ack?: { warning: string | null; required: boolean };
    /** What the active doc drives, in the resource's words. */
    blurb: string;
    onclose: () => void;
    /** After a successful activation (the dialog closes itself first). */
    onactivated?: (target: Pick<ConfigDocBase<unknown>, 'name' | 'revision'>) => void;
  }

  let { ed, target, ack, blurb, onclose, onactivated }: Props = $props();

  let force = $state(false);
  let acknowledged = $state(false);
  const refused = $derived(
    ed.active.errorDetail?.error === 'vlm_external_not_acknowledged',
  );
  const showAck = $derived(ack != null && (ack.required || refused));

  const rep = $derived(ed.active.activateReport ?? target.validation);

  function refText(ref: ActiveRef | null | undefined): string {
    if (!ref || ref.name == null) return 'none';
    return ref.revision == null ? ref.name : `${ref.name} r${ref.revision}`;
  }

  async function run(): Promise<void> {
    const ok = await ed.active.activate(
      target.name,
      target.revision,
      force,
      showAck && acknowledged ? { acknowledge_external: true } : undefined,
    );
    if (ok) {
      onclose();
      onactivated?.(target);
    } else if (!ed.active.activateReport?.force_allowed) {
      force = false;
    }
  }
</script>

<ConfirmDialog
  title="Activate {target.name}{target.revision != null
    ? ` revision ${target.revision}`
    : ''}"
  confirmLabel={force ? 'Activate anyway' : 'Activate'}
  danger={force}
  busy={ed.active.busy}
  onconfirm={() => void run()}
  oncancel={() => {
    ed.active.clearAction();
    onclose();
  }}
>
  <p data-testid="activate-from-to">
    <span class="font-mono">{refText(ed.active.active?.active)}</span>
    →
    <strong class="font-mono"
      >{refText({ name: target.name, revision: target.revision })}</strong
    >
  </p>
  <p class="text-xs text-zinc-400">{blurb}</p>
  {#if ack?.warning}
    <p
      class="rounded border border-red-500/40 bg-red-500/10 px-2 py-1 text-xs text-red-200"
      data-testid="activate-external-warning"
    >
      {ack.warning}
    </p>
  {/if}
  {#if showAck}
    <label class="flex items-center gap-2 text-xs text-red-200">
      <input type="checkbox" bind:checked={acknowledged} data-testid="activate-ack" />
      I understand crops will be sent outside this deployment
    </label>
  {/if}
  {#if ed.dirty && !ed.viewing}
    <p class="text-xs text-amber-300" data-testid="activate-unsaved-note">
      This activates the saved revision; your unsaved edits are not included.
    </p>
  {/if}
  {#if rep}
    <ConfigIssueList issues={[...rep.errors, ...rep.warnings]} showField />
  {/if}
  {#if ed.active.actionError}
    <p class="text-red-300" data-testid="activate-error">{ed.active.actionError}</p>
  {/if}
  {#if ed.active.activateReport?.force_allowed}
    <label class="flex items-center gap-2 text-xs text-amber-200">
      <input type="checkbox" bind:checked={force} data-testid="activate-force" />
      Activate anyway (the server allows overriding these errors)
    </label>
  {/if}
</ConfirmDialog>
