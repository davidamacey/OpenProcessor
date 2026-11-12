<script lang="ts">
  /**
   * Register a candidate space: an encoder the deployment can serve (the served catalog; one that
   * cannot be served now is listed with its reason and cannot be chosen), a target, and, when the
   * encoder serves smaller sizes, the size. The request carries one `Idempotency-Key` for the whole
   * dialog, so a double click answers the first job.
   */
  import ConfirmDialog from '$components/ConfirmDialog.svelte';
  import type { EmbeddingSpacesState } from '$lib/embeddingSpaces/embeddingSpacesController.svelte';
  import { EMBEDDING_TARGETS, type EmbeddingTarget } from '$lib/types_embeddingSpaces';

  interface Props {
    state: EmbeddingSpacesState;
    onregistered: (spaceKey: string) => void;
    oncancel: () => void;
  }

  let { state: spaces, onregistered, oncancel }: Props = $props();

  const idempotencyKey = `register-${crypto.randomUUID()}`;
  const models = $derived(spaces.models ?? []);
  let choice = $state('');
  let target = $state<EmbeddingTarget>('item');
  let dim = $state('');

  const picked = $derived(models.find((m) => `${m.model_key}@${m.version}` === choice));

  async function run(): Promise<void> {
    if (!picked) return;
    const ok = await spaces.register(
      {
        model_key: picked.model_key,
        model_version: picked.version,
        target,
        dim: dim === '' ? null : Number(dim),
      },
      idempotencyKey,
    );
    if (ok) onregistered(`${picked.model_key} v${picked.version}`);
  }
</script>

<ConfirmDialog
  title="Register an embedding space"
  confirmLabel="Register and build"
  busy={spaces.busy}
  confirmDisabled={!picked || !picked.available}
  onconfirm={() => void run()}
  {oncancel}
>
  {#if models.length === 0}
    <p class="text-fg-2" data-testid="register-no-models">
      {spaces.modelsError ?? 'This deployment serves no encoder to build a space from.'}
    </p>
  {/if}
  <label class="flex flex-col gap-1 text-xs text-fg-2">
    Encoder
    <select class="input input-sm" bind:value={choice} data-testid="register-model">
      <option value="" disabled>Choose an encoder</option>
      {#each models as m (`${m.model_key}@${m.version}`)}
        <option value={`${m.model_key}@${m.version}`} disabled={!m.available}>
          {m.model_key} v{m.version}{m.available
            ? ''
            : ` (unavailable${m.unavailable_reason ? `: ${m.unavailable_reason}` : ''})`}
        </option>
      {/each}
    </select>
  </label>
  <label class="flex flex-col gap-1 text-xs text-fg-2">
    Target
    <select class="input input-sm" bind:value={target} data-testid="register-target">
      {#each EMBEDDING_TARGETS as t (t)}<option value={t}>{t}</option>{/each}
    </select>
  </label>
  {#if picked && picked.matryoshka_dims.length > 0}
    <label class="flex flex-col gap-1 text-xs text-fg-2">
      Size
      <select class="input input-sm" bind:value={dim} data-testid="register-dim">
        <option value="">Native ({picked.native_dim})</option>
        {#each picked.matryoshka_dims as d (d)}<option value={String(d)}>{d}</option
          >{/each}
      </select>
    </label>
  {/if}
  {#if spaces.refusal}
    <p
      class="text-danger-text"
      data-testid="register-refusal"
      data-code={spaces.refusal.code}
    >
      {spaces.refusal.message}
    </p>
  {/if}
</ConfirmDialog>
