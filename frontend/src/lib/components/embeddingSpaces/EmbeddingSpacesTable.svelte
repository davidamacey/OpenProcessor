<script lang="ts">
  /**
   * The spaces of one target as served: encoder, size, role, status, when it became ready, the
   * documents holding a vector, the run followed in this tab, and the writes a row offers. A
   * primary space offers neither promote nor retire; every other refusal is the server's.
   */
  import ReindexRunProgress from '$components/embeddingSpaces/ReindexRunProgress.svelte';
  import { formatTimestamp } from '$lib/format/date';
  import type {
    EmbeddingSpacesState,
    SpaceAction,
  } from '$lib/embeddingSpaces/embeddingSpacesController.svelte';
  import type { EmbeddingSpaceView } from '$lib/types_embeddingSpaces';

  interface Props {
    state: EmbeddingSpacesState;
    target: string;
    spaces: EmbeddingSpaceView[];
    onaction: (action: SpaceAction, space: EmbeddingSpaceView) => void;
    onevaluation: (space: EmbeddingSpaceView) => void;
  }

  let { state: st, target, spaces, onaction, onevaluation }: Props = $props();
</script>

<section
  class="surface relative min-w-0 overflow-x-auto p-4"
  aria-label="{target} spaces"
  data-testid="space-table"
  data-target={target}
>
  <h2 class="mb-2 text-base font-semibold">{target} spaces</h2>
  <table class="w-full text-left text-sm">
    <thead class="text-xs text-zinc-500">
      <tr>
        <th class="py-1 pr-3 font-normal">Space</th>
        <th class="py-1 pr-3 font-normal">Encoder</th>
        <th class="py-1 pr-3 font-normal">Size</th>
        <th class="py-1 pr-3 font-normal">Role</th>
        <th class="py-1 pr-3 font-normal">Status</th>
        <th class="py-1 pr-3 font-normal">Ready</th>
        <th class="py-1 pr-3 font-normal">Documents with a vector</th>
        <th class="py-1 pr-3 font-normal">Build</th>
        <th class="py-1 font-normal"><span class="sr-only">Actions</span></th>
      </tr>
    </thead>
    <tbody>
      {#each spaces as s (s.space_key)}
        {@const run = st.runOf(s)}
        <tr
          class="border-t border-zinc-800 align-top"
          data-testid="space-row"
          data-space={s.space_key}
        >
          <td class="py-1.5 pr-3 font-mono text-xs">{s.space_key}</td>
          <td class="py-1.5 pr-3 text-xs">{s.model_key} v{s.model_version}</td>
          <td class="py-1.5 pr-3 font-mono text-xs">{s.dim}</td>
          <td class="py-1.5 pr-3 text-xs" data-testid="space-role">{s.role}</td>
          <td class="py-1.5 pr-3 text-xs" data-testid="space-status">{s.status}</td>
          <td class="py-1.5 pr-3 text-xs text-zinc-400"
            >{s.ready_at ? formatTimestamp(s.ready_at) : '—'}</td
          >
          <td class="py-1.5 pr-3 font-mono text-xs">{s.docs_with_vector ?? '—'}</td>
          <td class="py-1.5 pr-3">
            {#if run}
              <ReindexRunProgress {run} />
            {:else if s.run_status}
              <span
                class="flex flex-wrap items-center gap-2 text-xs"
                data-testid="last-run"
              >
                <span
                  class="rounded border border-zinc-700 px-1.5 text-[11px] text-zinc-200"
                  data-testid="last-run-status">{s.run_status}</span
                >
                <span class="text-zinc-400">{s.run_mode}</span>
              </span>
            {:else}
              <span class="text-xs text-zinc-500">—</span>
            {/if}
          </td>
          <td class="py-1.5 text-right whitespace-nowrap">
            <button type="button" class="btn btn-sm" onclick={() => onaction('build', s)}
              >Restart or backfill</button
            >
            {#if st.isFollowing(s)}
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => onaction('cancel', s)}>Cancel build</button
              >
            {/if}
            {#if s.role !== 'primary'}
              <button
                type="button"
                class="btn btn-sm"
                data-testid="space-evaluation"
                onclick={() => onevaluation(s)}>Evaluation</button
              >
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => onaction('promote', s)}>Promote</button
              >
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => onaction('retire', s)}>Retire</button
              >
            {/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>
