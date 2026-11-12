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
  <h2 class="mb-2 text-title font-semibold">{target} spaces</h2>
  <table class="w-full text-left text-body">
    <thead class="text-meta text-fg-3">
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
          class="border-t border-line align-top"
          data-testid="space-row"
          data-space={s.space_key}
        >
          <td class="py-1.5 pr-3 font-mono text-meta">{s.space_key}</td>
          <td class="py-1.5 pr-3 text-meta">{s.model_key} v{s.model_version}</td>
          <td class="py-1.5 pr-3 font-mono text-meta">{s.dim}</td>
          <td class="py-1.5 pr-3 text-meta" data-testid="space-role">{s.role}</td>
          <td class="py-1.5 pr-3 text-meta" data-testid="space-status">{s.status}</td>
          <td class="py-1.5 pr-3 text-meta text-fg-2"
            >{s.ready_at ? formatTimestamp(s.ready_at) : '—'}</td
          >
          <td class="py-1.5 pr-3 font-mono text-meta">{s.docs_with_vector ?? '—'}</td>
          <td class="py-1.5 pr-3">
            {#if run}
              <ReindexRunProgress {run} />
            {:else if s.run_status}
              <span
                class="flex flex-wrap items-center gap-2 text-meta"
                data-testid="last-run"
              >
                <span
                  class="rounded border border-line-strong px-1.5 text-caption text-fg-1"
                  data-testid="last-run-status">{s.run_status}</span
                >
                <span class="text-fg-2">{s.run_mode}</span>
              </span>
            {:else}
              <span class="text-meta text-fg-3">—</span>
            {/if}
          </td>
          <td class="py-1.5">
            <div class="flex flex-wrap justify-end gap-1">
              <button
                type="button"
                class="btn btn-sm"
                onclick={() => onaction('build', s)}>Restart or backfill</button
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
                  data-testid="space-evaluation-open"
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
            </div>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</section>
