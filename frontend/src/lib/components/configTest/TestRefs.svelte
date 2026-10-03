<script lang="ts">
  /**
   * The served refs of a test run: which pack/profile and which VLM
   * answered, and how long it took (W5). Rendered as served; the VLM's
   * `endpoint` is its own `name@revision`.
   */
  import { formatLatencyMs } from '$lib/formatLatency';
  import { refText } from '$lib/configTest/refText';
  import type { PackTestPackRef, PackTestVlmRef } from '$lib/types_configTest';

  interface Props {
    /** The pack (pack test) or the active pack used to verify. */
    pack?: PackTestPackRef | null;
    /** Row label for `pack`. */
    packLabel?: string;
    vlm?: PackTestVlmRef | null;
    latencyMs?: number | null;
  }

  let { pack = null, packLabel = 'Pack', vlm = null, latencyMs = null }: Props = $props();
</script>

<dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs" data-testid="test-refs">
  {#if pack}
    <dt class="text-zinc-500">{packLabel}</dt>
    <dd class="font-mono" data-testid="test-pack-ref">{refText(pack)}</dd>
  {/if}
  {#if vlm}
    <dt class="text-zinc-500">VLM</dt>
    <dd class="font-mono" data-testid="test-vlm-ref">
      {vlm.endpoint} · {vlm.model}{vlm.draft ? ' (draft)' : ''}
    </dd>
  {/if}
  {#if latencyMs != null}
    <dt class="text-zinc-500">Latency</dt>
    <dd class="font-mono" data-testid="test-latency">{formatLatencyMs(latencyMs)}</dd>
  {/if}
</dl>
