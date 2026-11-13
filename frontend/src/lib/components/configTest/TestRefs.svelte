<script lang="ts">
  /**
   * The served refs of a test run: which pack/profile and which VLM
   * answered, and how long it took. Rendered as served; the VLM's
   * `endpoint` is its own `name@revision`.
   */
  import { formatLatencyMs } from '$lib/format/latency';
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

<dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-meta" data-testid="test-refs">
  {#if pack}
    <dt class="text-fg-3">{packLabel}</dt>
    <dd data-testid="test-pack-ref"><code>{refText(pack)}</code></dd>
  {/if}
  {#if vlm}
    <dt class="text-fg-3">VLM</dt>
    <dd data-testid="test-vlm-ref">
      <code>{vlm.endpoint}</code> · <code>{vlm.model}</code>{vlm.draft ? ' (draft)' : ''}
    </dd>
  {/if}
  {#if latencyMs != null}
    <dt class="text-fg-3">Latency</dt>
    <dd class="tabular-nums" data-testid="test-latency">{formatLatencyMs(latencyMs)}</dd>
  {/if}
</dl>
