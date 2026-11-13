<script lang="ts">
  /**
   * A served probe result (`VlmProbeResult`): `ok`, latency, the models the
   * endpoint listed, whether the configured one is among them, the served
   * root and context length, and what the probe found (vision, JSON mode,
   * reasoning channel, image tokens, max images). Every value is the
   * server's; an unknown (`null`) fact prints "—". The probe's own issues
   * render through the shared issue list.
   */
  import ConfigIssueList from '$components/config/ConfigIssueList.svelte';
  import { formatLatencyMs } from '$lib/format/latency';
  import { formatTimestamp } from '$lib/format/date';
  import type { VlmProbeResult } from '$lib/types_vlm';

  interface Props {
    probe: VlmProbeResult;
  }

  let { probe }: Props = $props();

  const yn = (v: boolean | null | undefined): string =>
    v == null ? '—' : v ? 'yes' : 'no';
  const num = (v: number | null | undefined): string => (v == null ? '—' : String(v));
</script>

<div class="flex flex-col gap-2 text-meta" data-testid="vlm-probe">
  <div class="flex flex-wrap items-center gap-2">
    <span
      class="rounded border px-1.5 py-0.5 {probe.ok
        ? 'border-success/40 bg-success/10 text-success-text'
        : 'border-danger/40 bg-danger/10 text-danger-text'}"
      data-testid="vlm-probe-ok">{probe.ok ? 'probe ok' : 'probe failed'}</span
    >
    <span class="text-fg-3" title={probe.probed_at}
      >probed {formatTimestamp(probe.probed_at)}</span
    >
  </div>
  <dl class="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-0.5">
    <dt class="text-fg-3">Latency</dt>
    <dd class="tabular-nums">{formatLatencyMs(probe.latency_ms)}</dd>
    <dt class="text-fg-3">Models listed</dt>
    <dd class="break-all">
      <code
        >{(probe.models_listed ?? []).length > 0
          ? probe.models_listed!.join(', ')
          : '—'}</code
      >
    </dd>
    <dt class="text-fg-3">Configured model listed</dt>
    <dd>{yn(probe.model_listed)}</dd>
    <dt class="text-fg-3">Root</dt>
    <dd class="break-all"><code>{probe.root ?? '—'}</code></dd>
    <dt class="text-fg-3">Max model length</dt>
    <dd class="tabular-nums">{num(probe.max_model_len)}</dd>
    <dt class="text-fg-3">Reads images</dt>
    <dd>{yn(probe.vision_ok)}</dd>
    <dt class="text-fg-3">JSON mode</dt>
    <dd>{yn(probe.json_mode_supported)}</dd>
    <dt class="text-fg-3">Reasoning channel</dt>
    <dd>{yn(probe.reasoning_channel)}</dd>
    <dt class="text-fg-3">Tokens per image</dt>
    <dd class="tabular-nums">{num(probe.image_tokens)}</dd>
    <dt class="text-fg-3">Max images per call ok</dt>
    <dd>{yn(probe.max_images_ok)}</dd>
  </dl>
  <ConfigIssueList issues={probe.issues ?? []} showField />
</div>
