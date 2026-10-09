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
  import { formatLatencyMs } from '$lib/formatLatency';
  import { formatTimestamp } from '$lib/formatDate';
  import type { VlmProbeResult } from '$lib/types_vlm';

  interface Props {
    probe: VlmProbeResult;
  }

  let { probe }: Props = $props();

  const yn = (v: boolean | null | undefined): string =>
    v == null ? '—' : v ? 'yes' : 'no';
  const num = (v: number | null | undefined): string => (v == null ? '—' : String(v));
</script>

<div class="flex flex-col gap-2 text-xs" data-testid="vlm-probe">
  <div class="flex flex-wrap items-center gap-2">
    <span
      class="rounded border px-1.5 py-0.5 {probe.ok
        ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
        : 'border-red-500/40 bg-red-500/10 text-red-200'}"
      data-testid="vlm-probe-ok">{probe.ok ? 'probe ok' : 'probe failed'}</span
    >
    <span class="text-zinc-500" title={probe.probed_at}
      >probed {formatTimestamp(probe.probed_at)}</span
    >
  </div>
  <dl class="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-0.5">
    <dt class="text-zinc-500">Latency</dt>
    <dd class="font-mono">{formatLatencyMs(probe.latency_ms)}</dd>
    <dt class="text-zinc-500">Models listed</dt>
    <dd class="break-all font-mono">
      {(probe.models_listed ?? []).length > 0 ? probe.models_listed!.join(', ') : '—'}
    </dd>
    <dt class="text-zinc-500">Configured model listed</dt>
    <dd class="font-mono">{yn(probe.model_listed)}</dd>
    <dt class="text-zinc-500">Root</dt>
    <dd class="break-all font-mono">{probe.root ?? '—'}</dd>
    <dt class="text-zinc-500">Max model length</dt>
    <dd class="font-mono">{num(probe.max_model_len)}</dd>
    <dt class="text-zinc-500">Reads images</dt>
    <dd class="font-mono">{yn(probe.vision_ok)}</dd>
    <dt class="text-zinc-500">JSON mode</dt>
    <dd class="font-mono">{yn(probe.json_mode_supported)}</dd>
    <dt class="text-zinc-500">Reasoning channel</dt>
    <dd class="font-mono">{yn(probe.reasoning_channel)}</dd>
    <dt class="text-zinc-500">Tokens per image</dt>
    <dd class="font-mono">{num(probe.image_tokens)}</dd>
    <dt class="text-zinc-500">Max images per call ok</dt>
    <dd class="font-mono">{yn(probe.max_images_ok)}</dd>
  </dl>
  <ConfigIssueList issues={probe.issues ?? []} showField />
</div>
