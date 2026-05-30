<script lang="ts">
  /**
   * Quantization comparison panel for the /bakeoff cockpit.
   *
   * Surfaces the lighter/faster-vs-accuracy trade-off for our model's
   * quantized variants (produced by export/quantize.py + the Mac CoreML
   * workflow), reusing the same matrix.json the cockpit already loads — no
   * new endpoint. Variants are matrix rows named ours_<precision>_<target>;
   * we filter to those and show size, accuracy, ΔmAP vs FP32, and latency so
   * an operator can see which build to ship (FP16 ONNX is the default).
   */
  import { type BakeoffMatrix } from '$lib/api';
  import DetectorChip from '$components/DetectorChip.svelte';

  interface Props {
    matrix: BakeoffMatrix;
    dataset: string;
  }
  let { matrix, dataset }: Props = $props();

  interface Variant {
    name: string;
    precision: 'fp32' | 'fp16' | 'int8';
    target: 'onnx' | 'coreml';
    detector: string; // for the runtime chip
    order: number;
    size: number | null;
    map: number | null;
    apSmall: number | null;
    latency: number | null;
    delta: number | null; // mAP@.5:.95 vs the FP32 baseline
  }

  const PREC_LABEL: Record<string, string> = {
    fp32: 'FP32',
    fp16: 'FP16',
    int8: 'INT8',
  };
  const TARGET_LABEL: Record<string, string> = {
    onnx: 'ONNX (NVIDIA)',
    coreml: 'CoreML (Apple)',
  };
  const ORDER: Record<string, number> = {
    fp32_onnx: 0,
    fp16_onnx: 1,
    int8_onnx: 2,
    fp16_coreml: 3,
    int8_coreml: 4,
  };

  function cellVal(model: string, metric: string): number | null {
    const v = matrix?.cells?.[model]?.[dataset]?.[metric];
    return typeof v === 'number' ? v : null;
  }

  // The FP32 reference for ΔmAP: prefer the FP32 ONNX variant, else the .pt run.
  const fp32Key = $derived(
    matrix.models.find((m) => m.includes('ours_fp32_onnx')) ??
      matrix.models.find((m) => m.includes('ours') && m.includes('[full]')) ??
      matrix.models.find((m) => m === 'ours'),
  );
  const fp32Map = $derived(fp32Key ? cellVal(fp32Key, 'map_50_95') : null);

  const variants = $derived.by<Variant[]>(() => {
    const re = /ours_(fp32|fp16|int8)_(onnx|coreml)/;
    const out: Variant[] = [];
    for (const name of matrix.models) {
      const m = re.exec(name);
      if (!m) continue;
      const precision = m[1] as Variant['precision'];
      const target = m[2] as Variant['target'];
      const map = cellVal(name, 'map_50_95');
      out.push({
        name,
        precision,
        target,
        detector: target === 'coreml' ? 'coreml' : 'onnxruntime',
        order: ORDER[`${precision}_${target}`] ?? 99,
        size: cellVal(name, 'size_mb'),
        map,
        apSmall: cellVal(name, 'ap_small'),
        latency: cellVal(name, 'latency_ms'),
        delta: map != null && fp32Map != null ? map - fp32Map : null,
      });
    }
    return out.sort((a, b) => a.order - b.order);
  });

  const pct = (v: number | null): string => (v == null ? '—' : (v * 100).toFixed(1));
  const mb = (v: number | null): string => (v == null ? '—' : v.toFixed(1));
  const ms = (v: number | null): string => (v == null ? '—' : v.toFixed(0));
  const deltaPts = (v: number | null): string =>
    v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}`;
  const deltaCls = (v: number | null): string =>
    v == null ? 'text-zinc-500' : v >= -0.005 ? 'text-emerald-300' : v >= -0.02 ? 'text-amber-300' : 'text-red-300';
</script>

{#if variants.length}
  <section class="mt-6 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4">
    <h2 class="text-sm font-medium text-zinc-300">Quantization — lighter &amp; faster</h2>
    <p class="mb-3 text-xs text-zinc-500">
      Size / accuracy / latency by precision on <code>{dataset}</code>, with ΔmAP@.5:.95 vs FP32.
      <strong class="text-indigo-200">FP16 ONNX</strong> is the shipped NVIDIA artifact; INT8 is the
      lighter edge / Apple option. Lower size + latency = better.
    </p>
    <div class="overflow-x-auto rounded border border-zinc-800">
      <table class="w-full text-sm">
        <thead class="bg-zinc-900 text-xs uppercase text-zinc-400">
          <tr>
            <th class="px-3 py-2 text-left">Precision</th>
            <th class="px-2 py-2 text-left">Target</th>
            <th class="px-2 py-2 text-right">Size (MB)</th>
            <th class="px-2 py-2 text-right">mAP@.5:.95</th>
            <th class="px-2 py-2 text-right">AP small</th>
            <th class="px-2 py-2 text-right">ΔmAP (pp)</th>
            <th class="px-2 py-2 text-right">Latency (ms)</th>
          </tr>
        </thead>
        <tbody>
          {#each variants as v (v.name)}
            <tr class="border-t border-zinc-800 hover:bg-zinc-800/40">
              <td class="px-3 py-2 font-mono text-xs">{PREC_LABEL[v.precision]}</td>
              <td class="px-2 py-2">
                <DetectorChip detector={v.detector} size="sm" />
                <span class="ml-1 text-[10px] text-zinc-500">{TARGET_LABEL[v.target]}</span>
              </td>
              <td class="px-2 py-2 text-right">{mb(v.size)}</td>
              <td class="px-2 py-2 text-right">{pct(v.map)}</td>
              <td class="px-2 py-2 text-right text-zinc-400">{pct(v.apSmall)}</td>
              <td class="px-2 py-2 text-right {deltaCls(v.delta)}">{deltaPts(v.delta)}</td>
              <td class="px-2 py-2 text-right text-zinc-400">{ms(v.latency)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>
{/if}
