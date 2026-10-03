<!--
  The served ingest detector and policy from `GET {API_PREFIX}/ingest/config`,
  on `/ingest`. Read-only: every value is the served one. The policy is edited
  on `/settings/ingest-policy`.
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import { projectHref } from '$lib/projectPaths';
  import type { IngestConfig } from '$lib/types';

  let { config }: { config: IngestConfig } = $props();

  const detector = $derived(config.detector);
  const policy = $derived(config.policy ?? null);

  const detectFacts = $derived.by(() => {
    const d = policy?.detect;
    if (!d) return [] as string[];
    const out: string[] = [];
    if (d.min_confidence != null) out.push(`min confidence ${d.min_confidence}`);
    if (d.min_box_area_frac != null) out.push(`min box area ${d.min_box_area_frac}`);
    if (d.max_per_image != null) out.push(`max ${d.max_per_image} per image`);
    if (d.classes != null) out.push(`only ${d.classes.length} classes`);
    if (d.exclude_classes?.length) out.push(`excluding ${d.exclude_classes.length}`);
    return out;
  });
</script>

<section
  class="space-y-2 rounded border border-zinc-800 bg-zinc-900/40 p-3 text-sm"
  data-testid="ingest-detector-card"
>
  <h2 class="font-medium text-zinc-200">Detector</h2>
  {#if detector === null}
    <p class="text-zinc-400">No detector reported.</p>
  {:else}
    <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
      <dt class="text-zinc-500">Model</dt>
      <dd class="font-mono text-zinc-200">
        {detector.model} (version {detector.version})
      </dd>
      <dt class="text-zinc-500">Input size</dt>
      <dd class="text-zinc-200">{detector.input_size}</dd>
      <dt class="text-zinc-500">Assigns a class</dt>
      <dd class="text-zinc-200" data-testid="detector-assigns-class">
        {detector.assigns_class ? 'yes' : 'no'}
      </dd>
      <dt class="text-zinc-500">Confidence floor applies</dt>
      <dd class="text-zinc-200">{detector.confidence_floor_applies ? 'yes' : 'no'}</dd>
    </dl>
    <details class="text-xs">
      <summary class="cursor-pointer text-zinc-300">{detector.n_labels} labels</summary>
      <table class="mt-2 w-full text-left">
        <thead class="text-zinc-500">
          <tr><th class="pr-3">id</th><th class="pr-3">name</th><th>slug</th></tr>
        </thead>
        <tbody>
          {#each detector.labels as l (l.class_id)}
            <tr class="border-t border-zinc-900">
              <td class="pr-3 font-mono text-zinc-400">{l.class_id}</td>
              <td class="pr-3 text-zinc-200">{l.name}</td>
              <td class="font-mono text-zinc-400">{l.slug}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </details>
  {/if}
  {#if policy}
    <p class="text-xs text-zinc-400" data-testid="detector-policy-summary">
      Embedding: {policy.embedding?.mode ?? 'all'}{#if detectFacts.length > 0}; detect
        filter: {detectFacts.join(', ')}{/if}.
      <a
        class="text-blue-300 hover:underline"
        href={resolve(projectHref('/settings/ingest-policy'))}>Edit the ingest policy</a
      >
    </p>
  {/if}
</section>
