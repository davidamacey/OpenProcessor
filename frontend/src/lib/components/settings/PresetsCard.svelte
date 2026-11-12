<!--
  The `/settings` card linking to the research presets and the embedding spaces. The serving profile
  is deployment-wide; the rest belong to this project.
-->
<script lang="ts">
  import { resolve } from '$app/paths';
  import { PRESET_KINDS, presetPath } from '$lib/presets/presetKinds';
  import { projectHref } from '$lib/projectPaths';
</script>

<section class="surface flex flex-col gap-3 p-5" data-testid="presets-card">
  <div class="flex min-w-0 flex-col gap-1">
    <h2 class="text-base font-semibold">Presets and embedding spaces</h2>
    <p class="text-xs text-zinc-400">
      Revisioned settings for serving, clustering, the assist loop and near-duplicates,
      and the vector spaces this project searches and clusters in.
    </p>
  </div>
  <ul class="grid gap-2 sm:grid-cols-2">
    {#each PRESET_KINDS as k (k.id)}
      <li class="flex flex-wrap items-center gap-2">
        <a class="btn" href={resolve(projectHref(presetPath(k)))}>{k.title}</a>
        {#if k.scope === 'deployment'}
          <span class="text-[11px] text-zinc-400">deployment-wide</span>
        {/if}
      </li>
    {/each}
    <li>
      <a class="btn" href={resolve(projectHref('/settings/embedding-spaces'))}
        >Embedding spaces</a
      >
    </li>
  </ul>
</section>
