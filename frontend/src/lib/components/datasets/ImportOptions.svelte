<script lang="ts">
  /**
   * Import options (W10.14 `DatasetImportOptions`). Every enum control
   * renders the served vocabulary; an option left at "Server default" is
   * omitted from the request, so the server's own default applies (plan
   * §8 question 6).
   */
  import type { ImportWizard } from '$lib/datasets/importWizardController.svelte';
  import type { DatasetFormatsResponse, DatasetImportOptions } from '$lib/types_import';

  interface Props {
    wizard: ImportWizard;
    formats: DatasetFormatsResponse;
  }

  let { wizard, formats }: Props = $props();

  // `DatasetImportOptions.missing_label`'s wire values; the server serves
  // no labels for them yet (question 6).
  const MISSING_LABEL_VALUES = ['unlabeled', 'negative'];

  function setText(key: 'name' | 'source_tag', v: string): void {
    wizard.setOption(key, v.trim() === '' ? undefined : v);
  }

  function setTri(key: 'freeze_test_split' | 'region_negatives', v: string): void {
    const value = v === '' ? undefined : v === 'true';
    wizard.setOption(key, value as DatasetImportOptions[typeof key]);
  }

  function triValue(v: boolean | null | undefined): string {
    return v == null ? '' : String(v);
  }
</script>

<div class="space-y-4 text-xs" data-testid="import-options">
  <fieldset>
    <legend class="mb-1 font-semibold text-zinc-300">Processing</legend>
    <label class="flex items-start gap-2 py-0.5">
      <input
        type="radio"
        name="processing"
        checked={wizard.options.processing === undefined}
        onchange={() => wizard.setOption('processing', undefined)}
      />
      <span class="text-zinc-400">Server default</span>
    </label>
    {#each formats.processing_modes as p (p.value)}
      <label class="flex items-start gap-2 py-0.5">
        <input
          type="radio"
          name="processing"
          value={p.value}
          checked={wizard.options.processing === p.value}
          onchange={() => wizard.setOption('processing', p.value)}
        />
        <span>
          <span class="text-zinc-200">{p.label}</span>
          {#if p.description}<span class="block text-zinc-500">{p.description}</span>{/if}
        </span>
      </label>
    {/each}
  </fieldset>

  <div class="grid gap-3 sm:grid-cols-2">
    <label class="block">
      <span class="mb-0.5 block text-zinc-400">Label trust</span>
      <select
        class="select select-sm w-full"
        value={wizard.options.label_trust ?? ''}
        onchange={(e) =>
          wizard.setOption(
            'label_trust',
            (e.currentTarget as HTMLSelectElement).value || undefined,
          )}
      >
        <option value="">Server default</option>
        {#each formats.trust_levels as t (t.value)}
          <option value={t.value}>{t.label}</option>
        {/each}
      </select>
    </label>

    <label class="block">
      <span class="mb-0.5 block text-zinc-400">Freeze the test split as the holdout</span>
      <select
        class="select select-sm w-full"
        aria-label="Freeze the test split as the holdout"
        value={triValue(wizard.options.freeze_test_split)}
        onchange={(e) =>
          setTri('freeze_test_split', (e.currentTarget as HTMLSelectElement).value)}
      >
        <option value="">The dataset's default</option>
        <option value="true">Freeze it</option>
        <option value="false">Don't freeze it</option>
      </select>
      <span class="mt-0.5 block text-zinc-500">
        Frozen test images are held out: they never enter training.
      </span>
    </label>

    {#if wizard.preview?.region}
      <label class="block">
        <span class="mb-0.5 block text-zinc-400">Region parents</span>
        <select
          class="select select-sm w-full"
          value={wizard.options.parents ?? ''}
          onchange={(e) =>
            wizard.setOption(
              'parents',
              (e.currentTarget as HTMLSelectElement).value || undefined,
            )}
        >
          <option value="">Server default</option>
          {#each formats.parents_modes as p (p.value)}
            <option value={p.value}>{p.label}</option>
          {/each}
        </select>
      </label>
    {/if}

    <label class="block">
      <span class="mb-0.5 block text-zinc-400">Import name</span>
      <input
        class="input input-sm w-full"
        value={wizard.options.name ?? ''}
        onchange={(e) => setText('name', (e.currentTarget as HTMLInputElement).value)}
      />
    </label>
    <label class="block">
      <span class="mb-0.5 block text-zinc-400">Source tag</span>
      <input
        class="input input-sm w-full"
        value={wizard.options.source_tag ?? ''}
        onchange={(e) =>
          setText('source_tag', (e.currentTarget as HTMLInputElement).value)}
      />
    </label>
  </div>

  <details>
    <summary class="cursor-pointer text-zinc-400">Advanced</summary>
    <div class="mt-2 grid gap-3 sm:grid-cols-3">
      <label class="block">
        <span class="mb-0.5 block text-zinc-400">Images without a label file</span>
        <select
          class="select select-sm w-full"
          value={wizard.options.missing_label ?? ''}
          onchange={(e) =>
            wizard.setOption(
              'missing_label',
              (e.currentTarget as HTMLSelectElement).value || undefined,
            )}
        >
          <option value="">Server default</option>
          {#each MISSING_LABEL_VALUES as v (v)}
            <option value={v}>{v}</option>
          {/each}
        </select>
      </label>
      <label class="block">
        <span class="mb-0.5 block text-zinc-400">Reviewed-negative region parents</span>
        <select
          class="select select-sm w-full"
          value={triValue(wizard.options.region_negatives)}
          onchange={(e) =>
            setTri('region_negatives', (e.currentTarget as HTMLSelectElement).value)}
        >
          <option value="">Server default</option>
          <option value="true">Yes</option>
          <option value="false">No</option>
        </select>
      </label>
      <label class="block">
        <span class="mb-0.5 block text-zinc-400">Region containment</span>
        <input
          type="number"
          step="0.05"
          class="input input-sm w-full"
          placeholder="Server default"
          value={wizard.options.region_containment ?? ''}
          onchange={(e) => {
            const raw = (e.currentTarget as HTMLInputElement).value;
            wizard.setOption('region_containment', raw === '' ? undefined : Number(raw));
          }}
        />
      </label>
    </div>
  </details>

  {#if wizard.preview?.force_allowed}
    <label class="flex items-center gap-2 text-amber-200" data-testid="force-toggle">
      <input
        type="checkbox"
        checked={wizard.options.force === true}
        onchange={(e) =>
          wizard.setOption(
            'force',
            (e.currentTarget as HTMLInputElement).checked || undefined,
          )}
      />
      Start anyway (the blocking issues above can be forced)
    </label>
  {/if}
</div>
