<script lang="ts">
  /** The training form's collapsible hyperparameter override grid. Writes
   *  go straight into the bound overrides object the form owns. */
  import type { HyperparamOverrides } from '$lib/train/trainSpec';

  interface Props {
    hp: HyperparamOverrides;
  }

  let { hp = $bindable() }: Props = $props();

  let expanded = $state<boolean>(false);
</script>

<!-- Hyperparameters -->
<section class="rounded-md border border-zinc-800 bg-zinc-900">
  <button
    type="button"
    class="flex w-full items-center gap-3 px-3 py-2 text-left text-sm hover:bg-zinc-800"
    onclick={() => (expanded = !expanded)}
    aria-expanded={expanded}
  >
    <span class="font-mono text-xs text-zinc-500">{expanded ? '▼' : '▶'}</span>
    <span class="font-semibold text-zinc-100">Hyperparameters</span>
    <span class="grow text-xs text-zinc-400">
      epochs {hp.epochs ?? '—'} · batch {hp.batch ?? '—'} · imgsz {hp.imgsz ?? '—'}
    </span>
  </button>

  {#if expanded}
    <div
      class="grid grid-cols-2 gap-3 border-t border-zinc-800 p-3 text-sm sm:grid-cols-4"
    >
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">epochs</span>
        <input
          type="number"
          min="1"
          bind:value={hp.epochs}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 flex items-center justify-between text-xs text-zinc-400">
          <span>batch</span>
          <span class="flex items-center gap-1 text-[10px] text-zinc-500">
            <input
              type="checkbox"
              aria-label="batch: auto"
              checked={hp.batch === -1}
              onchange={(e) => (hp.batch = e.currentTarget.checked ? -1 : 64)}
            />
            auto
          </span>
        </span>
        <input
          type="number"
          min="-1"
          aria-label="batch"
          bind:value={hp.batch}
          disabled={hp.batch === -1}
          placeholder={hp.batch === -1 ? 'auto (−1)' : ''}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none disabled:opacity-50"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">imgsz</span>
        <input
          type="number"
          min="64"
          step="32"
          bind:value={hp.imgsz}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">optimizer</span>
        <input
          type="text"
          value="MuSGD"
          readonly
          class="w-full cursor-not-allowed rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 font-mono text-sm text-zinc-500"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">lr0</span>
        <input
          type="number"
          step="0.0001"
          bind:value={hp.lr0}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">lrf</span>
        <input
          type="number"
          step="0.001"
          bind:value={hp.lrf}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">momentum</span>
        <input
          type="number"
          step="0.001"
          bind:value={hp.momentum}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">weight_decay</span>
        <input
          type="number"
          step="0.00001"
          bind:value={hp.weight_decay}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">mosaic</span>
        <input
          type="number"
          step="0.001"
          min="0"
          max="1"
          bind:value={hp.mosaic}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">mixup</span>
        <input
          type="number"
          step="0.001"
          min="0"
          max="1"
          bind:value={hp.mixup}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">copy_paste</span>
        <input
          type="number"
          step="0.001"
          min="0"
          max="1"
          bind:value={hp.copy_paste}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
      <label class="block">
        <span class="mb-1 block text-xs text-zinc-400">patience</span>
        <input
          type="number"
          min="0"
          bind:value={hp.patience}
          class="w-full rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-blue-500 focus:outline-none"
        />
      </label>
    </div>
  {/if}
</section>
