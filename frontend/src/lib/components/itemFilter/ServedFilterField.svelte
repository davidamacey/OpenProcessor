<script lang="ts">
  /**
   * One filter control drawn from a served `ReviewFilterSpec`, chosen by its
   * `kind` and nothing else: `enum` (one `<select>`), `multi_enum` (toggle
   * chips), `class_names` (an add-a-class `<select>` over the registry),
   * `bool` (any / yes / no), `number` / `integer` (a bounded input) and
   * `text`. The label, the options, the bounds and the help text are the
   * served ones; no param name is read here.
   *
   * Values are strings (a list for `multi_enum` / `class_names`); `''` and
   * `[]` mean "not set", and the owner decides what to send for them.
   */
  import { enumFilterSelection, enumServedDefault } from '$lib/review/enumFilter';
  import { classesStore } from '$stores/classes.svelte';
  import type { ReviewFilterSpec } from '$lib/api';

  type Spec = Pick<
    ReviewFilterSpec,
    'param' | 'kind' | 'label' | 'options' | 'min' | 'max' | 'description'
  >;

  interface Props {
    spec: Spec;
    value: string | string[] | undefined;
    onchange: (param: string, value: string | string[]) => void;
    /** An `enum` value to show when nothing is picked (the tab's served default). */
    servedDefault?: string | null;
    disabled?: boolean;
  }

  let { spec, value, onchange, servedDefault = null, disabled = false }: Props = $props();

  const list = $derived(Array.isArray(value) ? value : value ? [value] : []);
  const text = $derived(Array.isArray(value) ? '' : (value ?? ''));
  const classNames = $derived(
    classesStore.classes.filter((c) => !c.deprecated).map((c) => c.name),
  );
  const addable = $derived(classNames.filter((n) => !list.includes(n)));
  const enumValue = $derived(enumFilterSelection(spec, text, servedDefault));
  // With no served default the backend applies no filter, so "any" is a real
  // state; without this option the select would show a value that is not sent.
  const enumHasAny = $derived(enumServedDefault(spec, servedDefault) == null);

  function toggle(v: string): void {
    onchange(spec.param, list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  }
</script>

<!-- A label (so the control is named by its text) except for the toggle chips,
     which are several buttons. -->
<svelte:element
  this={spec.kind === 'multi_enum' ? 'div' : 'label'}
  class="flex shrink-0 items-center gap-1.5"
  data-testid="served-filter-{spec.param}"
  title={spec.description || undefined}
>
  <span class="text-zinc-400">{spec.label}</span>
  {#if spec.kind === 'enum'}
    <select
      value={enumValue}
      {disabled}
      onchange={(e) => onchange(spec.param, e.currentTarget.value)}
      class="select-sm"
    >
      {#if enumHasAny}
        <option value="">any</option>
      {/if}
      {#each spec.options as opt (opt.value)}
        <option value={opt.value}>{opt.label}</option>
      {/each}
    </select>
  {:else if spec.kind === 'multi_enum'}
    <div class="inline-flex flex-wrap gap-1">
      {#each spec.options as opt (opt.value)}
        <button
          type="button"
          class="chip {list.includes(opt.value)
            ? 'border-blue-500/60 bg-blue-600 text-white'
            : 'bg-zinc-900 text-zinc-300 hover:bg-zinc-700'}"
          aria-pressed={list.includes(opt.value)}
          {disabled}
          onclick={() => toggle(opt.value)}
        >
          {opt.label}
        </button>
      {/each}
    </div>
  {:else if spec.kind === 'class_names'}
    <select
      value=""
      {disabled}
      aria-label={spec.label}
      onchange={(e) => {
        const v = e.currentTarget.value;
        e.currentTarget.value = '';
        if (v) onchange(spec.param, [...list, v]);
      }}
      class="select-sm"
    >
      <option value="">{list.length > 0 ? `${list.length} chosen, add…` : 'any'}</option>
      {#each addable as name (name)}
        <option value={name}>{name}</option>
      {/each}
    </select>
  {:else if spec.kind === 'bool'}
    <select
      value={text}
      {disabled}
      onchange={(e) => onchange(spec.param, e.currentTarget.value)}
      class="select-sm"
    >
      <option value="">any</option>
      <option value="true">yes</option>
      <option value="false">no</option>
    </select>
  {:else if spec.kind === 'number' || spec.kind === 'integer'}
    <input
      type="number"
      min={spec.min ?? undefined}
      max={spec.max ?? undefined}
      step={spec.kind === 'integer' ? 1 : 'any'}
      value={text}
      {disabled}
      placeholder="any"
      onchange={(e) => onchange(spec.param, e.currentTarget.value)}
      class="input-sm w-20"
    />
  {:else}
    <input
      type="text"
      value={text}
      {disabled}
      placeholder="any"
      onchange={(e) => onchange(spec.param, e.currentTarget.value)}
      class="input-sm w-32"
    />
  {/if}
</svelte:element>
