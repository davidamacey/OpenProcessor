<script lang="ts">
  /**
   * The sort control of a Workspace view. The options and the applied sort
   * are the server's; when it applied another sort than the one asked for,
   * the served reason is shown instead of silently re-ordering.
   */
  interface SortOption {
    id: string;
    label: string;
  }

  interface Props {
    options: SortOption[];
    applied: string;
    fallbackReason?: string | null;
    onchange: (id: string) => void;
    testid?: string;
  }

  let {
    options,
    applied,
    fallbackReason = null,
    onchange,
    testid = 'sort-select',
  }: Props = $props();
</script>

<label class="flex items-center gap-2 text-xs text-fg-2">
  <span>Sort</span>
  <select
    class="select"
    value={applied}
    onchange={(e) => onchange(e.currentTarget.value)}
    data-testid={testid}
  >
    {#each options as o (o.id)}
      <option value={o.id}>{o.label}</option>
    {/each}
  </select>
  {#if fallbackReason}
    <span class="text-warn" data-testid="sort-fallback">{fallbackReason}</span>
  {/if}
</label>
