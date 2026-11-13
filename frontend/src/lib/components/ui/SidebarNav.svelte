<script lang="ts">
  /**
   * A grouped vertical nav (the Settings sections): group headings, links with an optional
   * icon and count, the current one marked by an accent bar and `aria-current`. Below 768px
   * it collapses to a select, so a long list never takes the screen. `href`s are app paths
   * from `projectHref`.
   */
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import Badge from './Badge.svelte';
  import Icon from './Icon.svelte';
  import type { IconName } from './iconRegistry';

  export interface SidebarItem {
    id: string;
    label: string;
    href: string;
    icon?: IconName;
    count?: number | string;
  }
  export interface SidebarGroup {
    heading: string;
    items: SidebarItem[];
  }
  interface Props {
    groups: SidebarGroup[];
    current: string;
    label: string;
  }
  let { groups, current, label }: Props = $props();

  function onchange(e: Event): void {
    const item = groups
      .flatMap((g) => g.items)
      .find((i) => i.id === (e.target as HTMLSelectElement).value);
    if (item) void goto(resolve(item.href as '/'));
  }
</script>

<select
  class="select w-full md:hidden"
  aria-label={label}
  value={current}
  {onchange}
  data-testid="sidebar-select"
>
  {#each groups as g (g.heading)}
    <optgroup label={g.heading}>
      {#each g.items as item (item.id)}
        <option value={item.id}>{item.label}</option>
      {/each}
    </optgroup>
  {/each}
</select>

<nav class="hidden flex-col gap-4 md:flex" aria-label={label} data-testid="sidebar-nav">
  {#each groups as g (g.heading)}
    <div>
      <p class="mb-1 px-3 text-meta font-medium text-fg-2">
        {g.heading}
      </p>
      {#each g.items as item (item.id)}
        <a
          href={resolve(item.href as '/')}
          aria-current={item.id === current ? 'page' : undefined}
          class="flex items-center gap-2 rounded-control border-l-2 px-3 py-1.5 text-control {item.id ===
          current
            ? 'border-accent bg-accent-soft text-fg-1'
            : 'border-transparent text-fg-2 hover:bg-surface-2 hover:text-fg-1'}"
        >
          {#if item.icon}<Icon name={item.icon} size={14} />{/if}
          <span class="min-w-0 flex-1 truncate">{item.label}</span>
          {#if item.count !== undefined}<Badge>{item.count}</Badge>{/if}
        </a>
      {/each}
    </div>
  {/each}
</nav>
