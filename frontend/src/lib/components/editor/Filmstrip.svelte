<script lang="ts">
  /**
   * The neighbours of the open image in the list it came from, as served: the one before, then the
   * next few. Each is a link; the strip scrolls and shows when more is off-screen.
   */
  import { resolve } from '$app/paths';
  import ScrollStrip from '$components/ScrollStrip.svelte';
  import { getImageThumbUrl } from '$lib/api_images';

  interface Props {
    before: string | null;
    after: string[];
    currentId: string;
    hrefFor: (imageId: string) => string;
  }

  let { before, after, currentId, hrefFor }: Props = $props();

  const ids = $derived([...(before ? [before] : []), currentId, ...after]);
</script>

<ScrollStrip
  navLabel="Images in this list"
  activeKey={currentId}
  class="gap-2"
  testId="filmstrip"
>
  {#each ids as id (id)}
    {@const current = id === currentId}
    <a
      href={resolve(hrefFor(id))}
      class="block h-14 w-20 shrink-0 overflow-hidden rounded-md border-2 focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:outline-none {current
        ? 'border-sky-400'
        : 'border-zinc-800 opacity-80 hover:opacity-100'}"
      aria-current={current ? 'true' : undefined}
      data-active={current ? 'true' : undefined}
      aria-label={current ? 'This image' : `Open image ${id}`}
      data-image-id={id}
    >
      <img
        src={getImageThumbUrl(id, 96)}
        alt=""
        loading="lazy"
        class="h-full w-full object-cover"
      />
    </a>
  {/each}
</ScrollStrip>
