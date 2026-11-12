<script lang="ts">
  /** `/p/<slug>/images/<image_id>`: the image editor. The Workspace query rides along in the URL. */
  import { page } from '$app/state';
  import ImageEditor from '$components/editor/ImageEditor.svelte';
  import type { PageProps } from './$types';

  let { params }: PageProps = $props();
</script>

{#key params.image_id}
  <ImageEditor imageId={params.image_id} search={page.url.search} />
{/key}
