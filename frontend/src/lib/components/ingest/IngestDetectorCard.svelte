<!--
  Track B (v0.4.0 generic detector): the served ingest detector and policy
  from `GET {API_PREFIX}/ingest/config`, on `/ingest`. Empty until the
  track lands.
-->
<script lang="ts">
  import type { IngestConfig } from '$lib/types';

  let { config: _config }: { config: IngestConfig } = $props();
</script>
