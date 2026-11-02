<!--
  The project's active ingest profile, read-only: the profile names the
  detector an ingest runs. It is edited through the ingest-profile routes,
  not here.
-->
<script lang="ts">
  import type { ActiveConfigResponse } from '$lib/types_config';

  let { profile, error }: { profile: ActiveConfigResponse | null; error: string | null } =
    $props();

  const DOCS =
    'https://github.com/davidamacey/OpenProcessor/blob/main/docs/CURATION.md#ingest-profiles';
</script>

<section class="surface space-y-2 p-4" data-testid="active-ingest-profile">
  <h2 class="text-sm font-semibold">Detector (ingest profile)</h2>
  {#if error}
    <p class="text-xs text-red-300" data-testid="active-ingest-profile-error">{error}</p>
  {:else if profile === null}
    <p class="text-xs text-zinc-400">Loading the active ingest profile.</p>
  {:else if profile.active.name === null}
    <p class="text-xs text-zinc-300" data-testid="active-ingest-profile-none">
      No ingest profile is active (source {profile.source}); ingest is refused until one
      is activated.
    </p>
  {:else}
    <p class="text-xs text-zinc-300" data-testid="active-ingest-profile-name">
      Active profile
      <span class="font-mono text-zinc-100">{profile.active.name}</span>
      {#if profile.active.revision !== null}(revision {profile.active.revision}){/if}.
    </p>
  {/if}
  <p class="text-xs text-zinc-500">
    The detector a project ingests with is chosen by its active ingest profile, not by
    this policy. See
    <a class="text-blue-300 hover:underline" href={DOCS} target="_blank" rel="noopener"
      >Ingest profiles</a
    >.
  </p>
</section>
