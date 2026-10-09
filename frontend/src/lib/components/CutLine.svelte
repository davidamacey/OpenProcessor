<script lang="ts">
  interface Props {
    label?: string;
  }
  let { label = 'low similarity below' }: Props = $props();
</script>

<div class="my-4 flex items-center gap-3 col-span-full" aria-hidden="false">
  <div class="h-px flex-1 bg-zinc-700"></div>
  <span class="font-mono text-xs tracking-wide text-zinc-500 uppercase">— {label} —</span>
  <div class="h-px flex-1 bg-zinc-700"></div>
</div>
