<script lang="ts">
  import { keyboardStore } from '$stores/keyboard.svelte';
  $effect(() => {
    keyboardStore.setScope('classes');
  });
</script>

<div class="mx-auto max-w-3xl space-y-4 p-8">
  <h1 class="text-2xl font-semibold tracking-tight">Class management</h1>
  <p class="text-sm text-zinc-400">
    Coming in <strong>v1.1</strong>. Use the API directly meanwhile.
  </p>

  <pre
    class="overflow-auto rounded-md border border-zinc-800 bg-zinc-900 p-4 text-xs leading-relaxed text-zinc-200"><code
>{`# list
curl -s http://localhost:4603/op/classes | jq

# add
curl -s -X POST http://localhost:4603/op/classes \\
  -H 'Content-Type: application/json' \\
  -d '{"name":"new_class","group":"misc"}'

# rename / regroup
curl -s -X PUT http://localhost:4603/op/classes/42 \\
  -H 'Content-Type: application/json' \\
  -d '{"name":"renamed"}'

# merge
curl -s -X POST http://localhost:4603/op/classes/merge \\
  -H 'Content-Type: application/json' \\
  -d '{"from_id":7,"into_id":3}'`}</code></pre>
</div>
