<script lang="ts">
  import { keyboardStore } from '$stores/keyboard.svelte';
  $effect(() => {
    keyboardStore.setScope('export');
  });
</script>

<div class="mx-auto max-w-3xl space-y-4 p-8">
  <h1 class="text-2xl font-semibold tracking-tight">Export dataset</h1>
  <p class="text-sm text-zinc-400">
    The full export UI (per-class balance gap, augmentation plan, training command preview)
    arrives in <strong>v1.1</strong>. The Dashboard's "Export Dataset" button already triggers
    a YOLO export job; the snippet below is for scripting.
  </p>

  <pre
    class="overflow-auto rounded-md border border-zinc-800 bg-zinc-900 p-4 text-xs leading-relaxed text-zinc-200"><code
>{`# trigger export
curl -s -X POST http://localhost:4603/op/export/yolo | jq

# poll status
curl -s http://localhost:4603/op/export/status | jq

# class registry & data yaml
curl -s -OJ http://localhost:4603/op/export/registry/class_registry.json
curl -s -OJ http://localhost:4603/op/export/registry/data_v7.yaml`}</code></pre>
</div>
